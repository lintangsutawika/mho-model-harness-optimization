"""Harbor-compatible trial directories from a platoon evaluation, so harbor-era tools read platoon runs unchanged.

The old optimizer loop, the proposer, the gate (`utils/gate.py`, `utils/seq.py`) and ad-hoc analysis scripts read
harbor's per-trial layout. This module writes that layout from a `PlatoonBackend` run directory, incrementally while
it runs (`python -m harnessforge.evals run REQUEST.json --harbor-export DIR`) or once over a finished run
(`python -m harnessforge.evals export RUN_DIR DIR`):

    <dest>/<candidate>/                          a harbor job dir (what eval.sh's <cand>/eval/ was)
        config.json, result.json                 job level: n_attempts, agents, datasets / n_total_trials, stats
        <task>__r<kkk>/                          one trial (k = platoon's rollout index)
            config.json                          task.path, trial_name, agent; mtime = the trial's launch time
            result.json                          written last, once the trial finished (see `trial_result`)
            agent/mini-swe-agent.trajectory.json the agent's own trajectory (collected) or one rebuilt from platoon's
            agent/mini-swe-agent.txt             a readable transcript of that trajectory
            agent/platoon-harness.log            platoon's harness log (install output + the agent's stdout/stderr)
            verifier/test-stdout.txt             the tests' output (collected, else platoon's 4 KB tail), when known
            verifier/reward.txt                  the reward of a scored trial
            harnessforge.json                    the TrialRecord this trial was exported from

`result.json` follows harbor's `TrialResult`: `task_name` (the task.toml `[task].name`, else the task id, as harbor
reports it), `verifier_result.rewards.reward`, `agent_result.n_{input,output,cache}_tokens`, `exception_info`
(`exception_type`, `exception_message`), the phases `environment_setup` / `agent_setup` / `agent_execution` /
`verifier`, `started_at` / `finished_at`. Its exception mapping is chosen so that harbor's own rules give the same
answer as the TrialRecord:

- `utils.gate.env_failed(result)` (exception_info set and no `agent_setup`) == `TrialRecord.env_failed`: an
  env-failed trial has no `agent_setup`, every other trial has one;
- the reward harbor's readers compute (`verifier_result.rewards.reward`, missing = 0) == `TrialRecord.reward or 0`;
- env-failed trials carry harbor's infrastructure exception types (`EnvironmentStartTimeoutError`, or
  `AgentSetupTimeoutError` for a failed harness setup), which analysis scripts already drop as "never started"; the
  message names platoon's real cause.

Where the agent's trajectory and the tests' full output come from: platoon's rollout directory holds neither (its
trajectory has the model's replies, not the tool outputs; the tests' output tail is in the outcome's metadata, which
the event writer drops). `PodSingularityProvider(collect_dir=...)` (on by default in `PlatoonBackend`, option
`collect`) copies both out of each sandbox before it stops, to `<collect_dir>/<rollout id>/`; the platoon patch
`patches/platoon-outcome-metadata.patch` keeps the outcome metadata (`test_output`, `verifier_exit_code`, ...) in the
trajectory. Without either, the exporter rebuilds the trajectory from platoon's assistant turns and writes no
test-stdout. Phase times are estimates (see `phases`): platoon records the trial's start and end, and the model calls'
monotonic spans, not harbor's phase boundaries.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import tomllib
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from harnessforge.evals.reader import (COLLECTION, METADATA, ROLLOUT_PREFIX, RolloutReader, _load_json, outcome_of,
                                       root_trajectory, safe_task_id)
from harnessforge.records import TrialRecord

#: What a harbor reader drops as "the sandbox never started" (env_failed); the message carries platoon's cause.
ENV_EXCEPTION = "EnvironmentStartTimeoutError"
SETUP_EXCEPTION = "AgentSetupTimeoutError"
#: platoon failure kinds of a failed harness (agent) setup, before the agent ran.
SETUP_KINDS = frozenset({"harness_setup", "harness_config"})
TRAJECTORY_FORMAT = "mini-swe-agent-1.1"
#: The agent trajectory the harness writes (`harness.render_spec`: --output={state_dir}/trajectory.json).
AGENT_TRAJECTORY = "trajectory.json"
#: What PodSingularityProvider's collect hook writes per rollout, under <collect_dir>/<rollout id>/.
COLLECTED_TEST_STDOUT = "test-stdout.txt"
COLLECTED_SANDBOX = "sandbox.json"
NAMESPACE = uuid.UUID("6b2f8a52-6c1e-4f43-9a55-3a8c0e0b7f10")


def iso(ts: float | None) -> str | None:
    """Harbor's timestamp format (UTC, microseconds, Z)."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def span(start: float | None, end: float | None) -> dict[str, str] | None:
    if start is None or end is None:
        return None
    return {"started_at": iso(start), "finished_at": iso(max(start, end))}


def trial_name(safe_task: str, index: int) -> str:
    return f"{safe_task}__r{index:03d}"


def task_names(dataset_dir: str | Path | None) -> dict[str, str]:
    """{task dir name: the name harbor reports in result.json} (task.toml `[task].name`, else the dir name), as
    `utils.alloc.task_dirs` maps them."""
    out: dict[str, str] = {}
    if not dataset_dir or not Path(dataset_dir).is_dir():
        return out
    for d in Path(dataset_dir).iterdir():
        toml = d / "task.toml"
        if toml.is_file():
            try:
                name = (tomllib.loads(toml.read_text()).get("task") or {}).get("name")
            except (OSError, tomllib.TOMLDecodeError):
                name = None
            out[d.name] = name or d.name
    return out


# --------------------------------------------------------------------------------------------- exception mapping
def harbor_exception(record: TrialRecord, outcome: Mapping[str, Any], harness: Mapping[str, Any] | None,
                     error: str | None, finished_at: float | None) -> dict[str, Any] | None:
    """harbor's exception_info for a finished trial (None for a clean one); see the module docstring."""
    label = record.error
    kind = outcome.get("failure_kind")
    reason = outcome.get("reason")
    exit_code = (harness or {}).get("exit_code")
    timed_out = bool((harness or {}).get("timed_out"))
    detail = (error or "").strip()

    def info(kind_: str, message: str) -> dict[str, Any]:
        return {"exception_type": kind_, "exception_message": message, "exception_traceback": detail or None,
                "occurred_at": iso(finished_at)}

    if record.env_failed:
        typ = SETUP_EXCEPTION if kind in SETUP_KINDS else ENV_EXCEPTION
        return info(typ, f"[platoon {label}] " + (detail.splitlines()[0][:500] if detail else
                                                  "the agent never ran (no evidence about the candidate)"))
    if label is None:
        return None
    if "context_overflow" in label:
        # mini-swe-agent under harbor died on litellm's ContextWindowExceededError (a nonzero exit)
        return info("NonZeroAgentExitCodeError",
                    f"[platoon {label}] litellm.ContextWindowExceededError: the conversation outgrew the context")
    if "agent_timeout" in label or (label.startswith("scored:") and timed_out):
        return info("AgentTimeoutError", f"[platoon {label}] Agent execution timed out")
    if "harness_crash" in label or (label.startswith("scored:") and exit_code not in (0, None)):
        return info("NonZeroAgentExitCodeError", f"[platoon {label}] Command failed (exit {exit_code}): micro-swe-agent")
    if kind == "verifier_timeout":
        return info("VerifierTimeoutError", f"[platoon {label}] Verifier execution timed out")
    if kind in ("verifier_no_reward", "verifier_output"):
        return info("RewardFileNotFoundError", f"[platoon {label}] " + (reason or "the tests reported no reward"))
    if label.startswith("scored:"):
        return info("PlatoonGradedWithNote", f"[platoon {label}] " + (reason or ""))
    first_line = detail.splitlines()[0][:500] if detail else ""
    return info("PlatoonError", f"[platoon {label}] " + (reason or first_line))


# --------------------------------------------------------------------------------------------- timing
def phases(*, started: float | None, finished: float | None, env_failed: bool,
           call_spans: Iterable[Iterable[float]] = (), agent_timestamps: Iterable[float] = (),
           sandbox: Mapping[str, Any] | None = None, harness_log_mtime: float | None = None,
           verifier_seconds: float | None = None) -> dict[str, dict[str, str] | None]:
    """harbor's phase spans, estimated. platoon gives the trial's start (`created_at - wall_time_seconds`) and end, and
    each model call's `time.perf_counter` span; the calls are put on the wall clock by the agent's own message
    timestamps (collected trajectory), else by the harness log's last write (= the agent's end), else by the end of the
    trial minus the tests' time. agent_execution = first call start .. last call end; environment_setup ends when the
    sandbox started (collect hook's sandbox.json) or, without it, where agent_execution starts (agent_setup then has
    zero length); verifier = agent end .. trial end. An env-failed trial has only environment_setup."""
    out: dict[str, dict[str, str] | None] = {"environment_setup": None, "agent_setup": None, "agent_execution": None,
                                             "verifier": None}
    if started is None or finished is None:
        return out
    if env_failed:
        out["environment_setup"] = span(started, finished)
        return out
    spans = sorted((float(a), float(b)) for a, b in (tuple(s)[:2] for s in call_spans if len(tuple(s)) >= 2))
    stamps = sorted(float(t) for t in agent_timestamps)
    sandbox_started = (sandbox or {}).get("started")
    env_end = float(sandbox_started) if isinstance(sandbox_started, (int, float)) and \
        started <= sandbox_started <= finished else None
    if spans:
        offset = None
        if stamps and len(stamps) == len(spans):
            offset = statistics.median(t - e for t, (_, e) in zip(stamps, spans))
        elif harness_log_mtime is not None and started <= harness_log_mtime <= finished + 5:
            offset = harness_log_mtime - spans[-1][1]
        else:
            offset = (finished - (verifier_seconds or 0.0)) - spans[-1][1]
        a0 = min(max(spans[0][0] + offset, env_end or started), finished)
        a1 = min(max(spans[-1][1] + offset, a0), finished)
    else:
        a1 = harness_log_mtime if harness_log_mtime is not None and started <= harness_log_mtime <= finished \
            else finished - (verifier_seconds or 0.0)
        a0 = env_end if env_end is not None and env_end <= a1 else a1
    env_end = env_end if env_end is not None and env_end <= a0 else a0
    out["environment_setup"] = span(started, env_end)
    out["agent_setup"] = span(env_end, a0)
    out["agent_execution"] = span(a0, a1)
    out["verifier"] = span(a1, finished)
    return out


# --------------------------------------------------------------------------------------------- trajectories
def rebuild_trajectory(root: Mapping[str, Any] | None, *, exit_status: str) -> dict[str, Any]:
    """A mini-swe-agent-1.1 trajectory from platoon's root trajectory: the task, then every assistant reply (content,
    tool calls, usage). platoon records no system prompt and no tool output, so neither is here (info.reconstructed)."""
    root = root or {}
    task = root.get("task")
    goal = task.get("goal") if isinstance(task, Mapping) else task
    messages: list[dict[str, Any]] = []
    if goal:
        messages.append({"role": "user", "content": str(goal)})
    calls = 0
    for step in root.get("steps") or []:
        misc = step.get("misc") if isinstance(step, Mapping) else None
        msg = (misc or {}).get("message")
        if not isinstance(msg, Mapping):
            continue
        calls += 1
        action = (misc or {}).get("action_misc") or {}
        out = dict(msg)
        out["extra"] = {"usage": action.get("usage"), "response_id": action.get("response_id"),
                        "platoon_completion_id": action.get("completion_id"),
                        "actions": [{"command": _command(tc), "tool_call_id": tc.get("id")}
                                    for tc in msg.get("tool_calls") or [] if isinstance(tc, Mapping)]}
        messages.append(out)
    messages.append({"role": "exit", "content": "", "extra": {"exit_status": exit_status, "submission": ""}})
    return {"info": {"model_stats": {"instance_cost": 0.0, "api_calls": calls}, "exit_status": exit_status,
                     "submission": "",
                     "reconstructed": "from platoon's trajectory_collection.json: the task and the model's replies; "
                                      "platoon does not record the system prompt or the tool outputs"},
            "messages": messages, "trajectory_format": TRAJECTORY_FORMAT}


def _command(tool_call: Mapping[str, Any]) -> str | None:
    fn = tool_call.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return args
    return args.get("command") if isinstance(args, Mapping) else None


def transcript(traj: Mapping[str, Any]) -> str:
    """A readable turn-by-turn rendering of a mini-swe-agent trajectory."""
    lines: list[str] = []
    for i, m in enumerate(traj.get("messages") or []):
        if not isinstance(m, Mapping):
            continue
        role = m.get("role")
        lines.append(f"===== [{i}] {role} =====")
        content = m.get("content")
        if isinstance(content, list):  # content parts
            content = "\n".join(str(p.get("text", p)) if isinstance(p, Mapping) else str(p) for p in content)
        if content:
            lines.append(str(content))
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, Mapping):
                lines.append(f"$ {_command(tc)}")
        if role == "exit":
            lines.append(f"exit_status: {(m.get('extra') or {}).get('exit_status')}")
    info = traj.get("info") or {}
    if info.get("reconstructed"):
        lines.insert(0, f"(rebuilt: {info['reconstructed']})")
    return "\n".join(lines) + "\n"


def agent_timestamps(traj: Mapping[str, Any] | None) -> list[float]:
    """Wall-clock times of the assistant replies of a collected mini-swe-agent trajectory."""
    out = []
    for m in (traj or {}).get("messages") or []:
        if isinstance(m, Mapping) and m.get("role") == "assistant":
            ts = (m.get("extra") or {}).get("timestamp")
            if isinstance(ts, (int, float)):
                out.append(float(ts))
    return out


def exit_status_of(record: TrialRecord, harness: Mapping[str, Any] | None) -> str:
    if record.env_failed:
        return "NotStarted"
    if (harness or {}).get("timed_out"):
        return "AgentTimeout"
    code = (harness or {}).get("exit_code")
    if code not in (0, None):
        return f"ExitCode{code}"
    return "Submitted" if code == 0 else "Unknown"


# --------------------------------------------------------------------------------------------- one trial
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, obj: Any) -> None:
    _write(path, json.dumps(obj, indent=2, default=str))


def _collection_id(trial_dir: Path) -> str | None:
    coll = _load_json(trial_dir / COLLECTION)
    if isinstance(coll, Mapping) and coll.get("id"):
        return str(coll["id"])
    events = trial_dir / "events"
    if events.is_dir():
        for f in events.glob("events_*.jsonl"):
            # events_<safe task>_<rollout id>.jsonl, the rollout id being "<safe task>-<12 hex>"
            stem = f.stem[len("events_"):]
            task = trial_dir.parent.name
            if stem.startswith(task + "_"):
                return stem[len(task) + 1:]
    return None


class CandidateExport:
    """Exports one candidate's platoon output dir into one harbor job dir, incrementally (`sync`)."""

    def __init__(self, platoon_dir: str | Path, dest: str | Path, candidate: str, *, model: str | None = None,
                 policy: str | None = None, tasks: Iterable[str] | None = None, dataset_dir: str | Path | None = None,
                 collect_dir: str | Path | None = None, trials: int | None = None, job: Mapping[str, Any] | None = None
                 ) -> None:
        self.platoon_dir = Path(platoon_dir)
        self.dest = Path(dest)
        self.candidate = candidate
        self.model = model
        self.tasks = list(tasks or [])
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.collect_dir = Path(collect_dir) if collect_dir else None
        self.trials = trials
        self.job = dict(job or {})
        self.reader = RolloutReader(self.platoon_dir, candidate, policy=policy, tasks=self.tasks)
        self._names = task_names(self.dataset_dir)
        self._done: set[str] = set()       # trial names whose result.json this exporter wrote
        self._configured: set[str] = set()
        self._job_id = str(uuid.uuid5(NAMESPACE, str(self.dest.resolve())))
        self._started = time.time()

    def task_name(self, task: str) -> str:
        return self._names.get(task, task)

    def sync(self, *, stopped: bool = False, final: bool = False) -> dict[str, int]:
        """Write what is new: config.json of launched trials, the whole dir of finished ones, the job files.
        `stopped`: the candidate's run ended without finishing some trials (they become env_failed "stopped")."""
        records = self.reader.records(stopped=stopped)
        by_key = {(r.task, r.index): r for r in records}
        rollouts = self.platoon_dir / "rollouts"
        new = 0
        if rollouts.is_dir():
            for task_dir in sorted(p for p in rollouts.iterdir() if p.is_dir()):
                trial_dirs = []
                for trial_dir in task_dir.iterdir():
                    if trial_dir.is_dir() and trial_dir.name.startswith(ROLLOUT_PREFIX):
                        try:
                            trial_dirs.append((int(trial_dir.name[len(ROLLOUT_PREFIX):]), trial_dir))
                        except ValueError:
                            continue
                for k, trial_dir in sorted(trial_dirs):  # launch order: config mtimes must not go backwards
                    task = self.reader._task_of.get(task_dir.name, task_dir.name)
                    meta = _load_json(trial_dir / METADATA)
                    if isinstance(meta, Mapping) and meta.get("task_id"):
                        task = str(meta["task_id"])
                    record = by_key.get((task, k))
                    if record is None:
                        continue
                    new += self._sync_trial(trial_dir, task_dir.name, record)
        self._write_job(records, final=final)
        return {"trials": len(records), "finished": sum(r.finished for r in records), "new": new}

    def _sync_trial(self, trial_dir: Path, safe_task: str, record: TrialRecord) -> int:
        name = trial_name(safe_task, record.index)
        out = self.dest / name
        if name in self._done:
            return 0
        if record.finished and name not in self._configured and isinstance(_load_json(out / "result.json"), dict):
            self._done.add(name)  # exported by an earlier pass (a restarted run, or the export command)
            return 0
        if name not in self._configured:
            self._write_config(out, name, safe_task, record, trial_dir)
            self._configured.add(name)
        if not record.finished:
            return 0
        self.export_trial(trial_dir, out, name, safe_task, record)
        self._done.add(name)
        return 1

    def _write_config(self, out: Path, name: str, safe_task: str, record: TrialRecord, trial_dir: Path) -> None:
        task_path = (self.dataset_dir / safe_task) if self.dataset_dir else Path(safe_task)
        _write_json(out / "config.json", {
            "task": {"path": str(task_path), "source": self.dataset_dir.name if self.dataset_dir else None},
            "trial_name": name, "trials_dir": str(self.dest),
            "agent": {"name": "harnessforge.evals:platoon-micro-swe-agent",
                      "model_name": f"litellm_proxy/{self.model}" if self.model else None},
            "job_id": self._job_id, "harnessforge": {"candidate": self.candidate, "index": record.index,
                                                     "platoon_rollout": str(trial_dir)}})
        launched = record.started_at
        if launched is None:
            try:
                launched = trial_dir.stat().st_mtime
            except OSError:
                launched = time.time()
        # seq.launched orders a task's trials by config.json's mtime (harbor: the launch time), then by name. platoon
        # launches a task's rollouts in index order, but the times we have are estimates (an unfinished trial's dir
        # mtime moves as files appear), so keep them non-decreasing in the index: the order is then the index.
        earlier = out.parent / trial_name(safe_task, record.index - 1) / "config.json"
        if record.index > 0 and earlier.is_file():
            launched = max(launched, earlier.stat().st_mtime)
        os.utime(out / "config.json", (launched, launched))

    def export_trial(self, trial_dir: Path, out: Path, name: str, safe_task: str, record: TrialRecord) -> None:
        meta = _load_json(trial_dir / METADATA) or {}
        root = root_trajectory(_load_json(trial_dir / COLLECTION))
        outcome = outcome_of(root)
        misc = (root or {}).get("misc") or {}
        harness = misc.get("harness") if isinstance(misc.get("harness"), Mapping) else None
        outcome_meta = outcome.get("metadata") if isinstance(outcome.get("metadata"), Mapping) else {}
        collected = None
        rollout_id = _collection_id(trial_dir)
        if self.collect_dir is not None and rollout_id:
            d = self.collect_dir / rollout_id
            collected = d if d.is_dir() else None

        # agent/: the agent's own trajectory when collected, else rebuilt from platoon's
        agent_traj = None
        if collected is not None:
            for f in sorted(collected.glob("agent/*" + AGENT_TRAJECTORY)):
                t = _load_json(f)
                if isinstance(t, Mapping) and t.get("messages"):
                    agent_traj = dict(t)
                    break
        source = "collected"
        if agent_traj is None and not record.env_failed:
            agent_traj = rebuild_trajectory(root, exit_status=exit_status_of(record, harness))
            source = "rebuilt"
        if agent_traj is not None:
            _write_json(out / "agent" / "mini-swe-agent.trajectory.json", agent_traj)
            _write(out / "agent" / "mini-swe-agent.txt", transcript(agent_traj))
        log_mtime = None
        logs_path = (harness or {}).get("logs_path")
        log = trial_dir / "harness" / Path(logs_path).name if logs_path else None
        if log is None or not log.is_file():
            logs = sorted((trial_dir / "harness").glob("*.log")) if (trial_dir / "harness").is_dir() else []
            log = logs[0] if logs else None
        if log is not None and log.is_file():
            log_mtime = log.stat().st_mtime
            _write(out / "agent" / "platoon-harness.log", log.read_text(encoding="utf-8", errors="replace"))

        # verifier/: the tests' output, full when collected, else platoon's tail (patched platoon)
        test_out = None
        test_source = None
        if collected is not None and (collected / COLLECTED_TEST_STDOUT).is_file():
            test_out = (collected / COLLECTED_TEST_STDOUT).read_text(encoding="utf-8", errors="replace")
            test_source = "collected"
        elif isinstance(outcome_meta.get("test_output"), str):
            test_out = outcome_meta["test_output"]
            test_source = "platoon_outcome_tail"
        if test_out is not None:
            _write(out / "verifier" / "test-stdout.txt", test_out)
        if record.reward is not None and not record.env_failed and meta.get("outcome_status") == "scored":
            _write(out / "verifier" / "reward.txt", f"{record.reward:g}\n")

        sandbox = _load_json(collected / COLLECTED_SANDBOX) if collected is not None else None
        spans = (misc.get("model_call_spans") or {}).values() if isinstance(misc.get("model_call_spans"),
                                                                           Mapping) else ()
        timing = phases(started=record.started_at, finished=record.finished_at, env_failed=record.env_failed,
                        call_spans=spans, agent_timestamps=agent_timestamps(agent_traj if source == "collected"
                                                                            else None),
                        sandbox=sandbox if isinstance(sandbox, Mapping) else None, harness_log_mtime=log_mtime,
                        verifier_seconds=outcome_meta.get("verifier_seconds"))
        if record.finished_at is None and record.error == "stopped":  # never finished: stamp the export time
            timing["environment_setup"] = span(record.started_at, time.time())
        result = trial_result(record, task_name=self.task_name(record.task), trial_name=name, trials_dir=self.dest,
                              task_path=(self.dataset_dir / safe_task) if self.dataset_dir else Path(safe_task),
                              source=self.dataset_dir.name if self.dataset_dir else None, model=self.model,
                              job_id=self._job_id, outcome=outcome, harness=harness, metadata=meta, timing=timing)
        result["harnessforge"].update({"platoon_rollout": str(trial_dir), "rollout_id": rollout_id,
                                       "agent_trajectory": source if agent_traj is not None else None,
                                       "test_output": test_source,
                                       "verifier_exit_code": outcome_meta.get("verifier_exit_code")})
        _write_json(out / "harnessforge.json", {"schema": 1, **asdict(record)})
        _write_json(out / "result.json", result)  # last: a reader that sees result.json sees the whole dir

    def _write_job(self, records: list[TrialRecord], *, final: bool) -> None:
        self.dest.mkdir(parents=True, exist_ok=True)
        _write_json(self.dest / "config.json", {
            "job_name": self.dest.name, "jobs_dir": str(self.dest.parent), "n_attempts": self.trials,
            "agents": [{"name": "harnessforge.evals:platoon-micro-swe-agent",
                        "model_name": f"litellm_proxy/{self.model}" if self.model else None}],
            "datasets": [{"path": str(self.dataset_dir)}] if self.dataset_dir else [],
            "harnessforge": {"candidate": self.candidate, "platoon_dir": str(self.platoon_dir), **self.job}})
        finished = [r for r in records if r.finished]
        errored = [r for r in finished if r.env_failed or (r.error and not r.error.startswith("scored:"))]
        by_reward: dict[str, list[str]] = {}
        for r in finished:
            if not r.env_failed:
                by_reward.setdefault(str(float(r.reward or 0.0)), []).append(
                    trial_name(safe_task_id(r.task), r.index))
        scored = [float(r.reward or 0.0) for r in finished if not r.env_failed]
        total = len(self.tasks) * self.trials if self.tasks and self.trials else len(records)
        key = f"mini-swe-agent__{self.model or 'model'}__{self.dataset_dir.name if self.dataset_dir else 'dataset'}"
        now = time.time()
        _write_json(self.dest / "result.json", {
            "id": self._job_id, "started_at": iso(min((r.started_at for r in records if r.started_at), default=now)),
            "updated_at": iso(now), "finished_at": iso(now) if final else None, "n_total_trials": total,
            "stats": {"n_completed_trials": len(finished), "n_errored_trials": len(errored),
                      "n_running_trials": len(records) - len(finished),
                      "n_pending_trials": max(0, total - len(records)) if not final else 0,
                      "n_cancelled_trials": sum(r.error == "stopped" or r.error == "cancelled" for r in finished),
                      "n_retries": 0,
                      "evals": {key: {"n_trials": len(finished), "n_errors": len(errored),
                                      "metrics": [{"mean": (sum(scored) / len(scored)) if scored else 0.0}],
                                      "reward_stats": {"reward": by_reward}}}}})


def trial_result(record: TrialRecord, *, task_name: str, trial_name: str, trials_dir: Path, task_path: Path,
                 source: str | None, model: str | None, job_id: str, outcome: Mapping[str, Any],
                 harness: Mapping[str, Any] | None, metadata: Mapping[str, Any],
                 timing: Mapping[str, Any]) -> dict[str, Any]:
    """harbor's TrialResult for one finished TrialRecord (the fields harbor-era readers use, plus `harnessforge`)."""
    usage = metadata.get("usage") if isinstance(metadata.get("usage"), Mapping) else {}
    cached = sum(int((usage.get(b) or {}).get("cached_prompt_tokens") or 0) for b in ("policy", "aux")
                 if isinstance(usage.get(b), Mapping))
    cost = (metadata.get("cost") or {}).get("usd") if isinstance(metadata.get("cost"), Mapping) else None
    verifier_result = None
    if not record.env_failed and metadata.get("outcome_status") == "scored" and record.reward is not None:
        rewards = {k: v for k, v in (outcome.get("rewards") or {}).items() if isinstance(v, (int, float))}
        rewards["reward"] = float(record.reward)
        verifier_result = {"rewards": rewards}
    ran = not record.env_failed
    return {
        "id": str(uuid.uuid5(NAMESPACE, f"{trials_dir}/{trial_name}")),
        "task_name": task_name,
        "trial_name": trial_name,
        "trial_uri": (trials_dir / trial_name).resolve().as_uri() if trials_dir.is_absolute() else None,
        "task_id": {"path": str(task_path)},
        "source": source,
        "task_checksum": None,
        "config": {"task": {"path": str(task_path), "source": source}, "trial_name": trial_name,
                   "trials_dir": str(trials_dir),
                   "agent": {"name": "harnessforge.evals:platoon-micro-swe-agent",
                             "model_name": f"litellm_proxy/{model}" if model else None},
                   "job_id": job_id},
        "agent_info": {"name": "mini-swe-agent", "version": "platoon",
                       "model_info": {"name": model, "provider": "openai"}},
        "agent_result": {"n_input_tokens": record.input_tokens, "n_cache_tokens": cached if ran else None,
                         "n_output_tokens": record.output_tokens, "cost_usd": float(cost) if cost else 0.0,
                         "rollout_details": None, "metadata": None} if ran else None,
        "verifier_result": verifier_result,
        "exception_info": harbor_exception(record, outcome, harness, metadata.get("error"), record.finished_at),
        "started_at": iso(record.started_at),
        "finished_at": iso(record.finished_at if record.finished_at is not None else time.time()),
        **{k: timing.get(k) for k in ("environment_setup", "agent_setup", "agent_execution", "verifier")},
        "step_results": None,
        "harnessforge": {"candidate": record.candidate, "task": record.task, "index": record.index,
                         "env_failed": record.env_failed, "error": record.error, "reward": record.reward,
                         "outcome_status": metadata.get("outcome_status"),
                         "failure_kind": outcome.get("failure_kind"), "reason": outcome.get("reason"),
                         "timing": "estimated"},
    }


# --------------------------------------------------------------------------------------------- a whole run
class RunExport:
    """Every candidate of a `PlatoonBackend` run dir (<out_dir>/<run_id>/, with its request.json) into
    `<dest>/<candidate>/` harbor job dirs."""

    def __init__(self, run_dir: str | Path, dest: str | Path, *, candidates: Iterable[str] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.dest = Path(dest)
        request = json.loads((self.run_dir / "request.json").read_text())
        dataset = self.run_dir / "dataset" if (self.run_dir / "dataset").is_dir() else Path(request["dataset"])
        wanted = set(candidates) if candidates else None
        self.exports: dict[str, CandidateExport] = {}
        for c in request["candidates"]:
            cid = c["id"]
            if wanted is not None and cid not in wanted:
                continue
            base = self.run_dir / slug(cid)
            self.exports[cid] = CandidateExport(
                base / "platoon", self.dest / slug(cid), cid, model=request.get("model"), policy=request.get("policy"),
                tasks=request.get("tasks"), dataset_dir=dataset, collect_dir=base / "collected",
                trials=request.get("trials"), job={"run_id": request.get("run_id"), "run_dir": str(self.run_dir)})

    def sync(self, *, stopped: Iterable[str] = (), final: bool = False) -> dict[str, dict[str, int]]:
        stopped = set(stopped)
        return {cid: e.sync(stopped=cid in stopped or final, final=final) for cid, e in self.exports.items()}


def slug(text: str) -> str:
    """The candidate dir name PlatoonBackend uses (backend._slug)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def export_run(run_dir: str | Path, dest: str | Path, *, candidates: Iterable[str] | None = None,
               final: bool | None = None) -> dict[str, dict[str, int]]:
    """One pass over a run dir. `final` (default: the run's status.json says done): unfinished trials will never
    finish, so they are exported as env_failed "stopped"."""
    run_dir = Path(run_dir)
    if final is None:
        status = _load_json(run_dir / "status.json")
        final = bool(isinstance(status, Mapping) and status.get("done"))
    return RunExport(run_dir, dest, candidates=candidates).sync(final=final)
