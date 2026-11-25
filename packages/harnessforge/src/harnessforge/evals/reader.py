"""Turn a platoon evaluation's rollout directories into TrialRecords (stdlib only).

`platoon eval` writes one directory per launched trial and fills it as the trial ends:

    <output_dir>/rollouts/<safe task id>/rollout_<k>/
        metadata.json               written when the trial ends: outcome status, error, wall time, usage
        trajectory_collection.json  written with it (absent when the rollout raised): rewards, outcome
        harness/, events/           written while it runs

The directory is created when the trial takes its concurrency slot, i.e. at launch, so a directory without
`metadata.json` is an in-flight trial. `k` is the rollout index, which platoon launches task-major (every rollout of
task 1, then task 2, ... in the dataset order), so it is the TrialRecord `index`.

Which outcomes are evidence about the candidate (`env_failed=False`) follows the old harbor rule
(`utils/gate.py:env_failed`): a trial counts, as its reward or 0, once the agent got to run; it is left out
(`env_failed=True`) only when it failed before the agent ran (the sandbox or the session never came up, the harness
could not be launched) or when the failure is the evaluation's own (the model backend failed, the trial was
cancelled). See `classify` and `docs/evals.md`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

from harnessforge.records import TrialRecord

METADATA = "metadata.json"
COLLECTION = "trajectory_collection.json"
ROLLOUT_PREFIX = "rollout_"

# Failure kinds (platoon's Outcome.failure_kind) of failures that happen before the agent runs: the trial says
# nothing about the candidate.
PRE_AGENT_KINDS = frozenset({
    "sandbox_start", "env_setup", "env_config", "env_healthcheck", "sidecar_start",
    "harness_setup", "harness_config",
})
# Failures of the evaluation itself, not of the candidate's agent: the model backend failed one of the agent's calls,
# platoon's queue for the sandbox timed out, the session was closed under the trial, the sandbox was lost.
INFRA_KINDS = frozenset({"harness_crash_gateway", "env_busy", "session_closed",
                         # the sandbox died under the trial: its container, or the sandbox host it ran on
                         # (harnessforge.sandboxhost raises hpc_sandbox.SandboxLostError for a host that went away)
                         "unexpected:SandboxLostError"})
# Every other failure kind (harness_crash, context_overflow, verifier_timeout, verifier_no_reward, verifier_output,
# verifier_setup, artifact_collect, main_exited, ...) happens after the agent ran, and counts as reward 0, as harbor's
# missing reward did.


class Verdict(NamedTuple):
    reward: float | None
    env_failed: bool
    error: str | None


def safe_task_id(task_id: str) -> str:
    """The directory name platoon gives a task's rollouts (platoon.eval.runner._safe_task_id)."""
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in task_id)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def root_trajectory(collection: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The root agent's trajectory: the first of the collection's trajectories (platoon writes the root first)."""
    trajectories = collection.get("trajectories") if isinstance(collection, Mapping) else None
    if not isinstance(trajectories, Mapping) or not trajectories:
        return None
    root = next(iter(trajectories.values()))
    return root if isinstance(root, Mapping) else None


def outcome_of(root: Mapping[str, Any] | None) -> dict[str, Any]:
    misc = root.get("misc") if isinstance(root, Mapping) else None
    outcome = misc.get("outcome") if isinstance(misc, Mapping) else None
    return dict(outcome) if isinstance(outcome, Mapping) else {}


def classify(metadata: Mapping[str, Any], root: Mapping[str, Any] | None) -> Verdict:
    """Reward, env_failed and error of one finished trial (see the module docstring)."""
    status = metadata.get("outcome_status")
    error = metadata.get("error")
    outcome = outcome_of(root)
    kind = outcome.get("failure_kind")
    if status is None:
        # The rollout raised before it recorded an outcome: the session (sandbox) never opened, or the worker died.
        return Verdict(None, True, _short(error) or "no_outcome")
    if isinstance(error, str) and error.startswith("backend_unavailable"):
        return Verdict(None, True, "backend_unavailable")
    if status == "cancelled":
        return Verdict(None, True, "cancelled")
    if status == "scored":
        reward = root.get("reward") if isinstance(root, Mapping) else None
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            rewards = outcome.get("rewards") or {}
            reward = rewards.get("reward", 0.0) if isinstance(rewards, Mapping) else 0.0
        reason = outcome.get("reason")
        # A scored outcome with a reason (agent_timeout, harness_crash under on_harness_exit: grade) keeps it as a note.
        return Verdict(float(reward), False, f"scored:{reason}" if reason else None)
    label = f"{status}:{kind}" if kind else str(status)
    if kind in PRE_AGENT_KINDS or kind in INFRA_KINDS:
        return Verdict(None, True, label)
    return Verdict(0.0, False, label)


def _short(error: Any, limit: int = 300) -> str | None:
    if not isinstance(error, str) or not error:
        return None
    line = error.strip().splitlines()[0] if error.strip() else error
    return line[:limit]


def _tokens(usage: Any, key: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    total, seen = 0, False
    for bucket in ("policy", "aux"):
        counts = usage.get(bucket)
        if isinstance(counts, Mapping) and isinstance(counts.get(key), (int, float)):
            total += int(counts[key])
            seen = True
    return total if seen else None


def _timestamp(text: Any) -> float | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class RolloutReader:
    """Reads one candidate's platoon output directory into TrialRecords, caching finished trials (their files never
    change once metadata.json exists), so polling a long run stays cheap."""

    def __init__(self, output_dir: str | Path, candidate: str, *, policy: str | None = None,
                 tasks: Iterable[str] | None = None,
                 aliases: Mapping[str, tuple[str, int]] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.candidate = candidate
        self.policy = policy
        # safe dir name -> task id, for in-flight trials (which have no metadata naming their task yet)
        self._task_of = {safe_task_id(t): t for t in (tasks or ())}
        # a look-major launch (PlatoonBackend option order): task name -> the (task, index) unit it runs
        self._aliases = dict(aliases or {})
        self._alias_of = {safe_task_id(a): unit for a, unit in self._aliases.items()}
        self._finished: dict[Path, TrialRecord] = {}

    def records(self, *, stopped: bool = False) -> list[TrialRecord]:
        """Every launched trial: finished ones, and in-flight ones as finished=False (or, once the run is `stopped`
        and can no longer finish them, as cancelled: finished, env_failed)."""
        rollouts = self.output_dir / "rollouts"
        out: list[TrialRecord] = []
        if not rollouts.is_dir():
            return out
        for task_dir in sorted(p for p in rollouts.iterdir() if p.is_dir()):
            for trial_dir in sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith(ROLLOUT_PREFIX)):
                try:
                    index = int(trial_dir.name[len(ROLLOUT_PREFIX):])
                except ValueError:
                    continue
                record = self._finished.get(trial_dir) or self._read(trial_dir, task_dir.name, index, stopped)
                if record is not None:
                    out.append(record)
        out.sort(key=lambda r: (r.task, r.index))
        return out

    def _read(self, trial_dir: Path, safe_task: str, index: int, stopped: bool) -> TrialRecord | None:
        task = ""
        meta_path = trial_dir / METADATA
        metadata = _load_json(meta_path) if meta_path.exists() else None
        if safe_task in self._alias_of:
            task, index = self._alias_of[safe_task]
        if not isinstance(metadata, Mapping):
            # Not finished (or metadata.json is being written: the next poll reads it).
            task = task if safe_task in self._alias_of else self._task_of.get(safe_task, safe_task)
            try:
                started = trial_dir.stat().st_mtime
            except OSError:
                started = None
            if stopped:
                return TrialRecord(self.candidate, task, index, True, env_failed=True, error="stopped",
                                   started_at=started, policy=self.policy)
            return TrialRecord(self.candidate, task, index, False, started_at=started, policy=self.policy)
        unit = self._aliases.get(str(metadata.get("task_id"))) or self._alias_of.get(safe_task)
        if unit is not None:
            task, index = unit
        else:
            task = str(metadata.get("task_id") or self._task_of.get(safe_task, safe_task))
        root = root_trajectory(_load_json(trial_dir / COLLECTION))
        verdict = classify(metadata, root)
        wall = metadata.get("wall_time_seconds")
        wall = float(wall) if isinstance(wall, (int, float)) else None
        finished_at = _timestamp(metadata.get("created_at"))
        started_at = finished_at - wall if finished_at is not None and wall is not None else None
        outcome = outcome_of(root)
        harness = (root.get("misc") or {}).get("harness") if isinstance(root, Mapping) else None
        extra: dict[str, Any] = {"outcome_status": metadata.get("outcome_status")}
        for key in ("reason", "failure_kind"):
            if outcome.get(key):
                extra[key] = outcome[key]
        outcome_meta = outcome.get("metadata")  # kept only by a patched platoon (patches/platoon-outcome-metadata.patch)
        if isinstance(outcome_meta, Mapping) and outcome_meta.get("verifier_exit_code") is not None:
            extra["verifier_exit_code"] = outcome_meta["verifier_exit_code"]
        if isinstance(harness, Mapping):
            extra["exit_code"] = harness.get("exit_code")
            extra["timed_out"] = harness.get("timed_out")
        if isinstance(root, Mapping):
            extra["steps"] = len(root.get("steps") or [])
        usage = metadata.get("usage")
        record = TrialRecord(
            candidate=self.candidate, task=task,
            index=index if unit is not None else int(metadata.get("rollout_index", index)), finished=True,
            reward=verdict.reward, env_failed=verdict.env_failed, error=verdict.error,
            output_tokens=_tokens(usage, "completion_tokens"), input_tokens=_tokens(usage, "prompt_tokens"),
            wall_seconds=wall, started_at=started_at, finished_at=finished_at, policy=self.policy, extra=extra)
        self._finished[trial_dir] = record
        return record


def read_rollouts(output_dir: str | Path, candidate: str, *, policy: str | None = None,
                  tasks: Iterable[str] | None = None, stopped: bool = False) -> list[TrialRecord]:
    """One-shot `RolloutReader(...).records()`."""
    return RolloutReader(output_dir, candidate, policy=policy, tasks=tasks).records(stopped=stopped)

