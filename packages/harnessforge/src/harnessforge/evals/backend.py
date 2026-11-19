"""PlatoonBackend: evaluate micro-swe-agent candidates on Harbor tasks with `platoon eval`, one process per candidate.

    backend = PlatoonBackend(platoon="~/hf-dev/evals/venv/bin/platoon", site="~/hf-dev/evals/site.yaml")
    run = backend.start(EvalRequest(run_id="r1", candidates=[base, cand], tasks=order, dataset="/data/tmax-gate200",
                                    trials=2, endpoint="http://gpu-node:8000/v1", policy="Qwen3.5-4B",
                                    concurrency=12, out_dir=Path("runs"), options={...}))
    while not run.done():
        records = run.records()      # finished and in-flight trials of every candidate, read from disk
        ...                          # run.stop("cand") once its sequential test decides

**One platoon process per candidate.** `platoon eval` has no way to stop one environment entry while the others
continue (its runner's stop ends the whole evaluation), and its SIGTERM handling stops exactly the sandboxes that
process started (they carry its output directory as their label) and exits 143. So each candidate gets its own
`platoon eval` (own config, output directory, gateway and concurrency `request.concurrency`), started in a session
of its own; `stop(candidate)` sends that process SIGTERM (SIGKILL to its process group after `stop_grace_s`), and the
other candidates' processes are untouched. The cost is one gateway and one task listing per candidate, which is
small next to the trials; the toolchain is staged once per node and shared.

**Order.** platoon launches task-major in the dataset order (every rollout of task 1, then task 2, ...), and the
dataset order here is `request.tasks` (a split manifest lists them in that order), so `order = TASK_MAJOR` and a
look's task prefix is a prefix of `request.tasks`.

**Records** come from the rollout directories (`harnessforge.evals.reader`): finished trials from their
metadata.json and trajectory, launched-but-unfinished ones as `finished=False`, and, once a candidate is stopped
and its process has exited, those it never finished as `finished=True, env_failed=True, error="stopped"`.

Options (`EvalRequest.options`):
    model             served model name (default: the endpoint's first /models entry)
    agent_config      sampling YAML layered after `-c mini` (configs/sampling/<model>.yaml)
    api_key_env       variable holding the endpoint's key (none for a local vLLM/SGLang)
    provider          sandbox provider (default "harnessforge.evals.pods:PodSingularityProvider")
    provider_options  its options (image_cache, scratch_root, fakeroot, allow_network_none, pod_users, ...)
    agent_timeout_multiplier, verifier_timeout_multiplier   as harbor's --agent-timeout-multiplier (default 1.0)
    harbor_system_deps  apt/dnf/apk-install curl, bash, gcc/make, git, python3, pip3 when missing, before the agent
                      runs, as harbor's installed-agent setup does in every trial (default True: parity with harbor)
    materialize       copy the tasks into the run dir and give Dockerfile-only tasks their FROM image (default True)
    toolchain         toolchain name (default harness.TOOLCHAIN)
    hosted_timeout_s  seconds one model call may take at the gateway (default 3600, the agent's own litellm timeout)
    service_kwargs, rollout_config, workflow   merged into the generated config
    env               extra environment for the platoon processes
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from harnessforge.contracts import Candidate, EvalRequest, Order
from harnessforge.evals import dataset as datasets
from harnessforge.evals.harness import CANDIDATE_MOUNT, TOOLCHAIN, check_candidate, render_spec
from harnessforge.evals.reader import RolloutReader
from harnessforge.records import TrialRecord, write_jsonl

DEFAULT_PROVIDER = "harnessforge.evals.pods:PodSingularityProvider"
#: harbor's MiniSweAgent.install: ensure_system_dependencies(curl, bash, build_tools, git, python3, python_pip).
HARBOR_SYSTEM_DEPS = r"""
need=""
for c in curl bash gcc make git python3 pip3; do command -v "$c" >/dev/null 2>&1 || need=1; done
[ -z "$need" ] && exit 0
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update && apt-get install -y curl bash build-essential git python3 python3-pip
elif command -v dnf >/dev/null 2>&1; then dnf install -y curl bash gcc gcc-c++ make git python3 python3-pip
elif command -v yum >/dev/null 2>&1; then yum install -y curl bash gcc gcc-c++ make git python3 python3-pip
elif command -v apk >/dev/null 2>&1; then apk add --no-cache curl bash build-base git python3 py3-pip
else echo "no supported package manager; cannot ensure system dependencies" >&2
fi
""".strip()


def served_model(endpoint: str, timeout: float = 30.0) -> str:
    """The first model an OpenAI-compatible endpoint lists at /models."""
    with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=timeout) as reply:
        data = json.load(reply)
    return data["data"][0]["id"]


@dataclass
class _Member:
    candidate: Candidate
    out_dir: Path
    config: Path
    log: Path
    reader: RolloutReader
    process: subprocess.Popen | None = None
    stop_requested: float | None = None
    killed: bool = False
    returncode: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def exited(self) -> bool:
        """Its process ended (or none was launched)."""
        return self.process is None or self.returncode is not None


class PlatoonRun:
    order = Order.TASK_MAJOR

    def __init__(self, request: EvalRequest, members: dict[str, _Member], *, stop_grace_s: float) -> None:
        self.request = request
        self._members = members
        self._stop_grace_s = stop_grace_s

    # ------------------------------------------------------------------ EvalRun
    def records(self) -> list[TrialRecord]:
        self._reap()
        out: list[TrialRecord] = []
        for member in self._members.values():
            stopped = member.stop_requested is not None and member.exited
            out.extend(member.reader.records(stopped=stopped))
        return out

    def stop(self, candidate: str) -> None:
        member = self._members[candidate]
        if member.stop_requested is None:
            member.stop_requested = time.monotonic()
            if member.process is not None and member.process.poll() is None:
                member.process.send_signal(signal.SIGTERM)  # platoon stops its sandboxes, then exits 143

    def stop_all(self) -> None:
        for candidate in self._members:
            self.stop(candidate)

    def done(self) -> bool:
        self._reap()
        return all(m.exited for m in self._members.values())

    def wait(self, timeout: float | None = None, poll_s: float = 5.0) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.done():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(poll_s if deadline is None else max(0.0, min(poll_s, deadline - time.monotonic())))
        return True

    # ------------------------------------------------------------------ extras
    def status(self) -> dict[str, dict[str, Any]]:
        """Per candidate: process id, exit code, whether stopped, output dir and log."""
        self._reap()
        return {c: {"pid": m.process.pid if m.process else None, "returncode": m.returncode,
                    "stopped": m.stop_requested is not None, "out_dir": str(m.out_dir), "log": str(m.log)}
                for c, m in self._members.items()}

    def write_records(self, path: str | Path) -> list[TrialRecord]:
        records = self.records()
        write_jsonl(path, records)
        return records

    def _reap(self) -> None:
        for member in self._members.values():
            if member.process is None or member.returncode is not None:
                continue
            code = member.process.poll()
            if code is not None:
                member.returncode = code
                continue
            if (member.stop_requested is not None and not member.killed
                    and time.monotonic() - member.stop_requested > self._stop_grace_s):
                try:
                    os.killpg(member.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                member.killed = True


class PlatoonBackend:
    """EvalBackend over `platoon eval` (see the module docstring)."""

    def __init__(self, *, platoon: str | Path = "platoon", site: str | Path | None = None,
                 pythonpath: str | Path | None = None, stop_grace_s: float = 180.0,
                 launch: bool = True) -> None:
        self.platoon = str(Path(platoon).expanduser()) if os.sep in str(platoon) else str(platoon)
        self.site = Path(site).expanduser() if site else None
        # Where harnessforge is importable from, for the pod provider in platoon's processes (default: this copy).
        self.pythonpath = str(pythonpath or Path(__file__).resolve().parents[2])
        self.stop_grace_s = stop_grace_s
        self.launch = launch  # False: write the configs only (tests, dry runs)

    def start(self, request: EvalRequest) -> PlatoonRun:
        if request.out_dir is None:
            raise ValueError("PlatoonBackend needs EvalRequest.out_dir")
        if not request.tasks:
            raise ValueError("EvalRequest.tasks is empty")
        ids = [c.id for c in request.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"candidate ids must be unique: {ids}")
        for candidate in request.candidates:
            check_candidate(candidate.harness_dir)
        opts = dict(request.options)
        run_dir = Path(request.out_dir) / request.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = Path(request.dataset)
        prepare = [HARBOR_SYSTEM_DEPS] if opts.get("harbor_system_deps", True) else []
        if opts.get("materialize", True):
            replays = datasets.materialize(dataset_dir, request.tasks, run_dir / "dataset")
            commands = {datasets.replay_command(r) for r in replays.values()} - {None}
            if len(commands) > 1:
                raise ValueError("tasks replay different Dockerfile steps; one harness cannot replay them all: "
                                 + "; ".join(sorted(commands))[:500])
            prepare = [*commands, *prepare]
            dataset_dir = run_dir / "dataset"
        model = opts.get("model") or (served_model(request.endpoint) if self.launch else "policy")
        members: dict[str, _Member] = {}
        for candidate in request.candidates:
            out = run_dir / _slug(candidate.id)
            spec = render_spec(candidate, out / "harness", agent_config=opts.get("agent_config"),
                               toolchain=opts.get("toolchain", TOOLCHAIN), prepare=prepare)
            config = out / "eval.yaml"  # JSON, which is YAML
            config.write_text(json.dumps(self._config(request, candidate, model, spec, dataset_dir, out, opts),
                                         indent=1))
            members[candidate.id] = _Member(candidate, out / "platoon", config, out / "platoon.log",
                                            RolloutReader(out / "platoon", candidate.id, policy=request.policy,
                                                          tasks=request.tasks))
        (run_dir / "request.json").write_text(json.dumps(_request_json(request, model), indent=1))
        if self.launch:
            for member in members.values():
                member.process = self._spawn(member, opts)
        return PlatoonRun(request, members, stop_grace_s=self.stop_grace_s)

    # ------------------------------------------------------------------ helpers
    def _config(self, request: EvalRequest, candidate: Candidate, model: str, spec: Path, dataset_dir: Path,
                out: Path, opts: Mapping[str, Any]) -> dict[str, Any]:
        provider_options = {"fakeroot": True, "allow_network_none": True, **opts.get("provider_options", {})}
        service_kwargs = {
            "dataset_dir": str(dataset_dir),
            "splits": {"eval": list(request.tasks)},
            "provider": opts.get("provider", DEFAULT_PROVIDER),
            "provider_options": provider_options,
            "mounts": [{"source": str(Path(candidate.harness_dir).resolve()), "target": CANDIDATE_MOUNT,
                        "read_only": True}],
            "agent_timeout_multiplier": float(opts.get("agent_timeout_multiplier", 1.0)),
            "verifier_timeout_multiplier": float(opts.get("verifier_timeout_multiplier", 1.0)),
            "agent_user": "ignore",  # the sandbox user, root in our pods (fakeroot), as harbor ran them
            **opts.get("service_kwargs", {}),
        }
        hosted: dict[str, Any] = {"base_url": request.endpoint.rstrip("/"),
                                  "timeout_s": float(opts.get("hosted_timeout_s", 3600))}
        if opts.get("api_key_env"):
            hosted["api_key_env"] = opts["api_key_env"]
        return {
            "inference": {
                "model_name": model,
                "output_dir": str(out / "platoon"),
                "judges": False,
                "retry_errored": False,
                "hosted": hosted,
                "workflow": {
                    "num_rollouts_per_task": int(request.trials),
                    "num_concurrent_workers": int(request.concurrency),
                    "rollout_config": {"verbose": False, **opts.get("rollout_config", {})},
                    **opts.get("workflow", {}),
                },
            },
            "environments": [{
                "name": "harnessforge",
                "service": "harbor",
                "service_kwargs": service_kwargs,
                "harness": str(spec),
                # As harbor: the verifier grades whatever the agent left, after a timeout or a crash too.
                "on_harness_exit": {"timeout": "grade", "crash": "grade"},
            }],
            "split": "eval",
        }

    def _spawn(self, member: _Member, opts: Mapping[str, Any]) -> subprocess.Popen:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (self.pythonpath, env.get("PYTHONPATH")) if p)
        if self.site is not None:
            env["PLATOON_SITE"] = str(self.site)
        env.update({k: str(v) for k, v in opts.get("env", {}).items()})
        member.log.parent.mkdir(parents=True, exist_ok=True)
        log = member.log.open("ab")
        try:
            return subprocess.Popen([self.platoon, "eval", str(member.config)], stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, cwd=str(member.config.parent),
                                    start_new_session=True)
        finally:
            log.close()


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def _request_json(request: EvalRequest, model: str) -> dict[str, Any]:
    return {"run_id": request.run_id, "tasks": list(request.tasks), "dataset": request.dataset,
            "trials": request.trials, "endpoint": request.endpoint, "model": model, "policy": request.policy,
            "concurrency": request.concurrency,
            "candidates": [{"id": c.id, "harness_dir": str(c.harness_dir), "parent": c.parent}
                           for c in request.candidates]}
