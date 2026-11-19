"""Shared pieces of the batch launchers: the child-job environment, the command runner (swappable
for tests) and transient-failure retries around submissions."""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from harnessforge.contracts import JobSpec

# Variables that describe the *submitting* job, not the job being submitted. qsub -V /
# sbatch --export=ALL would otherwise copy them into the child: e.g. an rt_HC driver's RTYPE made
# ABCI's reservation hook put GPU jobs on CPU-only nodes, and a stale CUDA_VISIBLE_DEVICES hides
# GPUs.
JOB_LOCAL_PREFIXES = ("PBS_", "SLURM_", "TMUX")
JOB_LOCAL_VARS = frozenset({
    "RTYPE", "JOB_NAME",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "NCPUS", "OMP_NUM_THREADS", "TMPDIR", "ENVIRONMENT", "HOSTNAME",
    "SSH_TTY", "SSH_CONNECTION", "SSH_CLIENT", "OLDPWD",
})

TRANSIENT_SUBMIT_ERRORS = (
    "under maintenance", "cannot connect to server", "communication failure", "timed out",
    "timeout", "temporarily unavailable", "connection refused", "socket_resource_unavailable",
    "slurm_load", "try again")


def child_job_env(environ: Mapping[str, str], extra: Mapping[str, str],
                  resource_type: str = "") -> tuple[dict[str, str], list[str]]:
    """The env a submitted job inherits: `environ` minus job-local variables, plus `extra`
    (the job's own env), plus RTYPE only from this job's own resource type.
    Returns (env, dropped_names)."""
    dropped = sorted(k for k in environ
                     if k in JOB_LOCAL_VARS or k.startswith(JOB_LOCAL_PREFIXES))
    env = {k: v for k, v in environ.items() if k not in dropped}
    env.update({k: str(v) for k, v in extra.items()})
    if resource_type:
        env["RTYPE"] = resource_type
    return env, dropped


def is_transient(msg: str) -> bool:
    m = msg.lower()
    return any(t in m for t in TRANSIENT_SUBMIT_ERRORS)


@dataclass
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


# A runner executes argv with an env and returns (rc, stdout, stderr). Tests pass a fake.
Runner = Callable[[Sequence[str], "Mapping[str, str] | None"], Result]


def subprocess_runner(argv: Sequence[str], env: Mapping[str, str] | None = None) -> Result:
    r = subprocess.run(list(argv), env=dict(env) if env is not None else None,
                       capture_output=True, text=True)
    return Result(r.returncode, r.stdout, r.stderr)


def job_command(spec: JobSpec) -> list[str]:
    """The argv a job runs: spec.command, optionally inside `options["cwd"]`."""
    cmd = [str(c) for c in spec.command]
    if not cmd:
        raise ValueError(f"job {spec.name!r} has an empty command")
    cwd = spec.options.get("cwd")
    if cwd:
        cmd = ["/bin/bash", "-c", f"cd {shlex.quote(str(cwd))} && exec {shlex.join(cmd)}"]
    return cmd


class BatchLauncher:
    """Common submit loop: render argv, run it with the child env, retry transient failures."""

    kind = "batch"

    def __init__(self, *, runner: Runner | None = None, environ: Mapping[str, str] | None = None,
                 retry_seconds: float = 7200.0, retry_interval: float = 60.0,
                 sleep: Callable[[float], None] = time.sleep, defaults: Mapping | None = None):
        self.runner = runner or subprocess_runner
        self.environ = environ            # None -> os.environ at submit time
        self.retry_seconds = retry_seconds
        self.retry_interval = retry_interval
        self.sleep = sleep
        self.defaults = dict(defaults or {})   # option defaults (queue, account, ...)

    def opt(self, spec: JobSpec, *keys: str, default=None):
        for k in keys:
            if spec.options.get(k) not in (None, ""):
                return spec.options[k]
        for k in keys:
            if self.defaults.get(k) not in (None, ""):
                return self.defaults[k]
        return default

    # subclasses
    def render(self, spec: JobSpec) -> list[str]:
        raise NotImplementedError

    def parse_job_id(self, out: str) -> str:
        raise NotImplementedError

    def submit_env(self, spec: JobSpec) -> dict[str, str]:
        environ = os.environ if self.environ is None else self.environ
        env, _ = child_job_env(environ, spec.env, "")
        return env

    def submit(self, spec: JobSpec) -> str:
        argv = self.render(spec)
        env = self.submit_env(spec)
        if spec.log is not None:
            spec.log.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.retry_seconds
        while True:
            r = self.runner(argv, env)
            msg = (r.stderr.strip() or r.stdout.strip())
            if r.returncode == 0 or not is_transient(msg) or time.monotonic() > deadline:
                break
            self.sleep(self.retry_interval)
        if r.returncode != 0:
            raise RuntimeError(f"{self.kind} submit of {spec.name!r} failed rc={r.returncode}: "
                               f"{msg}")
        return self.parse_job_id(r.stdout)
