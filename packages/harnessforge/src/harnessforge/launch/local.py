"""Local launcher: each job is a subprocess on this machine (tests, laptops, inside one big job).

Job ids are "local-<pid>". Exit codes go to `<state_dir>/<id>.exit`, so a different process
(e.g. the pool CLI run later) can still read a job's status and cancel it.
JobSpec.options: cwd. Resources (gpus/cpus/walltime/nodes) are not enforced.
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from pathlib import Path

from harnessforge.contracts import JobSpec, JobState, JobStatus
from harnessforge.launch.base import child_job_env

import socket


class LocalLauncher:
    kind = "local"

    def __init__(self, state_dir: Path | str | None = None):
        self.state_dir = Path(state_dir or Path(tempfile.gettempdir()) / "harnessforge-local-jobs")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}

    def submit(self, spec: JobSpec) -> str:
        env, _ = child_job_env(os.environ, spec.env)
        if "CUDA_VISIBLE_DEVICES" in os.environ and "CUDA_VISIBLE_DEVICES" not in spec.env:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]  # same machine
        cmd = [str(c) for c in spec.command]
        if not cmd:
            raise ValueError(f"job {spec.name!r} has an empty command")
        log = spec.log
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
        out = open(log, "ab") if log is not None else subprocess.DEVNULL
        # sh records the command's exit code; the job id is known only after the fork, so the
        # exit file path is derived from the shell's own pid ($$).
        wrapper = ['/bin/sh', '-c', '"$@"; echo $? > "$HF_LOCAL_JOBS/local-$$.exit"', "sh", *cmd]
        env["HF_LOCAL_JOBS"] = str(self.state_dir)
        env["HF_JOB_NAME"] = spec.name
        try:
            p = subprocess.Popen(wrapper, env=env, cwd=spec.options.get("cwd"), stdout=out,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        finally:
            if log is not None:
                out.close()
        job_id = f"local-{p.pid}"
        self._procs[job_id] = p
        return job_id

    def _pid(self, job_id: str) -> int:
        if not job_id.startswith("local-"):
            raise ValueError(f"not a local job id: {job_id!r}")
        return int(job_id.removeprefix("local-"))

    def status(self, job_id: str) -> JobStatus:
        host = (socket.gethostname(),)
        exit_file = self.state_dir / f"{job_id}.exit"
        p = self._procs.get(job_id)
        if p is not None:
            rc = p.poll()
            if rc is None:
                return JobStatus(JobState.RUNNING, None, host)
        if exit_file.exists():
            try:
                return JobStatus(JobState.DONE, int(exit_file.read_text().strip()), host)
            except ValueError:
                return JobStatus(JobState.DONE, None, host)
        if p is not None:
            return JobStatus(JobState.DONE, p.returncode, host)
        try:
            os.kill(self._pid(job_id), 0)
        except ProcessLookupError:
            return JobStatus(JobState.DONE, None, host)
        except PermissionError:
            pass
        return JobStatus(JobState.RUNNING, None, host)

    def cancel(self, job_id: str) -> None:
        pid = self._pid(job_id)
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        p = self._procs.get(job_id)
        if p is not None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
