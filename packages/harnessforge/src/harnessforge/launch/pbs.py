"""PBS Pro launcher (ABCI 3.0). Ported from mho.jobs.Scheduler.

JobSpec.options understood (each may also come from the launcher's `defaults`):

    queue           -q            e.g. "R9920261300" (an ABCI reservation)
    account         -P            e.g. "gai51740"
    resource_type   -v RTYPE=...  ABCI resource type: rt_HG (1 GPU), rt_HF (8 GPUs), rt_HC (CPU).
                                  With a resource type and no `select`, the select string is just
                                  `select=<nodes>`: the type fixes the node shape.
    select          -l select=... raw select string (wins over nodes/gpus/cpus), e.g.
                                  "1:ncpus=192:mem=1920gb:ngpus=8"
    host                          pin to one node: select=1:host=<h>:...
    mem                           per-chunk mem, e.g. "160gb" (auto-built select only)
    extra_args                    list of extra qsub arguments
    cwd                           run the command in this directory

The job's environment is the submitter's minus its own PBS_*/RTYPE/CUDA_VISIBLE_DEVICES/...
(`child_job_env`) plus `JobSpec.env`, exported with -V.
"""
from __future__ import annotations

import os
import shutil

from harnessforge.contracts import JobSpec, JobState, JobStatus
from harnessforge.launch.base import BatchLauncher, child_job_env, job_command

_PBS_STATES = {"R": JobState.RUNNING, "E": JobState.RUNNING, "B": JobState.RUNNING,
               "Q": JobState.PENDING, "H": JobState.PENDING, "W": JobState.PENDING,
               "T": JobState.PENDING, "S": JobState.PENDING, "F": JobState.DONE,
               "X": JobState.DONE}


def pbs_select(spec: JobSpec, *, select: str | None = None, resource_type: str | None = None,
               host: str | None = None, mem: str | None = None) -> str:
    if select:
        sel = select if select.startswith("select=") else f"select={select}"
    elif resource_type:
        sel = f"select={spec.nodes}"
    else:
        sel = (f"select={spec.nodes}" + (f":ngpus={spec.gpus}" if spec.gpus else "")
               + (f":ncpus={spec.cpus}" if spec.cpus else "") + (f":mem={mem}" if mem else ""))
    if host:
        head, _, rest = sel.partition(":")
        sel = f"{head}:host={host}" + (f":{rest}" if rest else "")
    return sel


def parse_qstat_f(text: str) -> JobStatus:
    """JobStatus from `qstat -x -f <id>` output."""
    fields: dict[str, str] = {}
    key = None
    for raw in text.splitlines():
        if not raw.strip() or raw.startswith("Job Id:"):
            continue
        if raw.startswith(("\t", "        ")) and key and "=" not in raw:
            fields[key] += raw.strip()            # continuation line
            continue
        k, sep, v = raw.strip().partition("=")
        if sep:
            key = k.strip()
            fields[key] = v.strip()
    st = fields.get("job_state", "")
    exit_code = None
    if "Exit_status" in fields:
        try:
            exit_code = int(fields["Exit_status"])
        except ValueError:
            pass
    state = _PBS_STATES.get(st, JobState.DONE if exit_code is not None else JobState.PENDING)
    hosts: list[str] = []
    for chunk in fields.get("exec_host", "").split("+"):
        h = chunk.split("/")[0].strip()
        if h and h not in hosts:
            hosts.append(h)
    return JobStatus(state=state, exit_code=exit_code, hosts=tuple(hosts))


def pbs_bin(name: str, bin_dir: str | None = None) -> str:
    """Path of a PBS client command. Batch jobs often have a minimal PATH without the PBS
    clients (ABCI: /opt/pbs/bin), so fall back to $PBS_EXEC/bin and /opt/pbs/bin."""
    if bin_dir:
        return os.path.join(bin_dir, name)
    if shutil.which(name):
        return name
    for d in (os.path.join(os.environ.get("PBS_EXEC", "/opt/pbs"), "bin"), "/opt/pbs/bin"):
        if os.access(os.path.join(d, name), os.X_OK):
            return os.path.join(d, name)
    return name


class PBSLauncher(BatchLauncher):
    kind = "pbs"

    def __init__(self, *, bin_dir: str | None = None, **kw):
        super().__init__(**kw)
        self.bin_dir = bin_dir

    def render(self, spec: JobSpec) -> list[str]:
        rtype = self.opt(spec, "resource_type", "rtype")
        argv = [pbs_bin("qsub", self.bin_dir), "-V"]
        if rtype:
            argv += ["-v", f"RTYPE={rtype}"]
        argv += ["-N", spec.name]
        if q := self.opt(spec, "queue"):
            argv += ["-q", str(q)]
        if a := self.opt(spec, "account"):
            argv += ["-P", str(a)]
        argv += ["-l", pbs_select(spec, select=self.opt(spec, "select"), resource_type=rtype,
                                  host=self.opt(spec, "host"), mem=self.opt(spec, "mem"))]
        if spec.walltime:
            argv += ["-l", f"walltime={spec.walltime}"]
        if spec.log is not None:
            argv += ["-o", str(spec.log), "-j", "oe"]
        argv += [str(a) for a in (self.opt(spec, "extra_args") or [])]
        return argv + ["--", *job_command(spec)]

    def submit_env(self, spec: JobSpec) -> dict[str, str]:
        environ = os.environ if self.environ is None else self.environ
        env, _ = child_job_env(environ, spec.env, self.opt(spec, "resource_type", "rtype") or "")
        return env

    def parse_job_id(self, out: str) -> str:
        return out.strip().split()[0]            # "<id>.<server>"

    def status(self, job_id: str) -> JobStatus:
        # -x includes finished jobs, so Exit_status is read rather than guessed
        r = self.runner([pbs_bin("qstat", self.bin_dir), "-x", "-f", job_id], None)
        if r.returncode != 0:
            return JobStatus(JobState.DONE)       # purged from the server => finished
        return parse_qstat_f(r.stdout)

    def cancel(self, job_id: str) -> None:
        r = self.runner([pbs_bin("qdel", self.bin_dir), job_id], None)
        msg = (r.stderr + r.stdout).lower()
        if r.returncode != 0 and not any(s in msg for s in ("unknown job", "finished",
                                                             "job has finished")):
            raise RuntimeError(f"qdel {job_id} failed rc={r.returncode}: {msg.strip()}")
