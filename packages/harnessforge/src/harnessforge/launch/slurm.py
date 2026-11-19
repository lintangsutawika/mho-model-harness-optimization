"""SLURM launcher, kept at parity with PBSLauncher (ported from mho.jobs.Scheduler).

JobSpec.options: queue (-p), account (-A), host (-w), mem (--mem), extra_args, cwd.
`select` and `resource_type` are PBS-only and ignored here. The job env is exported with
--export=ALL after stripping the submitter's SLURM_*/PBS_*/CUDA_VISIBLE_DEVICES/....
"""
from __future__ import annotations

import shlex

from harnessforge.contracts import JobSpec, JobState, JobStatus
from harnessforge.launch.base import BatchLauncher, job_command

_LIVE_PENDING = ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED", "RESV_DEL_HOLD")


class SlurmLauncher(BatchLauncher):
    kind = "slurm"

    def render(self, spec: JobSpec) -> list[str]:
        argv = ["sbatch", "--parsable", "--export=ALL", "-J", spec.name]
        if q := self.opt(spec, "queue"):
            argv += ["-p", str(q)]
        if a := self.opt(spec, "account"):
            argv += ["-A", str(a)]
        argv += [f"--nodes={spec.nodes}"]
        if spec.gpus:
            argv += [f"--gres=gpu:{spec.gpus}"]
        if spec.cpus:
            argv += [f"--cpus-per-task={spec.cpus}"]
        if m := self.opt(spec, "mem"):
            argv += [f"--mem={m}"]
        if spec.walltime:
            argv += ["-t", spec.walltime]
        if h := self.opt(spec, "host"):
            argv += ["-w", str(h)]
        if spec.log is not None:
            argv += ["-o", str(spec.log)]
        argv += [str(a) for a in (self.opt(spec, "extra_args") or [])]
        return argv + [f"--wrap={shlex.join(job_command(spec))}"]

    def parse_job_id(self, out: str) -> str:
        return out.strip().split()[-1].split(";")[0]     # --parsable: "<id>[;cluster]"

    def _hosts(self, nodelist: str) -> tuple[str, ...]:
        nodelist = nodelist.strip()
        if not nodelist or nodelist in ("None assigned", "(null)"):
            return ()
        r = self.runner(["scontrol", "show", "hostnames", nodelist], None)
        if r.returncode == 0 and r.stdout.strip():
            return tuple(r.stdout.split())
        return (nodelist,)

    def status(self, job_id: str) -> JobStatus:
        r = self.runner(["squeue", "-j", str(job_id), "-h", "-o", "%T|%N"], None)
        live = r.stdout.strip().splitlines()
        if r.returncode == 0 and live:
            st, _, nodes = live[0].partition("|")
            state = JobState.PENDING if st.strip() in _LIVE_PENDING else JobState.RUNNING
            return JobStatus(state, None, self._hosts(nodes) if state == JobState.RUNNING else ())
        s = self.runner(["sacct", "-j", str(job_id), "-X", "-n", "-P", "-o",
                         "State,ExitCode,NodeList"], None)
        exit_code, hosts = None, ()
        for line in s.stdout.splitlines():
            parts = line.split("|")
            if len(parts) >= 2 and parts[1].strip():
                try:
                    exit_code = int(parts[1].split(":")[0])
                except ValueError:
                    pass
                if len(parts) >= 3:
                    hosts = self._hosts(parts[2])
                break
        return JobStatus(JobState.DONE, exit_code, hosts)

    def cancel(self, job_id: str) -> None:
        r = self.runner(["scancel", str(job_id)], None)
        if r.returncode != 0 and "invalid job id" not in (r.stderr + r.stdout).lower():
            raise RuntimeError(f"scancel {job_id} failed rc={r.returncode}: {r.stderr.strip()}")
