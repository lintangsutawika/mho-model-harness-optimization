"""Batch jobs for the optimization loop: the scheduler (PBS / SLURM) and the three phase
submitters (eval, train, propose). Each submitter turns (model, harness, data, ...) into the
phase script's MHO_* env and returns the job id; nothing here knows about loop state.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from utils.env import _ACCOUNT, _data_env, _phase_env, _phase_res, child_job_env
from utils.state import detect_scheduler

REPO_DIR = Path(__file__).resolve().parents[2]

# Phase job scripts submitted by the loop (one per phase). Each dispatches to the concrete
# entrypoint from env (eval.sh on MHO_TASK; train.sh on MHO_TRAINER). Overridable via env.
EVAL_SCRIPT = Path(os.environ.get("MHO_EVAL_SCRIPT", REPO_DIR / "scripts" / "optimize" / "eval.sh"))
TRAIN_SCRIPT = Path(os.environ.get("MHO_TRAIN_SCRIPT", REPO_DIR / "scripts" / "optimize" / "train.sh"))
# Propose job: the analysis/mutation step. Set MHO_PROPOSE_JOB (the run/*.sh front-ends point
# it at scripts/optimize/propose.sh). Absent -> the harness phase is skipped (model-only runs).
PROPOSE_JOB = os.environ.get("MHO_PROPOSE_JOB", "")

AGENT_IMPORT = os.environ.get("AGENT_IMPORT", "harness.micro_swe_agent:MicroSweAgent")
POLL_SECONDS = int(os.environ.get("MHO_POLL_SECONDS", 60))
LOG_VERBOSE = bool(os.environ.get("MHO_LOG_VERBOSE"))  # full job cmd in logs


# --------------------------------------------------------------------------- #
# Scheduler abstraction -- PBS (qsub/qstat) or SLURM (sbatch/squeue+sacct).
# scripts/hpc/ holds the per-scheduler shell wrappers; this is the Python driver side,
# so the loop is scheduler-agnostic. Force one with MHO_SCHEDULER=pbs|slurm.
class Scheduler:
    """Submit/poll batch jobs on PBS or SLURM behind one interface.

    Resources are given abstractly (ngpus/cpus/walltime/queue/account); each backend
    renders its own flags. Jobs export the full env (PBS -V / SLURM --export=ALL) so the
    phase config rides along without fragile per-key quoting.
    """

    def __init__(self, kind: str | None = None):
        self.kind = kind or detect_scheduler()

    def submit(self, script: Path, env: dict[str, str], *, name: str = "", ngpus: int = 0,
               cpus: int = 0, walltime: str = "", queue: str = "", account: str = "",
               select: str = "", resource_type: str = "", out_dir: Path | None = None,
               host: str = "") -> str:
        # `select` (raw PBS -l select string, e.g. ABCI's "1:ncpus=192:mem=1920gb:ngpus=8") wins
        # over the auto-built ngpus/cpus select -- clusters with full-node/reservation shapes need
        # the exact string. `resource_type` -> PBS `-v RTYPE=...`, required by ABCI reservation
        # (RESERVE) queues (e.g. rt_HF / rt_HC). On SLURM both are ignored (use gres/cpus).
        # qsub -V / --export=ALL export the SUBMITTER's env. Strip the driver job's own
        # PBS_*/RTYPE/CUDA_VISIBLE_DEVICES/TMPDIR/... first: e.g. an rt_HC driver's RTYPE made
        # ABCI's reservation hook put GPU evals on CPU-only nodes (ngpus=0).
        job_env, _dropped = child_job_env(dict(os.environ), env, resource_type)
        if LOG_VERBOSE and _dropped:
            print(f"  submit: not inheriting job-local env: {' '.join(_dropped)}", flush=True)
        # out_dir: a "*.out" path = this job's log file; otherwise a directory (legacy loop).
        log_file = bool(out_dir) and out_dir.suffix == ".out"
        if out_dir:
            (out_dir.parent if log_file else out_dir).mkdir(parents=True, exist_ok=True)
        if self.kind == "pbs":
            cmd = ["qsub", "-V"]
            if resource_type: cmd += ["-v", f"RTYPE={resource_type}"]
            if name: cmd += ["-N", name]
            if queue: cmd += ["-q", queue]
            if account: cmd += ["-P", account]
            sel = select or ("select=1" + (f":ngpus={ngpus}" if ngpus else "") + (f":ncpus={cpus}" if cpus else ""))
            if host:   # pin to one node: select=1:host=<h>:...
                head, _, rest = sel.partition(":")
                sel = f"{head}:host={host}" + (f":{rest}" if rest else "")
            cmd += ["-l", sel]
            if walltime: cmd += ["-l", f"walltime={walltime}"]
            if out_dir: cmd += ["-o", str(out_dir), "-j", "oe"]
            cmd.append(str(script))
        else:  # slurm
            cmd = ["sbatch", "--parsable", "--export=ALL"]
            if name: cmd += ["-J", name]
            if queue: cmd += ["-p", queue]
            if account: cmd += ["-A", account]
            if ngpus: cmd += [f"--gres=gpu:{ngpus}"]
            if cpus: cmd += [f"--cpus-per-task={cpus}"]
            if walltime: cmd += ["-t", walltime]
            if host: cmd += ["-w", host]
            if out_dir: cmd += ["-o", str(out_dir if log_file else out_dir / "%x-%j.out")]
            cmd.append(str(script))
        if LOG_VERBOSE:
            print(f"  submit[{self.kind}]: {' '.join(cmd)}", flush=True)
            # The harness/model/data ride via -V (inherited env), not visible in the cmd;
            # surface the selection vars so the log shows exactly what each job evaluated.
            _sel = {k: v for k, v in job_env.items()
                    if k in ("MHO_MODEL", "MHO_HARNESS", "MHO_DATA", "MHO_DATA_PATH",
                             "MHO_REPO_DIR", "MHO_TRAINER", "MHO_TASK",
                             "MHO_TRIALS_DIR", "MICRO_SCAFFOLD_DIR", "MINI_FORK_LOCAL",
                             "AGENT_IMPORT") and v}
            if _sel:
                print("      " + "  ".join(f"{k}={v}" for k, v in sorted(_sel.items())),
                      flush=True)
        else:
            _n = next((cmd[i+1] for i, c in enumerate(cmd) if c == "-N"), "job")
            print(f"  submit[{self.kind}] {_n}", flush=True)
        r = subprocess.run(cmd, cwd=str(REPO_DIR), env=job_env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"submit failed rc={r.returncode}: {r.stderr.strip() or r.stdout.strip()}")
        out = r.stdout.strip()
        # SLURM --parsable: "<id>[;cluster]"; PBS: "<id>.<server>".
        job_id = (out.split()[-1].split(";")[0]) if self.kind == "slurm" else out.split()[0]
        print(f"  -> {job_id}", flush=True)
        return job_id

    def state(self, job_id: str) -> tuple[str, int | None]:
        """Return (normalized_state, exit_status); normalized in PENDING/RUNNING/DONE."""
        if self.kind == "pbs":
            # -x: include finished jobs, so Exit_status is read instead of guessed
            r = subprocess.run(["qstat", "-x", "-f", job_id], capture_output=True, text=True)
            if r.returncode != 0:
                return "DONE", None  # purged from the queue => finished
            st, exit_status = "?", None
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith("job_state"):
                    st = line.split("=", 1)[1].strip()
                elif line.startswith("Exit_status"):
                    try:
                        exit_status = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
            norm = {"R": "RUNNING", "E": "RUNNING", "Q": "PENDING", "H": "PENDING",
                    "F": "DONE"}.get(st, "DONE" if exit_status is not None else "PENDING")
            return norm, exit_status
        # slurm: squeue for liveness, sacct for the final exit code
        r = subprocess.run(["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                           capture_output=True, text=True)
        live = r.stdout.strip()
        if live:
            return ("RUNNING" if live.startswith("RUN") else "PENDING"), None
        s = subprocess.run(["sacct", "-j", str(job_id), "-n", "-P", "-o", "State,ExitCode"],
                           capture_output=True, text=True)
        exit_status = None
        for line in s.stdout.splitlines():
            parts = line.split("|")
            if len(parts) >= 2 and parts[1].strip():
                try:
                    exit_status = int(parts[1].split(":")[0])
                except ValueError:
                    pass
                break
        return "DONE", exit_status

    def host(self, job_id: str) -> str | None:
        """The node a (running or finished) job ran on, if the scheduler still knows it."""
        if self.kind == "pbs":
            r = subprocess.run(["qstat", "-x", "-f", job_id], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.strip().startswith("exec_host"):
                    return line.split("=", 1)[1].strip().split("/")[0].split("+")[0] or None
            return None
        r = subprocess.run(["sacct", "-j", str(job_id), "-X", "-n", "-P", "-o", "NodeList"],
                           capture_output=True, text=True)
        node = (r.stdout.strip().splitlines() or [""])[0].strip()
        return node if node and node not in ("None assigned", "(null)") else None

    def wait(self, job_id: str) -> int:
        """Block until `job_id` finishes; return its exit status (0 if unreported)."""
        while True:
            norm, exit_status = self.state(job_id)
            if norm == "DONE":
                return exit_status if exit_status is not None else 0
            time.sleep(POLL_SECONDS)


def submit_eval(sched: Scheduler, task: str, *, model: str | None, harness: str | None,
                data: str, n_trials: int, n_concurrent: int, out_dir: Path, logs: Path,
                job_name: str, base_model: str | None = None) -> str:
    # The loop hands eval.sh the (model, harness, data) explicitly via MHO_* -- exactly the
    # manual form `MHO_MODEL=<model> MHO_DATA=<data> MHO_HARNESS=<harness> bash .../eval.sh`.
    # Trials land at MHO_OUT/<job_name>/ (eval.sh: EVAL_JOBS_DIR= MHO_OUT, EVAL_JOB_NAME= job_name).
    extra = {
        "AGENT_IMPORT": AGENT_IMPORT,
        "TASK_SET": "full",
        "N_ATTEMPTS": str(n_trials),
        "N_CONCURRENT": str(n_concurrent),
        "MHO_OUT": str(out_dir),
        "EVAL_JOB_NAME": job_name,
        **_data_env(data),
    }
    if model:
        extra["MHO_MODEL"] = model
    if harness:
        extra["MHO_HARNESS"] = harness
    if base_model:
        # `model` is a trained checkpoint dir: serve/sample it with its base model's configs
        # (configs/serve/<base>.json, configs/sampling/<base>.yaml).
        extra["MHO_BASE_MODEL"] = base_model
    if logs.suffix == ".out":       # <activity>.out -> <activity>-vllm.out
        extra["MHO_VLLM_LOG"] = str(logs.with_name(logs.stem + "-vllm.out"))
    # eval self-serves the model with vLLM on the node -> needs GPU(s). Default 1 (Qwen3.5-4B
    # fits on one H200); set MHO_EVAL_NGPUS=0 for an API/litellm model (no self-serve).
    return sched.submit(EVAL_SCRIPT, _phase_env(extra, None), name=f"mho-eval-{job_name}",
                        account=_ACCOUNT, out_dir=logs, **_phase_res("EVAL"))


def submit_train(sched: Scheduler, trainer: str, *, model: str | None, harness: str | None,
                 out_dir: Path, logs: Path, tag: str,
                 sigma_reward: str | None = None, train_data: str | None = None) -> str:
    extra = {"MHO_TRAINER": trainer, "MHO_OUT": str(out_dir)}
    if model:
        extra["MHO_MODEL"] = model
    if harness:
        extra["MHO_HARNESS"] = harness
    if sigma_reward:
        # The training script realizes this sigma-reward over the train cohort (injects it into
        # the staged train tasks just-in-time) so the model trains on sigma-shaped rewards.
        extra["MHO_SIGMA_REWARD"] = sigma_reward
    if train_data:
        extra["MHO_TRAIN_DATA"] = train_data
    return sched.submit(TRAIN_SCRIPT, _phase_env(extra, None), name=f"mho-train-{tag}",
                        account=_ACCOUNT, out_dir=logs, **_phase_res("TRAIN"))


def submit_propose(sched: Scheduler, *, run_root: Path, logs: Path, tag: str,
                   kind: str = "harness") -> str | None:
    import os as _os
    if _os.environ.get("MHO_LOOP_DEBUG"):
        print(f"  [DBG submit_propose] tag={tag} kind={kind} PROPOSE_JOB={PROPOSE_JOB!r}", flush=True)
    """Submit the HARNNES-mutation or MODEL-proposal job (MHO_PROPOSE_JOB). The proposer derives
    the parent scaffold + candidate dest from MHO_RUN_DIR/MHO_TAG, and -- per MHO_PROPOSE_KIND --
    writes either a harness edit (candidates/<tag>/harness) or a model sigma-reward config
    (candidates/<tag>/model/sigma_rewards/<sigma>.json). None if unconfigured."""
    if LOG_VERBOSE:
        print(f"  [TRACE submit_propose] tag={tag} PROPOSE_JOB={PROPOSE_JOB!r}", flush=True)
    if not PROPOSE_JOB:
        print("  MHO_PROPOSE_JOB unset -> skipping proposal this iteration")
        return None
    env = _phase_env({
        "MHO_RUN_DIR": str(run_root),
        "MHO_TAG": tag,
        "MHO_PROPOSE_KIND": kind,
    }, None)
    return sched.submit(Path(PROPOSE_JOB), env, name=f"mho-propose-{tag}",
                        account=_ACCOUNT, out_dir=logs, **_phase_res("PROPOSE"))


HOLD_SCRIPT = REPO_DIR / "scripts" / "utils" / "hold_node.sh"


def submit_hold(sched: Scheduler, host: str, seconds: int, logs: Path) -> str:
    """Park a sleeping job on a broken node so the retried eval cannot land on it again
    (PBS here cannot EXCLUDE a host, but it can pin a job to one). Same resources as an eval,
    so it takes the whole node."""
    res = _phase_res("EVAL")
    res["walltime"] = f"{(seconds + 600) // 3600:02d}:{(seconds + 600) % 3600 // 60:02d}:00"
    return sched.submit(HOLD_SCRIPT, _phase_env({"MHO_HOLD_SECONDS": str(seconds)}, None),
                        name=f"mho-hold-{host}", account=_ACCOUNT, out_dir=logs, host=host, **res)
