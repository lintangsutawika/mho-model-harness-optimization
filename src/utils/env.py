"""Phase / data environment builders shared by the loop's submit helpers."""
from __future__ import annotations

import os
from pathlib import Path

from . import REPO_DIR

# Variables that describe the SUBMITTING job/node/terminal, not the config a child job needs.
# Submits export the driver's env (qsub -V / sbatch --export=ALL); when the driver itself runs
# inside a batch job these would leak into every child (e.g. RTYPE=rt_HC put GPU evals on
# CPU nodes). The scheduler sets the child's own values for these at job start.
# SBATCH_*/SALLOC_* are sbatch/salloc INPUT variables (e.g. SBATCH_PARTITION = default -p): a
# driver launched with them would push its own partition/account onto every child job.
_JOB_LOCAL_PREFIXES = ("PBS_", "SLURM_", "SBATCH_", "SALLOC_", "TMUX")
_JOB_LOCAL_VARS = frozenset({
    "RTYPE", "JOB_NAME",                                     # submit_pbs.sh knobs
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "NCPUS", "OMP_NUM_THREADS", "TMPDIR", "ENVIRONMENT", "HOSTNAME",
    "SSH_TTY", "SSH_CONNECTION", "SSH_CLIENT", "OLDPWD",
})


def child_job_env(environ: dict[str, str], extra: dict[str, str],
                  resource_type: str = "") -> tuple[dict[str, str], list[str]]:
    """The env a submitted job inherits: the driver's env minus job-local variables, plus
    `extra` (this job's config), plus RTYPE only from this job's own resource type.
    Returns (env, dropped_names)."""
    dropped = sorted(k for k in environ
                     if k in _JOB_LOCAL_VARS or k.startswith(_JOB_LOCAL_PREFIXES))
    env = {k: v for k, v in environ.items() if k not in dropped}
    env.update({k: str(v) for k, v in extra.items()})
    if resource_type:
        env["RTYPE"] = resource_type
    return env, dropped


def _phase_env(extra: dict[str, str], scaffold_dir: Path | None) -> dict[str, str]:
    env = dict(extra)
    # PBS spools the job script to /var/spool/pbs/..., so the phase scripts' BASH_SOURCE-based
    # repo-dir resolution is wrong in the job. Pass the real repo dir explicitly.
    env["MHO_REPO_DIR"] = str(REPO_DIR)
    env["PYTHONPATH"] = f"{REPO_DIR / 'src'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    if scaffold_dir is not None:
        env["MICRO_SCAFFOLD_DIR"] = str(scaffold_dir)
        env["MINI_FORK_LOCAL"] = str(scaffold_dir)
    return env


_ACCOUNT = os.environ.get("MHO_ACCOUNT", "")  # PBS -P / SLURM -A


def _phase_res(phase: str) -> dict[str, str]:
    """Resolve one phase's resource vars with per-phase override -> shared base -> default.
    Scheduler-agnostic: NGPUS/CPUS are the PORTABLE primary signal -- the Scheduler builds
    PBS ``-l select=1:ngpus=N:ncpus=C`` and SLURM ``--gres=gpu:N --cpus-per-task=C`` from
    them, so the same MHO_<PHASE>_NGPUS/CPUS works on either backend. QUEUE/ACCOUNT/WALLTIME
    are already neutral (PBS -q/-P/-l walltime vs SLURM -p/-A/-t). SELECT (raw PBS
    ``-l select=`` for full-node/reservation shapes) and RTYPE (PBS ``-v RTYPE=``, ABCI RESERVE
    queues) are PBS-only escape hatches; the Scheduler ignores both on SLURM. QOS/MEM are the
    SLURM-only counterparts (--qos / --mem), e.g. a CPU-only QoS when the default one needs
    a GPU; PBS ignores them. Per-phase
    MHO_<PHASE>_<VAR> overrides the shared base MHO_<VAR>; unset falls back to phase defaults.
    """
    base = {k: os.environ.get(f"MHO_{k}") for k in ("QUEUE", "RTYPE", "SELECT", "WALLTIME", "NGPUS", "QOS", "MEM")}
    get = lambda k: os.environ.get(f"MHO_{phase}_{k}", base[k])
    return {
        "queue": get("QUEUE") or "",
        "select": get("SELECT") or "",
        "resource_type": get("RTYPE") or "",
        "qos": get("QOS") or "",
        "mem": get("MEM") or "",
        "walltime": get("WALLTIME") or ("08:00:00" if phase == "EVAL" else "24:00:00" if phase == "TRAIN" else "02:00:00"),
        "ngpus": int(get("NGPUS") or (1 if phase == "EVAL" else 8 if phase == "TRAIN" else 0)),
        "cpus": int(os.environ.get(f"MHO_{phase}_CPUS", 0) or 0),
    }


def _data_env(data: str) -> dict[str, str]:
    """Map a data spec to the phase script's -p/-d env: a path on disk -> MHO_DATA_PATH (-p),
    else a registry name -> MHO_DATA (-d)."""
    if data and Path(data).exists():
        return {"MHO_DATA_PATH": data}
    return {"MHO_DATA": data} if data else {}