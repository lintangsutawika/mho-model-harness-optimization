"""Phase / data environment builders shared by the loop's submit helpers."""
from __future__ import annotations

import os
from pathlib import Path

from . import REPO_DIR


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
    queues) are PBS-only escape hatches; the Scheduler ignores both on SLURM. Per-phase
    MHO_<PHASE>_<VAR> overrides the shared base MHO_<VAR>; unset falls back to phase defaults.
    """
    base = {k: os.environ.get(f"MHO_{k}") for k in ("QUEUE", "RTYPE", "SELECT", "WALLTIME", "NGPUS")}
    get = lambda k: os.environ.get(f"MHO_{phase}_{k}", base[k])
    return {
        "queue": get("QUEUE") or "",
        "select": get("SELECT") or "",
        "resource_type": get("RTYPE") or "",
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