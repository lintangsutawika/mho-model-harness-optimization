"""Launchers: submit `JobSpec`s to PBS, SLURM or local subprocesses (contracts.Launcher)."""
from __future__ import annotations

import os
import shutil
from typing import Any, Mapping

from harnessforge.launch.base import child_job_env
from harnessforge.launch.local import LocalLauncher
from harnessforge.launch.pbs import PBSLauncher
from harnessforge.launch.slurm import SlurmLauncher

__all__ = ["LocalLauncher", "PBSLauncher", "SlurmLauncher", "child_job_env", "detect",
           "make_launcher"]


def detect() -> str:
    """'pbs' | 'slurm' | 'local' from what this machine has."""
    if os.environ.get("HF_LAUNCHER"):
        return os.environ["HF_LAUNCHER"]
    if shutil.which("qsub"):
        return "pbs"
    if shutil.which("sbatch"):
        return "slurm"
    return "local"


def make_launcher(kind: str | None = None, defaults: Mapping[str, Any] | None = None, **kw):
    """A launcher by kind ('pbs', 'slurm', 'local'; None = detect). `defaults` are option
    defaults merged under every JobSpec.options (queue, account, resource_type, ...)."""
    kind = kind or detect()
    if kind == "pbs":
        return PBSLauncher(defaults=defaults, **kw)
    if kind == "slurm":
        return SlurmLauncher(defaults=defaults, **kw)
    if kind == "local":
        return LocalLauncher(**kw)
    raise ValueError(f"unknown launcher kind {kind!r}")
