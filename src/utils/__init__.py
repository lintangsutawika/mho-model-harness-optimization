"""Shared helper functions extracted from ``optimize_loop`` (scoring, env building, state IO).

Kept decoupled from the PBS/SLURM loop orchestration so they are unit-testable and
reusable. The loop imports these directly; nothing here imports ``optimize_loop``.
"""
from __future__ import annotations

from pathlib import Path

# Repository root: this file lives at <repo>/src/utils/, so climb 3 levels.
REPO_DIR = Path(__file__).resolve().parents[2]

from . import env, scores, state  # noqa: E402  (REPO_DIR must be defined first)

__all__ = ["REPO_DIR", "env", "scores", "state"]