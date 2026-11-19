"""Harness candidates: the proposer edits a copy of the current best harness into
candidates/<tag>/harness (an installable scaffold). No build step."""
from __future__ import annotations

import shutil
from pathlib import Path

from .base import Kind, Pair


class HarnessKind(Kind):
    name = "harness"

    def __init__(self, accept: str = "gate") -> None:
        if accept not in ("gate", "train"):
            raise ValueError(f"harness accept must be gate|train, got {accept!r}")
        self.accept_on_train = accept == "train"

    def proposal_ok(self, cand_dir: Path) -> tuple[bool, str]:
        ok = (cand_dir / "harness" / "pyproject.toml").exists()
        return ok, "" if ok else "no harness/pyproject.toml"

    def candidate_pair(self, cand_dir: Path, current: Pair) -> Pair:
        return Pair(harness=str(cand_dir / "harness"), model=current.model)

    def on_accept(self, cand_dir: Path, run_dir: Path, pair: Pair) -> None:
        # <run>/best/harness: the accepted scaffold, easy to grab after the run
        dst = run_dir / "best" / "harness"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(cand_dir / "harness", dst)
