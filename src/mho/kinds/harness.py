"""Harness candidates: the proposer edits a copy of the current best harness into
candidates/<tag>/harness (an installable scaffold). No build step."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .base import Kind, Pair


class HarnessKind(Kind):
    name = "harness"

    def proposal_ok(self, cand_dir: Path) -> tuple[bool, str]:
        # The proposer's own per-candidate verdict (validity + changed vs the parent). A joint
        # session pre-creates every sibling's harness/ as a copy of the parent, so the
        # pyproject check alone cannot tell an untouched sibling from a real proposal.
        verdict = cand_dir / "propose.json"
        if verdict.exists():
            try:
                v = json.loads(verdict.read_text())
                return bool(v.get("ok")), v.get("reason", "") if not v.get("ok") else ""
            except ValueError:
                return False, "unreadable propose.json"
        ok = (cand_dir / "harness" / "pyproject.toml").exists()
        return ok, "" if ok else "no harness/pyproject.toml"

    def candidate_pair(self, cand_dir: Path, current: Pair) -> Pair:
        return Pair(harness=str(cand_dir / "harness"), model=current.model)

    def on_accept(self, cand_dir: Path, run_dir: Path, pair: Pair) -> None:
        # <run>/best/harness: the accepted scaffold, easy to grab after the run
        dst = run_dir / "best" / "harness"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(cand_dir / "harness", dst)
