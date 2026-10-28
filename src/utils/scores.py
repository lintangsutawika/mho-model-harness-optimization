"""Scoring / frontier / summary helpers for the optimization loop.

Reuses ``scripts/optimize/aggregate_results.py`` (no duplication). Everything here is
pure (takes paths, returns values); the loop's submit/orchestration logic does not live
here.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from . import REPO_DIR

# Path to the shared aggregation script.
AGGREGATE_PY = REPO_DIR / "scripts" / "optimize" / "aggregate_results.py"

_AGG = None


def _agg():
    global _AGG
    if _AGG is None:
        spec = importlib.util.spec_from_file_location("aggregate_results", AGGREGATE_PY)
        _AGG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_AGG)  # type: ignore[union-attr]
    return _AGG


def score_trials(trials_dir: str | Path) -> dict[str, Any]:
    trials_dir = Path(trials_dir)
    return _agg().aggregate(str(trials_dir), str(trials_dir / "aggregated.jsonl"))


def latest_trials_dir() -> str | None:
    return os.environ.get("MHO_TRIALS_DIR") or _agg()._newest_trials_dir()


def update_frontier(frontier_path: Path, candidate: str, score: dict[str, Any]) -> bool:
    frontier = json.loads(frontier_path.read_text()) if frontier_path.exists() else {}
    metric = score.get("mean_reward") or score.get("solved_pass@1") or 0.0
    improved = metric > frontier.get("_best", {}).get("metric", -1)
    if improved:
        frontier["_best"] = {"candidate": candidate, "metric": metric, "score": score}
    frontier.setdefault("history", []).append({"candidate": candidate, "metric": metric})
    frontier_path.write_text(json.dumps(frontier, indent=2))
    return improved


def append_summary(summary_path: Path, row: dict[str, Any]) -> None:
    with summary_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")