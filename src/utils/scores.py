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


def metric_of(score: dict[str, Any]) -> float:
    """The scalar a score dict is ranked by: mean_reward, else solved_pass@1, else 0.0.
    Uses `is None` (not `or`) so a real 0.0 mean_reward is kept, not replaced by pass@1."""
    for key in ("mean_reward", "solved_pass@1"):
        if score.get(key) is not None:
            return float(score[key])
    return 0.0


def update_frontier(frontier_path: Path, candidate: str, score: dict[str, Any],
                    accept: bool | None = None) -> bool:
    """`accept` = the gate's verdict (utils.gate); None -> plain metric > best. `_anchor` is
    the most accurate accepted entry (the cheaper path must stay non-inferior to it)."""
    frontier = json.loads(frontier_path.read_text()) if frontier_path.exists() else {}
    metric = metric_of(score)
    best = frontier.get("_best", {}).get("metric")
    improved = best is None or (metric > best if accept is None else accept)
    if improved:
        frontier["_best"] = {"candidate": candidate, "metric": metric, "score": score}
        anc = frontier.get("_anchor")
        if anc is None or metric > anc.get("metric", float("-inf")):
            frontier["_anchor"] = {"candidate": candidate, "metric": metric}
    frontier.setdefault("history", []).append({"candidate": candidate, "metric": metric,
                                               "accepted": bool(improved)})
    frontier_path.write_text(json.dumps(frontier, indent=2))
    return improved


def append_summary(summary_path: Path, row: dict[str, Any]) -> None:
    with summary_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")