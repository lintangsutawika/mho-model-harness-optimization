"""Materialize per-candidate scaffold snapshots for the meta-harness.

The meta-loop continually rewrites the minimal scaffold (micro-swe-agent's agent loop /
prompts / tools) and evaluates each candidate. Git is a poor fit for that inner loop
(commit churn, sandbox clone/auth), so instead each candidate gets an IMMUTABLE local
SNAPSHOT copied from a base checkout:

    runs/run_<run_id>/candidate_<candidate_id>/micro-swe-agent/

The eval points `MINI_FORK_LOCAL` at the snapshot; harbor uploads it into each Modal
sandbox and `uv tool install`s it. Because every run has its own `run_<id>` subtree and
every candidate its own `candidate_<id>` dir, concurrent runs and candidates never collide
and each snapshot is a reproducible artifact (the dir *is* the version).

Typical use (from a Python meta-loop):

    from harness.scaffold import materialize
    path = materialize(run_id, cand_id, mutate=lambda d: edit_scaffold(d))
    # then: eval with MINI_FORK_LOCAL=<path>  (see scripts/eval/eval_terminal_bench_2.sh)

or from a shell meta-loop:

    DIR=$(python -m harness.scaffold materialize --run-id "$RUN" --candidate-id "$C")
    MINI_FORK_LOCAL="$DIR" bash scripts/eval/eval_terminal_bench_2.sh ...
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
#: The base scaffold's directory name, preserved inside each candidate dir.
SCAFFOLD_NAME = "micro-swe-agent"
#: Base scaffold to snapshot from. Set MICRO_SCAFFOLD_BASE to the micro-swe-agent checkout;
#: otherwise fall back to a sibling of this repo (works only if the layout has them side by
#: side). Cluster layouts differ, so prefer the env var (or pass --base).
DEFAULT_BASE = Path(os.environ.get("MICRO_SCAFFOLD_BASE") or (REPO_ROOT.parent / SCAFFOLD_NAME))

# Never copy VCS / build / cache / venv artifacts into a snapshot: keeps uploads small,
# keeps the snapshot a clean installable package, and avoids shipping stale build output.
_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", "*.pyo", ".venv", "venv", "env",
    "dist", "build", "*.egg-info", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "runs",
)


def snapshot_dir(run_id: str, candidate_id: str, *, runs_dir: Path | str = RUNS_DIR) -> Path:
    """The snapshot path for a (run, candidate) -- does not touch the filesystem."""
    return Path(runs_dir) / f"run_{run_id}" / f"candidate_{candidate_id}" / SCAFFOLD_NAME


def materialize(
    run_id: str,
    candidate_id: str,
    *,
    base: Path | str = DEFAULT_BASE,
    runs_dir: Path | str = RUNS_DIR,
    mutate: Callable[[Path], None] | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy the base scaffold to runs/run_<id>/candidate_<id>/micro-swe-agent/ and return it.

    `mutate(dst)` (optional) is called after the copy to apply this candidate's edits in
    place (rewrite prompts, agent loop, tools). The snapshot must remain a valid
    installable package (pyproject.toml + src/) for harbor to `uv tool install` it.
    """
    base = Path(base)
    if not (base / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"base scaffold {base} is not an installable package (no pyproject.toml). "
            "Point it at your micro-swe-agent checkout: set MICRO_SCAFFOLD_BASE=<path> "
            "or pass --base <path>."
        )
    dst = snapshot_dir(run_id, candidate_id, runs_dir=runs_dir)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"snapshot already exists: {dst} (pass overwrite=True to replace)")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base, dst, ignore=_IGNORE)
    if mutate is not None:
        mutate(dst)
    if not (dst / "pyproject.toml").exists():
        raise RuntimeError(f"snapshot {dst} lost its pyproject.toml after mutate")
    return dst


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="harness.scaffold", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialize", help="create a candidate snapshot; prints its path")
    m.add_argument("--run-id", required=True)
    m.add_argument("--candidate-id", required=True)
    m.add_argument("--base", default=str(DEFAULT_BASE))
    m.add_argument("--runs-dir", default=str(RUNS_DIR))
    m.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "materialize":
        path = materialize(
            args.run_id, args.candidate_id,
            base=args.base, runs_dir=args.runs_dir, overwrite=args.overwrite,
        )
        print(path)


if __name__ == "__main__":
    _main()
