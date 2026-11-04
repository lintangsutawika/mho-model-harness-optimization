"""State persistence, determinism, and small pure helpers for the optimization loop."""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path


def detect_scheduler() -> str:
    kind = os.environ.get("MHO_SCHEDULER", "").lower()
    if kind in ("pbs", "slurm"):
        return kind
    if shutil.which("sbatch"):
        return "slurm"
    if shutil.which("qsub"):
        return "pbs"
    raise RuntimeError("no batch scheduler found (need sbatch or qsub); set MHO_SCHEDULER")


def _load_state(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def _save_state(path: Path, st: dict) -> None:
    path.write_text(json.dumps(st, indent=2))


def _sample_subset(task_dir: str | Path, k: int, out_dir: Path, seed: int,
                   sigma_reward: Path | None = None) -> Path:
    """Copy `k` random problem dirs from a local harbor task dir (`task_dir`) into a fresh
    `out_dir` (a valid `harbor run -p` target). Each problem = a subdir (e.g. `<id>-python/`)
    holding instruction.md + task.toml + tests/. Gives the proposer a BOUNDED,
    fresh-random subset of the full training task instead of eval'ing the whole train set.
    The sample is deterministic for a given (k, seed); fresh per iteration via a new seed.
    `sigma_reward` (e.g. <RUN>/candidates/<tag>/model/sigma_rewards/<check>.json) is the ONE
    sigma config for this candidate, applied to ALL of its train tasks: when given, it is copied
    into every staged task as sigma_reward.json JUST-IN-TIME, so the sigma-aware test.sh applies
    the sigma reward for this rollout batch without mutating the base train cohort.
    """
    task_dir, out_dir = Path(task_dir), Path(out_dir)
    if k <= 0:
        raise ValueError(f"propose-sample k must be > 0 (got {k})")
    problems = sorted(p for p in task_dir.iterdir() if p.is_dir())
    if not problems:
        raise ValueError(f"no problem dirs under train task dir: {task_dir}")
    sample = problems  # if k covers them all, no need to randomize beyond order
    if k < len(problems):
        rng = random.Random(seed)
        sample = rng.sample(problems, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma_path = Path(sigma_reward) if sigma_reward else None
    for p in sample:
        dst = out_dir / p.name
        shutil.copytree(p, dst, dirs_exist_ok=True)
        if sigma_path and sigma_path.is_file():
            shutil.copy2(sigma_path, dst / "sigma_reward.json")   # JIT: same sigma to all tasks
    return out_dir


def _candidate_hypothesis(candidate_dir: Path) -> str | None:
    """The candidate's own description of what it changed (CANDIDATE.md), for the summary so the
    next proposer sees what was tried. None if the agent didn't write one.
    Harness candidates write <cand>/harness/CANDIDATE.md; model candidates write <cand>/CANDIDATE.md
    (synthesized from the model_spec). Check both so the record is uniform across kinds.
    Called with the candidate ROOT (<RUN>/candidates/<tag>) in the loop, so look under
    <tag>/harness/CANDIDATE.md too (the harness proposer's output location)."""
    for md in (candidate_dir / "CANDIDATE.md",
               candidate_dir.parent / "CANDIDATE.md",
               candidate_dir / "harness" / "CANDIDATE.md"):
        if md.exists():
            return md.read_text()[:1000].strip()
    return None


def _latest_hf_ckpt(hf_dir: Path) -> Path | None:
    """Newest loadable HF checkpoint under a miles --save-hf dir (step_N subdirs), or the dir
    itself if it's already a checkpoint, else None."""
    if not hf_dir.is_dir():
        return None
    steps = sorted((d for d in hf_dir.glob("step_*") if (d / "config.json").exists()),
                   key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else -1)
    if steps:
        return steps[-1]
    return hf_dir if (hf_dir / "config.json").exists() else None