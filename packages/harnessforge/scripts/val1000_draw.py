"""Draw the val1000 search-gate taskset from the tmax train pool, stratified by base-model difficulty (stdlib only).

Inputs (ABCI paths in the defaults):

  --pool     the GRPO train pool: a harbor dataset dir whose `task_*` subdirs are the tasks (6 726 tasks of the v2
             run's dataset_dir)
  --val200   the held-out val200 dataset dir (excluded; the script checks it does not overlap the pool)
  --sweep    harbor job dirs of a per-task pass-rate sweep (one dir per shard; each holds trial dirs
             `<task>__<suffix>/result.json`). Default: the Qwen3.5-4B hpcsb sweep (4 attempts, base harness, 1800 s
             agent timeout, vLLM, no 64k cap).

Per-task labels: successes / scored attempts, where an attempt is scored when its verifier produced a reward
(`verifier_result.rewards.reward`); attempts without one (sandbox/infra exceptions) are not evidence. Tasks with no
scored attempt are "unlabeled".

Strata (on the label rate): 0 | (0, .5) | .5 | (.5, 1) | 1 | unlabeled (with 4 scored attempts these are 0/4, 1/4,
2/4, 3/4, 4/4). The draw is proportional allocation (largest remainder) of `--n` tasks over the strata, simple random
sampling without replacement within each stratum (seeded), so val1000 has the pool's difficulty mix; the val1000 list
is then shuffled once (seeded): its order is the uniform task-major launch order. The rest of the pool is the screen
pool.

Outputs in --out:
  labels.json          {task: {"successes", "scored", "attempts", "rate", "errors": {exception: count}}} for every
                       task of the sweep (the step-0 difficulty labels: `python -m harnessforge.stats.sampling`)
  val1000.json         {"tasks": [...shuffled...], "labels": {task: rate | None}, "strata": ..., provenance}
  val1000.tasks.txt    one task per line, in that order
  screen_pool.json / screen_pool.tasks.txt   the remaining pool tasks (sorted)
  manifest.json        counts, strata mix (pool vs val1000), sweep error summary, inputs, seed

    python3 val1000_draw.py --out /groups/gai51740/aci18914wh/mho/tasksets/val1000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

G = "/groups/gai51740/aci18914wh/mho"
DEF_POOL = f"{G}/runs/platoon-grpo200-tmax-naive-uniform-noKL-64kcap-v2-ext--Qwen3.5-4B/dataset"
DEF_VAL200 = os.path.expanduser("~/mho-model-harness-optimization/data-harbor/tmax-v11-val200")
DEF_SWEEP = [f"{G}/sweeps/tmax-passrate-hpcsb/Qwen3.5-4B/shard{i}" for i in range(4)]
STRATA = ("0", "(0,.5)", ".5", "(.5,1)", "1", "unlabeled")


def task_dirs(d: str) -> list[str]:
    return sorted(e.name for e in os.scandir(d) if e.is_dir() and e.name.startswith("task_"))


def read_sweep(shards: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sh in shards:
        for e in os.scandir(sh):
            if not (e.is_dir() and "__" in e.name and e.name.startswith("task_")):
                continue
            task = e.name.split("__", 1)[0]
            row = out.setdefault(task, {"successes": 0, "scored": 0, "attempts": 0, "errors": Counter()})
            row["attempts"] += 1
            try:
                r = json.loads(Path(e.path, "result.json").read_text())
            except (OSError, ValueError):
                row["errors"]["no_result"] += 1
                continue
            exc = r.get("exception_info") or {}
            if exc:
                row["errors"][exc.get("exception_type") or "exception"] += 1
            reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            if reward is None:
                continue
            row["scored"] += 1
            row["successes"] += float(reward) > 0
    for row in out.values():
        row["rate"] = row["successes"] / row["scored"] if row["scored"] else None
        row["errors"] = dict(row["errors"])
    return out


def stratum(rate: float | None) -> str:
    if rate is None:
        return "unlabeled"
    if rate <= 0:
        return "0"
    if rate >= 1:
        return "1"
    if abs(rate - 0.5) < 1e-9:
        return ".5"
    return "(0,.5)" if rate < 0.5 else "(.5,1)"


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with largest remainders (ties by stratum order)."""
    total = sum(sizes.values())
    raw = {s: n * c / total for s, c in sizes.items()}
    out = {s: int(v) for s, v in raw.items()}
    rest = n - sum(out.values())
    for s in sorted(raw, key=lambda s: (-(raw[s] - out[s]), STRATA.index(s)))[:rest]:
        out[s] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=DEF_POOL)
    p.add_argument("--val200", default=DEF_VAL200)
    p.add_argument("--sweep", nargs="+", default=DEF_SWEEP)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    t0 = time.time()
    pool = task_dirs(a.pool)
    val200 = set(task_dirs(a.val200))
    overlap = sorted(val200 & set(pool))
    if overlap:
        print(f"val200 overlaps the pool on {len(overlap)} tasks; excluding them", file=sys.stderr)
    pool = [t for t in pool if t not in val200]
    sweep = read_sweep(a.sweep)
    print(f"pool {len(pool)} tasks, val200 {len(val200)}, sweep {len(sweep)} tasks "
          f"({sum(r['attempts'] for r in sweep.values())} attempts) in {time.time() - t0:.0f} s", flush=True)

    rate = {t: (sweep[t]["rate"] if t in sweep else None) for t in pool}
    by: dict[str, list[str]] = {s: [] for s in STRATA}
    for t in pool:
        by[stratum(rate[t])].append(t)
    sizes = {s: len(v) for s, v in by.items() if v}
    alloc = allocate(sizes, a.n)
    rng = random.Random(a.seed)
    chosen: list[str] = []
    for s in STRATA:
        if s in alloc:
            chosen += rng.sample(sorted(by[s]), alloc[s])
    rng.shuffle(chosen)
    rest = sorted(set(pool) - set(chosen))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"pool": a.pool, "val200": a.val200, "sweep": a.sweep, "seed": a.seed, "n": a.n,
              "sweep_settings": "Qwen3.5-4B, base harness (mini-swe-agent), 4 attempts, 1800 s agent timeout, vLLM, "
                                "no 64k cap (harbor; the default --sweep)"}
    (out / "labels.json").write_text(json.dumps(sweep, indent=0, sort_keys=True))
    val = {**inputs, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "order": "uniform (seeded shuffle)",
           "tasks": chosen, "labels": {t: rate[t] for t in chosen},
           "strata": {t: stratum(rate[t]) for t in chosen}}
    (out / "val1000.json").write_text(json.dumps(val, indent=1))
    (out / "val1000.tasks.txt").write_text("\n".join(chosen) + "\n")
    (out / "screen_pool.json").write_text(json.dumps({**inputs, "tasks": rest}, indent=1))
    (out / "screen_pool.tasks.txt").write_text("\n".join(rest) + "\n")

    def mix(ts: list[str]) -> dict[str, int]:
        c = Counter(stratum(rate[t]) for t in ts)
        return {s: c.get(s, 0) for s in STRATA}

    def mean_rate(ts: list[str]) -> float | None:
        v = [rate[t] for t in ts if rate[t] is not None]
        return round(sum(v) / len(v), 4) if v else None

    errs = Counter()
    for t in pool:
        errs.update((sweep.get(t) or {}).get("errors", {}))
    manifest = {**inputs, "pool_tasks": len(pool), "val200_overlap": len(overlap), "val1000": len(chosen),
                "screen_pool": len(rest), "pool_in_sweep": sum(t in sweep for t in pool),
                "strata_pool": mix(pool), "strata_val1000": mix(chosen), "strata_screen": mix(rest),
                "mean_rate_pool": mean_rate(pool), "mean_rate_val1000": mean_rate(chosen),
                "scored_attempts_hist_pool": dict(sorted(Counter((sweep.get(t) or {}).get("scored", 0)
                                                                 for t in pool).items())),
                "sweep_exceptions_pool": dict(errs.most_common())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sweep",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
