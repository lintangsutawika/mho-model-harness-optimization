"""Frontier-weighted task-major looks: a stratified look order and its Horvitz-Thompson weights.

A task-major gate (val1000: 1 000 tasks x 4 trials, looks after 250/500/750/1 000 tasks) normally looks at prefixes of
ONE uniformly shuffled task list. With `TaskSampling` the order is instead built so that every look is a stratified
sample of the FULL set: tasks are grouped into difficulty bands by a label (a per-task pass rate from data NOT used in
the comparison, see `labels_from_records`), and each look takes, from every band, a fixed number of tasks, more from
the "frontier" bands than proportional, fewer from the bands that rarely change:

    band (label pass rate)    [0, .25)   [.25, .5)   [.5, .75)   [.75, 1]
    priority (DEFAULT)           3           1           2           4        (1 = most over-sampled)

Within a band the tasks are a seeded random permutation, and a look adds the next n_{b,j} - n_{b,j-1} tasks of band b
(sampling without replacement across looks; the last look has every task). So the tasks of band b in look j are a
simple random sample of the band, and task i's inclusion probability by look j is exactly

    pi_i = n_{b(i), j} / N_b.

The estimator at look j (`paired(weights=, strata=)`) is the Hajek / Horvitz-Thompson mean of the paired per-task
differences with weights w_i = 1 / pi_i = N_b / n_{b,j}:

    D_j = sum_i w_i d_i / sum_i w_i          (= sum_b (N_b / N) mean_{i in band b} d_i: the full-set mean)

so EVERY look estimates the full-population mean difference, not the mean of the over-sampled bands. Its
design-consistent SE is the stratified with-replacement (sandwich) variance

    SE^2 = sum_b n_b / (n_b - 1) sum_{i in b} (u_i - mean_b u)^2 / (sum w)^2,   u_i = w_i (d_i - D_j)
         = sum_b (N_b / N)^2 s_b^2 / n_b      (s_b^2: sample variance of d in band b),

which treats the tasks as a sample from a super-population (no finite-population correction), as the unweighted
statistic does. Observed information for alpha spending is 1 / SE^2 of this estimator. At the last look every
pi_i = 1, the weights are equal and the estimator IS the plain paired mean with sd / sqrt(n): the final analysis is the
uniform full-set analysis. Looks whose weights are all equal (tilt 0, or the last look) use the plain statistic.

Allocation. Band multipliers m_b = (n_bands - priority_b + 1) ** tilt (tilt 0: proportional; tilt 1: the frontier
band at 4x the rate of the 0.75-1 band). At look j with n_j tasks, the cumulative per-band fraction is
f_{b,j} = clip(c_j m_b, floor * n_j / N, 1), c_j solved so that sum_b N_b f_{b,j} = n_j; `floor` keeps every band
present from look 1 (at least floor x its proportional share). Counts are rounded with largest remainders, kept
non-decreasing across looks, and the last look takes every task. Tasks without a label form their own band with
multiplier 1 (they are never dropped).

HT is design-unbiased for ANY labels (the labels only fix the inclusion probabilities, and within-band sampling is
random); labels matter for efficiency. Labels from the comparison data itself (the reference's own trials) keep the
point estimate unbiased but make the weights correlated with the reference's noise; use an independent pass (a
1-trial label pass, the base-model sweep at step 0) or the previous reference's val results (label chaining,
`stats.reference.Lineage`).

Pure and stdlib-only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["DEFAULT_EDGES", "DEFAULT_PRIORITY", "TaskSampling", "allocate", "band_of", "frontier_order",
           "labels_from_records", "load_labels"]

DEFAULT_EDGES = (0.25, 0.5, 0.75)          # bands [0,.25) [.25,.5) [.5,.75) [.75,1]
DEFAULT_PRIORITY = (3, 1, 2, 4)            # per band, 1 = most over-sampled
UNLABELED = -1


def band_of(rate: float | None, edges: Sequence[float] = DEFAULT_EDGES) -> int:
    """Band index of a label pass rate (UNLABELED = -1 for None)."""
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        return UNLABELED
    for b, e in enumerate(edges):
        if rate < e:
            return b
    return len(edges)


@dataclass(frozen=True)
class TaskSampling:
    """The sampling design of a task-major look order: `bands[i]` is the band of `tasks[i]` (the schedule's task
    order), `sizes` the cumulative task count of each look (last = len(tasks)). Look j holds tasks[:sizes[j]]; per-band
    counts and inclusion probabilities follow from the order. `meta`: provenance (labels source, tilt, seed, ...)."""
    tasks: tuple[str, ...]
    bands: tuple[int, ...]
    sizes: tuple[int, ...]
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "bands", tuple(int(b) for b in self.bands))
        object.__setattr__(self, "sizes", tuple(int(s) for s in self.sizes))
        n = len(self.tasks)
        if len(self.bands) != n:
            raise ValueError("one band per task")
        if not self.sizes or any(b <= a for a, b in zip(self.sizes, self.sizes[1:])) or self.sizes[0] < 1 \
                or self.sizes[-1] != n:
            raise ValueError(f"sizes must increase to {n}: {self.sizes}")

    @property
    def band_sizes(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for b in self.bands:
            out[b] = out.get(b, 0) + 1
        return out

    def counts(self, j: int) -> dict[int, int]:
        """Per-band task count of look j (1-based)."""
        out: dict[int, int] = {}
        for b in self.bands[:self.sizes[j - 1]]:
            out[b] = out.get(b, 0) + 1
        return out

    def inclusion(self, j: int) -> dict[str, float]:
        """pi_i of each task of look j."""
        nb, cb = self.band_sizes, self.counts(j)
        return {t: cb[b] / nb[b] for t, b in zip(self.tasks[:self.sizes[j - 1]], self.bands)}

    def weights(self, j: int) -> dict[str, float] | None:
        """HT weights 1 / pi_i of look j's tasks; None when they are all equal (the plain mean)."""
        pi = self.inclusion(j)
        if max(pi.values()) - min(pi.values()) < 1e-12:
            return None
        return {t: 1.0 / p for t, p in pi.items()}

    def strata(self, j: int) -> dict[str, int]:
        return dict(zip(self.tasks[:self.sizes[j - 1]], self.bands))

    def design_effect(self, j: int) -> float:
        """Var(HT mean at look j) / Var(mean of a uniform sample of the same size), for equal per-task variance:
        n_j * sum_b (N_b / N)^2 / n_{b,j} (1 for proportional allocation; > 1 for a tilt)."""
        n, nb, cb = len(self.tasks), self.band_sizes, self.counts(j)
        return self.sizes[j - 1] * math.fsum((nb[b] / n) ** 2 / cb[b] for b in nb if cb.get(b))

    def as_dict(self) -> dict[str, Any]:
        return {"tasks": list(self.tasks), "bands": list(self.bands), "sizes": list(self.sizes),
                "meta": dict(self.meta),
                "counts": [{str(b): c for b, c in sorted(self.counts(j).items())} for j in range(1, len(self.sizes) + 1)]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TaskSampling":
        return cls(tuple(d["tasks"]), tuple(d["bands"]), tuple(d["sizes"]), dict(d.get("meta") or {}))

    @classmethod
    def uniform(cls, tasks: Sequence[str], sizes: Sequence[int]) -> "TaskSampling":
        """One band: look j is the first sizes[j] tasks, equal weights (the plain task-major schedule)."""
        return cls(tuple(tasks), (0,) * len(tasks), tuple(sizes), {"order": "uniform"})


def sizes_for(n: int, checkpoints: Sequence[float]) -> tuple[int, ...]:
    """Cumulative look sizes for task-count checkpoints (as `LookSchedule` computes them)."""
    return tuple(sorted({min(n, max(1, math.ceil(c * n - 1e-9))) for c in checkpoints} | {n}))


def multipliers(priority: Sequence[int], tilt: float) -> list[float]:
    top = max(priority)
    return [float(top - p + 1) ** tilt for p in priority]


def allocate(band_sizes: Mapping[int, int], sizes: Sequence[int], mult: Mapping[int, float],
             floor: float = 0.5) -> list[dict[int, int]]:
    """Cumulative per-band counts n_{b,j} for each look (see the module docstring)."""
    n = sum(band_sizes.values())
    bands = sorted(band_sizes)
    out: list[dict[int, int]] = []
    prev = {b: 0 for b in bands}
    for j, nj in enumerate(sizes):
        if nj >= n:
            cur = dict(band_sizes)
        else:
            lo = min(1.0, floor * nj / n)

            def total(c: float) -> float:
                return math.fsum(band_sizes[b] * min(1.0, max(lo, c * mult[b])) for b in bands)
            a, z = 0.0, 1.0
            while total(z) < nj:
                z *= 2.0
            for _ in range(200):
                mid = (a + z) / 2.0
                a, z = (mid, z) if total(mid) < nj else (a, mid)
            c = (a + z) / 2.0
            raw = {b: band_sizes[b] * min(1.0, max(lo, c * mult[b])) for b in bands}
            cur = {b: max(prev[b], min(band_sizes[b], int(math.floor(raw[b])))) for b in bands}
            # largest remainder up to n_j, never above the band size; then fix any excess from monotonicity
            rest = nj - sum(cur.values())
            order = sorted(bands, key=lambda b: (-(raw[b] - math.floor(raw[b])), b))
            while rest > 0:
                moved = False
                for b in order:
                    if rest > 0 and cur[b] < band_sizes[b]:
                        cur[b] += 1
                        rest -= 1
                        moved = True
                if not moved:
                    break
            while rest < 0:
                moved = False
                for b in sorted(bands, key=lambda b: (raw[b] - cur[b], b)):
                    if rest < 0 and cur[b] > prev[b]:
                        cur[b] -= 1
                        rest += 1
                        moved = True
                if not moved:
                    break
        out.append(cur)
        prev = cur
    return out


def frontier_order(tasks: Sequence[str], labels: Mapping[str, float | None], *,
                   checkpoints: Sequence[float] = (0.25, 0.5, 0.75, 1.0), tilt: float = 1.0, floor: float = 0.5,
                   edges: Sequence[float] = DEFAULT_EDGES, priority: Sequence[int] = DEFAULT_PRIORITY,
                   seed: int = 0, meta: Mapping[str, Any] | None = None) -> TaskSampling:
    """The frontier-weighted look order of `tasks` (any order in; deterministic in `seed`): see the module docstring.
    `labels`: task -> label pass rate (missing or None: unlabeled band). tilt 0 = proportional stratified order."""
    if len(priority) != len(edges) + 1:
        raise ValueError("one priority per band")
    if not 0 <= floor <= 1 or tilt < 0:
        raise ValueError("need 0 <= floor <= 1 and tilt >= 0")
    ts = sorted(set(tasks))
    if len(ts) != len(tasks):
        raise ValueError("duplicate task ids")
    n = len(ts)
    sizes = sizes_for(n, checkpoints)
    band = {t: band_of(labels.get(t), edges) for t in ts}
    groups: dict[int, list[str]] = {}
    for t in ts:
        groups.setdefault(band[t], []).append(t)
    rng = random.Random(seed)
    for b in sorted(groups):
        rng.shuffle(groups[b])
    m = multipliers(priority, tilt)
    mult = {b: (m[b] if b != UNLABELED else 1.0) for b in groups}
    alloc = allocate({b: len(g) for b, g in groups.items()}, sizes, mult, floor)
    order: list[str] = []
    prev = {b: 0 for b in groups}
    for cur in alloc:
        block = [t for b in sorted(groups) for t in groups[b][prev[b]:cur[b]]]
        rng.shuffle(block)
        order += block
        prev = cur
    info = {"order": "frontier", "tilt": tilt, "floor": floor, "edges": list(edges), "priority": list(priority),
            "seed": seed, "checkpoints": list(checkpoints), **(meta or {})}
    return TaskSampling(tuple(order), tuple(band[t] for t in order), sizes, info)


# ------------------------------------------------------------------------------------------------ labels
def labels_from_records(records: Iterable[Any], candidate: str | None = None) -> dict[str, float]:
    """Per-task pass rate over scored trials (`TrialRecord`s; a reward > 0 is a success)."""
    s: dict[str, list[float]] = {}
    for r in records:
        if candidate is not None and r.candidate != candidate:
            continue
        if not r.scored:
            continue
        s.setdefault(r.task, []).append(1.0 if (r.reward or 0) > 0 else 0.0)
    return {t: sum(v) / len(v) for t, v in s.items()}


def load_labels(path: str | Path, candidate: str | None = None) -> dict[str, float | None]:
    """Labels from a file: {task: rate}, {task: {"rate": ...}} (val1000_draw labels.json), {"labels": {...}}, or a
    TrialRecord JSONL (per-task pass rate of `candidate`, or of all records)."""
    p = Path(path)
    if p.suffix == ".jsonl":
        from harnessforge.records import read_jsonl
        return dict(labels_from_records(read_jsonl(p), candidate))
    raw = json.loads(p.read_text())
    if isinstance(raw, dict) and isinstance(raw.get("labels"), dict):
        raw = raw["labels"]
    out: dict[str, float | None] = {}
    for t, v in raw.items():
        out[t] = v.get("rate") if isinstance(v, dict) else (None if v is None else float(v))
    return out


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    """python -m harnessforge.stats.sampling --tasks val1000.json --labels labels.json --tilt 1 --out sampling.json"""
    p = argparse.ArgumentParser(prog="python -m harnessforge.stats.sampling",
                                description="frontier-weighted task-major look order (see the module docstring)")
    p.add_argument("--tasks", required=True, help="JSON list, {'tasks': [...]}, or a text file with one task per line")
    p.add_argument("--labels", help="label file (see load_labels); omitted: every task unlabeled (uniform)")
    p.add_argument("--labels-candidate", help="with a JSONL label file: whose records")
    p.add_argument("--lineage", help="label chaining: labels from a reference lineage (stats.reference.Lineage): the "
                                     "previous reference's val results, else its step-0 label file")
    p.add_argument("--checkpoints", default="0.25,0.5,0.75,1")
    p.add_argument("--tilt", type=float, default=1.0)
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--edges", default=",".join(map(str, DEFAULT_EDGES)))
    p.add_argument("--priority", default=",".join(map(str, DEFAULT_PRIORITY)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="sampling JSON (TaskSampling.as_dict); `tasks` is the launch order")
    a = p.parse_args(argv)
    tp = Path(a.tasks)
    if tp.suffix == ".json":
        raw = json.loads(tp.read_text())
        tasks = raw["tasks"] if isinstance(raw, dict) else raw
    else:
        tasks = [x.strip() for x in tp.read_text().splitlines() if x.strip()]
    if a.lineage and a.labels:
        raise SystemExit("give --labels or --lineage")
    if a.lineage:
        from harnessforge.stats.reference import Lineage
        lin = Lineage.load(a.lineage)
        labels, src = lin.labels(), lin.labels_source()
    else:
        labels = load_labels(a.labels, a.labels_candidate) if a.labels else {}
        src = {"kind": "file", "path": a.labels}
    s = frontier_order(tasks, labels, checkpoints=[float(x) for x in a.checkpoints.split(",")], tilt=a.tilt,
                       floor=a.floor, edges=[float(x) for x in a.edges.split(",")],
                       priority=[int(x) for x in a.priority.split(",")], seed=a.seed,
                       meta={"labels": src})
    Path(a.out).write_text(json.dumps(s.as_dict(), indent=1))
    print(json.dumps({"tasks": len(s.tasks), "band_sizes": s.band_sizes,
                      "looks": [{"n": n, "counts": s.counts(j), "design_effect": round(s.design_effect(j), 3)}
                                for j, n in enumerate(s.sizes, start=1)]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
