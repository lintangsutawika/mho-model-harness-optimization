"""Discovery metrics and the planned measurement experiments of the toy benchmark.

Three questions, three measurements:

1. **Which features did the optimizer rediscover?** `discovery_report(candidate)` probes a
   candidate harness dir (no model, no sandbox). Over a search lineage,
   `discovery_timeline(steps)` gives, per feature, the first proposal / eval budget at which a
   proposed candidate had it and at which the accepted frontier kept it.
2. **How much of the start-to-gold gap did it close?** `gap_closed(start, gold, candidate)` on
   task-mean accuracies from `TrialRecord`s (`accuracy(records, candidate)`).
3. **Which features matter at all, and how much?** The ablation study: gold vs gold-minus-f per
   feature on the task set (`ablation_plan`, `targeted_ablation_plans`), and start vs gold
   (`gap_plan`). `ablation_effects(records, ...)` turns its records into paired per-task deltas.

The plans are runnable later: `EvalPlan.request(endpoint)` gives a contracts.EvalRequest for any
EvalBackend (workstream C) and `EvalPlan.units()` the flat (harness dir, tasks, trials) list.
Nothing here starts an evaluation.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from harnessforge.bench import builder
from harnessforge.bench.features import BY_NAME, FEATURE_NAMES
from harnessforge.bench.probes import ProbeResult, present_set, probe
from harnessforge.contracts import Candidate, EvalRequest
from harnessforge.records import TrialRecord, per_task

# ----------------------------------------------------------------------------- discovery
@dataclass(frozen=True)
class DiscoveryReport:
    candidate: str
    results: dict[str, ProbeResult]

    @property
    def present(self) -> frozenset[str]:
        return present_set(self.results)

    @property
    def n_present(self) -> int:
        return len(self.present)

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {"accuracy": 0, "efficiency": 0}
        for f in self.present:
            out[BY_NAME[f].kind] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"candidate": self.candidate, "n_present": self.n_present, "present": sorted(self.present),
                "by_kind": self.by_kind(),
                "features": {f: {"present": r.present, "evidence": r.evidence, "error": r.error}
                             for f, r in self.results.items()}}

    def summary(self) -> str:
        lines = [f"{self.candidate}: {self.n_present}/{len(self.results)} features present"]
        for f in FEATURE_NAMES:
            if f in self.results:
                r = self.results[f]
                lines.append(f"  [{'x' if r.present else ' '}] {f:24s} {r.evidence}")
        return "\n".join(lines)


def discovery_report(candidate: Candidate | str | Path, **probe_kw) -> DiscoveryReport:
    """Probe one candidate harness dir for all ten features."""
    if isinstance(candidate, Candidate):
        cid, path = candidate.id, candidate.harness_dir
    else:
        path = Path(candidate)
        cid = path.name
    return DiscoveryReport(cid, probe(path, **probe_kw))


@dataclass(frozen=True)
class LineageStep:
    """One proposal of a search run, in proposal order. `proposals` and `trials` are the
    cumulative counts spent up to and including this proposal's evaluation."""
    candidate: str
    present: frozenset[str]
    accepted: bool
    proposals: int
    trials: int


@dataclass(frozen=True)
class Discovery:
    feature: str
    first_proposed: LineageStep | None      # first proposal that had it (accepted or not)
    first_kept: LineageStep | None          # first accepted candidate that had it
    lost: bool                              # the final frontier no longer has it


def discovery_timeline(steps: Sequence[LineageStep], features: Iterable[str] = FEATURE_NAMES) -> dict[str, Discovery]:
    """Per feature: when it first appeared in any proposal, when the frontier first kept it, and
    whether the final frontier still has it."""
    final = next((s.present for s in reversed(steps) if s.accepted), frozenset())
    out = {}
    for f in features:
        proposed = next((s for s in steps if f in s.present), None)
        kept = next((s for s in steps if s.accepted and f in s.present), None)
        out[f] = Discovery(f, proposed, kept, lost=kept is not None and f not in final)
    return out


def discovery_curve(steps: Sequence[LineageStep], *, by: str = "trials") -> list[tuple[int, int]]:
    """[(budget, features kept by the frontier)] after each accepted step; `by` = trials|proposals."""
    out, frontier = [], frozenset()
    for s in steps:
        if s.accepted:
            frontier = s.present
        out.append((getattr(s, by), len(frontier)))
    return out


# ----------------------------------------------------------------------------- accuracy
def accuracy(records: Iterable[TrialRecord], candidate: str, *, tasks: set[str] | None = None) -> float:
    """Task-mean accuracy (each task weighted equally) over scored trials."""
    pt = per_task(records, candidate, tasks=tasks)
    return sum(s.reward for s in pt.values()) / len(pt) if pt else float("nan")


def gap_closed(start: float, gold: float, candidate: float) -> float:
    """Fraction of the start-to-gold accuracy gap a candidate closes (1 = as good as gold)."""
    if gold == start:
        return float("nan")
    return (candidate - start) / (gold - start)


@dataclass(frozen=True)
class PairedEffect:
    reference: str
    other: str
    delta: float        # mean over common tasks of (reference - other)
    se: float
    n_tasks: int

    @property
    def t(self) -> float:
        return self.delta / self.se if self.se > 0 else float("inf") if self.delta else 0.0


def paired_effect(records: Sequence[TrialRecord], reference: str, other: str, *,
                  tasks: set[str] | None = None) -> PairedEffect:
    a, b = per_task(records, reference, tasks=tasks), per_task(records, other, tasks=tasks)
    common = sorted(set(a) & set(b))
    d = [a[t].reward - b[t].reward for t in common]
    n = len(d)
    mean = sum(d) / n if n else float("nan")
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else float("nan")
    return PairedEffect(reference, other, mean, math.sqrt(var / n) if n > 1 else float("nan"), n)


def ablation_effects(records: Sequence[TrialRecord], *, gold: str = "gold",
                     families: dict[str, str] | None = None) -> dict[str, dict[str, PairedEffect]]:
    """Per feature: gold minus ablate_<feature>, on all tasks ("all") and, when `families`
    ({task: family}) is given, on the feature's own family ("family")."""
    out: dict[str, dict[str, PairedEffect]] = {}
    for f in FEATURE_NAMES:
        other = f"ablate_{f}"
        if not any(r.candidate == other for r in records):
            continue
        out[f] = {"all": paired_effect(records, gold, other)}
        if families:
            fam_tasks = {t for t, fam in families.items() if fam == BY_NAME[f].family}
            out[f]["family"] = paired_effect(records, gold, other, tasks=fam_tasks)
    return out


# ----------------------------------------------------------------------------- experiments
@dataclass(frozen=True)
class EvalPlan:
    """One planned evaluation: candidates x tasks x trials, compared against `reference`."""
    name: str
    candidates: tuple[Candidate, ...]
    tasks: tuple[str, ...]          # fixed order (the TASK_MAJOR look order)
    dataset: str                    # the generated task root (harbor dataset dir)
    trials: int
    reference: str | None = None
    note: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def units(self) -> list[tuple[Path, tuple[str, ...], int]]:
        """The flat [(harness dir, tasks, trials)] list."""
        return [(c.harness_dir, self.tasks, self.trials) for c in self.candidates]

    @property
    def n_trials(self) -> int:
        return len(self.candidates) * len(self.tasks) * self.trials

    def request(self, endpoint: str, *, run_id: str | None = None, policy: str | None = None,
                concurrency: int = 32, out_dir: Path | None = None) -> EvalRequest:
        return EvalRequest(run_id=run_id or self.name, candidates=self.candidates, tasks=self.tasks,
                           dataset=self.dataset, trials=self.trials, endpoint=endpoint, policy=policy,
                           concurrency=concurrency, out_dir=out_dir, options=dict(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "reference": self.reference, "dataset": self.dataset, "trials": self.trials,
                "n_tasks": len(self.tasks), "n_trials": self.n_trials, "note": self.note,
                "candidates": [{"id": c.id, "harness_dir": str(c.harness_dir)} for c in self.candidates]}


def harnesses(root: str | Path, *, build: bool = True) -> dict[str, Candidate]:
    """gold/start/ablate_* harness dirs under `root` as Candidates (materialized if missing)."""
    root = Path(root)
    names = list(builder.variants())
    if build:
        missing = [n for n in names if not (root / n).exists()]
        if missing:
            builder.build_all(root, names=missing)
    return {n: Candidate(id=n, harness_dir=root / n, meta={"features_on": sorted(builder.variants()[n])})
            for n in names}


def _tasks(tasks_root: str | Path, *, families: Iterable[str] | None = None, seed: int = 0) -> tuple[str, ...]:
    from harnessforge.bench.tasks.generate import read_manifest
    fams = set(families) if families is not None else None
    ids = [t["task_id"] for t in read_manifest(tasks_root) if fams is None or t["family"] in fams]
    random.Random(seed).shuffle(ids)            # one fixed shuffle: task-prefix looks are unbiased
    return tuple(ids)


def gap_plan(harness_root: str | Path, tasks_root: str | Path, *, trials: int = 8, seed: int = 0) -> EvalPlan:
    """start vs gold on every task: the size of the gap the optimizer should close."""
    h = harnesses(harness_root)
    return EvalPlan("gap_start_vs_gold", (h["gold"], h["start"]), _tasks(tasks_root, seed=seed), str(tasks_root),
                    trials, reference="gold", note="accuracy gap and token cost, start vs gold")


def ablation_plan(harness_root: str | Path, tasks_root: str | Path, *, trials: int = 8,
                  seed: int = 0) -> EvalPlan:
    """gold and every gold-minus-f on every task (11 candidates, one shared run)."""
    h = harnesses(harness_root)
    cands = (h["gold"], *(h[f"ablate_{f}"] for f in FEATURE_NAMES))
    return EvalPlan("ablation_all", cands, _tasks(tasks_root, seed=seed), str(tasks_root), trials,
                    reference="gold", note="per-feature effect on the whole task set (incl. spill-over)")


def targeted_ablation_plans(harness_root: str | Path, tasks_root: str | Path, *, trials: int = 16,
                            seed: int = 0) -> list[EvalPlan]:
    """Per feature: gold vs gold-minus-f on the feature's own family plus the control family
    (cheaper, more trials per task: the discoverability question)."""
    h = harnesses(harness_root)
    plans = []
    for f in FEATURE_NAMES:
        fams = {BY_NAME[f].family, "control"}
        plans.append(EvalPlan(f"ablation_{f}", (h["gold"], h[f"ablate_{f}"]),
                              _tasks(tasks_root, families=fams, seed=seed), str(tasks_root), trials,
                              reference="gold", note=f"effect of {f} on {sorted(fams)}"))
    return plans


def experiments(harness_root: str | Path, tasks_root: str | Path, *, trials: int = 8,
                targeted_trials: int = 16, seed: int = 0) -> list[EvalPlan]:
    """Every planned measurement, in the order to run them."""
    return [gap_plan(harness_root, tasks_root, trials=trials, seed=seed),
            ablation_plan(harness_root, tasks_root, trials=trials, seed=seed),
            *targeted_ablation_plans(harness_root, tasks_root, trials=targeted_trials, seed=seed)]


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m harnessforge.bench.metrics",
                                 description="Print the planned experiments (or probe candidates).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--harnesses", required=True, help="dir holding (or to receive) gold/start/ablate_*")
    p.add_argument("--tasks", required=True, help="generated task root (with manifest.json)")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--targeted-trials", type=int, default=16)
    r = sub.add_parser("report")
    r.add_argument("harness_dirs", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        plans = experiments(args.harnesses, args.tasks, trials=args.trials, targeted_trials=args.targeted_trials)
        print(json.dumps([pl.to_dict() for pl in plans], indent=2))
    else:
        for d in args.harness_dirs:
            print(discovery_report(d).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
