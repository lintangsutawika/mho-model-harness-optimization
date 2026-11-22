"""Reference reuse on a held-out gate: top-up plans and reference lineage.

The val1000 gate compares a candidate with the current reference on the same 1 000 tasks x 4 trials
(`ReferenceMode.ALL_TRIALS`: the reference's val1000 trials are on file). When a candidate is ACCEPTED at look k it
becomes the next reference, and its gate trials are reused: only the units it does not have (tasks after look k's
prefix, trials cancelled by the stop, env failures) are run, as a top-up.

    plan = topup_plan(records, "B1", tasks, trials=4)       # missing (task, index) units of B1
    # run plan.request_tasks() for B1 (one trial count per task), then
    merged = apply_topup(records_of_B1, topup_records, plan)  # top-up trials re-indexed onto the missing units

Winner's curse. A candidate accepted at an interim look was accepted BECAUSE its trials on that look's tasks came out
high; reused as a reference, those trials overstate it, and the next candidate's paired differences are biased down on
those tasks (docs/stats.md section 8.3: +0.26 to +0.40 points after an accept at look 1 or 2, about 0 after look 3,
and -0.20 points after an accept at the final look).
`Lineage.accept(..., fresh=...)` can therefore ask for a FRESH full evaluation of the new reference instead of a top-up:
"never" (always top up), "marginal" (fresh after a marginal accept: at the final look, or t within `margin` of the
efficacy boundary), "early" (fresh after an accept at look <= `early_look`, default 2: where the simulation puts the
upward bias), "looks" (fresh after an accept at one of `fresh_looks`, e.g. (1, 2, 4): every look except the one
where reuse is unbiased in the simulation), "always". An accept at the FINAL look biases the reused reference DOWNWARD (the candidate was not
accepted earlier, so its early-look trials came out low), which makes the next gate slightly anti-conservative; only
"always" removes both directions (section 8.3).

Label chaining. The frontier-weighted look order (`sampling.frontier_order`) needs per-task difficulty labels from data
NOT used in the comparison. The comparison uses the CURRENT reference's trials, so the labels come from the PREVIOUS
reference's val results (`Lineage.labels_source`), or, before any accept, from an independent pass (the base-model
sweep, or a 1-trial label pass of a harness).

Pure and stdlib-only.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from harnessforge.records import TrialRecord

__all__ = ["FRESH_POLICIES", "Lineage", "TopUp", "apply_topup", "is_marginal", "topup_plan"]

FRESH_POLICIES = ("never", "marginal", "early", "looks", "always")
CANCELLED = frozenset({"stopped", "cancelled"})


@dataclass(frozen=True)
class TopUp:
    """The units (task, trial index) `reference` still needs to have every task of `tasks` at `trials` scored trials."""
    reference: str
    tasks: tuple[str, ...]
    trials: int
    missing: dict[str, tuple[int, ...]]      # task -> missing indices (sorted), tasks in `tasks` order
    fresh: bool = False                      # the whole evaluation is to be re-run (no reuse)

    @property
    def n_units(self) -> int:
        return sum(len(v) for v in self.missing.values())

    @property
    def reused_units(self) -> int:
        return len(self.tasks) * self.trials - self.n_units

    def request_tasks(self) -> list[tuple[str, int]]:
        """(task, trials to run) for the top-up evaluation, in the gate's task order."""
        return [(t, len(ix)) for t, ix in self.missing.items()]

    def as_dict(self) -> dict[str, Any]:
        return {"reference": self.reference, "trials": self.trials, "tasks": len(self.tasks), "fresh": self.fresh,
                "units_missing": self.n_units, "units_reused": self.reused_units,
                "tasks_missing": sum(1 for v in self.missing.values() if v),
                "by_count": _hist(len(v) for v in self.missing.values()),
                "missing": {t: list(ix) for t, ix in self.missing.items()}}


def _hist(xs: Iterable[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _usable(r: TrialRecord) -> bool:
    return r.finished and r.scored and r.error not in CANCELLED


def topup_plan(records: Iterable[TrialRecord], reference: str, tasks: Sequence[str], trials: int, *,
               fresh: bool = False) -> TopUp:
    """Missing units of `reference`: indices 0..trials-1 of each task without a finished, scored (not env-failed,
    not cancelled) record. `fresh`: every unit (re-evaluate from scratch)."""
    have: dict[str, set[int]] = {}
    if not fresh:
        for r in records:
            if r.candidate == reference and _usable(r) and r.index < trials:
                have.setdefault(r.task, set()).add(r.index)
    missing = {}
    for t in tasks:
        ix = tuple(i for i in range(trials) if i not in have.get(t, ()))
        if ix:
            missing[t] = ix
    return TopUp(reference, tuple(tasks), trials, missing, fresh)


def apply_topup(base: Iterable[TrialRecord], topup: Iterable[TrialRecord], plan: TopUp) -> list[TrialRecord]:
    """The reference's records after the top-up: its usable base records (none when `plan.fresh`) plus the top-up
    run's records re-indexed onto the plan's missing units (the top-up's k-th launched trial of a task, by its own
    index, fills the k-th missing index) and relabelled as `plan.reference`. Top-up records beyond the plan are
    dropped; missing units the top-up did not fill stay missing (`topup_plan` on the result lists them)."""
    out = [] if plan.fresh else [r for r in base if r.candidate == plan.reference and _usable(r)
                                 and r.task in set(plan.tasks) and r.index < plan.trials]
    by: dict[str, list[TrialRecord]] = {}
    for r in topup:
        by.setdefault(r.task, []).append(r)
    for t, ix in plan.missing.items():
        rs = sorted(by.get(t, ()), key=lambda r: r.index)
        for slot, r in zip(ix, rs):
            out.append(replace(r, candidate=plan.reference, index=slot,
                               extra={**r.extra, "topup_index": r.index, "topup_candidate": r.candidate}))
    return out


def is_marginal(state: Any, margin: float = 0.5, n_looks: int | None = None) -> bool:
    """A pass that barely made it: decided at the final look (information fraction 1, or look `n_looks`), or
    t - efficacy < `margin` (`state`: a `race.CandidateState` or its as_dict)."""
    d = state if isinstance(state, Mapping) else state.as_dict()
    t = (d.get("acc") or {}).get("t")
    eff = d.get("efficacy")
    fr = d.get("fraction")
    final = (fr is not None and fr >= 1 - 1e-9) or (n_looks is not None and d.get("look") == n_looks)
    if final:
        return True
    if t is None or eff is None or not math.isfinite(t):
        return False
    return t - eff < margin


@dataclass
class Lineage:
    """Who the gate's reference is and where its trials and the labels come from.

    `current`: the reference the next candidate is compared with (its val records on file); `previous`: the
    reference before it (its val per-task pass rates are the difficulty labels of the next gate's frontier order);
    `step0_labels`: the label file used while there is no previous reference (an independent pass). `history`: one
    entry per accept (candidate, look, t, marginal, fresh, top-up size)."""
    current: str
    previous: str | None = None
    step0_labels: str | None = None
    records: dict[str, list[str]] = field(default_factory=dict)    # reference id -> record files (gate + top-up)
    history: list[dict[str, Any]] = field(default_factory=list)

    def labels_source(self) -> dict[str, Any]:
        """Where the next gate's labels come from: the previous reference's records, else the step-0 file."""
        if self.previous is not None:
            return {"kind": "previous_reference", "candidate": self.previous,
                    "records": list(self.records.get(self.previous, []))}
        return {"kind": "step0", "path": self.step0_labels}

    def labels(self) -> dict[str, float | None]:
        """Load the labels of `labels_source()`."""
        from harnessforge.stats.sampling import labels_from_records, load_labels
        src = self.labels_source()
        if src["kind"] == "previous_reference":
            from harnessforge.records import read_jsonl
            recs = [r for p in src["records"] for r in read_jsonl(p)]
            return dict(labels_from_records(recs, src["candidate"]))
        return load_labels(src["path"]) if src["path"] else {}

    def accept(self, candidate: str, state: Any, records: Iterable[TrialRecord], tasks: Sequence[str], trials: int,
               *, fresh: str = "never", margin: float = 0.5, n_looks: int | None = None, early_look: int = 2,
               fresh_looks: Sequence[int] = (1, 2, 4),
               record_files: Sequence[str] = ()) -> tuple["Lineage", TopUp]:
        """`candidate` passed the gate (`state`: its CandidateState): the new lineage (candidate = current, the old
        current = previous) and the top-up plan for the candidate's val trials (a full fresh plan when the `fresh`
        policy asks for one)."""
        if fresh not in FRESH_POLICIES:
            raise ValueError(f"fresh must be one of {FRESH_POLICIES}")
        marginal = is_marginal(state, margin, n_looks)
        look = state.get("look") if isinstance(state, Mapping) else state.look
        early = look is not None and look <= early_look and not (n_looks is not None and look == n_looks)
        go_fresh = (fresh == "always" or (fresh == "marginal" and marginal) or (fresh == "early" and early)
                    or (fresh == "looks" and look in set(fresh_looks)))
        plan = topup_plan(records, candidate, tasks, trials, fresh=go_fresh)
        d = state if isinstance(state, Mapping) else state.as_dict()
        entry = {"accepted": candidate, "over": self.current, "look": d.get("look"),
                 "t": (d.get("acc") or {}).get("t"), "efficacy": d.get("efficacy"), "marginal": marginal,
                 "fresh": go_fresh, "units_missing": plan.n_units, "units_reused": plan.reused_units}
        recs = dict(self.records)
        recs[candidate] = list(record_files)
        new = Lineage(candidate, self.current, self.step0_labels, recs, [*self.history, entry])
        return new, plan

    def add_records(self, candidate: str, *paths: str) -> None:
        self.records.setdefault(candidate, []).extend(paths)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "Lineage":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.as_dict(), indent=1))
        tmp.replace(p)
