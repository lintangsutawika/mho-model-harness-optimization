"""Racing K candidates against one reference on the same tasks, all evaluated concurrently.

`Race` is a pure state machine. Feed it the current records snapshot (`EvalRun.records()`,
finished and in-flight trials of the reference and every candidate) as often as you like:

    race = Race("A", ["B1", "B2", "B3"], LookSchedule(run.order, tasks, trials))
    while True:
        st = race.update(run.records())
        for c in st.newly_decided:          # pass or fail: its remaining trials are not needed
            run.stop(c)
        if st.reference_can_stop:           # no candidate alive
            run.stop("A"); break
        run.wait(timeout=60)
    st.winner                               # largest t among passers, or None

Each candidate is tested independently with the group-sequential `Design`: looks are evaluated
in order (never skipping one whose data are ready), the first PASS or FAIL is final and frozen.
Calling `update` again with the same snapshot (or a later one) never changes a decided verdict
and never re-reports it in `newly_decided`, so the loop may poll freely.

Pairing (which reference data a candidate's look j is compared with):

  ReferenceMode.SAME_UNITS (default)  the reference's own look j: the same (task, index) units.
      Both sides have the same information at every look, the reference's look is ready as soon
      as it has finished the same units, so early looks happen while everything is still
      running. Under the null (identical harnesses) the per-task difference is symmetric, which
      is the situation the calibration (`calibrate --reference same`) simulates. Units are
      chosen by LAUNCH order on both sides, so neither side's look is biased toward fast trials.
  ReferenceMode.ALL_TRIALS  all of the reference's trials of the look's tasks. Use when the
      reference was evaluated in full beforehand (e.g. its val results are on file): more
      information on the reference side, but with a concurrent ATTEMPT_MAJOR reference no look
      can be taken before the reference has finished everything.

In both modes a look is decided only when BOTH sides' units are all launched and finished, and
the paired t uses only tasks where both sides have scored (non-env-failed) trials.

Env failures: `max_env_fail` (off by default) skips an interim look whose env-failed share (on
either side) exceeds it -- skipping an efficacy look only lowers the false-accept rate. At the
final look the decision is made anyway and flagged `infra_suspect`.

Multiplicity: every candidate is tested at the design's alpha, so with K candidates the chance
that SOME null candidate passes is up to K * alpha, and picking the max-t passer inflates the
winner's apparent effect (winner's curse). The race is a screen: the winner must then pass a
held-out sequential val gate (`harnessforge.stats.gate.ValGate`) before it is adopted. For
K-way control on the race itself, build the design with a Bonferroni c:
`Design.for_schedule(schedule, c=simulation.c_for(alpha / K))`.

Optional cheaper path (`cheaper=GateConfig(cheaper_delta=...)`): a candidate that fails the
better path at the FINAL look may still pass via non-inferiority + lower cost
(`paired.decide`). It is a fixed-sample test applied once at the end and is not part of the
sequential calibration; off by default.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from harnessforge.records import TrialRecord
from harnessforge.stats.boundaries import Design, Verdict
from harnessforge.stats.looks import LookData, LookSchedule, UnitIndex, collect
from harnessforge.stats.paired import Decision, GateConfig, Paired, decide, paired

__all__ = ["CandidateState", "Race", "RaceStatus", "ReferenceMode"]


class ReferenceMode(str, enum.Enum):
    SAME_UNITS = "same"
    ALL_TRIALS = "all"


def _r(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else round(x, 4)


@dataclass(frozen=True)
class CandidateState:
    candidate: str
    verdict: Verdict
    look: int | None = None          # look decided at (or the last look evaluated)
    acc: Paired | None = None
    efficacy: float | None = None    # boundaries at that look
    futility: float | None = None
    env_fail_frac: float = 0.0       # max over candidate and reference at that look
    infra_suspect: bool = False
    path: str | None = None          # "better" | "cheaper" when passed
    skipped: tuple[int, ...] = ()    # interim looks skipped as infra-suspect
    cheaper: Decision | None = None

    @property
    def alive(self) -> bool:
        return self.verdict is Verdict.CONTINUE

    @property
    def t(self) -> float | None:
        return self.acc.t if self.acc else None

    @property
    def reason(self) -> str:
        if self.look is None:
            return "no look ready yet"
        if self.acc is None:
            s = f"look {self.look}: {self.verdict.value}; no paired data"
        else:
            s = (f"look {self.look}: {self.verdict.value}; B-A={self.acc.d:+.4f}"
                 f"±{self.acc.se:.4f} (t={self.acc.t:.2f}; pass > {self.efficacy}, "
                 f"stop <= {self.futility})")
        if self.path == "cheaper" and self.cheaper is not None:
            s += f"; cheaper: {self.cheaper.reason}"
        if self.infra_suspect:
            s += f"; infra-suspect (env failures {self.env_fail_frac:.0%})"
        return s

    def as_dict(self) -> dict:
        return {"candidate": self.candidate, "verdict": self.verdict.value, "look": self.look,
                "acc": self.acc.as_dict() if self.acc else None, "efficacy": self.efficacy,
                "futility": self.futility, "env_fail_frac": round(self.env_fail_frac, 4),
                "infra_suspect": self.infra_suspect, "path": self.path,
                "skipped": list(self.skipped), "reason": self.reason}


@dataclass(frozen=True)
class RaceStatus:
    states: dict[str, CandidateState]
    stop: tuple[str, ...]            # every decided candidate (stopping twice must be harmless)
    newly_decided: tuple[str, ...]   # decided by this update
    reference_can_stop: bool         # no candidate alive
    winner: str | None               # once done: max-t passer
    ranking: tuple[str, ...] = ()    # passers, best first (once done)

    @property
    def done(self) -> bool:
        return self.reference_can_stop

    @property
    def alive(self) -> tuple[str, ...]:
        return tuple(c for c, s in self.states.items() if s.alive)

    def as_dict(self) -> dict:
        return {"states": {c: s.as_dict() for c, s in self.states.items()},
                "stop": list(self.stop), "newly_decided": list(self.newly_decided),
                "done": self.done, "winner": self.winner, "ranking": list(self.ranking)}


@dataclass
class _Track:
    next_look: int = 1
    state: CandidateState | None = None
    skipped: list[int] = field(default_factory=list)


class Race:
    def __init__(self, reference: str, candidates: Sequence[str], schedule: LookSchedule,
                 design: Design | None = None, *,
                 reference_mode: ReferenceMode | str = ReferenceMode.SAME_UNITS,
                 max_env_fail: float | None = None, cheaper: GateConfig | None = None):
        cands = list(candidates)
        if not cands:
            raise ValueError("no candidates")
        if len(set(cands)) != len(cands) or reference in cands:
            raise ValueError("candidate ids must be unique and differ from the reference")
        self.reference = reference
        self.candidates = tuple(cands)
        self.schedule = schedule
        self.design = design or Design.for_schedule(schedule)
        if self.design.n_looks != len(schedule.looks):
            raise ValueError(f"design has {self.design.n_looks} looks, schedule "
                             f"{len(schedule.looks)}")
        self.reference_mode = ReferenceMode(reference_mode)
        self.max_env_fail = max_env_fail
        self.cheaper = cheaper if cheaper is not None and cheaper.cheaper_enabled else None
        self._tracks = {c: _Track(state=CandidateState(c, Verdict.CONTINUE)) for c in cands}
        self._last: RaceStatus | None = None

    # ------------------------------------------------------------------ public
    def update(self, records: Iterable[TrialRecord] | UnitIndex) -> RaceStatus:
        idx = records if isinstance(records, UnitIndex) else UnitIndex(records)
        ref_cache: dict[int, LookData] = {}
        newly: list[str] = []
        for c in self.candidates:
            tr = self._tracks[c]
            if tr.state is not None and not tr.state.alive:
                continue
            if self._advance(c, tr, idx, ref_cache):
                newly.append(c)
        self._last = self._status(tuple(newly))
        return self._last

    def status(self) -> RaceStatus:
        return self._last or self._status(())

    # ------------------------------------------------------------------ internals
    def _ref_look(self, j: int, idx: UnitIndex, cache: dict[int, LookData]) -> LookData:
        if j not in cache:
            look = self.schedule.look(j)
            if self.reference_mode is ReferenceMode.ALL_TRIALS:
                look = self.schedule.full(look)
            cache[j] = collect(idx, self.reference, look)
        return cache[j]

    def _advance(self, c: str, tr: _Track, idx: UnitIndex, cache: dict[int, LookData]) -> bool:
        """Evaluate every newly ready look in order; True if the candidate got decided."""
        d = self.design
        while tr.next_look <= d.n_looks:
            j = tr.next_look
            b = collect(idx, c, self.schedule.look(j))
            if not b.ready:
                return False
            a = self._ref_look(j, idx, cache)
            if not a.ready:
                return False
            env = max(a.env_fail_frac, b.env_fail_frac)
            suspect = self.max_env_fail is not None and env > self.max_env_fail
            tr.next_look = j + 1
            if suspect and not d.is_final(j):
                tr.skipped.append(j)
                tr.state = CandidateState(c, Verdict.CONTINUE, j, None, _r(d.efficacy(j)),
                                          _r(d.futility_bound(j)), env, True, None,
                                          tuple(tr.skipped))
                continue
            acc = paired(a.stats, b.stats)
            v = d.verdict(j, acc.t if acc else None)
            path, cheap = ("better" if v is Verdict.PASS else None), None
            if v is Verdict.FAIL and d.is_final(j) and self.cheaper is not None:
                cheap = decide(a.stats, b.stats, cfg=self.cheaper, z=math.inf)
                if cheap.accept:
                    v, path = Verdict.PASS, "cheaper"
            tr.state = CandidateState(c, v, j, acc, _r(d.efficacy(j)), _r(d.futility_bound(j)),
                                      env, suspect, path, tuple(tr.skipped), cheap)
            if v is not Verdict.CONTINUE:
                return True
        return False

    def _status(self, newly: tuple[str, ...]) -> RaceStatus:
        states = {c: self._tracks[c].state or CandidateState(c, Verdict.CONTINUE)
                  for c in self.candidates}
        stop = tuple(c for c in self.candidates if not states[c].alive)
        done = len(stop) == len(self.candidates)
        ranking: tuple[str, ...] = ()
        if done:
            order = {c: i for i, c in enumerate(self.candidates)}
            passers = [c for c in self.candidates if states[c].verdict is Verdict.PASS]
            ranking = tuple(sorted(passers, key=lambda c: _rank_key(states[c], order[c])))
        return RaceStatus(states, stop, newly, done, ranking[0] if ranking else None, ranking)


def _rank_key(s: CandidateState, pos: int) -> tuple:
    """Better-path passers before cheaper-path ones; then larger t, larger d, input order."""
    t = s.acc.t if s.acc else -math.inf
    dd = s.acc.d if s.acc else -math.inf
    return (s.path != "better", -t, -dd, pos)
