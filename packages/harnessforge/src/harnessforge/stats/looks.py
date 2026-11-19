"""Look schedules: which trials a sequential look uses, and when it is ready.

A look is a fixed set of *units* (task, trial index) chosen by LAUNCH ORDER, never by completion
order. Slow trials (timeouts) are mostly failures, so "whatever finished first" over-represents
fast successes and biases early looks in the candidate's favour. A look is therefore ready only
when every one of its units

  * exists (a record was produced: the trial was launched; a never-launched trial has no record),
  * and has finished (`TrialRecord.finished`).

Two schedules, one per `contracts.Order`:

  ATTEMPT_MAJOR (harbor: every task once, then every task again, ...)
      K = trials looks; look r (1..K) uses the trials with index < r of EVERY task.
      Information fraction r / K.
  TASK_MAJOR (platoon eval: all trials of task 1, then task 2, ... in `EvalRequest.tasks` order)
      looks at task-count checkpoints (default 25/50/75/100% of the ordered task list);
      look j uses ALL trials of the first n_j tasks. Information fraction n_j / N.

Env-failed trials (the sandbox never started) count as finished, so they never block a look, but
they are not evidence: `per_task` skips them. `LookData.env_fail_frac` reports their share so a
caller can treat a look as infra-suspect.

Pure and stdlib-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Iterable, Iterator, Sequence

from harnessforge.contracts import EvalRequest, Order
from harnessforge.records import TaskStats, TrialRecord, per_task

__all__ = ["DEFAULT_CHECKPOINTS", "Look", "LookData", "LookSchedule", "UnitIndex", "collect"]

DEFAULT_CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)

Unit = tuple[str, int]


@dataclass(frozen=True)
class Look:
    """One look: all trials with index < `trials` of each task in `tasks`."""
    number: int                  # 1-based
    fraction: float              # information fraction in (0, 1]
    tasks: tuple[str, ...]
    trials: int
    final: bool

    def units(self) -> Iterator[Unit]:
        for t in self.tasks:
            for i in range(self.trials):
                yield (t, i)

    @property
    def size(self) -> int:
        return len(self.tasks) * self.trials


class UnitIndex:
    """Records of one snapshot indexed by (candidate, task, index). When a unit has several
    records (a backend re-reported it), a finished one wins over an in-flight one, then the one
    that finished last."""

    def __init__(self, records: Iterable[TrialRecord]):
        self._by: dict[str, dict[Unit, TrialRecord]] = {}
        for r in records:
            units = self._by.setdefault(r.candidate, {})
            old = units.get((r.task, r.index))
            if old is None or _newer(r, old):
                units[(r.task, r.index)] = r

    def get(self, candidate: str, task: str, index: int) -> TrialRecord | None:
        return self._by.get(candidate, {}).get((task, index))

    def candidates(self) -> list[str]:
        return sorted(self._by)


def _newer(r: TrialRecord, old: TrialRecord) -> bool:
    if r.finished != old.finished:
        return r.finished
    return (r.finished_at or 0.0) >= (old.finished_at or 0.0)


@dataclass(frozen=True)
class LookData:
    """One candidate's data for one look. `stats` is filled only when `ready`."""
    look: Look
    candidate: str
    ready: bool
    expected: int                     # units the look covers
    launched: int                     # units with a record
    finished: int                     # units finished (including env-failed)
    env_failed: int                   # finished units whose sandbox never started
    stats: dict[str, TaskStats] = field(default_factory=dict)
    policies: frozenset[str] = frozenset()   # policy ids seen on finished units

    @property
    def missing(self) -> int:
        """Units never launched (no record) so far."""
        return self.expected - self.launched

    @property
    def env_fail_frac(self) -> float:
        return self.env_failed / self.finished if self.finished else 0.0

    def as_dict(self) -> dict:
        return {"look": self.look.number, "candidate": self.candidate, "ready": self.ready,
                "expected": self.expected, "launched": self.launched,
                "finished": self.finished, "env_failed": self.env_failed,
                "env_fail_frac": round(self.env_fail_frac, 4), "tasks": len(self.stats)}


def collect(index: UnitIndex | Iterable[TrialRecord], candidate: str, look: Look) -> LookData:
    """The candidate's records for `look`, with readiness (every unit launched and finished)."""
    if not isinstance(index, UnitIndex):
        index = UnitIndex(index)
    launched = finished = env_failed = 0
    used: list[TrialRecord] = []
    policies: set[str] = set()
    for task, i in look.units():
        r = index.get(candidate, task, i)
        if r is None:
            continue
        launched += 1
        if r.finished:
            finished += 1
            env_failed += r.env_failed
            used.append(r)
            if r.policy is not None:
                policies.add(r.policy)
    ready = finished == look.size
    stats = per_task(used, candidate) if ready else {}
    return LookData(look, candidate, ready, look.size, launched, finished, env_failed, stats,
                    frozenset(policies))


@dataclass(frozen=True)
class LookSchedule:
    """The looks of one evaluation: `tasks` in the request's (look) order, `trials` per task."""
    order: Order
    tasks: tuple[str, ...]
    trials: int
    checkpoints: tuple[float, ...] = DEFAULT_CHECKPOINTS   # TASK_MAJOR only

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", Order(self.order))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "checkpoints", tuple(float(c) for c in self.checkpoints))
        if not self.tasks:
            raise ValueError("no tasks")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("duplicate task ids")
        if self.trials < 1:
            raise ValueError("trials must be >= 1")
        if any(not 0 < c <= 1 for c in self.checkpoints):
            raise ValueError("checkpoints must be in (0, 1]")

    @classmethod
    def from_request(cls, request: EvalRequest, order: Order,
                     checkpoints: Sequence[float] = DEFAULT_CHECKPOINTS) -> "LookSchedule":
        return cls(order, tuple(request.tasks), request.trials, tuple(checkpoints))

    @cached_property
    def looks(self) -> tuple[Look, ...]:
        n, k = len(self.tasks), self.trials
        if self.order is Order.ATTEMPT_MAJOR:
            return tuple(Look(r, r / k, self.tasks, r, r == k) for r in range(1, k + 1))
        counts = sorted({min(n, max(1, math.ceil(c * n - 1e-9))) for c in self.checkpoints} | {n})
        return tuple(Look(j, m / n, self.tasks[:m], k, m == n)
                     for j, m in enumerate(counts, start=1))

    @property
    def fractions(self) -> tuple[float, ...]:
        return tuple(lk.fraction for lk in self.looks)

    def look(self, number: int) -> Look:
        return self.looks[number - 1]

    def full(self, look: Look) -> Look:
        """`look` widened to ALL trials of its tasks (for a reference evaluated in full)."""
        return replace(look, trials=self.trials)

    def collect(self, records: UnitIndex | Iterable[TrialRecord], candidate: str,
                number: int) -> LookData:
        return collect(records, candidate, self.look(number))

    def ready_looks(self, records: UnitIndex | Iterable[TrialRecord],
                    candidate: str) -> list[int]:
        idx = records if isinstance(records, UnitIndex) else UnitIndex(records)
        return [lk.number for lk in self.looks if collect(idx, candidate, lk).ready]
