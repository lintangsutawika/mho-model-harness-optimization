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
      With `trial_looks` (e.g. (2, 4, 8, 24)): look j uses the trials with index < k_j of every task.
      With `information` (planning variance components, `information.Information`) the information
      fractions are I(k_j) / I(K), I(k) = n / (tau2 + sigma2 / k), instead of k_j / K.
  TASK_MAJOR (platoon eval: all trials of task 1, then task 2, ... in `EvalRequest.tasks` order)
      looks at task-count checkpoints (default 25/50/75/100% of the ordered task list);
      look j uses ALL trials of the first n_j tasks. Information fraction n_j / N.
      With `sampling` (`sampling.TaskSampling`, a frontier-weighted stratified order) the looks are its sizes and
      each look is estimated with Horvitz-Thompson weights and stratified SE (`weights`, `strata`); with
      `information` the planned information of look j is 1 / var(HT mean) (n_j / var(d_i) for a uniform order).

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
from harnessforge.stats.information import Information
from harnessforge.stats.sampling import TaskSampling

__all__ = ["DEFAULT_CHECKPOINTS", "Look", "LookData", "LookSchedule", "UnitIndex", "collect"]

DEFAULT_CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)

Unit = tuple[str, int]


@dataclass(frozen=True)
class Look:
    """One look: the trials with index < `trials` of each task in `tasks`, or, with `counts` (one per task,
    a planned look of unequal depth), the trials with index < counts[i] of tasks[i]; `trials` is then the
    largest count."""
    number: int                  # 1-based
    fraction: float              # information fraction in (0, 1]
    tasks: tuple[str, ...]
    trials: int
    final: bool
    counts: tuple[int, ...] | None = None

    @property
    def uniform(self) -> bool:
        return self.counts is None or len(set(self.counts)) <= 1

    def per_task(self) -> Iterator[tuple[str, int]]:
        """(task, trials of it in this look)."""
        if self.counts is None:
            for t in self.tasks:
                yield t, self.trials
        else:
            yield from zip(self.tasks, self.counts)

    def units(self) -> Iterator[Unit]:
        for t, k in self.per_task():
            for i in range(k):
                yield (t, i)

    @property
    def size(self) -> int:
        return sum(self.counts) if self.counts is not None else len(self.tasks) * self.trials


def _counts_of(look: Sequence[Segment], n: int) -> list[int] | None:
    counts = [0] * n
    start = 0
    for end, k in look:
        if not start < end <= n or k < 1:
            return None
        counts[start:end] = [k] * (end - start)
        start = end
    return counts


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


Segment = tuple[int, int]            # (end task position, trials): tasks[previous end:end] get `trials` trials


@dataclass(frozen=True)
class LookSchedule:
    """The looks of one evaluation: `tasks` in the request's (look) order, `trials` per task (the most any
    task gets).

    ATTEMPT_MAJOR only (the backend launches look by look, see `launch_order`):

      trial_looks   the trials per task at each look, every task alike, e.g. (2, 4, 8, 24) (default every
                    trial 1..trials; `trials` is appended when missing);
      plan          the general form: one entry per look, each a sequence of segments (end, k) over the
                    ordered task list: tasks[previous end:end] have k trials in that look; tasks after the
                    last segment are not in it yet. A look may add tasks, add trials to tasks it has, or
                    both; a task's trials never decrease. E.g. ((100, 1),), ((200, 2),), ((200, 8),),
                    ((200, 24),) or ((50, 2),), ((50, 2), (100, 4)), ...;
      information   planning variance components (`information.Information`): look j's information is
                    I_j = sum over its tasks of 1 / (tau2 + sigma2 / k_task), its fraction I_j / I_J, and a
                    look of unequal depth is estimated with inverse-variance task weights (`weights`).
                    Without it, fractions are units_j / units_J (= k_j / K for trial looks).
    """
    order: Order
    tasks: tuple[str, ...]
    trials: int
    checkpoints: tuple[float, ...] = DEFAULT_CHECKPOINTS   # TASK_MAJOR only
    trial_looks: tuple[int, ...] | None = None             # ATTEMPT_MAJOR only
    information: Information | None = None                 # ATTEMPT_MAJOR only
    plan: tuple[tuple[Segment, ...], ...] | None = None    # ATTEMPT_MAJOR only
    sampling: TaskSampling | None = None                   # TASK_MAJOR only

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
        if self.trial_looks is not None and self.plan is not None:
            raise ValueError("give trial_looks or plan, not both")
        if (self.trial_looks is not None or self.plan is not None) and self.order is not Order.ATTEMPT_MAJOR:
            raise ValueError("trial_looks and plan need Order.ATTEMPT_MAJOR (task-major looks are task prefixes)")
        if self.sampling is not None:
            if self.order is not Order.TASK_MAJOR:
                raise ValueError("sampling needs Order.TASK_MAJOR")
            if tuple(self.sampling.tasks) != self.tasks:
                raise ValueError("sampling.tasks must be the schedule's task order")
            n = len(self.tasks)
            object.__setattr__(self, "checkpoints", tuple(m / n for m in self.sampling.sizes))
        if self.trial_looks is not None:
            ks = tuple(int(k) for k in self.trial_looks)
            if not ks or any(b <= a for a, b in zip(ks, ks[1:])) or ks[0] < 1 or ks[-1] > self.trials:
                raise ValueError(f"trial_looks must increase within 1..{self.trials}: {ks}")
            object.__setattr__(self, "trial_looks", ks if ks[-1] == self.trials else (*ks, self.trials))
        if self.plan is not None:
            plan = tuple(tuple((int(e), int(k)) for e, k in look) for look in self.plan)
            object.__setattr__(self, "plan", plan)
            self._check_plan(plan)

    def _check_plan(self, plan) -> None:
        n = len(self.tasks)
        if not plan:
            raise ValueError("empty plan")
        prev = [0] * n
        for j, look in enumerate(plan, start=1):
            counts = _counts_of(look, n)
            if counts is None:
                raise ValueError(f"plan look {j}: segment ends must increase within 1..{n} with trials >= 1")
            if max(counts) > self.trials:
                raise ValueError(f"plan look {j} has more than trials={self.trials} trials on a task")
            if any(c < p for c, p in zip(counts, prev)):
                raise ValueError(f"plan look {j} takes trials away from a task")
            if counts == prev:
                raise ValueError(f"plan look {j} adds nothing")
            prev = counts
        if max(prev) != self.trials:
            raise ValueError(f"no task reaches trials={self.trials} in the plan")

    @classmethod
    def from_request(cls, request: EvalRequest, order: Order,
                     checkpoints: Sequence[float] = DEFAULT_CHECKPOINTS,
                     trial_looks: Sequence[int] | None = None,
                     information: Information | None = None,
                     plan: Sequence[Sequence[Segment]] | None = None,
                     sampling: TaskSampling | None = None) -> "LookSchedule":
        return cls(order, tuple(request.tasks), request.trials, tuple(checkpoints),
                   None if trial_looks is None else tuple(trial_looks), information,
                   None if plan is None else tuple(tuple(lk) for lk in plan), sampling)

    @property
    def planned(self) -> bool:
        """Looks set by trial_looks or a plan (rather than one look per trial / task checkpoints)."""
        return self.trial_looks is not None or self.plan is not None

    @property
    def trial_counts(self) -> tuple[int, ...] | None:
        """ATTEMPT_MAJOR with every task alike at every look: trials per task at each look; else None."""
        if self.order is not Order.ATTEMPT_MAJOR:
            return None
        if self.plan is not None:
            if all(lk.uniform and len(lk.tasks) == len(self.tasks) for lk in self.looks):
                return tuple(lk.trials for lk in self.looks)
            return None
        return self.trial_looks or tuple(range(1, self.trials + 1))

    def _per_look_counts(self) -> list[tuple[int, ...]]:
        """Per look, trials of each task of the ordered list (0 = not in the look)."""
        n = len(self.tasks)
        if self.plan is not None:
            return [tuple(_counts_of(lk, n)) for lk in self.plan]
        return [(k,) * n for k in self.trial_counts]

    def planned_information(self, all_trials_reference: bool = False) -> tuple[float, ...] | None:
        """I_j = sum over look j's tasks of 1 / var(d_task) from `information` (None without it).
        `all_trials_reference`: the reference contributes each task's final trial count (ReferenceMode.ALL_TRIALS)."""
        if self.information is None:
            return None
        if self.order is Order.TASK_MAJOR:
            v = self.information.variance(self.trials)
            out = []
            for lk in self.looks:
                if self.sampling is None:
                    out.append(len(lk.tasks) / v)
                else:
                    out.append(len(lk.tasks) / (v * self.sampling.design_effect(lk.number)))
            return tuple(out)
        per = self._per_look_counts()
        final = per[-1]
        return tuple(math.fsum(1.0 / self.information.variance(c, f if all_trials_reference else None)
                               for c, f in zip(counts, final) if c > 0) for counts in per)

    @cached_property
    def looks(self) -> tuple[Look, ...]:
        n, k = len(self.tasks), self.trials
        if self.order is Order.ATTEMPT_MAJOR:
            per = self._per_look_counts()
            info = self.planned_information()
            if info is not None:
                fr = tuple(i / info[-1] for i in info)
            else:
                sizes = [sum(c) for c in per]
                fr = tuple(s_ / sizes[-1] for s_ in sizes)
            out = []
            for j, (counts, f) in enumerate(zip(per, fr), start=1):
                m = max(i for i, c in enumerate(counts) if c > 0) + 1 if self.plan is not None else n
                cs = counts[:m]
                final = j == len(per)
                if len(set(cs)) == 1 and all(c > 0 for c in cs):
                    out.append(Look(j, f, self.tasks[:m], cs[0], final))
                else:
                    keep = [i for i in range(m) if cs[i] > 0]
                    out.append(Look(j, f, tuple(self.tasks[i] for i in keep), max(cs), final,
                                    tuple(cs[i] for i in keep)))
            return tuple(out)
        counts = (list(self.sampling.sizes) if self.sampling is not None else
                  sorted({min(n, max(1, math.ceil(c * n - 1e-9))) for c in self.checkpoints} | {n}))
        looks = tuple(Look(j, m / n, self.tasks[:m], k, m == n) for j, m in enumerate(counts, start=1))
        if self.information is not None:
            v = self.information.variance(k)
            info = [len(lk.tasks) / v / (self.sampling.design_effect(lk.number) if self.sampling else 1.0)
                    for lk in looks]
            looks = tuple(replace(lk, fraction=i / info[-1]) for lk, i in zip(looks, info))
        return looks

    @property
    def fractions(self) -> tuple[float, ...]:
        return tuple(lk.fraction for lk in self.looks)

    def look(self, number: int) -> Look:
        return self.looks[number - 1]

    def full(self, look: Look) -> Look:
        """`look` widened to ALL trials its tasks get in the evaluation (for a reference evaluated in full)."""
        if self.plan is None:
            return replace(look, trials=self.trials)
        final = dict(self.looks[-1].per_task())
        counts = tuple(final[t] for t in look.tasks)
        return replace(look, trials=max(counts), counts=None if len(set(counts)) == 1 else counts)

    def weights(self, look: Look, all_trials_reference: bool = False) -> dict[str, float] | None:
        """Inverse-variance task weights 1 / (tau2 + sigma2 / k_task) for a look of unequal depth (None when
        every task of the look has the same trials, or without `information`: the plain mean)."""
        if self.sampling is not None:
            return self.sampling.weights(look.number)
        if self.information is None or look.uniform:
            return None
        final = dict(self.looks[-1].per_task())
        return {t: 1.0 / self.information.variance(k, final[t] if all_trials_reference else None)
                for t, k in look.per_task()}

    def strata(self, look: Look) -> dict[str, int] | None:
        """Task -> band for the stratified SE of a weighted (frontier-sampled) look; None otherwise."""
        if self.sampling is None or self.sampling.weights(look.number) is None:
            return None
        return self.sampling.strata(look.number)

    def launch_order(self) -> list[Unit]:
        """The units in the order a look-major backend should launch them: look 1's units, then the units look 2
        adds, ...; within a look by trial index, then task order (attempt-major)."""
        seen: set[Unit] = set()
        out: list[Unit] = []
        for lk in self.looks:
            new = [u for u in lk.units() if u not in seen]
            pos = {t: i for i, t in enumerate(self.tasks)}
            new.sort(key=lambda u: (u[1], pos[u[0]]))
            out += new
            seen.update(new)
        return out

    def collect(self, records: UnitIndex | Iterable[TrialRecord], candidate: str,
                number: int) -> LookData:
        return collect(records, candidate, self.look(number))

    def ready_looks(self, records: UnitIndex | Iterable[TrialRecord],
                    candidate: str) -> list[int]:
        idx = records if isinstance(records, UnitIndex) else UnitIndex(records)
        return [lk.number for lk in self.looks if collect(idx, candidate, lk).ready]
