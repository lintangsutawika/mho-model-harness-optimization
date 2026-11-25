"""Sequential val gate: one candidate vs the current best on a fixed val split.

The same machinery as the race (a one-candidate `Race`), so the val decision is a calibrated
group-sequential test too (it replaces the old 2-look adaptive val). Use it after a race: the
race winner's train-subset t is inflated by selection (max of K), the val split is held out.

    gate = ValGate(best="h12", candidate="h17", schedule=LookSchedule(order, val_tasks, trials))
    st = gate.update(records)            # CandidateState; st.verdict in pass/fail/continue
    if not st.alive: run.stop(...)

Default `reference_mode` is ALL_TRIALS: the current best's val trials are usually on file
already (it was evaluated when it was accepted), so compare against all of them. If the best is
re-evaluated concurrently, pass ReferenceMode.SAME_UNITS so looks can happen early.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from harnessforge.records import TrialRecord
from harnessforge.contracts import Order
from harnessforge.stats.boundaries import Design
from harnessforge.stats.information import VAL200, Information
from harnessforge.stats.looks import LookSchedule, UnitIndex
from harnessforge.stats.paired import GateConfig
from harnessforge.stats.race import CandidateState, Race, ReferenceMode
from harnessforge.stats.sampling import TaskSampling

__all__ = ["VAL1000_CHECKPOINTS", "ValGate", "val1000_schedule"]

VAL1000_CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)


def val1000_schedule(tasks: Sequence[str] | None = None, trials: int = 4, *, sampling: TaskSampling | None = None,
                     information: Information = VAL200) -> LookSchedule:
    """The search gate's schedule (docs/stats.md section 8): TASK_MAJOR over the val1000 tasks in launch order,
    looks at 25/50/75/100% of the tasks, planning information for Lan-DeMets OBF spending on observed information
    (`race.default_design`). `sampling`: a frontier-weighted order (`sampling.frontier_order`); its task order is the
    schedule's (`tasks` may then be omitted)."""
    if sampling is not None:
        if tasks is not None and tuple(tasks) != sampling.tasks:
            raise ValueError("tasks must be the sampling's order")
        tasks = sampling.tasks
    if tasks is None:
        raise ValueError("give tasks or sampling")
    return LookSchedule(Order.TASK_MAJOR, tuple(tasks), trials, VAL1000_CHECKPOINTS, information=information,
                        sampling=sampling)


class ValGate:
    def __init__(self, best: str, candidate: str, schedule: LookSchedule,
                 design: Design | None = None, *,
                 reference_mode: ReferenceMode | str = ReferenceMode.ALL_TRIALS,
                 max_env_fail: float | None = None, cheaper: GateConfig | None = None):
        self.best, self.candidate = best, candidate
        self._race = Race(best, [candidate], schedule, design, reference_mode=reference_mode,
                          max_env_fail=max_env_fail, cheaper=cheaper)

    @property
    def design(self) -> Design:
        return self._race.design

    def update(self, records: Iterable[TrialRecord] | UnitIndex) -> CandidateState:
        return self._race.update(records).states[self.candidate]

    def status(self) -> CandidateState:
        return self._race.status().states[self.candidate]
