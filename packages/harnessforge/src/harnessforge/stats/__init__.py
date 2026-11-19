"""Sequential paired testing over TrialRecords (stdlib-only).

    paired      per-task paired statistics, fixed-sample decide (better / optional cheaper)
    looks       look schedules for ATTEMPT_MAJOR and TASK_MAJOR evaluations, readiness
    boundaries  group-sequential designs (O'Brien-Fleming efficacy, non-binding futility)
    calibrate   Monte Carlo calibration of c + CLI (python -m harnessforge.stats.calibrate)
    race        K candidates vs one reference, concurrently, as a pure state machine
    gate        sequential val gate (candidate vs current best)

See docs/stats.md.
"""
from harnessforge.stats.boundaries import (DEFAULT_C, DEFAULT_FUTILITY, Design, Verdict,
                                           linear_futility)
from harnessforge.stats.gate import ValGate
from harnessforge.stats.looks import (DEFAULT_CHECKPOINTS, Look, LookData, LookSchedule,
                                      UnitIndex, collect)
from harnessforge.stats.paired import Decision, GateConfig, Paired, decide, paired, task_mean
from harnessforge.stats.race import CandidateState, Race, RaceStatus, ReferenceMode

__all__ = ["DEFAULT_C", "DEFAULT_CHECKPOINTS", "DEFAULT_FUTILITY", "CandidateState", "Decision",
           "Design", "GateConfig", "Look", "LookData", "LookSchedule", "Paired", "Race",
           "RaceStatus", "ReferenceMode", "UnitIndex", "ValGate", "Verdict", "collect", "decide",
           "linear_futility", "paired", "task_mean"]
