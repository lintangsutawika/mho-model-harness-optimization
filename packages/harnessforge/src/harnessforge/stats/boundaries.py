"""Group-sequential designs: efficacy boundaries, non-binding futility, the final look.

At look j (information fraction tau_j, see `looks.py`) the paired t of candidate vs reference is
compared with

    efficacy   t > c / sqrt(tau_j)      (O'Brien-Fleming shape; from look `first_efficacy` on)
    futility   t <= f_j                 (interim looks only; non-binding)
    final      pass iff t > c, else fail

For ATTEMPT_MAJOR with K trials, tau_r = r / K, so the efficacy boundary is c * sqrt(K / r):
with c = 1.72 and K = 5 the boundaries are (none), 2.72, 2.22, 1.92, 1.72.

c is calibrated by simulation (`harnessforge.stats.calibrate`) so the overall one-sided false
accept rate is alpha WITHOUT futility. Futility is non-binding: stopping for futility can only
lower the false-accept rate, so the caller may ignore a futility stop (e.g. keep evaluating)
without invalidating alpha. c = 1.72 gives alpha ~= 0.05 at K = 5 attempt looks on real tmax
pass rates (vs 1.645 for a single fixed look).

Default futility: f(tau) = -1 + 2.5 * tau, which is (-0.5, 0, 0.5, 1.0) at K = 5 attempt looks
(the rule validated on the old loop: -1 to -2 points of power, about half the trials for a
no-effect candidate).

Pure and stdlib-only.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

from harnessforge.stats.looks import LookSchedule

__all__ = ["DEFAULT_C", "DEFAULT_FUTILITY", "Design", "Verdict", "linear_futility"]

DEFAULT_C = 1.72
DEFAULT_FUTILITY = (-0.5, 0.0, 0.5, 1.0)     # = linear_futility at K = 5 attempt looks


def linear_futility(fraction: float) -> float:
    """Default futility threshold at information fraction `fraction`."""
    return -1.0 + 2.5 * fraction


class Verdict(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    CONTINUE = "continue"


@dataclass(frozen=True)
class Design:
    """Boundaries for looks with information fractions `fractions` (increasing, last = 1).

    `futility`: one threshold per interim look (missing entries = no futility stop at that look);
    None = `linear_futility`; () = no futility at all."""
    fractions: tuple[float, ...]
    c: float = DEFAULT_C
    futility: tuple[float, ...] | None = None
    first_efficacy: int = 2

    def __post_init__(self) -> None:
        fr = tuple(float(f) for f in self.fractions)
        object.__setattr__(self, "fractions", fr)
        if self.futility is not None:
            object.__setattr__(self, "futility", tuple(float(f) for f in self.futility))
        if not fr or any(b <= a for a, b in zip(fr, fr[1:])) or not 0 < fr[0] or \
                abs(fr[-1] - 1.0) > 1e-9:
            raise ValueError(f"fractions must increase to 1: {fr}")

    @classmethod
    def for_schedule(cls, schedule: LookSchedule, c: float = DEFAULT_C,
                     futility: Sequence[float] | None = None,
                     first_efficacy: int = 2) -> "Design":
        return cls(schedule.fractions, c, None if futility is None else tuple(futility),
                   first_efficacy)

    @classmethod
    def attempt_major(cls, k: int, c: float = DEFAULT_C,
                      futility: Sequence[float] | None = None) -> "Design":
        return cls(tuple(r / k for r in range(1, k + 1)), c,
                   None if futility is None else tuple(futility))

    def with_c(self, c: float) -> "Design":
        return replace(self, c=c)

    @property
    def n_looks(self) -> int:
        return len(self.fractions)

    def is_final(self, j: int) -> bool:
        return j == self.n_looks

    def efficacy(self, j: int) -> float:
        """Pass iff t > efficacy(j) (j is 1-based)."""
        self._check(j)
        if j < self.first_efficacy and not self.is_final(j):
            return math.inf
        return self.c / math.sqrt(self.fractions[j - 1])

    def futility_bound(self, j: int) -> float:
        """Fail iff t <= futility_bound(j). At the final look this is c (everything not passing
        fails)."""
        self._check(j)
        if self.is_final(j):
            return self.c
        if self.futility is None:
            return linear_futility(self.fractions[j - 1])
        return self.futility[j - 1] if j - 1 < len(self.futility) else -math.inf

    def verdict(self, j: int, t: float | None) -> Verdict:
        """Decision at look j for paired t (None = no paired data: continue, fail at the end)."""
        if t is None or math.isnan(t):
            return Verdict.FAIL if self.is_final(j) else Verdict.CONTINUE
        if t > self.efficacy(j):
            return Verdict.PASS
        if t <= self.futility_bound(j):
            return Verdict.FAIL
        return Verdict.CONTINUE

    def _check(self, j: int) -> None:
        if not 1 <= j <= self.n_looks:
            raise IndexError(f"look {j} not in 1..{self.n_looks}")

    def as_dict(self) -> dict[str, Any]:
        def r(x: float) -> float | None:
            return round(x, 4) if math.isfinite(x) else None
        return {"c": self.c, "fractions": [round(f, 4) for f in self.fractions],
                "efficacy": [r(self.efficacy(j)) for j in range(1, self.n_looks + 1)],
                "futility": [r(self.futility_bound(j)) for j in range(1, self.n_looks)]}
