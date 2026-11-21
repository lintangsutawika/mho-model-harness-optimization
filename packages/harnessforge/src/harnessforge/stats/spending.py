"""Lan-DeMets alpha-spending boundaries on information fractions (one-sided efficacy, non-binding futility).

A spending function alpha(t) gives the cumulative false-accept probability allowed by information fraction t
(alpha(0) = 0, alpha(1) = alpha). At look j with fraction t_j the efficacy boundary b_j solves

    P0(Z_1 <= b_1, ..., Z_{j-1} <= b_{j-1}, Z_j > b_j) = alpha(t_j) - alpha(t_{j-1})

under the canonical joint normal of the look statistics: Z_j ~ N(0, 1), corr(Z_i, Z_j) = sqrt(I_i / I_j). The
probability is computed by recursive numerical integration (Armitage, McPherson and Rowe; the grid and Simpson rule
of Jennison and Turnbull 2000, ch. 19), which is exact to about 1e-6 here and pure python.

Spending functions (`spending=`):

    "obf"      O'Brien-Fleming type, one-sided: alpha(t) = 2 - 2 Phi(z_{1 - alpha/2} / sqrt(t))
    "pocock"   Pocock type: alpha * log(1 + (e - 1) t)
    "power:R"  Kim-DeMets power family: alpha * t^R (R = 3 is close to "obf" late, less extreme early)

Information fractions come from one of two sources:

  planned    t_j = I(k_j) / I(K) from the planning variance components (`information.Information`), fixed up front;
  observed   t_j = I_obs,j / I_max with I_obs,j = 1 / SE_j^2 of the paired mean difference at look j and I_max the
             planned information of the last look. The correlations use the observed information ratios, so the
             boundaries are right whatever the true variance components are.

Observed information can miss the plan either way. Fractions are capped at 1: a look whose observed information
reaches I_max is the FINAL look (it spends the remainder and decides pass/fail; a design with no alpha left could
never pass later anyway). At the schedule's last look the remainder alpha - alpha(t_{J-1}) is spent whatever the
fraction is, so a last look below plan still uses the whole alpha (with less power), and one above plan too.
Looks before `first_efficacy` and looks skipped as infra-suspect spend nothing: their alpha moves to the next look
(the spending function is cumulative), and they are left out of the joint distribution.

Futility is non-binding (`boundaries.linear_futility` of the look's fraction by default): the efficacy boundaries
are computed without it, so a futility stop only lowers the false-accept rate.

Observed fractions are rounded to `resolution` (1e-3) so that boundary computations are shared between looks and
candidates with nearly the same information; the effect on a boundary is below 1e-3.

Pure and stdlib-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Sequence

from harnessforge.stats.boundaries import LookBounds, Verdict, linear_futility

__all__ = ["SpendingDesign", "crossing_probabilities", "spending_function"]

_N = NormalDist()
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _q(x: float) -> float:
    """Upper normal tail P(Z > x)."""
    return 0.5 * math.erfc(x / _SQRT2)


def _phi(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def spending_function(kind: str, alpha: float):
    """alpha(t) for `kind` "obf", "pocock" or "power:R" (see the module docstring)."""
    if kind == "obf":
        z = _N.inv_cdf(1.0 - alpha / 2.0)
        return lambda t: 0.0 if t <= 0 else (alpha if t >= 1 else 2.0 * _q(z / math.sqrt(t)))
    if kind == "pocock":
        return lambda t: 0.0 if t <= 0 else alpha * math.log(1.0 + (math.e - 1.0) * min(t, 1.0))
    if kind.startswith("power:"):
        rho = float(kind.split(":", 1)[1])
        if rho <= 0:
            raise ValueError("power spending needs R > 0")
        return lambda t: 0.0 if t <= 0 else alpha * min(t, 1.0) ** rho
    raise ValueError(f"unknown spending function {kind!r} (obf, pocock, power:R)")


# ----------------------------------------------------------------------------- recursive integration
def _grid(b: float, mu: float, r: int) -> tuple[list[float], list[float]]:
    """Jennison-Turnbull grid on (-inf, b] around mean mu, with Simpson weights (odd points are midpoints)."""
    x = []
    for i in range(1, 6 * r):
        if i < r:
            x.append(mu - 3.0 - 4.0 * math.log(r / i))
        elif i <= 5 * r:
            x.append(mu - 3.0 + 3.0 * (i - r) / (2.0 * r))
        else:
            x.append(mu + 3.0 + 4.0 * math.log(r / (6 * r - i)))
    if b >= x[-1]:
        z = x
    elif b <= x[0]:
        z = [b - 1.0, b]
    else:
        z = [v for v in x if v < b] + [b]
    pts: list[float] = []
    w: list[float] = []
    for i in range(len(z) - 1):
        lo, hi = z[i], z[i + 1]
        h = hi - lo
        if i == 0:
            pts.append(lo)
            w.append(h / 6.0)
        else:
            w[-1] += h / 6.0
        pts += [(lo + hi) / 2.0, hi]
        w += [4.0 * h / 6.0, h / 6.0]
    return pts, w


@dataclass
class _State:
    """Sub-density of Z_j on the continuation region of the looks so far (grid points, density x weight)."""
    u: float                 # information (in units of I_max) of the last look
    z: list[float]
    h: list[float]


def _first(u: float, b: float, drift: float, r: int) -> _State:
    mu = drift * math.sqrt(u)
    z, w = _grid(b, mu, r)
    return _State(u, z, [wi * _phi(zi - mu) for zi, wi in zip(z, w)])


def _cross(s: _State, u: float, b: float, drift: float) -> float:
    """P(continue so far, Z at information u > b)."""
    du = u - s.u
    sd = math.sqrt(du)
    su, sp = math.sqrt(u), math.sqrt(s.u)
    c = b * su - drift * du
    return math.fsum(h * _q((c - z * sp) / sd) for z, h in zip(s.z, s.h))


def _update(s: _State, u: float, b: float, drift: float, r: int) -> _State:
    du = u - s.u
    sd = math.sqrt(du)
    su, sp = math.sqrt(u), math.sqrt(s.u)
    z_new, w_new = _grid(b, drift * su, r)
    k = su / sd
    zs = [z * sp + drift * du for z in s.z]
    hs = s.h
    out = []
    for zn, wn in zip(z_new, w_new):
        a = zn * su
        acc = 0.0
        for zp, hp in zip(zs, hs):
            x = (a - zp) / sd
            if -40.0 < x < 40.0:
                acc += hp * math.exp(-0.5 * x * x)
        out.append(wn * k * _INV_SQRT_2PI * acc)
    return _State(u, z_new, out)


def _solve(s: _State | None, u: float, target: float) -> float:
    """b with P0(continue so far, Z_u > b) = target."""
    if target <= 1e-15:
        return math.inf
    if s is None:
        return _N.inv_cdf(1.0 - target) if target < 1 else -math.inf
    lo, hi = -12.0, 40.0
    if _cross(s, u, lo, 0.0) <= target:
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _cross(s, u, mid, 0.0) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return (lo + hi) / 2.0


def crossing_probabilities(fractions: Sequence[float], bounds: Sequence[float], drift: float = 0.0,
                           grid: int = 16) -> list[float]:
    """P(first efficacy crossing at look j) for looks at information `fractions` (any scale, increasing) with upper
    boundaries `bounds` (inf = no efficacy at that look), when E[Z_j] = drift * sqrt(fractions[j]). Without
    futility. sum() of the result is the power (drift = delta * sqrt(I at fraction 1)) or, at drift 0, the alpha."""
    out: list[float] = []
    s: _State | None = None
    for u, b in zip(fractions, bounds):
        if s is None:
            out.append(0.0 if math.isinf(b) else _q(b - drift * math.sqrt(u)))
            s = _first(u, b, drift, grid)
        else:
            out.append(0.0 if math.isinf(b) else _cross(s, u, b, drift))
            s = _update(s, u, b, drift, grid)
    return out


# ----------------------------------------------------------------------------- design
@dataclass(frozen=True)
class SpendingDesign:
    """Lan-DeMets alpha-spending efficacy boundaries plus non-binding futility on information fractions.

    `fractions`: planned information fractions (increasing, last = 1). `max_information`: the planned information
    at the last look (1 / SE^2 units); when set, `bounds` takes each look's observed information and the
    boundaries follow the observed fractions (see the module docstring); None = planned fractions only.
    `futility`: as `Design` (None = linear_futility of the fraction; () = none)."""
    fractions: tuple[float, ...]
    alpha: float = 0.05
    spending: str = "obf"
    max_information: float | None = None
    futility: tuple[float, ...] | None = None
    first_efficacy: int = 1
    resolution: float = 1e-3
    grid: int = 16
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)    # boundaries
    _states: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)   # densities

    def __post_init__(self) -> None:
        fr = tuple(float(f) for f in self.fractions)
        object.__setattr__(self, "fractions", fr)
        if self.futility is not None:
            object.__setattr__(self, "futility", tuple(float(f) for f in self.futility))
        if not fr or any(b <= a for a, b in zip(fr, fr[1:])) or not 0 < fr[0] or abs(fr[-1] - 1.0) > 1e-9:
            raise ValueError(f"fractions must increase to 1: {fr}")
        if not 0 < self.alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")
        spending_function(self.spending, self.alpha)          # validate
        if self.max_information is not None and not self.max_information > 0:
            raise ValueError("max_information must be > 0")

    @classmethod
    def for_schedule(cls, schedule: Any, *, alpha: float = 0.05, spending: str = "obf", observed: bool = True,
                     all_trials_reference: bool = False, futility: Sequence[float] | None = None,
                     first_efficacy: int = 1) -> "SpendingDesign":
        """The design for a `LookSchedule`. With planning `information` the fractions are I_j / I_J
        (`LookSchedule.planned_information`) and, if `observed`, the boundaries follow the observed information
        with I_max = I_J. `all_trials_reference`: the reference is compared on all its trials of each task
        (ReferenceMode.ALL_TRIALS) rather than on the same units."""
        info = schedule.planned_information(all_trials_reference)
        fr, imax = schedule.fractions, None
        if info is not None:
            fr = tuple(i / info[-1] for i in info)
            if observed:
                imax = info[-1]
        return cls(fr, alpha, spending, imax, None if futility is None else tuple(futility), first_efficacy)

    # ------------------------------------------------------------------ the Design interface
    @property
    def n_looks(self) -> int:
        return len(self.fractions)

    @property
    def c(self) -> float:
        """The final look's planned boundary (comparable to Design.c)."""
        return self.efficacy(self.n_looks)

    @property
    def observed(self) -> bool:
        return self.max_information is not None

    def is_final(self, j: int) -> bool:
        return j == self.n_looks

    def spend(self, t: float) -> float:
        return spending_function(self.spending, self.alpha)(t)

    def planned_bounds(self) -> tuple[float, ...]:
        """Efficacy boundaries at the planned fractions."""
        out, hist = [], []
        for j in range(1, self.n_looks + 1):
            b = self.bounds(j, history=tuple(hist))
            out.append(b.efficacy)
            hist.append((j, b.ratio))
        return tuple(out)

    def efficacy(self, j: int) -> float:
        self._check(j)
        return self.planned_bounds()[j - 1]

    def futility_bound(self, j: int) -> float:
        self._check(j)
        return self._futility(j, self.fractions[j - 1], self.is_final(j), self.efficacy(j))

    def verdict(self, j: int, t: float | None) -> Verdict:
        """Decision at look j at the PLANNED fractions (for simulation of a planned-information design)."""
        hist = tuple((i, self.fractions[i - 1]) for i in range(1, j))
        return self.bounds(j, history=hist).verdict(t)

    def bounds(self, j: int, info: float | None = None,
               history: Sequence[tuple[int, float | None]] = ()) -> LookBounds:
        """Boundaries at look j given the observed information `info` (1 / SE^2) at this look and `history`, the
        (look, ratio) of the looks evaluated before it (`LookBounds.ratio` of each; skipped looks left out)."""
        self._check(j)
        u = self._ratio(j, info)
        final = self.is_final(j) or (self.observed and info is not None and u >= 1.0)
        t = min(u, 1.0)
        if j < self.first_efficacy and not final:
            return LookBounds(math.inf, self._futility(j, t, False, math.inf), False, t, u)
        prior = tuple(r for i, r in history if i < j and i >= self.first_efficacy and r is not None)
        if prior and u <= prior[-1]:
            u = prior[-1] * (1.0 + 1e-3)          # no information gained: a token increment keeps the joint law proper
            t = min(u, 1.0)
        b = self._boundary(prior, u, final)
        return LookBounds(b, self._futility(j, t, final, b), final, t, u)

    def as_dict(self) -> dict[str, Any]:
        def r(x: float) -> float | None:
            return round(x, 4) if math.isfinite(x) else None
        eff = self.planned_bounds()
        return {"kind": "spending", "alpha": self.alpha, "spending": self.spending,
                "fractions": [round(f, 4) for f in self.fractions],
                "max_information": None if self.max_information is None else round(self.max_information, 2),
                "efficacy": [r(e) for e in eff],
                "futility": [r(self._futility(j, self.fractions[j - 1], False, eff[j - 1]))
                             for j in range(1, self.n_looks)],
                "first_efficacy": self.first_efficacy}

    # ------------------------------------------------------------------ internals
    def _check(self, j: int) -> None:
        if not 1 <= j <= self.n_looks:
            raise IndexError(f"look {j} not in 1..{self.n_looks}")

    def _ratio(self, j: int, info: float | None) -> float:
        if not self.observed or info is None or not math.isfinite(info) or info <= 0:
            return self.fractions[j - 1]
        q = self.resolution
        return max(q, round(info / self.max_information / q) * q)

    def _futility(self, j: int, t: float, final: bool, b: float) -> float:
        if final:
            return b
        if self.futility is None:
            return linear_futility(t)
        return self.futility[j - 1] if j - 1 < len(self.futility) else -math.inf

    def _boundary(self, prior: tuple[float, ...], u: float, final: bool) -> float:
        key = (prior, u, final)
        b = self._cache.get(key)
        if b is None:
            spent = self.spend(min(prior[-1], 1.0)) if prior else 0.0
            target = self.alpha - spent if final else self.spend(min(u, 1.0)) - spent
            b = _solve(self._state(prior), u, target)
            if len(self._cache) > 1_000_000:
                self._cache.clear()
            self._cache[key] = b
        return b

    def _state(self, prior: tuple[float, ...]) -> _State | None:
        if not prior:
            return None
        s = self._states.get(prior)
        if s is None:
            b = self._boundary(prior[:-1], prior[-1], False)
            parent = self._state(prior[:-1])
            s = (_first(prior[-1], b, 0.0, self.grid) if parent is None
                 else _update(parent, prior[-1], b, 0.0, self.grid))
            if len(self._states) > 50_000:      # about 200 floats each
                self._states.clear()
            self._states[prior] = s
        return s
