"""Paired per-task statistics and the fixed-sample accept decision.

A candidate B is compared with a reference A TASK BY TASK: both ran the same tasks, so the
per-task differences cancel the task-difficulty variance that dominates a single mean. Per task
the trials are averaged first (`harnessforge.records.per_task`), so the unit of variation is the
task, not the trial:

    D  = mean_i(B_i - A_i)          paired accuracy difference over tasks both have data for
    SE = sd_i(B_i - A_i) / sqrt(n)
    t  = D / SE

`decide` has two ways to accept B:

  better   t > z                              (clearly more accurate)
  cheaper  D - z_c*SE > -delta                (not worse than A by more than delta ...)
           [and the same vs an anchor, so a chain of "slightly worse but cheaper" cannot drift]
           and cost(A) - cost(B) clears z_c*SE  (... clearly cheaper ...)
           and (cost(A) - cost(B)) / cost(A) >= min_gain   (... by a meaningful margin)

The cheaper path is OFF by default (`GateConfig.cheaper_delta = 0`). Cost is
`TaskStats.cost` (mean output tokens per scored trial).

Pure and stdlib-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from harnessforge.records import TaskStats

__all__ = ["Decision", "GateConfig", "Paired", "decide", "paired", "task_mean"]


@dataclass(frozen=True)
class Paired:
    """Paired comparison of B against A over `n` common tasks (A = reference)."""
    n: int
    mean_a: float
    mean_b: float
    d: float        # mean over tasks of (B - A)
    se: float       # sd(B - A) / sqrt(n)

    @property
    def t(self) -> float:
        """d / se; +-inf when se == 0 and d != 0, 0 when both are 0."""
        if self.se > 0:
            return self.d / self.se
        return math.inf if self.d > 0 else -math.inf if self.d < 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        t = self.t
        return {"n": self.n, "A": round(self.mean_a, 6), "B": round(self.mean_b, 6),
                "d": round(self.d, 6), "se": round(self.se, 6),
                "t": round(t, 3) if math.isfinite(t) else None}


def _value(s: TaskStats, key: str) -> float | None:
    v = getattr(s, key)
    return None if v is None else float(v)


def paired(a: Mapping[str, TaskStats], b: Mapping[str, TaskStats],
           key: str = "reward", weights: Mapping[str, float] | None = None) -> Paired | None:
    """Paired stats of `b` minus `a` over the tasks both have (and, for key="cost", that both
    report a cost for). None with fewer than 2 common tasks.

    `weights` (task -> w > 0; tasks without one are left out): the weighted mean D = sum w d / sum w with
    the robust (sandwich) standard error SE = sqrt(n / (n - 1) * sum w^2 (d - D)^2) / sum w. This is the
    estimator of a look whose tasks have unequal trials (inverse-variance weights from
    `LookSchedule.weights`); the SE does not rely on the weights being right, and with equal weights it is
    the plain sd / sqrt(n)."""
    xa: list[float] = []
    xb: list[float] = []
    ws: list[float] = []
    for k in sorted(set(a) & set(b)):
        va, vb = _value(a[k], key), _value(b[k], key)
        if va is None or vb is None:
            continue
        w = 1.0 if weights is None else weights.get(k)
        if w is None or not w > 0:
            continue
        xa.append(va)
        xb.append(vb)
        ws.append(float(w))
    n = len(xa)
    if n < 2:
        return None
    d = [y - x for x, y in zip(xa, xb)]
    if weights is None:
        md = math.fsum(d) / n
        sd = math.sqrt(math.fsum((x - md) ** 2 for x in d) / (n - 1))
        if sd < 1e-12:                       # identical differences up to float noise
            sd = 0.0
        return Paired(n, math.fsum(xa) / n, math.fsum(xb) / n, md, sd / math.sqrt(n))
    sw = math.fsum(ws)
    md = math.fsum(w * x for w, x in zip(ws, d)) / sw
    se = math.sqrt(n / (n - 1) * math.fsum((w * (x - md)) ** 2 for w, x in zip(ws, d))) / sw
    if se < 1e-12:
        se = 0.0
    return Paired(n, math.fsum(w * x for w, x in zip(ws, xa)) / sw, math.fsum(w * x for w, x in zip(ws, xb)) / sw,
                  md, se)


def task_mean(pt: Mapping[str, TaskStats]) -> float:
    """Accuracy as the mean over tasks of per-task means."""
    return math.fsum(s.reward for s in pt.values()) / len(pt) if pt else 0.0


@dataclass(frozen=True)
class GateConfig:
    z: float = 1.645               # better path: one-sided 95% for a single fixed look
    cheaper_delta: float = 0.0     # max accuracy loss tolerated on the cheaper path; 0 = off
    cheaper_z: float = 1.0         # z for the cheaper path (non-inferiority + cost tests)
    cheaper_min_gain: float = 0.10  # min relative cost reduction on the cheaper path

    @property
    def cheaper_enabled(self) -> bool:
        return self.cheaper_delta > 0


@dataclass(frozen=True)
class Decision:
    accept: bool
    path: str | None                 # "better" | "cheaper" | None
    reason: str
    acc: Paired | None = None
    cost: Paired | None = None       # A = reference cost, B = candidate cost
    anchor: Paired | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"accept": self.accept, "path": self.path, "reason": self.reason,
                "acc": self.acc.as_dict() if self.acc else None,
                "cost": self.cost.as_dict() if self.cost else None,
                "anchor": self.anchor.as_dict() if self.anchor else None, **self.extra}


def decide(a: Mapping[str, TaskStats], b: Mapping[str, TaskStats], *,
           cfg: GateConfig = GateConfig(), z: float | None = None,
           anchor: Mapping[str, TaskStats] | None = None) -> Decision:
    """Accept B over reference A? `z` overrides `cfg.z` for the better path (a sequential
    design passes its look's efficacy boundary here). Without at least 2 paired tasks, B is
    rejected."""
    z = cfg.z if z is None else z
    acc = paired(a, b, "reward")
    if acc is None:
        return Decision(False, None, "fewer than 2 paired tasks")
    if acc.t > z:
        return Decision(True, "better", f"B-A={acc.d:+.4f}±{acc.se:.4f} (t={acc.t:.2f} > {z:.3g})",
                        acc=acc)
    why = [f"B-A={acc.d:+.4f}±{acc.se:.4f} (t={acc.t:.2f}, need > {z:.3g})"]
    if not cfg.cheaper_enabled:
        return Decision(False, None, why[0], acc=acc)

    zc, delta = cfg.cheaper_z, cfg.cheaper_delta
    ni_ok = acc.d - zc * acc.se > -delta
    anc = paired(anchor, b, "reward") if anchor is not None else None
    anc_ok = anc is None or anc.d - zc * anc.se > -delta
    cost = paired(a, b, "cost")
    gain = 0.0
    if cost is None or cost.mean_a <= 0:
        cheaper_ok = False
        why.append("cheaper: no cost data")
    else:
        save_d = -cost.d                                    # cost(A) - cost(B), per task
        gain = save_d / cost.mean_a
        save_t = save_d / cost.se if cost.se > 0 else (math.inf if save_d > 0 else -math.inf)
        cheaper_ok = save_d - zc * cost.se > 0 and gain >= cfg.cheaper_min_gain
        why.append(f"cost {cost.mean_a:.0f}->{cost.mean_b:.0f} ({-gain:+.0%}, t={save_t:.2f}; "
                   f"need <= -{cfg.cheaper_min_gain:.0%}, t > {zc:g})")
    why.append(f"non-inferior (lower bound {acc.d - zc * acc.se:+.4f} > -{delta:g}): {ni_ok}")
    if anc is not None:
        why.append(f"vs anchor {anc.d:+.4f}±{anc.se:.4f} non-inferior: {anc_ok}")
    ok = ni_ok and anc_ok and cheaper_ok
    return Decision(ok, "cheaper" if ok else None, "; ".join(why), acc=acc, cost=cost,
                    anchor=anc, extra={"cost_gain": round(gain, 4)})
