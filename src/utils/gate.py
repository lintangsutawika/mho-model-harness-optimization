"""Accept/reject gates shared by the harness loop and the model loop.

A candidate B is compared with a reference A (its parent on the train subset, or the frontier
best on val) TASK BY TASK: both ran the same tasks, so the per-task differences cancel the
task-difficulty variance that dominates a single mean. Per task, the --trials attempts are
averaged first, so the unit of variation is the task.

    D  = mean_i(B_i - A_i)          paired accuracy difference
    SE = sd_i(B_i - A_i) / sqrt(n)

Two ways to be accepted:

  better   D - z*SE > 0                       (clearly more accurate; z=0 -> plain B > A)
  cheaper  D - z_c*SE > -delta                (not worse than A by more than delta ...)
           and, if an anchor is given, the same vs the anchor (the most accurate harness/model
           accepted so far, so a chain of "slightly worse but cheaper" cannot drift down)
           and cost(A) - cost(B) clears z_c*SE  (... and clearly cheaper ...)
           and (cost(A) - cost(B)) / cost(A) >= min_gain   (... by a meaningful margin)

Cost is per-trial agent output tokens by default (independent of node load / concurrency;
a trial killed at the timeout still reports its tokens), or agent wall seconds.

Everything here is pure: dirs in, numbers out.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

COST_METRICS = ("tokens", "seconds")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class GateConfig:
    train_z: float = 1.0           # better-path z on the train subset (a cheap screen)
    val_z: float = 1.645           # better-path z on val (one-sided 95%)
    cheaper_delta: float = 0.03    # max accuracy loss tolerated on the cheaper path; 0 = off
    cheaper_z: float = 1.0         # z for the cheaper path (non-inferiority + cost tests)
    cheaper_min_gain: float = 0.10  # min relative cost reduction on the cheaper path
    cost_metric: str = "tokens"

    @classmethod
    def from_args(cls, args: Any) -> "GateConfig":
        d = cls()
        return cls(**{f: getattr(args, f"gate_{f}", getattr(d, f)) for f in
                      ("train_z", "val_z", "cheaper_delta", "cheaper_z", "cheaper_min_gain",
                       "cost_metric")})

    @classmethod
    def env_defaults(cls) -> dict[str, Any]:
        """CLI defaults, overridable from the environment (MHO_GATE_*)."""
        d = cls()
        return {
            "train_z": _env_float("MHO_GATE_TRAIN_Z", d.train_z),
            "val_z": _env_float("MHO_GATE_VAL_Z", d.val_z),
            "cheaper_delta": _env_float("MHO_GATE_CHEAPER_DELTA", d.cheaper_delta),
            "cheaper_z": _env_float("MHO_GATE_CHEAPER_Z", d.cheaper_z),
            "cheaper_min_gain": _env_float("MHO_GATE_CHEAPER_MIN_GAIN", d.cheaper_min_gain),
            "cost_metric": os.environ.get("MHO_GATE_COST", d.cost_metric),
        }


# --------------------------------------------------------------------------- per-task data
def _seconds(span: dict | None) -> float | None:
    if not span or not span.get("started_at") or not span.get("finished_at"):
        return None
    try:
        a = datetime.fromisoformat(span["started_at"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(span["finished_at"].replace("Z", "+00:00"))
    except ValueError:
        return None
    return (b - a).total_seconds()


def trial_cost(result: dict, metric: str = "tokens") -> float | None:
    if metric == "seconds":
        return _seconds(result.get("agent_execution"))
    tok = (result.get("agent_result") or {}).get("n_output_tokens")
    return float(tok) if tok is not None else None


def per_task(trials_dir: str | Path, cost_metric: str = "tokens") -> dict[str, dict[str, float]]:
    """{task: {"reward": mean over attempts (missing reward = 0), "cost": mean over attempts
    that reported one (absent if none), "n": attempts}} from <dir>/*/result.json."""
    rewards: dict[str, list[float]] = {}
    costs: dict[str, list[float]] = {}
    for f in glob.glob(str(Path(trials_dir) / "*" / "result.json")):
        try:
            j = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        task = j.get("task_name") if isinstance(j, dict) else None
        if not task:
            continue
        r = ((j.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rewards.setdefault(task, []).append(float(r) if r is not None else 0.0)
        c = trial_cost(j, cost_metric)
        if c is not None:
            costs.setdefault(task, []).append(c)
    out: dict[str, dict[str, float]] = {}
    for t, rs in rewards.items():
        row = {"reward": sum(rs) / len(rs), "n": float(len(rs))}
        if costs.get(t):
            row["cost"] = sum(costs[t]) / len(costs[t])
        out[t] = row
    return out


# --------------------------------------------------------------------------- statistics
@dataclass(frozen=True)
class Paired:
    n: int
    mean_a: float
    mean_b: float
    d: float        # mean(B - A)
    se: float       # sd(B - A) / sqrt(n)

    def t(self) -> float:
        return self.d / self.se if self.se > 0 else (math.inf if self.d > 0 else
                                                     -math.inf if self.d < 0 else 0.0)

    def as_dict(self) -> dict[str, float]:
        return {"n": self.n, "A": round(self.mean_a, 6), "B": round(self.mean_b, 6),
                "d": round(self.d, 6), "se": round(self.se, 6),
                "t": round(self.t(), 3) if math.isfinite(self.t()) else None}


def paired(a: dict[str, dict], b: dict[str, dict], key: str = "reward") -> Paired | None:
    ks = sorted(k for k in set(a) & set(b) if key in a[k] and key in b[k])
    n = len(ks)
    if n < 2:
        return None
    xa = [a[k][key] for k in ks]
    xb = [b[k][key] for k in ks]
    d = [y - x for x, y in zip(xa, xb)]
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
    return Paired(n, sum(xa) / n, sum(xb) / n, md, sd / math.sqrt(n))


# --------------------------------------------------------------------------- decision
@dataclass
class Decision:
    accept: bool
    path: str | None                 # "better" | "cheaper" | None
    reason: str
    acc: Paired | None = None
    cost: Paired | None = None       # A = reference cost, B = candidate cost
    anchor: Paired | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def fields(self) -> dict[str, Any]:
        return {"accept": self.accept, "path": self.path, "reason": self.reason,
                "acc": self.acc.as_dict() if self.acc else None,
                "cost": self.cost.as_dict() if self.cost else None,
                "anchor": self.anchor.as_dict() if self.anchor else None, **self.extra}


def decide(a: dict[str, dict], b: dict[str, dict], *, z: float, cfg: GateConfig,
           anchor: dict[str, dict] | None = None,
           fallback: tuple[float | None, float] | None = None) -> Decision:
    """Accept B over reference A? `fallback` = (A metric, B metric) is used when there is no
    usable per-task data (e.g. trials without task_name): then plain B > A, as before."""
    acc = paired(a, b, "reward")
    if acc is None:
        fa, fb = fallback if fallback else (None, 0.0)
        ok = fa is None or fb > fa
        return Decision(ok, "better" if ok else None,
                        f"no per-task data; scalar {fb:.4f} {'>' if ok else '<='} "
                        f"{fa if fa is not None else float('nan'):.4f}")
    lo = acc.d - z * acc.se
    if lo > 0 or (z <= 0 and acc.d > 0):
        return Decision(True, "better", f"B-A={acc.d:+.4f}±{acc.se:.4f} (t={acc.t():.2f} > {z:g})",
                        acc=acc)
    why = [f"B-A={acc.d:+.4f}±{acc.se:.4f} (t={acc.t():.2f}, need > {z:g})"]
    if cfg.cheaper_delta <= 0:
        return Decision(False, None, why[0], acc=acc)

    zc, delta = cfg.cheaper_z, cfg.cheaper_delta
    ni_ok = acc.d - zc * acc.se > -delta
    anc = paired(anchor, b, "reward") if anchor is not None else None
    anc_ok = anc is None or anc.d - zc * anc.se > -delta
    cost = paired(a, b, "cost")
    if cost is None or cost.mean_a <= 0:
        cheaper_ok, gain = False, 0.0
        why.append("cheaper: no cost data")
    else:
        save = Paired(cost.n, cost.mean_a, cost.mean_b, -cost.d, cost.se)   # A - B
        gain = save.d / cost.mean_a
        cheaper_ok = save.d - zc * save.se > 0 and gain >= cfg.cheaper_min_gain
        why.append(f"cost {cfg.cost_metric} {cost.mean_a:.0f}->{cost.mean_b:.0f} "
                   f"({-gain:+.0%}, t={save.t():.2f}; need <= -{cfg.cheaper_min_gain:.0%}, "
                   f"t > {zc:g})")
    why.append(f"non-inferior (lower bound {acc.d - zc * acc.se:+.4f} > -{delta:g}): {ni_ok}")
    if anc is not None:
        why.append(f"vs anchor {anc.d:+.4f}±{anc.se:.4f} non-inferior: {anc_ok}")
    ok = ni_ok and anc_ok and cheaper_ok
    return Decision(ok, "cheaper" if ok else None, "; ".join(why), acc=acc, cost=cost, anchor=anc,
                    extra={"cost_gain": round(gain, 4), "cost_metric": cfg.cost_metric})


def decide_dirs(a_dir: str | Path | None, b_dir: str | Path, *, z: float, cfg: GateConfig,
                anchor_dir: str | Path | None = None,
                fallback: tuple[float | None, float] | None = None) -> Decision:
    a = per_task(a_dir, cfg.cost_metric) if a_dir else {}
    b = per_task(b_dir, cfg.cost_metric)
    anc = (per_task(anchor_dir, cfg.cost_metric)
           if anchor_dir and Path(anchor_dir) != Path(a_dir or "") else None)
    return decide(a, b, z=z, cfg=cfg, anchor=anc, fallback=fallback)
