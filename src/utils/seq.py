"""Sequential train gate: look at B while its eval runs, stop as soon as the answer is clear.

Harbor launches trials attempt-major: every task once, then every task again, ... (--trials
times). So look r (r = 1..K, K = --trials) uses the FIRST r LAUNCHED trials of every task and
is taken once all of them have finished. Using launch order -- never "whatever finished
first" -- keeps slow trials (timeouts, which are mostly failures) from being left out of the
early looks, which would bias them in B's favour.

At look r, the paired per-task t of B (first r trials per task) vs A (all of A's trials) is
compared with two boundaries:

    efficacy  t > c * sqrt(K / r)        (r >= 2; O'Brien-Fleming shape)  -> pass
    futility  t <= f_r                   (r < K)                          -> fail
    else continue; at r = K: pass iff t > c

c = 1.72 keeps the overall one-sided false-accept rate at 5% for K = 5 (the fixed one-look
test it replaces used z = 1.645), calibrated by simulation on a real 128-task tmax subset.
Futility is non-binding (ignored in that calibration), so it only lowers the false-accept
rate. Default f = (-0.5, 0, 0.5, 1.0): -1 to -2 points of power, about half the B trials for
a no-effect candidate and about two thirds less for a worse one.

Everything here is pure: dirs in, numbers out.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from utils.gate import Paired, env_failed, paired, trial_cost

DEFAULT_C = 1.72
DEFAULT_FUTILITY = (-0.5, 0.0, 0.5, 1.0)


@dataclass(frozen=True)
class SeqConfig:
    k: int                                   # looks = trials per task
    c: float = DEFAULT_C
    futility: tuple[float, ...] = DEFAULT_FUTILITY

    def efficacy(self, r: int) -> float:
        return math.inf if r < 2 and self.k > 1 else self.c * math.sqrt(self.k / r)

    def fut(self, r: int) -> float:
        if r >= self.k:
            return math.inf                   # last look: everything not passing fails
        return self.futility[r - 1] if r - 1 < len(self.futility) else -math.inf


def parse_futility(s: str | None) -> tuple[float, ...]:
    if not s:
        return DEFAULT_FUTILITY
    return tuple(float(x) for x in s.split(",") if x.strip())


def _task_key(trial_dir: Path) -> str | None:
    try:
        cfg = json.loads((trial_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    p = ((cfg.get("task") or {}).get("path")) if isinstance(cfg, dict) else None
    return Path(p).name if p else None


def launched(trials_dir: str | Path) -> dict[str, list[dict]]:
    """{task dir name: [{"t0", "result"}, ...] in launch order} for every trial harbor has
    started under `trials_dir` (result None = still running). Launch time = config.json mtime
    (harbor writes it when the trial starts)."""
    out: dict[str, list[dict]] = {}
    d = Path(trials_dir)
    if not d.is_dir():
        return out
    for t in d.iterdir():
        if not t.is_dir() or not (t / "config.json").is_file():
            continue
        key = _task_key(t)
        if key is None:
            continue
        res = None
        if (t / "result.json").is_file():
            try:
                res = json.loads((t / "result.json").read_text())
            except (OSError, ValueError):
                res = None                   # being written: treat as not finished yet
            if not isinstance(res, dict) or not res.get("finished_at"):
                res = None
        out.setdefault(key, []).append({"t0": os.path.getmtime(t / "config.json"),
                                        "name": t.name, "result": res})
    for v in out.values():
        v.sort(key=lambda x: (x["t0"], x["name"]))
    return out


def look_pt(trials_dir: str | Path, r: int, n_tasks: int,
            cost_metric: str = "tokens") -> tuple[dict | None, dict]:
    """(per_task dict like utils.gate.per_task over the first r launched trials of every task,
    or None if look r is not ready yet; info {"env_bad", "n"}). Ready = all n_tasks tasks have
    r launched trials and all of those have finished."""
    rows = launched(trials_dir)
    info = {"env_bad": 0, "n": 0}
    if len(rows) < n_tasks:
        return None, info
    out: dict[str, dict] = {}
    for trials in rows.values():
        first = trials[:r]
        if len(first) < r or any(t["result"] is None for t in first):
            return None, info
        rs, cs, name = [], [], None
        for t in first:
            j = t["result"]
            info["n"] += 1
            if env_failed(j):
                info["env_bad"] += 1
                continue
            name = j.get("task_name") or name
            rew = ((j.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            rs.append(float(rew) if rew is not None else 0.0)
            c = trial_cost(j, cost_metric)
            if c is not None:
                cs.append(c)
        if not rs or not name:
            continue
        row = {"reward": sum(rs) / len(rs), "n": float(len(rs))}
        if cs:
            row["cost"] = sum(cs) / len(cs)
        out[name] = row
    return out, info


@dataclass
class SeqDecision:
    verdict: str            # "pass" | "fail" | "continue"
    r: int
    acc: Paired | None
    bound: float            # efficacy boundary at this look
    fut: float              # futility boundary at this look

    @property
    def reason(self) -> str:
        if self.acc is None:
            return f"look {self.r}: no paired data"
        t = self.acc.t()
        return (f"look {self.r}: B-A={self.acc.d:+.4f}±{self.acc.se:.4f} (t={t:.2f}; "
                f"pass > {self.bound:.2f}, stop <= {self.fut:.2f})")

    def fields(self) -> dict:
        return {"accept": self.verdict == "pass", "path": "better" if self.verdict == "pass"
                else None, "reason": self.reason, "acc": self.acc.as_dict() if self.acc else None,
                "cost": None, "anchor": None,
                "seq": {"look": self.r, "verdict": self.verdict,
                        "bound": round(self.bound, 3) if math.isfinite(self.bound) else None,
                        "futility": round(self.fut, 3) if math.isfinite(self.fut) else None}}


def decide_look(a: dict, b: dict, r: int, cfg: SeqConfig) -> SeqDecision:
    acc = paired(a, b, "reward")
    eff, fut = cfg.efficacy(r), cfg.fut(r)
    if acc is None:
        return SeqDecision("fail" if r >= cfg.k else "continue", r, None, eff, fut)
    t = acc.t()
    if t > eff:
        v = "pass"
    elif t <= fut:
        v = "fail"
    else:
        v = "continue"
    return SeqDecision(v, r, acc, eff, fut)
