"""Monte Carlo calibration of a group-sequential design, and a CLI to print it.

Model. Each task i has a true pass probability p_i (smoothed from real per-task results:
(successes + 0.5) / (trials + 1)). The reference A passes task i with p_i, the candidate B with
logistic(logit(p_i) + shift): shift = 0 is the null (no effect). Every trial is an independent
Bernoulli draw. For every simulated evaluation we compute the paired t (per-task means, unit =
task, see `paired.py`) at each look of the schedule:

  ATTEMPT_MAJOR  look r: B's first r trials of every task;
  TASK_MAJOR     look j: all of B's trials of the first n_j tasks of a random task order;

and the reference side is either
  reference="all"   all `reference_trials` of A's trials of the look's tasks (A evaluated in full,
                    e.g. before the race, or a val reference already on file), or
  reference="same"  A's trials on the SAME units as B's look (A raced concurrently).
For TASK_MAJOR with reference_trials == trials the two coincide.

Calibration. Without futility, B is accepted iff t_j > c / sqrt(tau_j) at some look j that can
pass, i.e. iff M = max_j t_j * sqrt(tau_j) > c. So c for one-sided alpha is the (1 - alpha)
quantile of M under the null: no search needed, and any alpha (e.g. Bonferroni alpha / K) can be
read off the same simulation. Futility is non-binding and only lowers alpha; its cost in power
and its saving in trials are reported by `Simulation.operating`.

Pure python by default (fast enough for tests and a few thousand simulations); pass
backend="numpy" (or "auto") for large runs if numpy is installed (optional extra
`harnessforge[calibrate]`).

    python -m harnessforge.stats.calibrate                      # bundled 128-task tmax sample
    python -m harnessforge.stats.calibrate --rewards r.json --order task_major --trials 2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnessforge.contracts import Order
from harnessforge.records import read_jsonl
from harnessforge.stats.boundaries import Design, Verdict
from harnessforge.stats.information import Information
from harnessforge.stats.looks import DEFAULT_CHECKPOINTS, LookSchedule

__all__ = ["Operating", "Simulation", "load_rates", "simulate", "smoothed_rates"]


def smoothed_rates(results: Mapping[str, Sequence[float]] | Sequence[Sequence[float]]) -> list[float]:
    """Per-task pass probabilities from per-task reward lists: (sum + 0.5) / (n + 1)."""
    rows = results.values() if isinstance(results, Mapping) else results
    return [(math.fsum(v) + 0.5) / (len(v) + 1) for v in rows if len(v)]


def load_rates(path: str | Path | None = None, candidate: str | None = None) -> list[float]:
    """Smoothed per-task rates from a JSON file {task: [rewards]} (or {task: rate}), a JSONL of
    TrialRecords (`candidate` picks one; default: all records), or the bundled tmax sample."""
    if path is None:
        from harnessforge.stats._tmax128 import SUCCESSES, TRIALS
        return [(s + 0.5) / (TRIALS + 1) for s in SUCCESSES]
    p = Path(path)
    if p.suffix == ".jsonl":
        res: dict[str, list[float]] = {}
        for r in read_jsonl(p):
            if r.scored and (candidate is None or r.candidate == candidate):
                res.setdefault(r.task, []).append(float(r.reward or 0.0))
        return smoothed_rates(res)
    text = p.read_text()
    obj = json.loads(text.splitlines()[0]) if not text.lstrip().startswith("{\n") else json.loads(text)
    if all(isinstance(v, (int, float)) for v in obj.values()):
        return [float(v) for v in obj.values()]
    return smoothed_rates(obj)


def _shifted(p: float, shift: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) + shift)))


@dataclass(frozen=True)
class Operating:
    """Operating characteristics of a design on one simulation."""
    accept: float                  # fraction of simulations accepted
    mean_fraction: float           # mean information fraction at stopping (B's share of units)
    stop_looks: tuple[float, ...]  # distribution of the stopping look


@dataclass
class Simulation:
    fractions: tuple[float, ...]
    t: list[list[float]]           # sims x looks; nan = look had < 2 tasks
    acc_a: float                   # expected accuracy of A (mean p)
    acc_b: float                   # expected accuracy of B

    @property
    def sims(self) -> int:
        return len(self.t)

    def max_stat(self, first_efficacy: int = 2) -> list[float]:
        J = len(self.fractions)
        js = [j for j in range(J) if j + 1 >= first_efficacy or j == J - 1]
        sq = [math.sqrt(self.fractions[j]) for j in js]
        out = []
        for row in self.t:
            vals = [row[j] * s for j, s in zip(js, sq) if not math.isnan(row[j])]
            out.append(max(vals) if vals else -math.inf)
        return out

    def c_for(self, alpha: float, first_efficacy: int = 2) -> float:
        """Smallest simulated c with P(accept | no futility) <= alpha on this (null) simulation."""
        m = sorted(self.max_stat(first_efficacy))
        k = math.floor(alpha * len(m))                 # at most k simulations may exceed c
        return m[len(m) - k - 1]

    def operating(self, design: Design, futility: bool = True) -> Operating:
        if design.n_looks != len(self.fractions):
            raise ValueError("design and simulation have different numbers of looks")
        acc = 0
        frac = 0.0
        stops = [0] * design.n_looks
        nf = design if futility else Design(design.fractions, design.c, (),
                                            design.first_efficacy)
        for row in self.t:
            for j in range(1, design.n_looks + 1):
                v = nf.verdict(j, row[j - 1])
                if v is not Verdict.CONTINUE:
                    acc += v is Verdict.PASS
                    frac += design.fractions[j - 1]
                    stops[j - 1] += 1
                    break
        n = len(self.t)
        return Operating(acc / n, frac / n, tuple(s / n for s in stops))

    def fixed_accept(self, z: float = 1.645) -> float:
        """Accept rate of a single fixed test at the last look."""
        return sum(row[-1] > z for row in self.t) / len(self.t)


def _t(s: float, ss: float, n: int) -> float:
    if n < 2:
        return math.nan
    md = s / n
    var = max(ss - s * s / n, 0.0) / (n - 1)
    se = math.sqrt(var / n)
    if se <= 1e-12:
        return math.inf if md > 1e-12 else -math.inf if md < -1e-12 else 0.0
    return md / se


def _counts(order: Order, n_tasks: int, trials: int, checkpoints: Sequence[float]) -> list[int]:
    sched = LookSchedule(order, tuple(range(n_tasks)), trials, tuple(checkpoints))  # type: ignore[arg-type]
    return [len(lk.tasks) for lk in sched.looks]


def _simulate_python(pa, pb, order, trials, counts, same, ra, sims, seed):
    rng = random.Random(seed)
    rnd = rng.random
    T, K = len(pa), trials
    out: list[list[float]] = []
    if order is Order.ATTEMPT_MAJOR:
        for _ in range(sims):
            s = [0.0] * K
            ss = [0.0] * K
            for p, q in zip(pa, pb):
                a = 0.0 if same else sum(rnd() < p for _ in range(ra)) / ra
                cb = ca = 0
                for r in range(K):
                    cb += rnd() < q
                    if same:
                        ca += rnd() < p
                        a = ca / (r + 1)
                    d = cb / (r + 1) - a
                    s[r] += d
                    ss[r] += d * d
            out.append([_t(s[r], ss[r], T) for r in range(K)])
        return out
    idx = list(range(T))
    for _ in range(sims):
        rng.shuffle(idx)
        row, s, ss, n = [], 0.0, 0.0, 0
        ci = 0
        for i in idx:
            d = sum(rnd() < pb[i] for _ in range(K)) / K - sum(rnd() < pa[i] for _ in range(ra)) / ra
            s += d
            ss += d * d
            n += 1
            while ci < len(counts) and counts[ci] == n:
                row.append(_t(s, ss, n))
                ci += 1
        out.append(row)
    return out


def _simulate_numpy(pa, pb, order, trials, counts, same, ra, sims, seed):
    import numpy as np  # optional dependency
    rng = np.random.default_rng(seed)
    pa_, pb_ = np.asarray(pa), np.asarray(pb)
    T, K = len(pa), trials
    out = np.empty((sims, len(counts)))
    chunk = max(1, 2_000_000 // max(1, T * K))
    for lo in range(0, sims, chunk):
        S = min(chunk, sims - lo)
        if order is Order.ATTEMPT_MAJOR:
            B = (rng.random((S, T, K)) < pb_[None, :, None]).cumsum(2) / np.arange(1, K + 1)
            if same:
                A = (rng.random((S, T, K)) < pa_[None, :, None]).cumsum(2) / np.arange(1, K + 1)
            else:
                A = np.repeat((rng.binomial(ra, pa_, size=(S, T)) / ra)[:, :, None], K, 2)
            d = B - A                                        # S x T x K
            n = T
            md = d.mean(1)
            sd = d.std(1, ddof=1)
        else:
            perm = np.argsort(rng.random((S, T)), axis=1)
            d = rng.binomial(K, pb_, size=(S, T)) / K - rng.binomial(ra, pa_, size=(S, T)) / ra
            d = np.take_along_axis(d, perm, 1)
            md = np.stack([d[:, :c].mean(1) for c in counts], 1)
            sd = np.stack([d[:, :c].std(1, ddof=1) if c > 1 else np.full(S, np.nan)
                           for c in counts], 1)
            n = np.asarray(counts, float)[None, :]
        se = sd / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 1e-12, md / se, np.sign(md) * np.inf)
            t = np.where((se <= 1e-12) & (np.abs(md) <= 1e-12), 0.0, t)
            t = np.where(np.isnan(sd), np.nan, t)
        out[lo:lo + S] = t
    return out.tolist()


def simulate(rates: Sequence[float], *, order: Order | str = Order.ATTEMPT_MAJOR,
             trials: int = 5, checkpoints: Sequence[float] = DEFAULT_CHECKPOINTS,
             reference: str = "all", reference_trials: int | None = None, shift: float = 0.0,
             sims: int = 2000, seed: int = 0, backend: str = "python",
             trial_looks: Sequence[int] | None = None, information: Information | None = None) -> Simulation:
    """Simulate paired t at every look (see module docstring). ATTEMPT_MAJOR: `trial_looks` picks the looks
    (trials per task; default every trial) and `information` makes their fractions I(k_j) / I(K)."""
    order = Order(order)
    if reference not in ("all", "same"):
        raise ValueError("reference must be 'all' or 'same'")
    ra = trials if reference_trials is None or reference == "same" else reference_trials
    pa = [min(max(p, 0.0), 1.0) for p in rates]
    pb = [_shifted(p, shift) if shift else p for p in pa]
    counts = _counts(order, len(pa), trials, checkpoints)
    if order is Order.ATTEMPT_MAJOR:
        sched = LookSchedule(order, tuple(map(str, range(len(pa)))), trials,
                             trial_looks=None if trial_looks is None else tuple(trial_looks),
                             information=information)
        fractions, picks = sched.fractions, [k - 1 for k in sched.trial_counts]
    else:
        fractions, picks = tuple(c / len(pa) for c in counts), None
    if backend == "auto":
        try:
            import numpy  # noqa: F401
            backend = "numpy"
        except ImportError:
            backend = "python"
    fn = _simulate_numpy if backend == "numpy" else _simulate_python
    t = fn(pa, pb, order, trials, counts, reference == "same", ra, sims, seed)
    if picks is not None and len(picks) != trials:
        t = [[row[k] for k in picks] for row in t]
    return Simulation(fractions, t, sum(pa) / len(pa), sum(pb) / len(pb))


# ----------------------------------------------------------------------------- CLI
def _floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _futility_rule(s: str) -> tuple[str, tuple[float, ...] | None]:
    s = s.strip()
    if s == "none":
        return "none", ()
    if s == "linear":
        return "linear (-1 + 2.5*tau)", None
    return s, _floats(s)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessforge.stats.calibrate",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--rewards", help="JSON {task: [rewards]} / {task: rate}, or TrialRecord "
                    "JSONL; default: bundled 128-task tmax sample")
    ap.add_argument("--candidate", help="with a JSONL: the candidate whose results to use")
    ap.add_argument("--order", default="attempt_major", choices=[o.value for o in Order])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--checkpoints", type=_floats, default=DEFAULT_CHECKPOINTS,
                    help="TASK_MAJOR look checkpoints (fractions of the task list)")
    ap.add_argument("--trial-looks", type=lambda s: tuple(int(x) for x in s.split(",")),
                    help="ATTEMPT_MAJOR looks at these trials per task (e.g. 2,4,8,24)")
    ap.add_argument("--tau2", type=float, help="with --trial-looks: planning tau2 for information fractions")
    ap.add_argument("--sigma2", type=float, help="with --trial-looks: planning sigma2 (both sides)")
    ap.add_argument("--reference", default="all", choices=["all", "same"])
    ap.add_argument("--reference-trials", type=int)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=1, help="candidates raced (Bonferroni alpha/k)")
    ap.add_argument("--c", type=float, help="c to report operating characteristics for "
                    "(default: the calibrated one)")
    ap.add_argument("--c-grid", type=_floats, default=(1.60, 1.65, 1.70, 1.75))
    ap.add_argument("--first-efficacy", type=int, default=2)
    ap.add_argument("--futility", action="append",
                    help="'none', 'linear' or comma-separated thresholds; repeatable "
                         "(default: none, linear)")
    ap.add_argument("--shifts", type=_floats, default=(0.0, -0.2, 0.1, 0.2, 0.3),
                    help="logit shifts of the candidate's per-task pass rates")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="auto", choices=["auto", "python", "numpy"])
    a = ap.parse_args(argv)

    rates = load_rates(a.rewards, a.candidate)
    info = Information(a.tau2 or 0.0, a.sigma2) if a.sigma2 else None
    kw: dict[str, Any] = dict(order=a.order, trials=a.trials, checkpoints=a.checkpoints,
                              reference=a.reference, reference_trials=a.reference_trials,
                              sims=a.sims, backend=a.backend, trial_looks=a.trial_looks,
                              information=info)
    null = simulate(rates, shift=0.0, seed=a.seed, **kw)
    alpha = a.alpha / a.k
    c_cal = null.c_for(alpha, a.first_efficacy)
    c = a.c if a.c is not None else round(c_cal, 2)
    rules = [_futility_rule(s) for s in (a.futility or ["none", "linear"])]
    base = Design(null.fractions, c, (), a.first_efficacy)
    se = math.sqrt(alpha * (1 - alpha) / a.sims)

    print(f"tasks {len(rates)}  mean p {sum(rates) / len(rates):.3f}  order {a.order}  "
          f"trials {a.trials}  reference {a.reference}  sims {a.sims}  seed {a.seed}")
    print(f"looks: information fractions {[round(f, 3) for f in null.fractions]}")
    print(f"fixed one-look z=1.645 alpha {null.fixed_accept(1.645):.4f}")
    for cg in a.c_grid:
        d = base.with_c(cg)
        print(f"OBF c {cg:.2f} efficacy {d.as_dict()['efficacy']}  alpha (no futility) "
              f"{null.operating(d, futility=False).accept:.4f}")
    print(f"calibrated c for one-sided alpha {alpha:.4g}"
          f"{f' (= {a.alpha:g}/{a.k})' if a.k > 1 else ''}: {c_cal:.3f}  (MC se of alpha "
          f"{se:.4f})")
    print(f"\noperating characteristics at c = {c:.2f}  (fraction = share of B's units used)")
    for shift in a.shifts:
        sim = null if shift == 0.0 else simulate(rates, shift=shift, seed=a.seed + 1, **kw)
        print(f"shift {shift:+.2f} (acc {sim.acc_b - sim.acc_a:+.3f})  fixed-power "
              f"{sim.fixed_accept(1.645):.3f}")
        for name, fut in rules:
            d = Design(null.fractions, c, fut, a.first_efficacy)
            oc = sim.operating(d)
            ks = a.trial_looks or tuple(range(1, a.trials + 1))
            if a.trial_looks and ks[-1] != a.trials:
                ks = (*ks, a.trials)
            mean_k = sum(p * k for p, k in zip(oc.stop_looks, ks))
            extra = (f" ({mean_k:.2f} of {a.trials} trials/task)"
                     if Order(a.order) is Order.ATTEMPT_MAJOR else "")
            print(f"   futility {name:24s} accept {oc.accept:.3f}  mean fraction "
                  f"{oc.mean_fraction:.3f}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
