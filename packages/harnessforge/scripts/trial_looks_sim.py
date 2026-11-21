"""Operating characteristics of look schedules on val200 (docs/stats.md section 7). Needs numpy.

Generative model (per simulated evaluation; SAME_UNITS pairing: the reference A raced concurrently on the same
(task, trial) units as the candidate B):

  * 200 base rates p_i resampled with replacement from the val200 seed-harness results (`stats._val200`: successes
    out of 8 per task), shrunk as (s + a) / (8 + 2a) with a fitted so that the per-trial variance summed over both
    sides, 2 mean p(1 - p), is the planning sigma2 = 0.249 (the unbiased within-task estimate on val200). The task
    order is random (resampling), as a shuffled request's is;
  * B's rates q_i = logistic(logit p_i + s + b z_i), z_i ~ N(0, 1), with (s, b) solved on a large sample so that the
    mean true difference is delta and the variance of the true per-task differences is tau2 (effect heterogeneity;
    when the shift alone gives more than tau2, b = 0 and the achieved tau2 is reported);
  * every trial an independent Bernoulli. At a look, task i contributes d_i = mean of B's first k_i trials - mean of
    A's first k_i trials; the estimate is `paired` (plain mean and sd / sqrt(n) when every task of the look has the
    same k; inverse-variance weights w_i = 1 / (tau2 + sigma2 / k_i) and the sandwich SE otherwise).

"identical" is the sharp null q_i = p_i (tau2 = 0). Every schedule is a `LookSchedule` and the spending designs run
through `SpendingDesign.bounds`, the code a `Race` uses.

    uv run --no-project -p 3.12 --with numpy --with-editable packages/harnessforge \\
        python packages/harnessforge/scripts/trial_looks_sim.py --sims 20000 --out sim.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from harnessforge.contracts import Order
from harnessforge.stats._val200 import SUCCESSES, TRIALS
from harnessforge.stats.boundaries import Design
from harnessforge.stats.calibrate import Simulation
from harnessforge.stats.information import VAL200
from harnessforge.stats.looks import LookSchedule
from harnessforge.stats.spending import SpendingDesign

K = 48                  # trials simulated per task (most schedules use up to 24; (4,8,16,32) 32; (a2) 48)
N = len(SUCCESSES)
TASKS = tuple(str(i) for i in range(N))
DELTAS = (0.0, 0.024, 0.05, 0.14, -0.02)
PLANNING = VAL200
SIM_RESOLUTION = 2e-3   # observed fractions rounded to this here (the library default is 1e-3): fewer distinct
                        # boundary computations; moves a boundary by < 0.01 at the earliest looks, less later
replace = dataclasses.replace


def schedules() -> dict[str, LookSchedule]:
    am = Order.ATTEMPT_MAJOR
    out = {f"({','.join(map(str, ks))})": LookSchedule(am, TASKS, ks[-1], trial_looks=ks, information=PLANNING)
           for ks in ((2, 4, 8, 24), (2, 4, 8, 16, 24), (4, 8, 16, 24), (4, 8, 16, 32))}
    out["(a) 50x2 +50x4 +50x8 +50x24"] = LookSchedule(
        am, TASKS, 24, information=PLANNING,
        plan=(((50, 2),), ((50, 2), (100, 4)), ((50, 2), (100, 4), (150, 8)), ((50, 2), (100, 4), (150, 8), (200, 24))))
    out["(a2) 50x4 +50x8 +50x16 +50x48"] = LookSchedule(
        am, TASKS, 48, information=PLANNING,
        plan=(((50, 4),), ((50, 4), (100, 8)), ((50, 4), (100, 8), (150, 16)), ((50, 4), (100, 8), (150, 16), (200, 48))))
    out["(b) 100x1, 200x2, 200x8, 200x24"] = LookSchedule(
        am, TASKS, 24, information=PLANNING, plan=(((100, 1),), ((200, 2),), ((200, 8),), ((200, 24),)))
    return out


def baselines() -> dict[str, tuple[LookSchedule, Design]]:
    """Fixed-boundary references: today's designs and a fixed test."""
    am, tm = Order.ATTEMPT_MAJOR, Order.TASK_MAJOR
    out = {}
    s = LookSchedule(tm, TASKS, 4)
    out["current: task-major 4 looks x 4 trials, c=1.77"] = (s, Design.for_schedule(s, c=1.77))
    s = LookSchedule(am, TASKS, 24)
    out["current r/K, 24 looks, c=1.77"] = (s, Design.for_schedule(s, c=1.77))
    s = LookSchedule(am, TASKS, 24, trial_looks=(2, 4, 8, 24))          # no information: fractions k/K
    out["k/K fractions at (2,4,8,24), c=1.77"] = (s, Design.for_schedule(s, c=1.77))
    s = LookSchedule(am, TASKS, 24, trial_looks=(24,))
    out["fixed 24 trials (z > 1.645)"] = (s, Design.for_schedule(s, c=1.645))
    return out


def look_specs():
    """Every distinct look of every schedule: (counts per task, weights or None)."""
    specs: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}
    keys: dict[str, list[tuple]] = {}
    scheds = {**schedules(), **{n: s for n, (s, _) in baselines().items()}}
    for name, s in scheds.items():
        keys[name] = []
        for lk in s.looks:
            counts = np.zeros(N, int)
            for t, k in lk.per_task():
                counts[int(t)] = k
            w = s.weights(lk)
            wv = None if w is None else np.array([w.get(str(i), 0.0) for i in range(N)])
            key = (tuple(counts), None if wv is None else tuple(np.round(wv, 9)))
            specs[key] = (counts, wv)
            keys[name].append(key)
    return specs, keys


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


def base_rates(sigma2: float) -> tuple[np.ndarray, float]:
    s = np.asarray(SUCCESSES, float)
    lo, hi = 1e-4, 4.0
    for _ in range(60):
        a = (lo + hi) / 2
        p = (s + a) / (TRIALS + 2 * a)
        lo, hi = (a, hi) if 2 * np.mean(p * (1 - p)) < sigma2 else (lo, a)
    a = (lo + hi) / 2
    return (s + a) / (TRIALS + 2 * a), a


def solve_effect(p: np.ndarray, delta: float, tau2: float, seed: int = 99) -> tuple[float, float, float]:
    """(s, b, achieved tau2) for mean difference delta and true-difference variance tau2."""
    rng = np.random.default_rng(seed)
    pp = rng.choice(p, 200_000)
    zz = rng.standard_normal(pp.size)
    lp = logit(pp)

    def moments(s, b):
        d = expit(lp + s + b * zz) - pp
        return d.mean(), d.var()

    def s_for(b):
        lo, hi = -12.0, 12.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if moments(mid, b)[0] < delta else (lo, mid)
        return (lo + hi) / 2

    s0 = s_for(0.0)
    v0 = moments(s0, 0.0)[1]
    if v0 >= tau2:
        return s0, 0.0, v0
    lo, hi = 0.0, 8.0
    for _ in range(40):
        b = (lo + hi) / 2
        lo, hi = (b, hi) if moments(s_for(b), b)[1] < tau2 else (lo, b)
    b = (lo + hi) / 2
    s = s_for(b)
    return s, b, moments(s, b)[1]


def simulate(p, effect, specs, sims, seed, chunk=250):
    """t and SE at every look spec for `sims` evaluations."""
    rng = np.random.default_rng(seed)
    out = {k: (np.empty(sims), np.empty(sims)) for k in specs}
    ks = np.arange(1, K + 1)
    for lo in range(0, sims, chunk):
        C = min(chunk, sims - lo)
        pa = rng.choice(p, (C, N))
        pb = pa if effect is None else expit(logit(pa) + effect[0] + effect[1] * rng.standard_normal((C, N)))
        A = (rng.random((C, N, K)) < pa[..., None]).cumsum(2) / ks
        B = (rng.random((C, N, K)) < pb[..., None]).cumsum(2) / ks
        D = B - A
        for key, (counts, w) in specs.items():
            m = counts > 0
            idx = np.maximum(counts, 1) - 1
            d = np.take_along_axis(D, np.broadcast_to(idx[None, :, None], (C, N, 1)), 2)[..., 0][:, m]
            n = d.shape[1]
            if w is None:
                md = d.mean(1)
                se = d.std(1, ddof=1) / math.sqrt(n)
            else:
                ww = w[m]
                sw = ww.sum()
                md = (d * ww).sum(1) / sw
                se = np.sqrt(n / (n - 1) * (((d - md[:, None]) * ww) ** 2).sum(1)) / sw
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(se > 1e-12, md / se, np.sign(md) * np.inf)
            out[key][0][lo:lo + C], out[key][1][lo:lo + C] = t, se
    return out


# ----------------------------------------------------------------------------- designs
def run_fixed(stats, keys, eff, fut):
    S = len(stats[keys[0]][0])
    alive = np.ones(S, bool)
    acc = np.zeros(S, bool)
    stop_j = np.full(S, len(keys) - 1)
    for j, key in enumerate(keys):
        t = stats[key][0]
        final = j == len(keys) - 1
        p = alive & (t > eff[j])
        f = alive & ~p & ((t <= fut[j]) | final)
        acc |= p
        stop_j[p | f] = j
        alive &= ~(p | f)
    return acc, stop_j


def run_spending(stats, keys, design):
    """Observed-information spending design through `bounds`, with linear futility and without futility.
    Vectorized: simulations are grouped by their (rounded) information history, and `bounds` is called once per
    group, the same call a Race makes per candidate."""
    S, J = len(stats[keys[0]][0]), len(keys)
    q, imax = design.resolution, design.max_information
    R = np.zeros((S, J))                               # ratios used (LookBounds.ratio)
    res = {}
    alive = {"fut": np.ones(S, bool), "nofut": np.ones(S, bool)}
    acc = {m: np.zeros(S, bool) for m in alive}
    stop = {m: np.full(S, J - 1) for m in alive}
    for j, key in enumerate(keys, start=1):
        t, se = stats[key]
        any_alive = alive["fut"] | alive["nofut"]
        if not any_alive.any():
            break
        u = np.maximum(q, np.round(1.0 / se ** 2 / imax / q) * q)
        rows = np.concatenate([R[:, :j - 1], u[:, None]], 1)[any_alive]
        uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        eff, fut, fin, rat = (np.empty(len(uniq)) for _ in range(4))
        for g, row in enumerate(uniq):
            lb = design.bounds(j, row[-1] * imax, tuple((i + 1, row[i]) for i in range(j - 1)))
            eff[g], fut[g], fin[g], rat[g] = lb.efficacy, lb.futility, lb.final, lb.ratio
        idx = np.flatnonzero(any_alive)
        inv = inv.reshape(-1)
        R[idx, j - 1] = rat[inv]
        e, f, fn = np.full(S, np.inf), np.full(S, -np.inf), np.zeros(S, bool)
        e[idx], f[idx], fn[idx] = eff[inv], fut[inv], fin[inv].astype(bool)
        for m in ("fut", "nofut"):
            a = alive[m]
            passed = a & (t > e)
            stopped = passed | (a & fn) | (a & (t <= f) if m == "fut" else np.zeros(S, bool))
            acc[m] |= passed
            stop[m][stopped] = j - 1
            alive[m] = a & ~stopped
    for m in alive:
        res[m] = (acc[m], stop[m])
    return res


def summarize(res, sizes):
    acc, stop_j = res
    units = np.asarray(sizes)[stop_j]
    return {"accept": float(acc.mean()), "units": float(units.mean()), "units_p90": float(np.percentile(units, 90)),
            "units_max": int(units.max()), "stop_look": [float((stop_j == j).mean()) for j in range(len(sizes))]}


def calibrate_c(null_stats, keys):
    """c of the c/sqrt(t) design (efficacy from look 1) for one-sided 0.05 on the identical null, per schedule."""
    cal = {}
    for name, s in schedules().items():
        rows = np.stack([null_stats[k][0] for k in keys[name]], 1).tolist()
        cal[name] = Simulation(s.fractions, rows, 0, 0).c_for(0.05, first_efficacy=1)
    return cal


def build_designs(cal):
    """name -> (schedule, evaluator spec)."""
    out = {}
    for name, s in schedules().items():
        sp = replace(SpendingDesign.for_schedule(s), resolution=SIM_RESOLUTION)
        out[name + " | spending, observed info"] = (s, ("spend", sp))
        pl = SpendingDesign.for_schedule(s, observed=False)
        n = len(s.looks)
        out[name + " | spending, planned info"] = (s, ("fixed", pl.planned_bounds(),
                                                       [pl.futility_bound(j) for j in range(1, n + 1)]))
        c = cal[name]
        d = Design(s.fractions, round(c, 3), None, first_efficacy=1)
        out[name + f" | c/sqrt(t), c={c:.3f}"] = (s, ("fixed", [d.efficacy(j) for j in range(1, n + 1)],
                                                     [d.futility_bound(j) for j in range(1, n + 1)]))
    for name, (s, d) in baselines().items():
        n = d.n_looks
        out[name] = (s, ("fixed", [d.efficacy(j) for j in range(1, n + 1)],
                         [d.futility_bound(j) for j in range(1, n + 1)]))
    return out


def evaluate(stats, designs, keys):
    rows = {}
    for name, (s, spec) in designs.items():
        sched_name = name.split(" | ")[0]
        ks = keys[sched_name]
        sizes = [lk.size for lk in s.looks]
        if spec[0] == "spend":
            r = run_spending(stats, ks, spec[1])
            rows[name] = summarize(r["fut"], sizes)
            rows[name.replace("observed info", "observed info, no futility")] = summarize(r["nofut"], sizes)
        else:
            rows[name] = summarize(run_fixed(stats, ks, spec[1], spec[2]), sizes)
    return rows


def _scenario(args):
    key, effect, sims, seed, cal = args
    p, _ = base_rates(PLANNING.sigma2)
    specs, keys = look_specs()
    t0 = time.time()
    stats = simulate(p, effect, specs, sims, seed)
    rows = evaluate(stats, build_designs(cal), keys)
    return key, rows, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000, help="per scenario (null: this; alternatives: half)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-mults", default="1,0.5,2", help="true tau2 = mult x planning tau2")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    p, shrink = base_rates(PLANNING.sigma2)
    print(f"# val200 base rates: mean {p.mean():.4f}, shrink a = {shrink:.3f}, sigma2 = {2 * np.mean(p * (1 - p)):.4f}; "
          f"planning tau2 {PLANNING.tau2}, sigma2 {PLANNING.sigma2}", flush=True)
    results: dict = {"planning": {"tau2": PLANNING.tau2, "sigma2": PLANNING.sigma2}, "shrink": shrink,
                     "schedules": {}, "scenarios": {}}
    for name, s in {**schedules(), **{n: s for n, (s, _) in baselines().items()}}.items():
        info = s.planned_information()
        sd = SpendingDesign.for_schedule(s)
        results["schedules"][name] = {
            "looks": [{"tasks": len(lk.tasks), "units": lk.size, "fraction": round(lk.fraction, 4)} for lk in s.looks],
            "info": None if info is None else [round(i, 1) for i in info],
            "final_se_pts": None if info is None else round(100 / math.sqrt(info[-1]), 3),
            "spending": sd.as_dict() if s.planned else None}
        print(f"# {name}: looks {[(len(lk.tasks), lk.size) for lk in s.looks]} fractions "
              f"{[round(f, 3) for f in s.fractions]}"
              + (f" info {[round(i) for i in info]} final SE {100 / math.sqrt(info[-1]):.2f} pts; spending bounds "
                 f"{sd.as_dict()['efficacy']} futility {sd.as_dict()['futility']}" if info else ""), flush=True)
    specs, keys = look_specs()
    cal = calibrate_c(simulate(p, None, specs, a.sims, a.seed), keys)
    results["c_calibrated"] = cal
    print(f"# calibrated c (c/sqrt(t), efficacy from look 1, identical null): "
          f"{ {k: round(v, 3) for k, v in cal.items()} }", flush=True)
    jobs = [("identical", None, a.sims, a.seed, cal)]
    meta = {"identical": {"delta": 0.0, "tau2": 0.0}}
    for m in (float(x) for x in a.tau_mults.split(",")):
        for di, delta in enumerate(DELTAS):
            s_, b_, achieved = solve_effect(p, delta, PLANNING.tau2 * m)
            key = f"tau2x{m:g} delta{delta:+.3f}"
            meta[key] = {"delta": delta, "tau2": PLANNING.tau2 * m, "tau2_achieved": achieved, "s": s_, "b": b_}
            jobs.append((key, (s_, b_), a.sims if delta == 0 else a.sims // 2, a.seed + 1 + 10 * di + int(100 * m),
                         cal))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for key, rows, dt in ex.map(_scenario, jobs):
            results["scenarios"][key] = {**meta[key], "rows": rows}
            print(f"# {key} done in {dt:.0f} s ({time.time() - t0:.0f} s)", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    print_tables(results)
    return 0


def print_tables(results) -> None:
    sc = results["scenarios"]
    rows = list(sc["identical"]["rows"])
    print("\n## tau2 as planned (units = trials per arm at the deciding look; 4800 = 200 x 24)\n")
    print("| design | alpha identical | alpha delta 0 | power +2.4 | +5 | +14 | E[units] -2 | 0 | +2.4 | +14 "
          "| max units | P(look 1) at +14 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        g = lambda d, f: sc[f"tau2x1 delta{d:+.3f}"]["rows"][r][f]
        mx = max(g(d, "units_max") for d in DELTAS)
        print(f"| {r} | {sc['identical']['rows'][r]['accept']:.3f} | {g(0, 'accept'):.3f} | {g(0.024, 'accept'):.3f} | "
              f"{g(0.05, 'accept'):.3f} | {g(0.14, 'accept'):.3f} | {g(-0.02, 'units'):.0f} | {g(0, 'units'):.0f} | "
              f"{g(0.024, 'units'):.0f} | {g(0.14, 'units'):.0f} | {mx} | {g(0.14, 'stop_look')[0]:.2f} |")
    for m in ("0.5", "2"):
        if f"tau2x{m} delta+0.000" not in sc:
            continue
        print(f"\n## true tau2 = {m} x planned\n")
        print("| design | alpha delta 0 | power +2.4 | +5 | +14 | E[units] -2 | 0 | +2.4 |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            g = lambda d, f: sc[f"tau2x{m} delta{d:+.3f}"]["rows"][r][f]
            print(f"| {r} | {g(0, 'accept'):.3f} | {g(0.024, 'accept'):.3f} | {g(0.05, 'accept'):.3f} | "
                  f"{g(0.14, 'accept'):.3f} | {g(-0.02, 'units'):.0f} | {g(0, 'units'):.0f} | {g(0.024, 'units'):.0f} |")


if __name__ == "__main__":
    raise SystemExit(main())
