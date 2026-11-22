"""Operating characteristics of the val1000 gate (docs/stats.md section 8). Needs numpy.

Gate: 1 000 tasks x 4 trials per side, TASK_MAJOR looks after 250/500/750/1 000 tasks, Lan-DeMets O'Brien-Fleming
spending (one-sided alpha 0.05) on OBSERVED information 1 / SE^2 with I_max = 1000 / (tau2 + sigma2 / 4) at the val200
planning values, non-binding linear futility -1 + 2.5 t, efficacy from look 1. Decisions go through
`SpendingDesign.bounds`, the code a `Race` uses (observed fractions rounded to --resolution).

Model (per replicate):
  * 1 000 true reference rates p_i drawn from the NPMLE (Kiefer-Wolfowitz, grid of 199 rates) of the val200 seed-harness
    results (`stats._val200`, successes out of 8): it reproduces val200's mix (35/200 always solved at 8 trials, 47
    never, 118 mixed; mean 0.50) and its within-task variance (2 E p(1 - p) = sigma2 ~ 0.249);
  * the candidate's rates q_i = logistic(logit p_i + s + b z_i), z ~ N(0, 1), (s, b) solved for the mean true
    difference delta and its variance across tasks tau2 (effect heterogeneity; when the shift alone gives more than
    tau2, b = 0 and the achieved tau2 is reported);
  * "null": q = p; "mixed null": +s_mid on logit for tasks with true p in [.25, .75) (about +6 points there), -s_easy for
    p >= .75, s_easy solved PER REPLICATE so that the 1 000-task mean of q - p is exactly 0; "mixed harmful": the same
    with a full-set mean of -1 point;
  * 4 Bernoulli trials per task and side (the reference on the same units); d_i = mean B - mean A;
  * difficulty labels from an INDEPENDENT pass of the reference: 1 trial or 4 trials per task (Binomial), or (a
    diagnostic) the reference's own comparison trials.

Designs: uniform order (look j = a uniform random prefix: the plain paired statistic) and frontier-weighted orders
(`stats.sampling`: bands by label [0,.25) [.25,.5) [.5,.75) [.75,1], priority 0.25-0.5 > 0.5-0.75 > 0-0.25 > 0.75-1,
multipliers (4, 3, 2, 1) ** tilt, floor 0.5 of the proportional share) with the HT estimate and stratified SE.
Allocation uses `sampling.allocate`; within-band membership is a random permutation (exactly the library's design).

Winner's curse (--curse): candidate B1 at +2.4 vs A; when B1 is accepted at look k it becomes the reference, its
gate trials (the tasks of looks <= k, 4 trials each) are reused and the other tasks topped up with fresh trials; a
second candidate B2 at +2.4 over B1 is then gated against that reused reference (frontier orders: labels from A's
trials, label chaining), vs against a fresh full evaluation of B1.

    uv run --no-project -p 3.12 --with numpy --with-editable packages/harnessforge \\
        python packages/harnessforge/scripts/val1000_sim.py --sims 10000 --out val1000_sim.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from harnessforge.contracts import Order
from harnessforge.stats._val200 import SUCCESSES, TRIALS
from harnessforge.stats.boundaries import Design
from harnessforge.stats.information import VAL200
from harnessforge.stats.looks import LookSchedule
from harnessforge.stats.sampling import allocate, multipliers, DEFAULT_PRIORITY
from harnessforge.stats.spending import SpendingDesign, crossing_probabilities

N, K = 1000, 4
SIZES = (250, 500, 750, 1000)
J = len(SIZES)
EDGES = np.array([0.25, 0.5, 0.75])
NB = 4
TASKS = tuple(f"t{i:04d}" for i in range(N))
PLANNING = VAL200

# design name -> (label source or None, tilt)
DESIGNS = {
    "uniform": (None, 0.0),
    "frontier L4 tilt 0.5": ("L4", 0.5),
    "frontier L4 tilt 1": ("L4", 1.0),
    "frontier L4 tilt 2": ("L4", 2.0),
    "frontier L1 tilt 1": ("L1", 1.0),
    "frontier L1 tilt 2": ("L1", 2.0),
    "frontier own-ref labels tilt 1 (diagnostic)": ("own", 1.0),
}
FLOOR = 0.5


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


# ----------------------------------------------------------------------------------------------- population
def npmle(succ=SUCCESSES, n=TRIALS, iters=5000):
    grid = np.concatenate([[0.002], np.linspace(0.005, 0.995, 197), [0.998]])
    s = np.asarray(succ)
    from math import comb
    lik = np.array([[comb(n, k) * g ** k * (1 - g) ** (n - k) for g in grid] for k in s])   # tasks x grid
    w = np.full(grid.size, 1.0 / grid.size)
    for _ in range(iters):
        post = lik * w
        post /= post.sum(1, keepdims=True)
        w = post.mean(0)
    return grid, w


GRID, WEIGHTS = npmle()


def draw_p(rng, shape):
    return rng.choice(GRID, size=shape, p=WEIGHTS)


def solve_effect(delta, tau2, seed=99):
    rng = np.random.default_rng(seed)
    pp = draw_p(rng, 400_000)
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


MID_GAIN = 0.06


def s_mid_for(gain=MID_GAIN, seed=98):
    rng = np.random.default_rng(seed)
    pp = draw_p(rng, 400_000)
    pp = pp[(pp >= 0.25) & (pp < 0.75)]
    lo, hi = 0.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (expit(logit(pp) + mid) - pp).mean() < gain else (lo, mid)
    return (lo + hi) / 2


def mixed_q(p, s_mid, target):
    """Mid band +s_mid, easy band -s_easy with s_easy per row so that mean(q - p) = target exactly."""
    mid = (p >= 0.25) & (p < 0.75)
    easy = p >= 0.75
    lp = logit(p)
    q = np.where(mid, expit(lp + s_mid), p)
    gain = (q - p).sum(1)
    lo, hi = np.zeros(p.shape[0]), np.full(p.shape[0], 20.0)
    for _ in range(60):
        m = (lo + hi) / 2
        loss = np.where(easy, p - expit(lp - m[:, None]), 0.0).sum(1)
        over = gain - loss > target * p.shape[1]
        lo, hi = np.where(over, m, lo), np.where(over, hi, m)
    m = (lo + hi) / 2
    return np.where(easy, expit(lp - m[:, None]), q)


# ----------------------------------------------------------------------------------------------- frontier masks
_ALLOC: dict = {}


def alloc_counts(bsizes, tilt):
    """(J, NB) cumulative per-band counts for each row's band sizes (sampling.allocate, cached)."""
    m = multipliers(DEFAULT_PRIORITY, tilt)
    out = np.empty((bsizes.shape[0], J, NB), int)
    for r, row in enumerate(map(tuple, bsizes)):
        key = (row, tilt)
        a = _ALLOC.get(key)
        if a is None:
            bs = {b: c for b, c in enumerate(row) if c > 0}
            al = allocate(bs, SIZES, {b: m[b] for b in bs}, FLOOR)
            a = np.array([[al[j].get(b, 0) for b in range(NB)] for j in range(J)])
            _ALLOC[key] = a
        out[r] = a
    return out


def frontier_masks(band, keys, tilt):
    """Inclusion masks (J, R, N) and weights (J, R, N) of the stratified look order."""
    R = band.shape[0]
    bsizes = np.stack([(band == b).sum(1) for b in range(NB)], 1)            # R x NB
    counts = alloc_counts(bsizes, tilt)                                       # R x J x NB
    order = np.argsort(band + keys, axis=1, kind="stable")
    inv = np.empty_like(order)
    np.put_along_axis(inv, order, np.arange(N)[None, :].repeat(R, 0), 1)
    start = np.concatenate([np.zeros((R, 1), int), np.cumsum(bsizes, 1)[:, :-1]], 1)
    rank = inv - np.take_along_axis(start, band, 1)
    masks, weights = [], []
    for j in range(J):
        nb = np.take_along_axis(counts[:, j, :], band, 1)                    # R x N: n_{b(i), j}
        Nb = np.take_along_axis(bsizes, band, 1)
        masks.append(rank < nb)
        weights.append(np.where(nb > 0, Nb / np.maximum(nb, 1), 0.0))
    return np.array(masks), np.array(weights)


def look_stats_frontier(d, band, mask, w):
    """HT mean and stratified SE (plain when the look's weights are all equal)."""
    R = d.shape[0]
    wm = np.where(mask, w, 0.0)
    sw = wm.sum(1)
    D = (wm * d).sum(1) / sw
    u = wm * (d - D[:, None])
    v = np.zeros(R)
    for b in range(NB):
        mb = mask & (band == b)
        n = mb.sum(1)
        ub = np.where(mb, u, 0.0)
        mu = ub.sum(1) / np.maximum(n, 1)
        ss = (np.where(mb, (u - mu[:, None]) ** 2, 0.0)).sum(1)
        v += np.where(n > 1, n / np.maximum(n - 1, 1) * ss, np.where(n == 1, ub.sum(1) ** 2, 0.0))
    se = np.sqrt(v) / sw
    equal = (np.where(mask, w, -np.inf).max(1) - np.where(mask, w, np.inf).min(1)) < 1e-12
    if equal.any():
        n = mask.sum(1)
        md = np.where(mask, d, 0.0).sum(1) / n
        sd = np.sqrt(np.where(mask, (d - md[:, None]) ** 2, 0.0).sum(1) / (n - 1))
        D = np.where(equal, md, D)
        se = np.where(equal, sd / np.sqrt(n), se)
    return D, se


def look_stats_uniform(d):
    out = []
    for n in SIZES:
        x = d[:, :n]
        out.append((x.mean(1), x.std(1, ddof=1) / math.sqrt(n)))
    return out


def bands_of(rates):
    return np.searchsorted(EDGES, rates, side="right")


def design_masks(labels, keys, own_band):
    out = {}
    for name, (src, tilt) in DESIGNS.items():
        if src is None:
            continue
        band = own_band if src == "own" else labels[src]
        out[name] = (band, *frontier_masks(band, keys, tilt))
    return out


def all_designs(d, dm):
    """{design: (D (R, J), se (R, J))}."""
    out = {}
    u = look_stats_uniform(d)
    out["uniform"] = (np.stack([x[0] for x in u], 1), np.stack([x[1] for x in u], 1))
    for name, (band, masks, ws) in dm.items():
        Ds, ses = [], []
        for j in range(J):
            D, se = look_stats_frontier(d, band, masks[j], ws[j])
            Ds.append(D)
            ses.append(se)
        out[name] = (np.stack(Ds, 1), np.stack(ses, 1))
    return out


# ----------------------------------------------------------------------------------------------- scenarios
def scenario_effects(tau2s):
    eff = {"null": ("null",)}
    for tau2 in tau2s:
        for delta in (0.016, 0.024, 0.05, 0.14, -0.02):
            s, b, ach = solve_effect(delta, tau2)
            eff[f"{delta * 100:+.1f} tau2={tau2}"] = ("shift", s, b, ach, delta, tau2)
    sm = s_mid_for()
    eff["mixed null (mid +6, easy -, mean 0)"] = ("mixed", sm, 0.0)
    eff["mixed harmful (mid +6, easy -, mean -1)"] = ("mixed", sm, -0.01)
    return eff


def worker(args):
    """One chunk of replicates: per scenario and design, (D, se) at every look."""
    seed, R, effects = args
    rng = np.random.default_rng(seed)
    p = draw_p(rng, (R, N))
    A = rng.binomial(K, p) / K
    L1 = rng.binomial(1, p).astype(float)
    L4 = rng.binomial(4, p) / 4
    labels = {"L1": bands_of(L1), "L4": bands_of(L4)}
    own_band = bands_of(A)
    keys = rng.random((R, N))
    dm = design_masks(labels, keys, own_band)
    out = {}
    for name, e in effects.items():
        r2 = np.random.default_rng([seed, zlib.crc32(name.encode())])
        if e[0] == "null":
            q = p
        elif e[0] == "shift":
            q = expit(logit(p) + e[1] + e[2] * r2.standard_normal((R, N)))
        else:
            q = mixed_q(p, e[1], e[2])
        B = r2.binomial(K, q) / K
        out[name] = all_designs(B - A, dm)
    return out


# ----------------------------------------------------------------------------------------------- decisions
def make_design(resolution):
    sch = LookSchedule(Order.TASK_MAJOR, TASKS, K, information=PLANNING)
    return replace(SpendingDesign.for_schedule(sch), resolution=resolution)


def decide(D, se, design, futility=True):
    """Sequential decisions for rows of (D, se) (S, J). Returns accept, stop look (0-based), t at stop, efficacy at
    stop. Rows grouped by their rounded information history; one `bounds` call per group."""
    S = D.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(se > 1e-12, D / se, np.sign(D) * np.inf)
    q, imax = design.resolution, design.max_information
    Rt = np.zeros((S, J))
    alive = np.ones(S, bool)
    acc = np.zeros(S, bool)
    stop = np.full(S, J - 1)
    t_at = np.full(S, np.nan)
    e_at = np.full(S, np.nan)
    for j in range(1, J + 1):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        u = np.maximum(q, np.round(1.0 / np.maximum(se[idx, j - 1], 1e-9) ** 2 / imax / q) * q)
        rows = np.concatenate([Rt[idx, :j - 1], u[:, None]], 1)
        uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        eff, fut, fin, rat = (np.empty(len(uniq)) for _ in range(4))
        for g, row in enumerate(uniq):
            lb = design.bounds(j, row[-1] * imax, tuple((i + 1, row[i]) for i in range(j - 1)))
            eff[g], fut[g], fin[g], rat[g] = lb.efficacy, lb.futility, lb.final, lb.ratio
        Rt[idx, j - 1] = rat[inv]
        t = T[idx, j - 1]
        e, f, fn = eff[inv], fut[inv], fin[inv].astype(bool)
        passed = t > e
        stopped = passed | fn | ((t <= f) if futility else np.zeros(idx.size, bool))
        acc[idx[passed]] = True
        stop[idx[stopped]] = j - 1
        t_at[idx[stopped]] = t[stopped]
        e_at[idx[stopped]] = e[stopped]
        alive[idx[stopped]] = False
    return acc, stop, t_at, e_at


def summarize(acc, stop):
    units = K * np.asarray(SIZES)[stop]
    return {"accept": float(acc.mean()), "units": float(units.mean()), "looks": float((stop + 1).mean()),
            "stop_look": [float((stop == j).mean()) for j in range(J)],
            "accept_look": [float((acc & (stop == j)).mean()) for j in range(J)]}


def sanity(design):
    """Boundaries and exact normal-theory alphas of the fixed-boundary references."""
    fr = (0.25, 0.5, 0.75, 1.0)
    ld = SpendingDesign(fr).planned_bounds()
    out = {"ld_obf_4_equal_looks": [round(b, 4) for b in ld],
           "ld_obf_alpha": round(sum(crossing_probabilities(fr, ld)), 5)}
    for c, first in ((1.77, 2), (1.77, 1)):
        b = [math.inf if j + 1 < first else c / math.sqrt(f) for j, f in enumerate(fr)]
        out[f"c={c} first_efficacy={first}"] = {"bounds": [round(x, 4) for x in b],
                                                "alpha": round(sum(crossing_probabilities(fr, b)), 5)}
    for first in (1, 2):
        lo, hi = 1.0, 3.0
        for _ in range(60):
            c = (lo + hi) / 2
            b = [math.inf if j + 1 < first else c / math.sqrt(f) for j, f in enumerate(fr)]
            lo, hi = (c, hi) if sum(crossing_probabilities(fr, b)) > 0.05 else (lo, c)
        out[f"exact OBF c for alpha 0.05, first_efficacy={first}"] = round((lo + hi) / 2, 4)
    out["design"] = design.as_dict()
    return out


# ----------------------------------------------------------------------------------------------- winner's curse
def curse_chunk(args):
    """B1 (+delta over A) gated against A; on accept at look k: reuse B1's look-<=k trials + top-up, vs fresh B1;
    B2 (+delta over B1) gated against each. Returns per-row arrays."""
    seed, R, (s1, b1), (s2, b2), designs = args
    rng = np.random.default_rng(seed)
    p = draw_p(rng, (R, N))
    q1 = expit(logit(p) + s1 + b1 * rng.standard_normal((R, N)))
    q2 = expit(logit(q1) + s2 + b2 * rng.standard_normal((R, N)))
    A = rng.binomial(K, p) / K
    B1 = rng.binomial(K, q1) / K
    B1_top = rng.binomial(K, q1) / K          # top-up trials of B1 (used where B1 had no gate trials)
    B1_fresh = rng.binomial(K, q1) / K        # a fresh full evaluation of B1
    B2 = rng.binomial(K, q2) / K
    L4 = bands_of(rng.binomial(4, p) / 4)     # step-0 labels (independent pass)
    labA = bands_of(A)                        # label chaining for gate 2: the previous reference A
    keys1, keys2 = rng.random((R, N)), rng.random((R, N))
    out = {"q1": q1.mean(1), "d12": (q2 - q1).mean(1)}
    for name in designs:
        src, tilt = DESIGNS[name]
        # gate 1
        if src is None:
            u = look_stats_uniform(B1 - A)
            D1 = np.stack([x[0] for x in u], 1)
            se1 = np.stack([x[1] for x in u], 1)
            masks1 = np.array([np.broadcast_to(np.arange(N) < n, (R, N)) for n in SIZES])
        else:
            masks1, ws1 = frontier_masks(L4, keys1, tilt)
            st = [look_stats_frontier(B1 - A, L4, masks1[j], ws1[j]) for j in range(J)]
            D1, se1 = np.stack([x[0] for x in st], 1), np.stack([x[1] for x in st], 1)
        out[name] = {"D1": D1, "se1": se1, "masks1": masks1, "B2": B2}
        # gate 2 inputs for every possible accept look k (reuse) and fresh
        refs = {f"reuse@{k + 1}": np.where(masks1[k], B1, B1_top) for k in range(J)}
        refs["fresh"] = B1_fresh
        res = {}
        if src is not None:
            m2, w2 = frontier_masks(labA, keys2, tilt)
        for rname, ref in refs.items():
            d2 = B2 - ref
            if src is None:
                u = look_stats_uniform(d2)
                res[rname] = (np.stack([x[0] for x in u], 1), np.stack([x[1] for x in u], 1),
                              (ref - q1).mean(1))
            else:
                st = [look_stats_frontier(d2, labA, m2[j], w2[j]) for j in range(J)]
                res[rname] = (np.stack([x[0] for x in st], 1), np.stack([x[1] for x in st], 1), (ref - q1).mean(1))
        out[name]["gate2"] = res
        out[name].pop("masks1")
        out[name].pop("B2")
    return out


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resolution", type=float, default=2e-3)
    ap.add_argument("--tau2", default="0.0025,0.0081")
    ap.add_argument("--curse-sims", type=int, default=20000)
    ap.add_argument("--curse-designs", default="uniform,frontier L4 tilt 1")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    t0 = time.time()
    design = make_design(a.resolution)
    res = {"sanity": sanity(design)}
    print("# sanity", json.dumps(res["sanity"]), flush=True)

    # population check
    rng = np.random.default_rng(1)
    pp = draw_p(rng, 200_000)
    s8 = rng.binomial(8, pp)
    res["population"] = {"mean_p": float(pp.mean()), "sigma2": float(2 * np.mean(pp * (1 - pp))),
                         "per200_always": float((s8 == 8).mean() * 200), "per200_never": float((s8 == 0).mean() * 200),
                         "band_mix_true_p": [float(((pp >= lo) & (pp < hi)).mean()) for lo, hi in
                                             ((0, .25), (.25, .5), (.5, .75), (.75, 1.01))]}
    print("# population", json.dumps(res["population"]), flush=True)

    tau2s = [float(x) for x in a.tau2.split(",")]
    effects = scenario_effects(tau2s)
    res["effects"] = {k: [float(x) if isinstance(x, (int, float)) else x for x in v] for k, v in effects.items()}
    print("# effects", json.dumps(res["effects"]), flush=True)
    jobs = [(a.seed * 100000 + i, min(a.chunk, a.sims - lo), effects) for i, lo in enumerate(range(0, a.sims, a.chunk))]
    stats: dict = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(worker, jobs):
            for sc, dd in part.items():
                for dn, (D, se) in dd.items():
                    stats.setdefault(sc, {}).setdefault(dn, []).append((D, se))
    print(f"# simulated in {time.time() - t0:.0f} s", flush=True)
    res["rows"] = {}
    for sc, dd in stats.items():
        for dn, parts in dd.items():
            D = np.concatenate([x[0] for x in parts])
            se = np.concatenate([x[1] for x in parts])
            acc, stop, _, _ = decide(D, se, design)
            row = summarize(acc, stop)
            acc_nf, stop_nf, _, _ = decide(D, se, design, futility=False)
            row["accept_nofut"] = float(acc_nf.mean())
            row["mean_D_look1"] = float(D[:, 0].mean())
            row["sd_D_look1"] = float(D[:, 0].std())
            row["mean_se_look1"] = float(se[:, 0].mean())
            row["mean_info_frac"] = [float(np.mean(1 / se[:, j] ** 2 / design.max_information)) for j in range(J)]
            res["rows"].setdefault(sc, {})[dn] = row
        print(f"# decided {sc} ({time.time() - t0:.0f} s)", flush=True)

    # fixed-boundary reference rows on the uniform order
    res["fixed"] = {}
    sch = LookSchedule(Order.TASK_MAJOR, TASKS, K)
    classic = Design.for_schedule(sch, c=1.77)
    for sc, dd in stats.items():
        D = np.concatenate([x[0] for x in dd["uniform"]])
        se = np.concatenate([x[1] for x in dd["uniform"]])
        T = D / se
        alive = np.ones(len(T), bool)
        acc = np.zeros(len(T), bool)
        stop = np.full(len(T), J - 1)
        for j in range(1, J + 1):
            e, f = classic.efficacy(j), classic.futility_bound(j)
            p_ = alive & (T[:, j - 1] > e)
            s_ = p_ | (alive & (T[:, j - 1] <= f)) | (alive & (j == J))
            acc |= p_
            stop[s_ & alive] = j - 1
            alive &= ~s_
        res["fixed"].setdefault(sc, {})["uniform, c/sqrt(t) c=1.77 (efficacy from look 2)"] = summarize(acc, stop)

    if a.curse_sims:
        s1, b1, _ = solve_effect(0.024, PLANNING.tau2)
        # B2's +2.4 is relative to B1's rates: solve on a q1-like population (approximately the same shift)
        designs = a.curse_designs.split(",")
        cj = [(a.seed * 100000 + 50000 + i, min(a.chunk, a.curse_sims - lo), (s1, b1), (s1, b1), designs)
              for i, lo in enumerate(range(0, a.curse_sims, a.chunk))]
        parts = []
        with ProcessPoolExecutor(a.workers) as ex:
            parts = list(ex.map(curse_chunk, cj))
        res["curse"] = curse_report(parts, designs, design)
        print("# curse", json.dumps(res["curse"]), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print_tables(res)
    print(f"# total {time.time() - t0:.0f} s")
    return 0


def curse_report(parts, designs, design):
    out = {}
    for name in designs:
        D1 = np.concatenate([p[name]["D1"] for p in parts])
        se1 = np.concatenate([p[name]["se1"] for p in parts])
        acc1, stop1, t1, e1 = decide(D1, se1, design)
        marginal = acc1 & ((stop1 == J - 1) | (t1 - e1 < 0.5))
        g2 = {}
        for rname in parts[0][name]["gate2"]:
            D2 = np.concatenate([p[name]["gate2"][rname][0] for p in parts])
            se2 = np.concatenate([p[name]["gate2"][rname][1] for p in parts])
            bias = np.concatenate([p[name]["gate2"][rname][2] for p in parts])
            acc2, stop2, _, _ = decide(D2, se2, design)
            g2[rname] = (acc2, bias)
        rows = {"power_gate1": float(acc1.mean()), "P(marginal | accept)": float(marginal[acc1].mean()),
                "true_B2_minus_B1_pts": float(np.concatenate([p["d12"] for p in parts]).mean() * 100)}
        for k in range(J):
            sel = acc1 & (stop1 == k)
            if sel.sum() < 20:
                rows[f"accept@{k + 1}"] = {"P": float(sel.mean()), "n": int(sel.sum())}
                continue
            acc_r, bias_r = g2[f"reuse@{k + 1}"]
            acc_f, bias_f = g2["fresh"]
            rows[f"accept@{k + 1}"] = {
                "P": float(sel.mean()), "n": int(sel.sum()),
                "D1_hat_pts": float(D1[sel, k].mean() * 100),
                "ref_bias_pts_reuse": float(bias_r[sel].mean() * 100),
                "power2_reuse": float(acc_r[sel].mean()), "power2_fresh": float(acc_f[sel].mean())}
        # policies over all accepts
        pol = {}
        acc_f = g2["fresh"][0]
        reuse_acc = np.zeros_like(acc1)
        reuse_bias = np.zeros(len(acc1))
        for k in range(J):
            m = acc1 & (stop1 == k)
            reuse_acc[m] = g2[f"reuse@{k + 1}"][0][m]
            reuse_bias[m] = g2[f"reuse@{k + 1}"][1][m]
        pol["never fresh (top-up)"] = float(reuse_acc[acc1].mean())
        pol["fresh after marginal"] = float(np.where(marginal, acc_f, reuse_acc)[acc1].mean())
        pol["always fresh"] = float(acc_f[acc1].mean())
        rows["power2_by_policy"] = pol
        rows["ref_bias_pts_reuse_all_accepts"] = float(reuse_bias[acc1].mean() * 100)
        rows["ref_bias_pts_reuse_marginal"] = float(reuse_bias[marginal].mean() * 100) if marginal.any() else None
        rows["ref_bias_pts_reuse_nonmarginal"] = (float(reuse_bias[acc1 & ~marginal].mean() * 100)
                                                  if (acc1 & ~marginal).any() else None)
        out[name] = rows
    return out


def print_tables(res):
    rows = res["rows"]
    designs = list(next(iter(rows.values())))
    t2 = sorted({k.split("tau2=")[1] for k in rows if "tau2=" in k})
    for tau2 in t2:
        print(f"\n## tau2 = {tau2} (units = trials per arm at the deciding look, max 4000)\n")
        print("| design | alpha null | alpha mixed null | accept mixed harmful | power +1.6 | +2.4 | +5 | +14 | "
              "E[units] null | -2 | +2.4 | +14 | E[looks] +2.4 | P(look 1) +14 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for dn in designs:
            g = lambda sc: rows[sc][dn]
            k = lambda d: f"{d:+.1f} tau2={tau2}"
            print(f"| {dn} | {g('null')['accept']:.3f} | {g('mixed null (mid +6, easy -, mean 0)')['accept']:.3f} | "
                  f"{g('mixed harmful (mid +6, easy -, mean -1)')['accept']:.3f} | {g(k(1.6))['accept']:.3f} | "
                  f"{g(k(2.4))['accept']:.3f} | {g(k(5.0))['accept']:.3f} | {g(k(14.0))['accept']:.3f} | "
                  f"{g('null')['units']:.0f} | {g(k(-2.0))['units']:.0f} | {g(k(2.4))['units']:.0f} | "
                  f"{g(k(14.0))['units']:.0f} | {g(k(2.4))['looks']:.2f} | {g(k(14.0))['stop_look'][0]:.2f} |")
        for fn, fr in res["fixed"]["null"].items():
            g = lambda sc: res["fixed"][sc][fn]
            k = lambda d: f"{d:+.1f} tau2={tau2}"
            print(f"| {fn} | {g('null')['accept']:.3f} | {g('mixed null (mid +6, easy -, mean 0)')['accept']:.3f} | "
                  f"{g('mixed harmful (mid +6, easy -, mean -1)')['accept']:.3f} | {g(k(1.6))['accept']:.3f} | "
                  f"{g(k(2.4))['accept']:.3f} | {g(k(5.0))['accept']:.3f} | {g(k(14.0))['accept']:.3f} | "
                  f"{g('null')['units']:.0f} | {g(k(-2.0))['units']:.0f} | {g(k(2.4))['units']:.0f} | "
                  f"{g(k(14.0))['units']:.0f} | {g(k(2.4))['looks']:.2f} | {g(k(14.0))['stop_look'][0]:.2f} |")
    print("\n## look-1 estimate (mean D, sd D, mean SE; points) and mean observed information fraction per look\n")
    for sc in ("null", "mixed null (mid +6, easy -, mean 0)"):
        for dn in designs:
            r = rows[sc][dn]
            print(f"{sc} | {dn}: D {r['mean_D_look1'] * 100:+.3f} sd {r['sd_D_look1'] * 100:.3f} "
                  f"se {r['mean_se_look1'] * 100:.3f}; info fraction {[round(x, 3) for x in r['mean_info_frac']]}")


if __name__ == "__main__":
    raise SystemExit(main())
