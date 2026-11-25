"""Wall-time and cost forecast of look schedules (docs/stats.md section 7.3). Pure python + the stats JSON.

Time model (calibrated on the val200 runs, Qwen3.5-4B, 64k cap, pool node 0 = 8 GPUs, 2 rt_HC sandbox hosts x 96
slots = 192 concurrent trials):

  * every arm (candidate or reference) is one platoon process with a FIXED concurrency (`request.concurrency`):
    192 / arms slots each (a stopped arm's slots are not handed to the others: that is how the backend works today);
  * each arm launches its units in the schedule's launch order (`LookSchedule.launch_order`: look by look, within a
    look by trial index then task; task-major for the task-major baseline) as soon as a slot is free;
  * a trial's duration is drawn from the empirical durations of THAT task (all candidates, both runs, finished and
    not cancelled), so slow tasks stay slow; the task order is reshuffled per replicate;
  * look j is ready when all its units of both arms have finished; the driver sees it at its next poll (uniform 0 to
    `--poll` s later) and decides; the stop takes `--stop-latency` s (20 s measured: SIGTERM, sandboxes reaped);
  * trials an arm launched beyond its deciding look are waste (cancelled in flight, or finished but unused).

The deciding look comes from the statistical simulation (`trial_looks_sim.py --out sim.json`): stop-look
distribution per design and true delta, assumed independent of the durations.

    python packages/harnessforge/scripts/trial_looks_cost.py --sim sim.json \\
        --records runs/defects-val200-qwen3.5-4b-64k/main/records.jsonl runs/.../stack-r2/records.jsonl
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trial_looks_sim import TASKS, baselines, schedules  # noqa: E402

SLOTS = 192
GPUS = 8
HOSTS = 2


def load_durations(paths):
    pools: dict[str, list[float]] = {}
    runs = []
    for p in paths:
        recs = [json.loads(line) for line in open(p)]
        runs.append(recs)
        for r in recs:
            if r.get("finished") and r.get("started_at") and r.get("finished_at") and \
                    r.get("error") not in ("stopped", "cancelled"):
                pools.setdefault(r["task"], []).append(r["finished_at"] - r["started_at"])
    return pools, runs


def run_arm(order, durations, slots):
    """Finish time of every unit (task position, index) launched in `order` with `slots` concurrent; FIFO."""
    free = [0.0] * slots
    heapq.heapify(free)
    out = {}
    for u in order:
        start = heapq.heappop(free)
        end = start + durations[u]
        out[u] = (start, end)
        heapq.heappush(free, end)
    return out


def arm_times(sched, order, pools, task_ids, slots, rng):
    """Per look: (ready time, units launched by then) for one arm, with one draw of durations."""
    tid = (lambda t: task_ids[t]) if isinstance(task_ids, dict) else (lambda t: task_ids[int(t)])
    dur = {u: rng.choice(pools[tid(u[0])]) for u in order}
    fin = run_arm(order, dur, slots)
    looks = []
    for lk in sched.looks:
        ready = max(fin[u][1] for u in lk.units())
        looks.append(ready)
    return looks, fin


def launched_by(fin, t):
    return sum(1 for s, _ in fin.values() if s <= t)


def order_of(sched):
    from harnessforge.contracts import Order
    if sched.order is Order.TASK_MAJOR:
        return [(t, i) for t in sched.tasks for i in range(sched.trials)]
    return sched.launch_order()


def calibrate(pools, runs, rng, reps=20):
    """Simulated vs measured look times of the measured runs (task-major, 4 looks)."""
    from harnessforge.contracts import Order
    from harnessforge.stats.looks import LookSchedule
    rows = []
    for recs, (arms, conc, trials) in zip(runs, ((8, 24, 4), (3, 64, 8))):
        tasks = sorted({r["task"] for r in recs})
        sched = LookSchedule(Order.TASK_MAJOR, tuple(tasks), trials)
        sim = []
        for _ in range(reps):
            ts = []
            order = list(sched.tasks)
            rng.shuffle(order)
            s2 = LookSchedule(Order.TASK_MAJOR, tuple(order), trials)
            for _ in range(2):           # the race's two arms of interest
                lt, _ = arm_times(s2, order_of(s2), pools, {t: t for t in order}, conc, rng)
                ts.append(lt)
            sim.append([max(a, b) / 60 for a, b in zip(*ts)])
        mean = [statistics.mean(x[j] for x in sim) for j in range(4)]
        done = sum(1 for r in recs if r["finished"])
        t0 = min(r["started_at"] for r in recs if r.get("started_at"))
        t1 = max(r["finished_at"] for r in recs if r.get("finished_at"))
        rows.append({"run": f"{arms} arms x {conc} slots, {trials} trials task-major",
                     "sim_look_min": [round(m) for m in mean],
                     "measured_trials_per_h": round(done / ((t1 - t0) / 3600))})
    return rows


def forecast(sim, pools, rng, reps, poll, stop_latency):
    scheds = schedules()
    base = baselines()
    designs = {f"{n} | spending, observed info": s for n, s in scheds.items()}
    designs.update({n: s for n, (s, _) in base.items()})
    task_ids = list(pools)
    assert len(task_ids) == len(TASKS)
    sc = sim["scenarios"]
    deltas = {"-2": "tau2x1 delta-0.020", "0": "tau2x1 delta+0.000", "+2.4": "tau2x1 delta+0.024",
              "+14": "tau2x1 delta+0.140"}
    single, rounds = {}, {}
    for name, sched in designs.items():
        order = order_of(sched)
        # one candidate + reference: 2 arms x 96 slots; a round: 4 candidates + reference, 5 arms x 38 slots
        reps1 = []
        reps5 = []
        for _ in range(reps):
            ids = task_ids[:]
            rng.shuffle(ids)
            a = [arm_times(sched, order, pools, ids, SLOTS // 2, rng) for _ in range(2)]
            reps1.append(([max(x, y) for x, y in zip(a[0][0], a[1][0])], a[0][1]))
            b = [arm_times(sched, order, pools, ids, SLOTS // 5, rng) for _ in range(5)]
            reps5.append(b)
        out = {}
        for dl, key in deltas.items():
            probs = sc[key]["rows"][name]["stop_look"]
            times, waste, launched = [], [], []
            for looks, fin in reps1:
                for _ in range(20):
                    j = rng.choices(range(len(probs)), probs)[0]
                    t = looks[j] + rng.uniform(0, poll)
                    times.append(t + stop_latency)
                    n = launched_by(fin, t)
                    launched.append(n)
                    waste.append(1 - sched.looks[j].size / n if n else 0.0)
            times.sort()
            h = statistics.mean(times) / 3600
            out[dl] = {"mean_h": round(h, 2), "p90_h": round(times[int(0.9 * len(times))] / 3600, 2),
                       "gpu_h": round(GPUS * h, 1), "rthc_node_h": round(HOSTS * h, 1),
                       "launched_per_arm": round(statistics.mean(launched)), "waste": round(statistics.mean(waste), 3)}
        single[name] = out
        # rounds
        rr = {}
        for comp_name, comp in (("0,0,-2,-2", ("0", "0", "-2", "-2")), ("0,0,-2,+2.4", ("0", "0", "-2", "+2.4")),
                                ("0,0,-2,+14", ("0", "0", "-2", "+14"))):
            ends = []
            for arms in reps5:
                ref_looks = arms[0][0]
                for _ in range(20):
                    end = 0.0
                    for i, dl in enumerate(comp):
                        probs = sc[deltas[dl]]["rows"][name]["stop_look"]
                        j = rng.choices(range(len(probs)), probs)[0]
                        end = max(end, max(arms[i + 1][0][j], ref_looks[j]) + rng.uniform(0, poll))
                    ends.append(end + stop_latency)
            ends.sort()
            h = statistics.mean(ends) / 3600
            rr[comp_name] = {"mean_h": round(h, 2), "p90_h": round(ends[int(0.9 * len(ends))] / 3600, 2),
                             "rounds_per_24h": round(24 / h, 1), "gpu_h": round(GPUS * h, 1),
                             "gpu_h_per_candidate": round(GPUS * h / 4, 1), "rthc_node_h": round(HOSTS * h, 1)}
        rounds[name] = rr
        print(f"# {name}: {out['0']['mean_h']} h at delta 0", file=sys.stderr, flush=True)
    return single, rounds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="trial_looks_sim.py --out JSON")
    ap.add_argument("--records", nargs="+", required=True, help="records.jsonl of the measured runs (main, stack-r2)")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--stop-latency", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    pools, runs = load_durations(a.records)
    allw = sorted(x for v in pools.values() for x in v)
    q = lambda p: allw[int(p * (len(allw) - 1))]
    print(f"# durations: {len(allw)} trials on {len(pools)} tasks; median {q(.5):.0f} s, p90 {q(.9):.0f} s, "
          f"max {allw[-1]:.0f} s, mean {statistics.mean(allw):.0f} s")
    cal = calibrate(pools, runs, rng)
    for r in cal:
        print(f"# calibration {r}")
    sim = json.loads(Path(a.sim).read_text())
    single, rounds = forecast(sim, pools, rng, a.reps, a.poll, a.stop_latency)
    if a.out:
        Path(a.out).write_text(json.dumps({"calibration": cal, "single": single, "rounds": rounds}, indent=1))
    print("\n## one candidate + reference on one node (2 arms x 96 slots): wall time to decision, cost\n")
    print("| design | mean / p90 h at -2 | 0 | +2.4 | +14 | GPU-h at 0 | +2.4 | rt_HC node-h at 0 | trials launched/arm at 0 | waste at 0 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, o in single.items():
        f = lambda d: f"{o[d]['mean_h']:.2f} / {o[d]['p90_h']:.2f}"
        print(f"| {n} | {f('-2')} | {f('0')} | {f('+2.4')} | {f('+14')} | {o['0']['gpu_h']} | {o['+2.4']['gpu_h']} | "
              f"{o['0']['rthc_node_h']} | {o['0']['launched_per_arm']} | {o['0']['waste']:.0%} |")
    print("\n## search round: 4 candidates + reference on one node (5 arms x 38 slots)\n")
    print("| design | round h (mean / p90): 0,0,-2,-2 | 0,0,-2,+2.4 | 0,0,-2,+14 | rounds / 24 h (0,0,-2,+2.4) | GPU-h per round | per candidate |")
    print("|---|---|---|---|---|---|---|")
    for n, o in rounds.items():
        f = lambda c: f"{o[c]['mean_h']:.2f} / {o[c]['p90_h']:.2f}"
        m = o["0,0,-2,+2.4"]
        print(f"| {n} | {f('0,0,-2,-2')} | {f('0,0,-2,+2.4')} | {f('0,0,-2,+14')} | {m['rounds_per_24h']} | "
              f"{m['gpu_h']} | {m['gpu_h_per_candidate']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
