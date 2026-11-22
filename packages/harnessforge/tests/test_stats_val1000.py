"""val1000 gate: frontier-weighted task-major looks (HT estimator), reference top-up and lineage."""
import json
import math
import random

import pytest

from harnessforge import Order, TrialRecord
from harnessforge.records import TaskStats, write_jsonl
from harnessforge.stats import (VAL200, Lineage, LookSchedule, Race, ReferenceMode, SpendingDesign, TaskSampling,
                                Verdict, apply_topup, frontier_order, is_marginal, labels_from_records, paired,
                                topup_plan, val1000_schedule)
from harnessforge.stats.sampling import band_of


def population(n=400, seed=0):
    """Fixed tasks with labels in 4 bands and fixed per-task differences that depend strongly on the band."""
    rng = random.Random(seed)
    tasks = [f"t{i:04d}" for i in range(n)]
    labels = {t: rng.choice([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]) for t in tasks}
    effect = {0: 0.0, 1: 0.20, 2: 0.10, 3: -0.08}
    d = {t: effect[band_of(labels[t])] + rng.uniform(-0.05, 0.05) for t in tasks}
    return tasks, labels, d


def stats_of(d, tasks):
    a = {t: TaskStats(0.5, 4) for t in tasks}
    b = {t: TaskStats(0.5 + d[t], 4) for t in tasks}
    return a, b


# ------------------------------------------------------------------------------------------------ sampling
def test_frontier_order_allocation_floor_monotone_deterministic():
    tasks, labels, _ = population()
    s = frontier_order(tasks, labels, tilt=1.0, floor=0.5, seed=7)
    assert sorted(s.tasks) == sorted(tasks) and s.sizes == (100, 200, 300, 400)
    nb = s.band_sizes
    prev = {b: 0 for b in nb}
    for j in range(1, 5):
        c = s.counts(j)
        assert sum(c.values()) == s.sizes[j - 1]
        for b in nb:                                     # every band present from look 1, at least the floor
            assert c[b] >= prev[b] and c[b] >= math.floor(0.5 * s.sizes[j - 1] / 400 * nb[b])
        prev = c
    assert s.counts(4) == nb
    c1 = s.counts(1)                                      # the frontier band is over-sampled, 0.75-1 under-sampled
    assert c1[1] / nb[1] > 0.25 > c1[3] / nb[3]
    assert frontier_order(list(reversed(tasks)), labels, tilt=1.0, seed=7) == s      # deterministic, input order free
    assert frontier_order(tasks, labels, tilt=1.0, seed=8).tasks != s.tasks
    assert TaskSampling.from_dict(json.loads(json.dumps(s.as_dict()))) == s


def test_ht_unbiased_on_fixed_population():
    """Over the design's randomisation, the look-1 HT estimate averages to the full-population mean, while the plain
    mean of the over-sampled look is biased toward the frontier bands."""
    tasks, labels, d = population()
    target = sum(d.values()) / len(d)
    ht, plain = [], []
    for seed in range(1500):
        s = frontier_order(tasks, labels, tilt=1.5, floor=0.3, seed=seed)
        look = s.tasks[:s.sizes[0]]
        a, b = stats_of(d, look)
        ht.append(paired(a, b, weights=s.weights(1), strata=s.strata(1)).d)
        plain.append(paired(a, b).d)
    m = sum(ht) / len(ht)
    sd = math.sqrt(sum((x - m) ** 2 for x in ht) / (len(ht) - 1))
    assert abs(m - target) < 4 * sd / math.sqrt(len(ht))
    assert sum(plain) / len(plain) - target > 0.02
    # Sum of weights is N exactly (stratified SRS): Hajek = HT
    s = frontier_order(tasks, labels, tilt=1.5, floor=0.3, seed=0)
    assert sum(s.weights(1).values()) == pytest.approx(len(tasks))


def test_stratified_se_matches_formula_and_reduces_to_plain():
    tasks, labels, d = population()
    s = frontier_order(tasks, labels, tilt=1.0, seed=1)
    look = s.tasks[:s.sizes[1]]
    a, b = stats_of(d, look)
    p = paired(a, b, weights=s.weights(2), strata=s.strata(2))
    n_all, by = len(tasks), {}
    for t in look:
        by.setdefault(s.strata(2)[t], []).append(d[t])
    var = 0.0
    for band, xs in by.items():
        m = sum(xs) / len(xs)
        var += (s.band_sizes[band] / n_all) ** 2 * (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / len(xs)
    assert p.se == pytest.approx(math.sqrt(var), rel=1e-9)
    # uniform weights / one stratum / tilt 0 / the last look: the existing paired statistic
    plain = paired(a, b)
    one = paired(a, b, weights={t: 3.0 for t in look}, strata={t: "x" for t in look})
    assert one.d == pytest.approx(plain.d) and one.se == pytest.approx(plain.se)
    assert TaskSampling.uniform(tasks, (100, 200, 300, 400)).weights(1) is None
    w0 = frontier_order(tasks, labels, tilt=0.0, seed=1).weights(1)      # proportional: equal up to rounding
    assert w0 is None or max(w0.values()) / min(w0.values()) < 1.1
    assert s.weights(4) is None
    sch = LookSchedule(Order.TASK_MAJOR, s.tasks, 4, information=VAL200, sampling=s)
    assert sch.weights(sch.look(4)) is None and sch.strata(sch.look(4)) is None
    assert sch.weights(sch.look(1)) is not None


def test_val1000_schedule_and_spending_bounds():
    tasks = [f"t{i:04d}" for i in range(1000)]
    sch = val1000_schedule(tasks)
    assert [len(lk.tasks) for lk in sch.looks] == [250, 500, 750, 1000]
    d = SpendingDesign.for_schedule(sch)
    assert d.observed
    assert [round(b, 2) for b in d.planned_bounds()] == [3.75, 2.54, 2.02, 1.72]
    assert d.max_information == pytest.approx(1000 / VAL200.variance(4))
    with pytest.raises(ValueError, match="need Order.ATTEMPT_MAJOR"):
        LookSchedule(Order.TASK_MAJOR, tasks, 4, trial_looks=(2, 4))
    labels = {t: (i % 5) / 4 for i, t in enumerate(tasks)}
    s = frontier_order(tasks, labels, tilt=1.0, seed=0)
    sch2 = val1000_schedule(sampling=s)
    fr = sch2.fractions
    assert fr[-1] == 1.0 and fr[0] < 0.25           # unequal weights: less planned information per task
    with pytest.raises(ValueError, match="sampling.tasks"):
        LookSchedule(Order.TASK_MAJOR, tasks, 4, sampling=s)


def test_race_on_frontier_schedule_passes_clear_winner():
    rng = random.Random(3)
    tasks = [f"t{i:03d}" for i in range(400)]
    p = {t: rng.random() for t in tasks}
    labels = {t: sum(rng.random() < p[t] for _ in range(4)) / 4 for t in tasks}     # independent label pass
    s = frontier_order(tasks, labels, tilt=1.0, seed=0)
    sch = val1000_schedule(sampling=s)
    recs = []
    for t in s.tasks[:100]:
        for i in range(4):
            recs.append(TrialRecord("A", t, i, True, float(rng.random() < p[t])))
            recs.append(TrialRecord("B", t, i, True, float(rng.random() < min(1.0, p[t] + 0.4))))
    race = Race("A", ["B"], sch, reference_mode=ReferenceMode.ALL_TRIALS)
    assert isinstance(race.design, SpendingDesign)
    st = race.update(recs).states["B"]
    assert st.verdict is Verdict.PASS and st.look == 1 and st.acc.n == 100


# ------------------------------------------------------------------------------------------------ top-up / lineage
def test_topup_plan_and_apply():
    tasks = ["a", "b", "c", "d"]
    recs = [TrialRecord("B", "a", i, True, 1.0) for i in range(4)]
    recs += [TrialRecord("B", "b", 0, True, 0.0), TrialRecord("B", "b", 1, True, env_failed=True, error="X"),
             TrialRecord("B", "b", 2, True, env_failed=True, error="stopped"), TrialRecord("B", "b", 3, False)]
    recs += [TrialRecord("A", "c", i, True, 1.0) for i in range(4)]          # other candidate: ignored
    plan = topup_plan(recs, "B", tasks, 4)
    assert plan.missing == {"b": (1, 2, 3), "c": (0, 1, 2, 3), "d": (0, 1, 2, 3)}
    assert plan.n_units == 11 and plan.reused_units == 5
    assert plan.request_tasks() == [("b", 3), ("c", 4), ("d", 4)]
    top = [TrialRecord("B-topup", t, i, True, 0.5) for t, k in plan.request_tasks() for i in range(k)]
    merged = apply_topup(recs, top, plan)
    assert topup_plan(merged, "B", tasks, 4).n_units == 0
    assert {(r.task, r.index) for r in merged if r.task == "b"} == {("b", i) for i in range(4)}
    assert all(r.candidate == "B" for r in merged)
    moved = [r for r in merged if r.extra.get("topup_candidate")]
    assert len(moved) == 11 and {r.extra["topup_index"] for r in moved if r.task == "b"} == {0, 1, 2}
    fresh = topup_plan(recs, "B", tasks, 4, fresh=True)
    assert fresh.n_units == 16 and fresh.fresh
    assert len(apply_topup(recs, top, fresh)) == 11              # fresh: base records not reused


def test_lineage_accept_fresh_policy_and_label_chaining(tmp_path):
    tasks = ["a", "b"]
    recs = [TrialRecord("B1", t, i, True, 1.0) for t in tasks for i in range(2)]
    strong = {"look": 1, "acc": {"t": 5.0}, "efficacy": 3.75, "fraction": 0.25, "verdict": "pass"}
    weak = {"look": 2, "acc": {"t": 2.7}, "efficacy": 2.54, "fraction": 0.5, "verdict": "pass"}
    final = {"look": 4, "acc": {"t": 2.5}, "efficacy": 1.72, "fraction": 1.0, "verdict": "pass"}
    assert not is_marginal(strong) and is_marginal(weak) and is_marginal(final)
    assert is_marginal({**final, "fraction": 0.97}, n_looks=4)
    lab = tmp_path / "labels.json"
    lab.write_text(json.dumps({"a": {"rate": 0.25}, "b": {"rate": None}}))
    lin = Lineage("A", step0_labels=str(lab))
    assert lin.labels_source()["kind"] == "step0" and lin.labels() == {"a": 0.25, "b": None}
    l1, p1 = lin.accept("B1", strong, recs, tasks, 4, fresh="marginal", record_files=[str(tmp_path / "B1.jsonl")])
    assert not p1.fresh and p1.n_units == 4 and l1.current == "B1" and l1.previous == "A"
    write_jsonl(tmp_path / "A.jsonl", [TrialRecord("A", "a", 0, True, 1.0), TrialRecord("A", "a", 1, True, 0.0)])
    l1.add_records("A", str(tmp_path / "A.jsonl"))
    assert l1.labels_source()["kind"] == "previous_reference" and l1.labels() == {"a": 0.5}
    _, p2 = l1.accept("B2", weak, [], tasks, 4, fresh="marginal")
    assert p2.fresh and p2.n_units == 8
    l1.save(tmp_path / "lin.json")
    assert Lineage.load(tmp_path / "lin.json") == l1


def test_cli_topup_accept_and_sampling(tmp_path):
    from harnessforge.race_run import main
    from harnessforge.stats.sampling import main as smain
    tasks = [f"t{i}" for i in range(8)]
    (tmp_path / "req.json").write_text(json.dumps({"tasks": tasks, "trials": 2}))
    recs = [TrialRecord("B", t, i, True, 1.0) for t in tasks[:4] for i in range(2)]
    write_jsonl(tmp_path / "B.jsonl", recs)
    assert main(["topup", "--records", str(tmp_path / "B.jsonl"), "--reference", "B", "--tasks",
                 str(tmp_path / "req.json"), "--out", str(tmp_path / "top.json")]) == 0
    assert json.loads((tmp_path / "top.json").read_text())["units_missing"] == 8
    race = {"looks": [{}, {}, {}, {}], "decisions": {"states": {"B": {
        "verdict": "pass", "look": 2, "acc": {"t": 4.0}, "efficacy": 2.54, "fraction": 0.5}}}}
    (tmp_path / "race.json").write_text(json.dumps(race))
    assert main(["accept", "--lineage", str(tmp_path / "lin.json"), "--reference", "A", "--race",
                 str(tmp_path / "race.json"), "--candidate", "B", "--records", str(tmp_path / "B.jsonl"),
                 "--tasks", str(tmp_path / "req.json"), "--fresh", "marginal",
                 "--out-topup", str(tmp_path / "top2.json")]) == 0
    lin = Lineage.load(tmp_path / "lin.json")
    assert lin.current == "B" and lin.previous == "A" and lin.history[0]["marginal"] is False
    write_jsonl(tmp_path / "A.jsonl", [TrialRecord("A", t, 0, True, float(i % 2)) for i, t in enumerate(tasks)])
    lin.add_records("A", str(tmp_path / "A.jsonl"))
    lin.save(tmp_path / "lin.json")
    assert smain(["--tasks", str(tmp_path / "req.json"), "--lineage", str(tmp_path / "lin.json"), "--tilt", "1",
                  "--out", str(tmp_path / "s.json")]) == 0
    s = TaskSampling.from_dict(json.loads((tmp_path / "s.json").read_text()))
    assert s.meta["labels"]["kind"] == "previous_reference" and len(s.tasks) == 8
    # the race CLI reads it
    from harnessforge.race_run import _schedule
    import argparse
    a = argparse.Namespace(checkpoints="0.25,0.5,0.75,1", sampling=str(tmp_path / "s.json"), design="auto",
                           tau2=VAL200.tau2, sigma2=VAL200.sigma2, trial_looks=None, plan=None)
    sch = _schedule(Order.TASK_MAJOR, s.tasks, 2, a)
    assert sch.sampling == s and sch.information is not None


def test_labels_from_records():
    recs = [TrialRecord("A", "x", 0, True, 1.0), TrialRecord("A", "x", 1, True, 0.0),
            TrialRecord("A", "y", 0, True, env_failed=True), TrialRecord("B", "x", 0, True, 1.0)]
    assert labels_from_records(recs, "A") == {"x": 0.5}
