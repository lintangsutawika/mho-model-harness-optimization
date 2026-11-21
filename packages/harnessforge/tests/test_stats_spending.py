import math
import random

import pytest

from harnessforge import Order, TrialRecord
from harnessforge.records import TaskStats
from harnessforge.stats import Design, LookSchedule, Race, Verdict, paired
from harnessforge.stats.information import VAL200, Information
from harnessforge.stats.race import default_design
from harnessforge.stats.spending import SpendingDesign, crossing_probabilities, spending_function

TASKS = tuple(f"t{i:03d}" for i in range(200))


# ------------------------------------------------------------------ information
def test_information_grows_much_less_than_trials():
    i2, i24 = VAL200.info(200, 2), VAL200.info(200, 24)
    assert i24 / i2 == pytest.approx(7.18, abs=0.01)                # not 12
    assert VAL200.fractions((2, 4, 8, 24)) == pytest.approx((0.139, 0.263, 0.471, 1.0), abs=1e-3)
    assert Information(0.0, 0.25).fractions((1, 2, 4)) == pytest.approx((0.25, 0.5, 1.0))   # no tau2: linear
    assert VAL200.variance(4, 24) == pytest.approx(0.0081 + 0.1245 / 4 + 0.1245 / 24)
    with pytest.raises(ValueError):
        Information(-1, 0.2)


# ------------------------------------------------------------------ spending boundaries
def test_obf_spending_matches_gsdesign():
    # gsDesign(k=4, test.type=1, alpha=0.025, sfu=sfLDOF)$upper$bound
    d = SpendingDesign((0.25, 0.5, 0.75, 1.0), alpha=0.025)
    assert d.planned_bounds() == pytest.approx((4.3326, 2.9631, 2.3590, 2.0141), abs=2e-4)
    assert sum(crossing_probabilities(d.fractions, d.planned_bounds())) == pytest.approx(0.025, abs=1e-6)


@pytest.mark.parametrize("kind", ["obf", "pocock", "power:3"])
def test_spending_functions_spend_alpha_exactly(kind):
    f = spending_function(kind, 0.05)
    assert f(0) == 0 and f(1) == pytest.approx(0.05) and f(0.3) < f(0.6) < 0.05
    d = SpendingDesign(VAL200.fractions((2, 4, 8, 24)), spending=kind)
    b = d.planned_bounds()
    probs = crossing_probabilities(d.fractions, b)
    assert sum(probs) == pytest.approx(0.05, abs=1e-6)
    cum = 0.0
    for j, p in enumerate(probs):
        cum += p
        assert cum == pytest.approx(f(d.fractions[j]), abs=1e-6)


def test_power_by_recursion_matches_closed_form_for_one_look():
    assert sum(crossing_probabilities((1.0,), (1.645,), drift=2.5)) == pytest.approx(0.80, abs=0.005)


def test_planned_bounds_and_linear_futility_on_information_fractions():
    sched = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 24, trial_looks=(2, 4, 8), information=VAL200)
    assert sched.trial_looks == (2, 4, 8, 24) and [lk.trials for lk in sched.looks] == [2, 4, 8, 24]
    d = SpendingDesign.for_schedule(sched, observed=False)
    assert d.fractions == pytest.approx(VAL200.fractions((2, 4, 8, 24)))
    assert d.as_dict()["efficacy"] == pytest.approx([5.1217, 3.6504, 2.6318, 1.6583], abs=1e-3)
    assert [d.futility_bound(j) for j in (1, 2, 3)] == pytest.approx([-1 + 2.5 * f for f in d.fractions[:3]])
    assert d.futility_bound(4) == d.efficacy(4)
    assert d.verdict(1, 5.2) is Verdict.PASS and d.verdict(1, -0.7) is Verdict.FAIL
    assert d.verdict(4, 1.6) is Verdict.FAIL and d.verdict(4, 1.7) is Verdict.PASS


def test_observed_information_shifts_boundaries_and_caps_at_one():
    d = SpendingDesign(VAL200.fractions((2, 4, 8, 24)), max_information=10_000)
    planned = d.planned_bounds()
    lb1 = d.bounds(1, info=1_390)                       # = plan (0.139)
    assert lb1.efficacy == pytest.approx(planned[0], abs=0.01) and not lb1.final
    # less information than planned at look 2: a higher boundary (less alpha spent so far)
    lb2 = d.bounds(2, info=2_000, history=((1, lb1.ratio),))
    assert lb2.efficacy > planned[1] and lb2.fraction == pytest.approx(0.2)
    # reaching I_max at an interim look makes it final: the remainder of alpha is spent there
    lb3 = d.bounds(3, info=12_000, history=((1, lb1.ratio), (2, lb2.ratio)))
    assert lb3.final and lb3.fraction == 1.0 and lb3.futility == lb3.efficacy
    assert sum(crossing_probabilities((lb1.ratio, lb2.ratio, lb3.ratio),
                                      (lb1.efficacy, lb2.efficacy, lb3.efficacy))) == pytest.approx(0.05, abs=1e-5)
    # final look below plan still spends everything
    lb4 = d.bounds(4, info=8_000, history=((1, lb1.ratio), (2, lb2.ratio), (3, 0.45)))
    assert lb4.final and lb4.fraction == pytest.approx(0.8)
    assert sum(crossing_probabilities((lb1.ratio, lb2.ratio, 0.45, 0.8),
                                      (lb1.efficacy, lb2.efficacy, d.bounds(3, 4_500, ((1, lb1.ratio), (2, lb2.ratio))).efficacy,
                                       lb4.efficacy))) == pytest.approx(0.05, abs=1e-5)


def test_skipped_or_early_looks_spend_nothing():
    d = SpendingDesign((0.25, 0.5, 1.0), first_efficacy=2)
    assert math.isinf(d.efficacy(1))
    # look 2 then has all alpha(0.5) available: the one-look boundary
    assert d.efficacy(2) == pytest.approx(SpendingDesign((0.5, 1.0)).efficacy(1), abs=1e-6)


# ------------------------------------------------------------------ plans and the weighted estimator
PLAN_A = (((50, 2),), ((50, 2), (100, 4)), ((50, 2), (100, 4), (150, 8)), ((50, 2), (100, 4), (150, 8), (200, 24)))
PLAN_B = (((100, 1),), ((200, 2),), ((200, 8),), ((200, 24),))


def test_plan_looks_information_and_launch_order():
    a = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 24, information=VAL200, plan=PLAN_A)
    assert [(len(lk.tasks), lk.size, lk.uniform) for lk in a.looks] == \
        [(50, 100, True), (100, 300, False), (150, 700, False), (200, 1900, False)]
    info = a.planned_information()
    assert info[-1] == pytest.approx(sum(50 / VAL200.variance(k) for k in (2, 4, 8, 24)))
    assert 100 / math.sqrt(info[-1]) == pytest.approx(1.40, abs=0.01)          # final SE in points
    assert a.trial_counts is None and a.planned
    order = a.launch_order()
    assert len(order) == 1900 and order[:3] == [("t000", 0), ("t001", 0), ("t002", 0)]
    assert order[100:102] == [("t050", 0), ("t051", 0)]                    # look 2 adds new tasks first ...
    w = a.weights(a.look(2))
    assert w["t000"] == pytest.approx(1 / VAL200.variance(2)) and w["t099"] == pytest.approx(1 / VAL200.variance(4))
    full = a.full(a.look(2))
    assert full.counts[:50] == (2,) * 50 and full.counts[50:] == (4,) * 50
    b = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 24, information=VAL200, plan=PLAN_B)
    assert [(len(lk.tasks), lk.trials, lk.counts) for lk in b.looks] == \
        [(100, 1, None), (200, 2, None), (200, 8, None), (200, 24, None)]
    assert b.trial_counts is None and b.look(2).units().__next__() == ("t000", 0)
    assert b.launch_order()[100] == ("t100", 0) and b.launch_order()[200] == ("t000", 1)


@pytest.mark.parametrize("plan,err", [
    ((((50, 2),), ((40, 4),)), "takes trials away"),
    ((((50, 2),), ((50, 2),)), "adds nothing"),
    ((((250, 2),),), "segment ends"),
    ((((200, 30),),), "more than trials"),
])
def test_plan_validation(plan, err):
    with pytest.raises(ValueError, match=err):
        LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 24, plan=plan)
    with pytest.raises(ValueError, match="need Order.ATTEMPT_MAJOR"):
        LookSchedule(Order.TASK_MAJOR, TASKS, 24, plan=PLAN_B)


def ts(reward, n=1):
    return TaskStats(reward=reward, n=n)


def test_weighted_paired_is_inverse_variance_mean_with_sandwich_se():
    a = {f"t{i}": ts(0.5) for i in range(4)}
    b = {"t0": ts(0.6), "t1": ts(0.4), "t2": ts(0.9), "t3": ts(0.5)}
    w = {"t0": 1.0, "t1": 1.0, "t2": 3.0, "t3": 3.0}
    p = paired(a, b, weights=w)
    d = [0.1, -0.1, 0.4, 0.0]
    md = sum(wi * x for wi, x in zip(w.values(), d)) / 8
    se = math.sqrt(4 / 3 * sum((wi * (x - md)) ** 2 for wi, x in zip(w.values(), d))) / 8
    assert p.d == pytest.approx(md) and p.se == pytest.approx(se)
    assert p.mean_b - p.mean_a == pytest.approx(p.d)
    plain = paired(a, b)
    eq = paired(a, b, weights={k: 2.0 for k in w})
    assert eq.d == pytest.approx(plain.d) and eq.se == pytest.approx(plain.se)    # equal weights = plain
    assert paired(a, b, weights={"t0": 1.0}) is None                             # tasks without weight dropped


# ------------------------------------------------------------------ race
def records(cand, sched, p, seed, upto=None):
    rng = random.Random(seed)
    out = []
    for t, i in sched.launch_order()[:upto]:
        rate = p(t)
        out.append(TrialRecord(cand, t, i, True, float(rng.random() < rate)))
    return out


def test_race_with_spending_design_on_trial_looks():
    sched = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 24, trial_looks=(2, 4, 8, 24), information=VAL200)
    race = Race("A", ["B", "C"], sched)
    assert isinstance(race.design, SpendingDesign) and race.design.observed
    base = {t: random.Random(t).random() for t in TASKS}
    a = records("A", sched, base.get, 1)
    better = records("B", sched, lambda t: min(1.0, base[t] + 0.3), 2)          # +~25 points
    same = records("C", sched, base.get, 3)
    st = race.update(a + better + same)
    b = st.states["B"]
    assert b.verdict is Verdict.PASS and b.look == 1 and b.fraction is not None
    c = st.states["C"]
    assert c.look is not None and c.verdict in (Verdict.FAIL, Verdict.CONTINUE, Verdict.PASS)
    lb = race.look_bounds("B", 1)
    assert lb.efficacy == pytest.approx(b.efficacy, abs=1e-3)
    # a look only uses trials with index < k of every task: 1 trial per task is not enough for look 1
    race2 = Race("A", ["B"], sched)
    st2 = race2.update([r for r in a + better if r.index < 1])
    assert st2.states["B"].look is None and st2.states["B"].alive


def test_race_on_plan_uses_weights_and_default_design():
    sched = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 24, information=VAL200, plan=PLAN_A)
    race = Race("A", ["B"], sched)
    assert isinstance(race.design, SpendingDesign)
    base = {t: random.Random(t).random() for t in TASKS}
    n2 = sched.look(2).size
    a = records("A", sched, base.get, 1, upto=n2)
    b = records("B", sched, lambda t: min(1.0, base[t] + 0.05), 2, upto=n2)
    st = race.update(a + b)
    s = st.states["B"]
    assert s.look in (1, 2)
    if s.look == 2:
        from harnessforge.stats.looks import collect
        la, lb_ = collect(a, "A", sched.look(2)), collect(b, "B", sched.look(2))
        assert s.acc.d == pytest.approx(paired(la.stats, lb_.stats, weights=sched.weights(sched.look(2))).d)


def test_default_design_without_plan_is_unchanged():
    sched = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 5)
    d = default_design(sched, "all")
    assert isinstance(d, Design) and d.c == 1.72 and d.fractions == pytest.approx((0.2, 0.4, 0.6, 0.8, 1.0))
    # trial looks without information: k/K fractions, spending on them (planned)
    s2 = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 24, trial_looks=(2, 4, 8, 24))
    d2 = default_design(s2)
    assert isinstance(d2, SpendingDesign) and not d2.observed and d2.fractions == pytest.approx((1 / 12, 1 / 6, 1 / 3, 1))


def test_calibrate_simulates_trial_looks_on_information_fractions():
    from harnessforge.stats.calibrate import load_rates, simulate
    sim = simulate(load_rates(), trials=8, trial_looks=(2, 4, 8), information=VAL200, reference="same", sims=200)
    assert sim.fractions == pytest.approx(VAL200.fractions((2, 4, 8)))
    assert len(sim.t[0]) == 3 and 1.0 < sim.c_for(0.05, first_efficacy=1) < 2.5
