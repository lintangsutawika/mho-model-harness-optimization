import math
import time

import pytest

from harnessforge import Order
from harnessforge.stats import DEFAULT_FUTILITY, Design, LookSchedule, Verdict, linear_futility
from harnessforge.stats.calibrate import load_rates, main, simulate, smoothed_rates


def test_reference_boundaries_k5():
    d = Design.attempt_major(5, c=1.72)
    assert d.efficacy(1) == math.inf
    assert [round(d.efficacy(r), 2) for r in (2, 3, 4, 5)] == [2.72, 2.22, 1.92, 1.72]
    assert [d.futility_bound(r) for r in (1, 2, 3, 4)] == pytest.approx(list(DEFAULT_FUTILITY))
    assert d.futility_bound(5) == 1.72
    explicit = Design.attempt_major(5, futility=DEFAULT_FUTILITY)
    assert [explicit.futility_bound(r) for r in range(1, 5)] == list(DEFAULT_FUTILITY)
    assert linear_futility(0.2) == pytest.approx(-0.5)


def test_verdicts():
    d = Design.attempt_major(5)
    assert d.verdict(1, 10.0) is Verdict.CONTINUE        # no efficacy at look 1
    assert d.verdict(1, -0.6) is Verdict.FAIL
    assert d.verdict(2, 2.8) is Verdict.PASS
    assert d.verdict(2, 1.0) is Verdict.CONTINUE
    assert d.verdict(4, 1.0) is Verdict.FAIL              # futility 1.0 at look 4 (t <= f)
    assert d.verdict(5, 1.73) is Verdict.PASS
    assert d.verdict(5, 1.72) is Verdict.FAIL
    assert d.verdict(3, None) is Verdict.CONTINUE
    assert d.verdict(5, None) is Verdict.FAIL
    assert d.verdict(2, math.inf) is Verdict.PASS
    nofut = Design.attempt_major(5, futility=())
    assert nofut.verdict(1, -50) is Verdict.CONTINUE
    partial = Design.attempt_major(5, futility=(-1.0,))
    assert partial.futility_bound(2) == -math.inf
    with pytest.raises(IndexError):
        d.efficacy(6)


def test_design_for_task_major_schedule_and_single_look():
    s = LookSchedule(Order.TASK_MAJOR, tuple("abcdefgh"), 2)
    d = Design.for_schedule(s)
    assert d.fractions == (0.25, 0.5, 0.75, 1.0)
    assert d.efficacy(2) == pytest.approx(1.72 / math.sqrt(0.5))
    one = Design((1.0,), c=1.645)
    assert one.efficacy(1) == 1.645 and one.verdict(1, 1.7) is Verdict.PASS
    assert d.as_dict()["efficacy"][0] is None
    with pytest.raises(ValueError):
        Design((0.5, 0.4, 1.0))
    with pytest.raises(ValueError):
        Design((0.5, 0.9))


def test_rates_loading(tmp_path):
    r = load_rates()
    assert len(r) == 128 and 0.42 < sum(r) / len(r) < 0.46
    assert smoothed_rates({"a": [1, 1], "b": [0, 0]}) == [2.5 / 3, 0.5 / 3]
    p = tmp_path / "r.json"
    p.write_text('{"a": [1.0, 0.0], "b": [0.0]}')
    assert load_rates(p) == [1.5 / 3, 0.5 / 2]
    p.write_text('{"a": 0.3, "b": 0.6}')
    assert load_rates(p) == [0.3, 0.6]
    from harnessforge import TrialRecord, write_jsonl
    write_jsonl(tmp_path / "r.jsonl", [TrialRecord("A", "a", 0, True, 1.0),
                                       TrialRecord("B", "a", 0, True, 0.0)])
    assert load_rates(tmp_path / "r.jsonl", "A") == [1.5 / 2]


def test_calibration_sanity_seeded():
    t0 = time.time()
    rates = load_rates()
    cal = simulate(rates, trials=5, sims=2000, seed=1)
    c = cal.c_for(0.05)
    assert 1.55 < c < 1.90                               # reference: 1.72 at 20k sims
    # c read off for Bonferroni alpha/3 is larger
    assert cal.c_for(0.05 / 3) > c
    # alpha on an INDEPENDENT null simulation, without futility, within MC error
    check = simulate(rates, trials=5, sims=2000, seed=2)
    d = Design.attempt_major(5, c=c)
    alpha = check.operating(d, futility=False).accept
    assert abs(alpha - 0.05) < 0.02                      # ~4 MC standard errors
    # non-binding futility only lowers alpha and roughly halves the null's trials
    oc = check.operating(d)
    assert oc.accept <= alpha and 0.35 < oc.mean_fraction < 0.65
    assert sum(oc.stop_looks) == pytest.approx(1.0)
    # a real effect has power above alpha
    alt = simulate(rates, trials=5, sims=500, seed=3, shift=0.4)
    assert alt.operating(d).accept > 0.4 and alt.acc_b > alt.acc_a
    assert time.time() - t0 < 5


def test_simulation_task_major_and_reference_modes():
    rates = load_rates()[:40]
    tm = simulate(rates, order=Order.TASK_MAJOR, trials=2, sims=300, seed=0)
    assert tm.fractions == (0.25, 0.5, 0.75, 1.0) and len(tm.t[0]) == 4
    same = simulate(rates, trials=3, sims=300, seed=0, reference="same")
    assert len(same.t[0]) == 3 and 1.0 < same.c_for(0.05) < 2.5
    with pytest.raises(ValueError):
        simulate(rates, reference="bogus")
    with pytest.raises(ValueError):
        tm.operating(Design.attempt_major(5))


def test_cli_prints_table(capsys):
    assert main(["--sims", "200", "--backend", "python", "--shifts", "0,0.3",
                 "--futility", "none", "--futility=-0.5,0,0.5,1.0"]) == 0
    out = capsys.readouterr().out
    assert "calibrated c for one-sided alpha 0.05" in out
    assert "futility -0.5,0,0.5,1.0" in out and "shift +0.30" in out
