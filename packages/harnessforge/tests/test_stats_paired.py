import math

import pytest

from harnessforge.records import TaskStats
from harnessforge.stats import GateConfig, decide, paired, task_mean


def ts(reward, cost=None, n=1):
    return TaskStats(reward, n, cost)


def test_paired_basic_and_unit_is_task():
    a = {"t1": ts(0.0), "t2": ts(0.5), "t3": ts(1.0), "t4": ts(0.2)}
    b = {"t1": ts(0.5), "t2": ts(0.5), "t3": ts(1.0), "t4": ts(0.6), "extra": ts(1.0)}
    p = paired(a, b)
    assert p.n == 4                                   # only common tasks
    d = [0.5, 0.0, 0.0, 0.4]
    md = sum(d) / 4
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / 3)
    assert p.d == pytest.approx(md) and p.se == pytest.approx(sd / 2)
    assert p.t == pytest.approx(md / (sd / 2))
    assert (p.mean_a, p.mean_b) == (pytest.approx(0.425), pytest.approx(0.65))
    assert p.as_dict()["n"] == 4


def test_paired_degenerate_cases():
    assert paired({"t": ts(0.0)}, {"t": ts(1.0)}) is None           # < 2 tasks
    same = {"t1": ts(0.3), "t2": ts(0.7)}
    assert paired(same, same).t == 0.0
    up = {"t1": ts(0.4), "t2": ts(0.8)}
    assert paired(same, up).t == math.inf                            # constant +0.1
    assert paired(up, same).t == -math.inf
    assert paired(up, same).as_dict()["t"] is None


def test_paired_cost_skips_tasks_without_cost():
    a = {"t1": ts(0, 100), "t2": ts(0, 200), "t3": ts(0, None)}
    b = {"t1": ts(0, 50), "t2": ts(0, 150), "t3": ts(0, 10)}
    p = paired(a, b, "cost")
    assert p.n == 2 and p.d == -50


def test_task_mean():
    assert task_mean({"a": ts(1.0), "b": ts(0.0)}) == 0.5
    assert task_mean({}) == 0.0


def _ab(n=20, better=3):
    a = {f"t{i}": ts(0.0, 100.0) for i in range(n)}
    b = {f"t{i}": ts(1.0 if i < better else 0.0, 100.0) for i in range(n)}
    return a, b


def test_decide_better_path():
    a, b = _ab(better=3)                               # t = 1.83
    assert decide(a, b).accept and decide(a, b).path == "better"
    assert not decide(a, b, z=2.0).accept
    assert not decide(a, {}).accept


def test_cheaper_path_off_by_default_and_on_when_configured():
    n = 20
    a = {f"t{i}": ts(0.5, 1000.0 + 10 * i) for i in range(n)}
    b = {f"t{i}": ts(0.5, 600.0 + 13 * i) for i in range(n)}    # same accuracy, ~35% cheaper
    off = decide(a, b)
    assert not off.accept and off.path is None
    cfg = GateConfig(cheaper_delta=0.03)
    on = decide(a, b, cfg=cfg)
    assert on.accept and on.path == "cheaper" and on.extra["cost_gain"] > 0.3
    # not cheaper enough
    assert not decide(a, {k: ts(0.5, v.cost * 0.97) for k, v in a.items()}, cfg=cfg).accept
    # cheaper but clearly worse
    worse = {k: ts(0.0 if i % 2 else 0.5, 500.0) for i, k in enumerate(a)}
    assert not decide(a, worse, cfg=cfg).accept
    # anchor blocks drift: much better anchor
    anchor = {k: ts(1.0, 1000.0) for k in a}
    assert not decide(a, b, cfg=cfg, anchor=anchor).accept
