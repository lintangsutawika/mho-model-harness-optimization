import pytest

from harnessforge import Order, TrialRecord
from harnessforge.contracts import EvalRequest
from harnessforge.stats import LookSchedule, UnitIndex, collect


def rec(cand, task, i, reward=1.0, finished=True, **kw):
    return TrialRecord(cand, task, i, finished, reward if finished else None, **kw)


TASKS = ("t0", "t1", "t2")


def full(cand="B", tasks=TASKS, k=3, reward=lambda t, i: 1.0):
    return [rec(cand, t, i, reward(t, i)) for t in tasks for i in range(k)]


# ------------------------------------------------------------------ ATTEMPT_MAJOR
def test_attempt_major_looks():
    s = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 3)
    assert [(lk.number, lk.trials, len(lk.tasks), lk.final) for lk in s.looks] == \
        [(1, 1, 3, False), (2, 2, 3, False), (3, 3, 3, True)]
    assert s.fractions == pytest.approx((1 / 3, 2 / 3, 1.0))


def test_attempt_major_readiness_in_flight_and_never_launched():
    s = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 3)
    rs = [rec("B", t, 0) for t in TASKS]                         # round 1 done
    rs += [rec("B", "t0", 1, 0.0), rec("B", "t1", 1, finished=False)]   # t2's 2nd never launched
    assert s.ready_looks(rs, "B") == [1]
    d2 = s.collect(rs, "B", 2)
    assert not d2.ready and d2.stats == {}
    assert (d2.expected, d2.launched, d2.finished, d2.missing) == (6, 5, 4, 1)
    # a LATER-launched trial that finished early is not used by look 1
    rs.append(rec("B", "t2", 2, 0.0))
    d1 = s.collect(rs, "B", 1)
    assert d1.ready and d1.stats["t2"].reward == 1.0 and d1.stats["t2"].n == 1
    # finishing the in-flight one and launching t2's 2nd -> look 2 ready
    rs += [rec("B", "t1", 1, 0.0), rec("B", "t2", 1, 0.0)]
    assert s.ready_looks(rs, "B") == [1, 2]
    assert s.collect(rs, "B", 2).stats["t1"].reward == 0.5


def test_finished_record_wins_over_stale_in_flight_record():
    s = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 1)
    rs = [rec("B", t, 0) for t in TASKS] + [rec("B", "t0", 0, finished=False)]
    assert s.collect(rs, "B", 1).ready
    idx = UnitIndex(rs)
    assert idx.get("B", "t0", 0).finished and idx.candidates() == ["B"]


def test_env_failed_counts_as_finished_but_not_evidence():
    s = LookSchedule(Order.ATTEMPT_MAJOR, ("t0", "t1"), 2)
    rs = [rec("B", "t0", 0, 0.0, env_failed=True), rec("B", "t0", 1, 1.0),
          rec("B", "t1", 0, 0.0, env_failed=True), rec("B", "t1", 1, 0.0, env_failed=True)]
    d = s.collect(rs, "B", 2)
    assert d.ready and d.env_failed == 3 and d.env_fail_frac == 0.75
    assert set(d.stats) == {"t0"} and d.stats["t0"].reward == 1.0
    assert d.as_dict()["env_fail_frac"] == 0.75


def test_records_beyond_trials_and_other_candidates_ignored():
    s = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 1)
    rs = [rec("B", t, 0, 0.0) for t in TASKS] + [rec("B", "t0", 5, 1.0), rec("A", "t0", 0, 1.0)]
    d = s.collect(rs, "B", 1)
    assert d.stats["t0"].reward == 0.0 and d.expected == 3


# ------------------------------------------------------------------ TASK_MAJOR
def test_task_major_checkpoints_follow_request_order():
    tasks = tuple(f"t{i}" for i in (7, 3, 5, 1, 0, 2, 4, 6))       # request order, not sorted
    s = LookSchedule(Order.TASK_MAJOR, tasks, 2)
    assert [len(lk.tasks) for lk in s.looks] == [2, 4, 6, 8]
    assert s.look(1).tasks == ("t7", "t3") and s.look(1).trials == 2
    assert s.fractions == (0.25, 0.5, 0.75, 1.0)
    odd = LookSchedule(Order.TASK_MAJOR, tasks[:7], 1, (0.5, 0.5, 0.9))  # dup, final appended
    assert [len(lk.tasks) for lk in odd.looks] == [4, 7]
    assert odd.looks[-1].final


def test_task_major_readiness():
    tasks = ("a", "b", "c", "d")
    s = LookSchedule(Order.TASK_MAJOR, tasks, 2, (0.5, 1.0))
    rs = [rec("B", "a", 0), rec("B", "a", 1, 0.0), rec("B", "b", 0),
          rec("B", "b", 1, finished=False),                 # in flight
          rec("B", "c", 0), rec("B", "c", 1), rec("B", "d", 0), rec("B", "d", 1)]
    assert s.ready_looks(rs, "B") == []                     # b's 2nd not finished
    rs.append(rec("B", "b", 1, 1.0))
    assert s.ready_looks(rs, "B") == [1, 2]
    d1 = s.collect(rs, "B", 1)
    assert set(d1.stats) == {"a", "b"} and d1.stats["a"].reward == 0.5
    # a task later in the order finishing does not make an earlier look ready
    rs2 = [r for r in rs if r.task != "a"]
    assert s.ready_looks(rs2, "B") == []
    assert s.collect(rs2, "B", 1).missing == 2


def test_from_request_and_validation(tmp_path):
    req = EvalRequest("r", [], ["x", "y"], "ds", 3, "http://e")
    s = LookSchedule.from_request(req, Order.ATTEMPT_MAJOR)
    assert s.tasks == ("x", "y") and len(s.looks) == 3
    assert LookSchedule("task_major", ("x",), 1).order is Order.TASK_MAJOR
    with pytest.raises(ValueError):
        LookSchedule(Order.TASK_MAJOR, ("x", "x"), 1)
    with pytest.raises(ValueError):
        LookSchedule(Order.TASK_MAJOR, ("x",), 0)
    with pytest.raises(ValueError):
        LookSchedule(Order.TASK_MAJOR, (), 1)


def test_collect_accepts_plain_records():
    s = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 3)
    assert collect(full(), "B", s.look(3)).ready
    assert collect(full(), "B", s.look(3)).policies == frozenset()
    rs = [rec("B", t, i, policy="p1") for t in TASKS for i in range(3)]
    assert collect(rs, "B", s.look(3)).policies == {"p1"}
