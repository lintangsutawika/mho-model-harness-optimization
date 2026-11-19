import pytest

from harnessforge import Order, TrialRecord
from harnessforge.stats import (Design, GateConfig, LookSchedule, Race, ReferenceMode, ValGate,
                                Verdict)

N, K = 40, 5
TASKS = tuple(f"t{i:02d}" for i in range(N))


def ref_reward(t, i):                     # reference: solves even tasks only
    return 1.0 if int(t[1:]) % 2 == 0 else 0.0


def flip(m):
    """Reference plus m odd tasks flipped to solved: t = 1.43 (m=2), 1.78 (m=3), 6.2 (m=20)."""
    odd = [t for t in TASKS if int(t[1:]) % 2][:m]
    return lambda t, i: 1.0 if t in odd else ref_reward(t, i)


def trials(cand, reward, rounds=K, tokens=None):
    """Attempt-major records of the first `rounds` rounds (all finished)."""
    return [TrialRecord(cand, t, i, True, reward(t, i), output_tokens=tokens)
            for i in range(rounds) for t in TASKS]


def schedule():
    return LookSchedule(Order.ATTEMPT_MAJOR, TASKS, K)


def test_loser_stops_at_look1_winner_passes_early_borderline_goes_to_end():
    race = Race("A", ["lose", "win", "border"], schedule())
    snap = trials("A", ref_reward, 1) + trials("lose", lambda t, i: 0.0, 1) + \
        trials("win", flip(20), 1) + trials("border", flip(2), 1)
    st = race.update(snap)
    assert st.states["lose"].verdict is Verdict.FAIL and st.states["lose"].look == 1
    assert st.newly_decided == ("lose",) and st.stop == ("lose",)
    assert st.states["win"].alive and st.states["win"].look == 1      # no efficacy at look 1
    assert not st.reference_can_stop and st.winner is None

    snap2 = trials("A", ref_reward, 2) + trials("win", flip(20), 2) + trials("border", flip(2), 2)
    st = race.update(snap + snap2)
    assert st.states["win"].verdict is Verdict.PASS and st.states["win"].look == 2
    assert st.states["win"].path == "better" and st.newly_decided == ("win",)
    assert st.states["border"].alive and not st.reference_can_stop    # reference keeps running

    for r in (3, 4):
        st = race.update(trials("A", ref_reward, r) + trials("border", flip(2), r))
        assert st.states["border"].alive and st.states["border"].look == r
        assert st.states["border"].t == pytest.approx(1.43, abs=0.01)
    st = race.update(trials("A", ref_reward, 5) + trials("border", flip(2), 5))
    assert st.states["border"].verdict is Verdict.FAIL and st.states["border"].look == 5
    assert st.done and st.reference_can_stop and st.winner == "win"
    assert st.ranking == ("win",) and st.as_dict()["winner"] == "win"


def test_borderline_passes_only_at_the_final_look():
    race = Race("A", ["b"], schedule())
    st = race.update(trials("A", ref_reward) + trials("b", flip(3)))   # all looks ready at once
    s = st.states["b"]
    assert s.verdict is Verdict.PASS and s.look == 5 and s.efficacy == 1.72
    assert s.t == pytest.approx(1.78, abs=0.01)


def test_looks_are_evaluated_in_order_when_several_become_ready():
    race = Race("A", ["win"], schedule())
    st = race.update(trials("A", ref_reward) + trials("win", flip(20)))
    assert st.states["win"].look == 2                                   # not 5


def test_idempotent_on_repeated_and_later_snapshots():
    race = Race("A", ["lose", "b"], schedule())
    snap = trials("A", ref_reward, 1) + trials("lose", lambda t, i: 0.0, 1) + \
        trials("b", flip(2), 1)
    st1 = race.update(snap)
    st2 = race.update(snap)
    assert st1.states == st2.states and st2.newly_decided == () and st2.stop == ("lose",)
    # later data for a decided candidate (it kept running a bit) never changes its verdict
    st3 = race.update(snap + trials("lose", lambda t, i: 1.0, 5)[N:])
    assert st3.states["lose"] == st1.states["lose"] and race.status() is st3


def test_never_decides_without_complete_reference_units():
    race = Race("A", ["lose"], schedule())
    a = trials("A", ref_reward, 1)
    a[0] = TrialRecord("A", a[0].task, 0, False)                         # one in flight
    st = race.update(a + trials("lose", lambda t, i: 0.0, 1))
    assert st.states["lose"].alive and st.states["lose"].look is None
    assert race.update(a[1:] + trials("lose", lambda t, i: 0.0, 1)).states["lose"].alive  # missing
    assert not race.update(trials("A", ref_reward, 1) +
                           trials("lose", lambda t, i: 0.0, 1)).states["lose"].alive


def test_reference_mode_same_units_vs_all_trials():
    a = [TrialRecord("A", t, i, True, 0.0 if i == 0 else 1.0) for i in range(K) for t in TASKS]
    b = trials("B", flip(20), 1)
    same = Race("A", ["B"], schedule()).update(a + b).states["B"]
    assert same.acc.mean_a == 0.0                                       # A's look-1 units only
    allt = Race("A", ["B"], schedule(), reference_mode="all").update(a + b).states["B"]
    assert allt.acc.mean_a == pytest.approx(0.8)                        # all 5 of A's trials
    # ALL_TRIALS with an unfinished reference cannot look at all
    part = Race("A", ["B"], schedule(), reference_mode=ReferenceMode.ALL_TRIALS)
    assert part.update(a[:-1] + b).states["B"].look is None


def test_winner_is_max_t_with_deterministic_ties():
    race = Race("A", ["x", "y", "z"], schedule())
    st = race.update(trials("A", ref_reward) + trials("x", flip(3)) + trials("y", flip(20)) +
                     trials("z", flip(20)))
    assert st.ranking == ("y", "z", "x") and st.winner == "y"


def test_env_failure_skips_interim_look_and_flags_final():
    race = Race("A", ["lose"], schedule(), max_env_fail=0.2)
    bad = [TrialRecord("lose", t, 0, True, 0.0, env_failed=j < 10) for j, t in enumerate(TASKS)]
    st = race.update(trials("A", ref_reward, 1) + bad)
    assert st.states["lose"].alive and st.states["lose"].skipped == (1,)
    st = race.update(trials("A", ref_reward, 2) + bad + trials("lose", lambda t, i: 0.0, 2)[N:])
    assert st.states["lose"].verdict is Verdict.FAIL and st.states["lose"].look == 2
    assert not st.states["lose"].infra_suspect
    one = Race("A", ["b"], LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 1), max_env_fail=0.2)
    s = one.update(trials("A", ref_reward, 1) +
                   [TrialRecord("b", t, 0, True, 0.0, env_failed=True) for t in TASKS]).states["b"]
    assert s.verdict is Verdict.FAIL and s.infra_suspect and "infra" in s.reason


def test_task_major_race():
    s = LookSchedule(Order.TASK_MAJOR, TASKS, 2)
    race = Race("A", ["win", "lose"], s)
    first = TASKS[:20]                                     # 50% checkpoint
    snap = [r for r in trials("A", ref_reward, 2) + trials("win", flip(20), 2) +
            trials("lose", lambda t, i: 0.0, 2) if r.task in first]
    st = race.update(snap)
    assert st.states["lose"].verdict is Verdict.FAIL and st.states["lose"].look == 1
    assert st.states["win"].verdict is Verdict.PASS and st.states["win"].look == 2
    assert st.done and st.winner == "win"


def test_cheaper_path_at_final_look_only_when_enabled():
    sched = LookSchedule(Order.ATTEMPT_MAJOR, TASKS, 1)
    a = [TrialRecord("A", t, 0, True, ref_reward(t, 0), output_tokens=1000 + j)
         for j, t in enumerate(TASKS)]
    b = [TrialRecord("B", t, 0, True, ref_reward(t, 0), output_tokens=500 + 2 * j)
         for j, t in enumerate(TASKS)]
    assert Race("A", ["B"], sched).update(a + b).states["B"].verdict is Verdict.FAIL
    s = Race("A", ["B"], sched, cheaper=GateConfig(cheaper_delta=0.03)).update(a + b).states["B"]
    assert s.verdict is Verdict.PASS and s.path == "cheaper" and "cheaper" in s.reason


def test_validation():
    with pytest.raises(ValueError):
        Race("A", [], schedule())
    with pytest.raises(ValueError):
        Race("A", ["A"], schedule())
    with pytest.raises(ValueError):
        Race("A", ["B"], schedule(), Design.attempt_major(3))


def test_val_gate():
    sched = LookSchedule(Order.TASK_MAJOR, TASKS, 2)
    gate = ValGate("best", "new", sched)
    assert gate.design.n_looks == 4 and gate.status().alive
    best = trials("best", ref_reward, 2)                    # on file in full
    new = [r for r in trials("new", flip(20), 2) if r.task in TASKS[:10]]
    s = gate.update(best + new)
    assert s.alive and s.look == 1                          # 25%: no efficacy look yet
    s = gate.update(best + trials("new", flip(20), 2))
    assert s.verdict is Verdict.PASS and s.look == 2
