import json
import random

import pytest

from harnessforge import Candidate, EvalRequest, Order, TrialRecord, write_jsonl
from harnessforge.evals import FakeBackend
from harnessforge.race_run import FileRun, RaceDriver, main, replay, side_counts, snapshot_at
from harnessforge.stats import LookSchedule, Race, Verdict

N, K = 48, 4
TASKS = [f"t{i:02d}" for i in range(N)]


def shuffled(seed=0):
    tasks = list(TASKS)
    random.Random(seed).shuffle(tasks)
    return tasks


def start(rewards, *, concurrency=8, durations=None, seed=0, tasks=None):
    backend = FakeBackend(rewards, durations=durations, seed=seed)
    tasks = tasks or shuffled()
    req = EvalRequest("r", [Candidate("A", "a"), Candidate("B", "b")], tasks, "ds", K, "http://x/v1",
                      concurrency=concurrency)
    run = backend.start(req)
    return run, LookSchedule(run.order, tuple(tasks), K)


def drive(run, race, **kw):
    logs = []
    driver = RaceDriver(run, race, poll_s=kw.pop("poll_s", 1), clock=lambda: float(run.tick), log=logs.append, **kw)
    return driver, driver.drive(), logs


# A solves a fixed half of the tasks; B solves A's tasks plus most others (clearly better).
A_REWARDS = {t: (1.0 if i % 2 == 0 else 0.0) for i, t in enumerate(TASKS)}
B_BETTER = {t: (1.0 if i % 2 == 0 or i % 3 else 0.0) for i, t in enumerate(TASKS)}


def test_better_candidate_passes_at_look_2_and_both_sides_stop(tmp_path):
    run, sched = start({"A": A_REWARDS, "B": B_BETTER})
    race = Race("A", ["B"], sched)
    driver, s, logs = drive(run, race, summary_path=tmp_path / "race.json")
    st = s["decisions"]["states"]["B"]
    assert st["verdict"] == "pass" and st["look"] == 2            # no efficacy at look 1
    assert s["decisions"]["winner"] == "B"
    assert [(e["look"], e["verdict"]) for e in s["timeline"]] == [(1, "continue"), (2, "pass")]
    e1, e2 = s["timeline"]
    assert e1["tasks"] == 12 and e2["tasks"] == 24 and e1["efficacy"] is None
    assert e2["efficacy"] == pytest.approx(1.77 / 0.5 ** 0.5, abs=1e-3)
    assert e2["t"] == pytest.approx(st["acc"]["t"], abs=1e-3)
    # B stopped when decided, A once no candidate is alive; both settle, and trials were saved
    assert set(s["stops"]) == {"A", "B"} and all(v["latency_s"] is not None for v in s["stops"].values())
    for side in ("A", "B"):
        k = s["sides"][side]
        assert k["in_flight"] == 0 and k["launched"] == k["finished"]
        assert k["never_launched"] > 0 and k["planned"] == N * K
        assert k["scored"] + k["cancelled"] + k["env_failed"] == k["finished"]
    assert json.loads((tmp_path / "race.json").read_text())["decisions"]["winner"] == "B"
    assert any("look 2 B vs A" in line and "-> pass" in line for line in logs)


def test_intermediate_looks_logged_when_several_become_ready_at_once():
    run, sched = start({"A": A_REWARDS, "B": B_BETTER}, concurrency=1000)
    race = Race("A", ["B"], sched)
    _, s, _ = drive(run, race, poll_s=50)        # everything finishes before the second poll
    assert [e["look"] for e in s["timeline"]] == [1, 2]
    assert s["decisions"]["states"]["B"]["look"] == 2


def test_worse_candidate_fails_on_futility_at_look_1_and_reverse_replay():
    run, sched = start({"A": B_BETTER, "B": A_REWARDS})
    race = Race("A", ["B"], sched)
    driver, s, _ = drive(run, race)
    assert s["decisions"]["states"]["B"]["verdict"] == "fail"
    assert s["decisions"]["states"]["B"]["look"] == 1
    assert s["timeline"][0]["t"] <= s["timeline"][0]["futility"]
    # reverse orientation offline from the same records: A (the better one) vs reference B
    rev = replay(driver.records, Race("B", ["A"], sched))
    ra = rev["decisions"]["states"]["A"]
    assert ra["verdict"] == "continue" and ra["look"] == 1     # not decidable at look 1 (no efficacy)
    assert rev["timeline"][0]["t"] == pytest.approx(-s["timeline"][0]["t"], abs=1e-3)
    assert "A" not in rev["decided_at_s"]


def test_replay_matches_the_live_driver():
    run, sched = start({"A": A_REWARDS, "B": B_BETTER}, durations={t: 1 + i % 5 for i, t in enumerate(TASKS)})
    driver, s, _ = drive(run, Race("A", ["B"], sched))
    rep = replay(driver.records, Race("A", ["B"], sched))
    assert rep["decisions"]["states"]["B"]["verdict"] == s["decisions"]["states"]["B"]["verdict"]
    assert [(e["look"], e["t"]) for e in rep["timeline"]] == [(e["look"], e["t"]) for e in s["timeline"]]


def test_null_race_runs_to_the_final_look_and_settles():
    same = {t: 0.5 for t in TASKS}
    run, sched = start({"A": same, "B": same}, seed=3)
    _, s, _ = drive(run, Race("A", ["B"], sched, ))
    st = s["decisions"]["states"]["B"]
    assert st["verdict"] in ("fail", "pass")
    assert [e["look"] for e in s["timeline"]] == list(range(1, st["look"] + 1))
    assert s["sides"]["A"]["in_flight"] == s["sides"]["B"]["in_flight"] == 0


def test_snapshot_at_and_side_counts():
    recs = [TrialRecord("A", "t", 0, True, 1.0, started_at=0, finished_at=10),
            TrialRecord("A", "t", 1, True, 0.0, started_at=5, finished_at=20),
            TrialRecord("A", "t", 2, True, env_failed=True, error="stopped", started_at=15),
            TrialRecord("A", "u", 0, True, 0.0, started_at=30, finished_at=40)]
    snap = snapshot_at(recs, 12, end=40)
    assert [(r.index, r.finished) for r in snap] == [(0, True), (1, False)]
    snap = snapshot_at(recs, 40, end=40)
    assert [r.finished for r in snap] == [True, True, False, True]   # a cancelled trial never finishes
    k = side_counts(recs, "A", planned=8)
    assert k == {"planned": 8, "launched": 4, "finished": 4, "scored": 3, "env_failed": 0, "cancelled": 1,
                 "in_flight": 0, "never_launched": 4, "successes": 1}


def test_file_run_reads_jobs_and_stops_by_file(tmp_path):
    a, b = tmp_path / "ra", tmp_path / "rb"
    a.mkdir(), b.mkdir()
    write_jsonl(a / "records.jsonl", [TrialRecord("A", "t1", 0, True, 1.0), TrialRecord("X", "t1", 0, False)])
    write_jsonl(b / "records.jsonl", [TrialRecord("B", "t1", 0, False)])
    run = FileRun({"A": a, "B": b}, poll_s=0.01)
    assert run.order is Order.TASK_MAJOR
    assert sorted((r.candidate, r.finished) for r in run.records()) == [("A", True), ("B", False)]
    run.stop("B")
    assert (b / "B.stop").exists() and not (a / "A.stop").exists()
    assert not run.done() and not run.wait(timeout=0.05)
    (a / "status.json").write_text(json.dumps({"done": True}))
    (b / "status.json").write_text(json.dumps({"done": False}))
    assert run.side_done("A") and not run.done()
    (b / "status.json").write_text(json.dumps({"done": True}))
    assert run.wait(timeout=1)


def test_cli_replay(tmp_path, capsys):
    run, sched = start({"A": B_BETTER, "B": A_REWARDS})
    driver, s, _ = drive(run, Race("A", ["B"], sched))
    write_jsonl(tmp_path / "records.jsonl", driver.records)
    (tmp_path / "request.json").write_text(json.dumps({"tasks": list(sched.tasks), "trials": K}))
    assert main(["replay", str(tmp_path / "records.jsonl"), "--tasks", str(tmp_path / "request.json"),
                 "--reference", "A", "--candidates", "B", "--out", str(tmp_path / "rep.json")]) == 0
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["decisions"]["states"]["B"]["verdict"] == "fail"
    assert "look 1 B" in capsys.readouterr().out


def test_cli_drive_over_file_run_jobs(tmp_path, capsys):
    """The pilot's path: one `evals run` job per side (records.jsonl, request.json, status.json), the driver attached
    with --side; here both jobs have already finished everything."""
    run, sched = start({"A": A_REWARDS, "B": B_BETTER})
    run.wait()
    recs = run.records()
    for side in ("A", "B"):
        d = tmp_path / f"race-{side}"
        d.mkdir()
        write_jsonl(d / "records.jsonl", [r for r in recs if r.candidate == side])
        (d / "request.json").write_text(json.dumps({"tasks": list(sched.tasks), "trials": K}))
        (d / "status.json").write_text(json.dumps({"done": True}))
    out = tmp_path / "race.json"
    assert main(["drive", "--side", f"A={tmp_path / 'race-A'}", "--side", f"B={tmp_path / 'race-B'}",
                 "--reference", "A", "--out", str(out), "--poll", "0.01", "--reverse"]) == 0
    s = json.loads(out.read_text())
    assert s["decisions"]["states"]["B"]["verdict"] == "pass" and s["decisions"]["states"]["B"]["look"] == 2
    assert s["design"]["c"] == pytest.approx(1.77)
    assert (tmp_path / "race-B" / "B.stop").exists() and (tmp_path / "race-A" / "A.stop").exists()
    assert s["reverse"]["decisions"]["states"]["A"]["verdict"] == "fail"
    assert s["reverse"]["decisions"]["states"]["A"]["look"] == 1
    assert (tmp_path / "race.records.jsonl").exists()


def test_reverse_replay_after_a_stop_does_not_use_cancelled_units():
    """Forward race decided at look 2 and stopped both sides; the reverse race must not take look 3 on the
    cancelled trials (it would see them as finished env-failures)."""
    run, sched = start({"A": A_REWARDS, "B": B_BETTER})
    driver, s, _ = drive(run, Race("A", ["B"], sched))
    assert s["sides"]["A"]["cancelled"] + s["sides"]["B"]["cancelled"] > 0
    rev = replay(driver.records, Race("B", ["A"], sched))
    assert rev["decisions"]["states"]["A"]["verdict"] == "fail" and rev["decisions"]["states"]["A"]["look"] == 1
    near_null = {t: 0.5 for t in TASKS}
    run, sched = start({"A": near_null, "B": B_BETTER})
    driver, s, _ = drive(run, Race("A", ["B"], sched))
    rev = replay(driver.records, Race("B", ["A"], sched))
    looks = [e["look"] for e in rev["timeline"]]
    assert max(looks) <= s["decisions"]["states"]["B"]["look"]


def test_attempt_major_trial_looks_drive_and_replay(tmp_path):
    """A look-major evaluation with trial looks: the race uses a spending design, decides on whole rounds of trials,
    and an offline replay reproduces the live decision."""
    from harnessforge.stats.information import VAL200
    from harnessforge.stats.spending import SpendingDesign
    backend = FakeBackend({"A": A_REWARDS, "B": B_BETTER}, seed=0)
    tasks = shuffled()
    req = EvalRequest("r", [Candidate("A", "a"), Candidate("B", "b")], tasks, "ds", 8, "http://x/v1",
                      concurrency=8, options={"order": "attempt_major", "trial_looks": [2, 4, 8]})
    run = backend.start(req)
    assert run.order is Order.ATTEMPT_MAJOR
    sched = LookSchedule(run.order, tuple(tasks), 8, trial_looks=(2, 4, 8), information=VAL200)
    race = Race("A", ["B"], sched)
    assert isinstance(race.design, SpendingDesign)
    driver, s, logs = drive(run, race)
    st = s["decisions"]["states"]["B"]
    assert st["verdict"] == "pass" and st["look"] == 1 and st["fraction"] is not None
    e = s["timeline"][0]
    assert e["trials"] == 2 and e["units"] == 2 * N and e["info_fraction"] == pytest.approx(st["fraction"], abs=1e-3)
    assert s["design"]["kind"] == "spending"
    rep = replay(driver.records, Race("A", ["B"], sched))
    assert rep["decisions"]["states"]["B"]["verdict"] == "pass" and rep["decisions"]["states"]["B"]["look"] == 1
    # CLI replay with the trial looks
    recs = tmp_path / "r.jsonl"
    write_jsonl(recs, driver.records)
    (tmp_path / "req.json").write_text(json.dumps({"tasks": tasks, "trials": 8, "order": "attempt_major",
                                                   "trial_looks": [2, 4, 8]}))
    assert main(["replay", str(recs), "--tasks", str(tmp_path / "req.json"), "--reference", "A",
                 "--out", str(tmp_path / "rep.json")]) == 0
    out = json.loads((tmp_path / "rep.json").read_text())
    assert out["design"]["kind"] == "spending" and out["decisions"]["states"]["B"]["verdict"] == "pass"
