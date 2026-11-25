"""--seq-gate: the sequential train gate (utils.seq) and its use in the loop's eval_b."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_loop import Fake, make, summary  # noqa: E402
from utils.seq import SeqConfig, decide_look, launched, look_pt  # noqa: E402

K = 5


def write_trial(job_dir: Path, task_dir: Path, i: int, t0: float, reward: float | None,
                *, env_fail: bool = False) -> None:
    """One harbor-like trial dir: config.json (mtime = launch time) + result.json if finished
    (reward None = still running)."""
    t = job_dir / f"{task_dir.name}__{i:03d}"
    t.mkdir(parents=True, exist_ok=True)
    (t / "config.json").write_text(json.dumps({"task": {"path": str(task_dir)}}))
    os.utime(t / "config.json", (t0, t0))
    if reward is None:
        return
    res = {"task_name": task_dir.name, "finished_at": "x", "agent_setup": {} if not env_fail
           else None, "verifier_result": {"rewards": {"reward": reward}},
           "agent_result": {"n_output_tokens": 100}}
    if env_fail:
        res["exception_info"] = {"exception_type": "RuntimeError"}
    (t / "result.json").write_text(json.dumps(res))


# --------------------------------------------------------------------------- utils.seq
def test_look_uses_first_launched_trials_and_waits_for_them(tmp_path):
    job, tasks = tmp_path / "job", [tmp_path / "data" / f"p{i}" for i in range(3)]
    # attempt-major launch: round 1 at t=1000+i, round 2 at t=2000+i
    for i, td in enumerate(tasks):
        write_trial(job, td, 0, 1000 + i, 1.0)
    # task p0's round-2 trial is still running; the others finished
    write_trial(job, tasks[0], 1, 2000, None)
    write_trial(job, tasks[1], 1, 2001, 0.0)
    write_trial(job, tasks[2], 1, 2002, 0.0)
    assert [t["t0"] for t in launched(job)["p0"]] == [1000, 2000]
    b1, _ = look_pt(job, 1, 3)
    assert b1 == {f"p{i}": {"reward": 1.0, "n": 1.0, "cost": 100.0} for i in range(3)}
    assert look_pt(job, 2, 3)[0] is None           # p0's 2nd launched trial not finished
    assert look_pt(job, 1, 4)[0] is None           # a task not launched at all yet
    # a later-launched trial that finished first is NOT used by look 1
    write_trial(job, tasks[0], 2, 3000, 0.0)
    assert look_pt(job, 1, 3)[0]["p0"]["reward"] == 1.0


def test_env_failed_trials_are_counted_but_not_scored(tmp_path):
    job, td = tmp_path / "job", tmp_path / "data" / "p0"
    write_trial(job, td, 0, 1000, 0.0, env_fail=True)
    write_trial(job, td, 1, 1001, 1.0)
    b, info = look_pt(job, 2, 1)
    assert b == {"p0": {"reward": 1.0, "n": 1.0, "cost": 100.0}}
    assert info == {"env_bad": 1, "n": 2}


def test_boundaries():
    cfg = SeqConfig(k=5)
    assert cfg.efficacy(1) == math.inf
    assert [round(cfg.efficacy(r), 2) for r in (2, 3, 4, 5)] == [2.72, 2.22, 1.92, 1.72]
    assert [cfg.fut(r) for r in (1, 2, 3, 4)] == [-0.5, 0.0, 0.5, 1.0]
    assert cfg.fut(5) == math.inf
    a = {f"p{i}": {"reward": 0.0} for i in range(20)}
    b = {f"p{i}": {"reward": 1.0 if i < 3 else 0.0} for i in range(20)}    # t = 1.83
    assert decide_look(a, b, 2, cfg).verdict == "continue"
    assert decide_look(a, b, 5, cfg).verdict == "pass"
    worse = {f"p{i}": {"reward": -1.0 if i < 3 else 0.0} for i in range(20)}
    assert decide_look(a, worse, 1, cfg).verdict == "fail"
    assert decide_look(a, a | {"p0": {"reward": 0.1}}, 2, cfg).verdict == "continue"   # t>0


# --------------------------------------------------------------------------- the loop
class RoundFake(Fake):
    """eval_a writes all K trials at once. eval_b reveals ONE attempt round per poll of its job
    (like harbor: every task once, then again, ...) and is DONE after the last round.
    `a_rw(task_index, trial)` / `b_rw(tag, task_index, trial)` give the rewards."""

    def __init__(self, run_dir, evals, a_rw, b_rw, **kw):
        super().__init__(run_dir, evals, **kw)
        self.a_rw, self.b_rw, self.rounds = a_rw, b_rw, {}

    def submit_eval(self, sched, task, *, model, harness, data, n_trials, n_concurrent,
                    out_dir, logs, job_name, base_model=None):
        if job_name not in ("eval_before", "eval_after"):
            return super().submit_eval(sched, task, model=model, harness=harness, data=data,
                                       n_trials=n_trials, n_concurrent=n_concurrent,
                                       out_dir=out_dir, logs=logs, job_name=job_name,
                                       base_model=base_model)
        stage = "eval_a" if job_name == "eval_before" else "eval_b"
        self.calls.append({"stage": stage, "harness": harness, "out": str(Path(out_dir) / job_name),
                           "log": logs})
        tasks = sorted(p for p in Path(data).iterdir() if p.is_dir())
        job_dir = Path(out_dir) / job_name
        jid = self._job()
        if stage == "eval_a":
            for k in range(n_trials):
                for i, td in enumerate(tasks):
                    write_trial(job_dir, td, k, 1000 * (k + 1) + i, self.a_rw(i, k))
            self._write_score(job_dir)
        else:
            tag = Path(harness).parent.name
            self.rounds[jid] = {"tag": tag, "tasks": tasks, "dir": job_dir, "done": 0,
                                "k": n_trials}
        return jid

    def _write_score(self, job_dir):
        rs = [json.loads(f.read_text())["verifier_result"]["rewards"]["reward"]
              for f in job_dir.glob("*/result.json")]
        (job_dir / "fake_score.json").write_text(json.dumps(sum(rs) / len(rs)))

    def state(self, jid):
        r = self.rounds.get(jid)
        if r is None:
            return super().state(jid)
        if r["done"] < r["k"]:
            k = r["done"]
            for i, td in enumerate(r["tasks"]):
                write_trial(r["dir"], td, k, 1000 * (k + 1) + i, self.b_rw(r["tag"], i, k))
            r["done"] += 1
            self._write_score(r["dir"])
            return "RUNNING", None
        return "DONE", 0


def seq_make(tmp_path, b_rw, *, proposals=1, iterations=1, val=(0.5, 0.6)):
    a_rw = lambda i, k: float((i + k) % 2)          # every task ~half solved by A
    plan = lambda stage, harness, n: list(val)[n]   # only val evals go through the plan
    fake = RoundFake(tmp_path / "run", plan, a_rw, b_rw)
    loop, fake = make(tmp_path, plan, iterations=iterations, proposals=proposals, fake=fake,
                      limit=1)
    loop.args.trials, loop.args.propose_sample = K, 20
    loop.args.seq_gate, loop.args.seq_c, loop.args.seq_futility = True, 1.72, "-0.5,0,0.5,1.0"
    return loop, fake


def seq_events(loop):
    return [json.loads(l) for l in loop.events_path.read_text().splitlines()
            if json.loads(l)["stage"] == "seq_look"]


def test_clearly_worse_candidate_stops_at_look_1(tmp_path):
    loop, fake = seq_make(tmp_path, lambda tag, i, k: 0.0)
    loop.run()
    rows = summary(loop)
    assert [r["outcome"] for r in rows] == ["baseline", "train_fail"]
    assert rows[1]["look"] == 1 and rows[1]["train_gate"]["seq"]["verdict"] == "fail"
    assert [e["seq"]["look"] for e in seq_events(loop)] == [1]
    assert (loop.cands / "candidate_1_0_1" / "eval_after.stop").exists()
    (jid,) = fake.rounds
    assert fake.rounds[jid]["done"] == 1           # stopped after the first attempt round


def test_clearly_better_candidate_passes_at_look_2_and_goes_to_val(tmp_path):
    loop, fake = seq_make(tmp_path, lambda tag, i, k: 1.0)
    loop.run()
    rows = summary(loop)
    assert [r["outcome"] for r in rows] == ["baseline", "accepted"]
    assert rows[1]["train_gate"]["seq"] == {"look": 2, "verdict": "pass", "bound": 2.72,
                                            "futility": 0.0}
    assert rows[1]["ts_B"] == 1.0
    assert [e["seq"]["verdict"] for e in seq_events(loop)] == ["continue", "pass"]
    assert [c["stage"] for c in fake.calls].count("val") == 2


def test_borderline_candidate_runs_to_the_end_and_uses_the_last_boundary(tmp_path):
    # B = A, plus 3 of 20 tasks always solved -> t ~ 1.8 at every look: above the futility
    # bounds, below the early efficacy bounds, above the final 1.72
    a_rw = lambda i, k: float((i + k) % 2)
    loop, fake = seq_make(tmp_path, lambda tag, i, k: 1.0 if i < 3 else a_rw(i, k))
    loop.run()
    rows = summary(loop)
    assert rows[1]["outcome"] == "accepted"
    tg = rows[1]["train_gate"]
    assert tg["seq"] == {"look": 5, "verdict": "pass", "bound": 1.72, "futility": None}
    assert [e["seq"]["verdict"] for e in seq_events(loop)] == ["continue"] * 4
    assert not (loop.cands / "candidate_1_0_1" / "eval_after.stop").exists()


def test_siblings_are_decided_independently(tmp_path):
    def b_rw(tag, i, k):
        return {"candidate_1_0_1a": 0.0, "candidate_1_0_1b": 1.0}[tag]
    loop, fake = seq_make(tmp_path, b_rw, proposals=2, iterations=2)
    loop.run()
    out = {r["candidate"]: (r["outcome"], r.get("look")) for r in summary(loop)[1:]}
    assert out == {"candidate_1_0_1a": ("train_fail", 1), "candidate_1_0_1b": ("accepted", None)}


def test_resubmit_clears_a_stale_stop_file(tmp_path):
    loop, fake = seq_make(tmp_path, lambda tag, i, k: 1.0)
    stale = loop.cands / "candidate_1_0_1" / "eval_after.stop"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.touch()
    seen = []
    orig = fake.submit_eval

    def spy(*a, **kw):
        if kw.get("job_name") == "eval_after":
            seen.append(stale.exists())
        return orig(*a, **kw)
    loop._submit_eval = spy
    loop.run()
    assert seen == [False]


def test_seq_off_keeps_the_fixed_gate(tmp_path):
    loop, fake = seq_make(tmp_path, lambda tag, i, k: 0.0)
    loop.args.seq_gate = False
    loop.run()
    rows = summary(loop)
    assert rows[1]["outcome"] == "train_fail" and "look" not in rows[1]
    assert seq_events(loop) == []
    (jid,) = fake.rounds
    assert fake.rounds[jid]["done"] == K
