"""State-machine tests for src/harness_loop.py with a fake scheduler (no PBS, no GPUs).

Each fake eval writes <out_dir>/<job_name>/t0/result.json plus the score the test's `plan`
chose; each fake propose writes candidates/<tag>/harness/pyproject.toml + CANDIDATE.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_loop import HarnessLoop, LoopAbort  # noqa: E402
from utils.scores import metric_of  # noqa: E402


class Fake:
    """Scheduler + submitters. `evals(stage, harness, call_no)` -> score or None (no trials);
    `proposes(tag)` -> True (write a harness) / False (fail)."""

    def __init__(self, run_dir: Path, evals, proposes=lambda tag: True, pending: int = 0):
        self.run_dir, self.evals, self.proposes, self.pending = run_dir, evals, proposes, pending
        self.calls: list[dict] = []
        self.jobs: dict[str, dict] = {}

    # scheduler
    def state(self, jid):
        j = self.jobs[jid]
        if j["polls"] < self.pending:
            j["polls"] += 1
            return "RUNNING", None
        return "DONE", j["rc"]

    def _job(self, rc=0):
        jid = f"{len(self.jobs) + 1}.pbs"
        self.jobs[jid] = {"rc": rc, "polls": 0}
        return jid

    # submitters (same signatures as optimize_loop.submit_eval / submit_propose)
    def submit_eval(self, sched, task, *, model, harness, data, n_trials, n_concurrent,
                    out_dir, logs, job_name):
        stage = ("eval_a" if job_name == "eval_before" else "eval_b" if job_name == "eval_after"
                 else "val")
        n = sum(c["stage"] == stage for c in self.calls)
        score = self.evals(stage, harness, n)
        self.calls.append({"stage": stage, "harness": harness, "data": data,
                           "out": str(Path(out_dir) / job_name), "score": score, "log": logs})
        if score is not None:
            t = Path(out_dir) / job_name / "t0"
            t.mkdir(parents=True, exist_ok=True)
            (t / "result.json").write_text("{}")
            (Path(out_dir) / job_name / "fake_score.json").write_text(json.dumps(score))
        return self._job()

    def submit_propose(self, sched, *, run_root, logs, tag, kind):
        ok = self.proposes(tag)
        self.calls.append({"stage": "propose", "tag": tag, "ok": ok, "log": logs})
        if ok:
            h = run_root / "candidates" / tag / "harness"
            h.mkdir(parents=True, exist_ok=True)
            (h / "pyproject.toml").write_text("")
            (h / "CANDIDATE.md").write_text(f"# CANDIDATE.md\nidea for {tag}\n")
        return self._job()


def fake_score(d: Path) -> dict:
    return {"mean_reward": json.loads((Path(d) / "fake_score.json").read_text())}


def make(tmp_path, evals, *, iterations=3, limit=3, proposes=lambda t: True, pending=0,
         fresh=False, fake=None):
    train = tmp_path / "train"
    for i in range(20):
        (train / f"p{i}").mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(model="m", trials=1, concurrent=1, train_data=str(train),
                           val_data="VAL", propose_sample=4, iterations=iterations,
                           ts_repropose_limit=limit, fresh=fresh, skip_baseline=False)
    run_dir = tmp_path / "run"
    fake = fake or Fake(run_dir, evals, proposes, pending)
    loop = HarnessLoop(args, run_dir, fake, submit_eval=fake.submit_eval,
                       submit_propose=fake.submit_propose, score_fn=fake_score,
                       sleep=lambda s: None, poll_seconds=0)
    return loop, fake


def summary(loop):
    return [json.loads(l) for l in loop.summary_path.read_text().splitlines()]


def scripted(val, a, b):
    """evals plan from per-stage score lists (indexed by call number within the stage)."""
    return lambda stage, harness, n: {"val": val, "eval_a": a, "eval_b": b}[stage][n]


# --------------------------------------------------------------------------- happy path
def test_accept_then_next_A_uses_accepted_parent(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5, 0.6], a=[0.3, 0.4], b=[0.5, 0.4]),
                      iterations=2)
    loop.run()
    rows = summary(loop)
    assert [r["outcome"] for r in rows] == ["baseline", "accepted", "train_fail"]
    acc = rows[1]
    assert acc["candidate"] == "candidate_1_0_1" and acc["val"] == 0.6 and acc["prev_best"] == 0.5
    assert acc["ts_A"] == 0.3 and acc["ts_B"] == 0.5 and isinstance(acc["subset_seed"], int)
    # status.json == the summary row (one terminal record per candidate)
    assert json.loads((loop.cands / "candidate_1_0_1" / "status.json").read_text()) == acc
    fr = json.loads(loop.frontier_path.read_text())
    assert fr["_best"] == {"candidate": "candidate_1_0_1", "metric": 0.6}
    assert (loop.run_dir / "best" / "harness" / "pyproject.toml").exists()
    # iteration 2: A is the ACCEPTED harness, and the candidate tag bumped version, reset try
    a_calls = [c for c in fake.calls if c["stage"] == "eval_a"]
    assert a_calls[0]["harness"] is None
    assert a_calls[1]["harness"].endswith("candidate_1_0_1/harness")
    assert rows[2]["candidate"] == "candidate_2_1_1"


def test_A_after_val_reject_is_still_the_parent(tmp_path):
    """Regression: the legacy loop evaluated the REJECTED candidate as next iteration's A."""
    loop, fake = make(tmp_path, scripted(val=[0.5, 0.4], a=[0.3, 0.3], b=[0.5, 0.1]),
                      iterations=2)
    loop.run()
    assert [r["outcome"] for r in summary(loop)] == ["baseline", "val_fail", "train_fail"]
    a_calls = [c for c in fake.calls if c["stage"] == "eval_a"]
    assert [c["harness"] for c in a_calls] == [None, None]          # base both times
    assert summary(loop)[2]["candidate"] == "candidate_2_0_2"       # try bumped after reject


def test_tie_fails_train_gate(tmp_path):
    loop, _ = make(tmp_path, scripted(val=[0.5], a=[0.3], b=[0.3]), iterations=1)
    loop.run()
    assert summary(loop)[-1]["outcome"] == "train_fail"


# --------------------------------------------------------------------------- re-proposals
def test_reproposes_on_same_subset_then_fresh_subset(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5], a=[0.3, 0.3], b=[0.1] * 4),
                      iterations=4, limit=3)
    loop.run()
    rows = summary(loop)[1:]
    assert [r["candidate"] for r in rows] == [
        "candidate_1_0_1", "candidate_1_0_2", "candidate_1_0_3", "candidate_2_0_4"]
    assert all(r["outcome"] == "train_fail" for r in rows)
    seeds = [r["subset_seed"] for r in rows]
    assert seeds[0] == seeds[1] == seeds[2] != seeds[3]               # fresh subset after 3
    b_data = [c["data"] for c in fake.calls if c["stage"] == "eval_b"]
    assert b_data[0] == b_data[1] == b_data[2] != b_data[3]
    assert len([c for c in fake.calls if c["stage"] == "eval_a"]) == 2  # A not re-evaluated
    # every try is self-contained: eval_before links to the subset's A traces
    link = loop.cands / "candidate_1_0_2" / "eval_before"
    assert link.is_symlink() and (link / "t0" / "result.json").exists()


def test_failed_proposal_is_recorded_and_not_charged(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5], a=[0.3], b=[0.1, 0.1]), iterations=2,
                      limit=2, proposes=lambda tag: tag != "candidate_1_0_1")
    loop.run()
    rows = summary(loop)
    # the failed proposal neither uses the budget nor a slot on the subset
    assert [(r["candidate"], r["outcome"]) for r in rows[1:]] == [
        ("candidate_1_0_1", "no_proposal"), ("candidate_1_0_2", "train_fail"),
        ("candidate_1_0_3", "train_fail")]
    assert rows[1]["subset_seed"] == rows[2]["subset_seed"] == rows[3]["subset_seed"]
    assert loop.st["propose_count"] == 2


def test_proposer_down_aborts_after_consecutive_failures(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5], a=[0.3], b=[]), iterations=2,
                      proposes=lambda tag: False)
    with pytest.raises(LoopAbort, match="consecutive proposals"):
        loop.run()
    assert [r["outcome"] for r in summary(loop)[1:]] == ["no_proposal"] * 3
    assert loop.st["propose_count"] == 0


def test_one_log_file_per_job_under_run_logs(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[None, 0.5, 0.6], a=[0.3], b=[0.5]), iterations=1)
    loop.run()
    logs = [Path(c["log"]).relative_to(loop.run_dir).as_posix() for c in fake.calls]
    assert logs == ["logs/baseline.out", "logs/baseline-attempt2.out",
                    "logs/candidate_1_0_1-eval_before.out", "logs/candidate_1_0_1-propose.out",
                    "logs/candidate_1_0_1-eval_after.out", "logs/candidate_1_0_1-val.out"]


# --------------------------------------------------------------------------- budget
def test_last_proposal_is_fully_evaluated(tmp_path):
    """Regression: the legacy loop exited right after submitting the final propose."""
    loop, fake = make(tmp_path, scripted(val=[0.5, 0.7], a=[0.3], b=[0.5]), iterations=1)
    loop.run()
    assert [c["stage"] for c in fake.calls] == ["val", "eval_a", "propose", "eval_b", "val"]
    assert summary(loop)[-1]["outcome"] == "accepted"
    assert loop.st["propose_count"] == 1


# --------------------------------------------------------------------------- infra failures
def test_baseline_failure_aborts_after_attempts(tmp_path):
    loop, fake = make(tmp_path, lambda stage, h, n: None)
    with pytest.raises(LoopAbort, match="baseline"):
        loop.run()
    assert len(fake.calls) == 3
    st = json.loads(loop.state_path.read_text())
    assert st["stage"] == "baseline" and st["job"] is None    # restart retries the baseline


def test_eval_b_no_score_then_abort_after_consecutive(tmp_path):
    def evals(stage, h, n):
        return {"val": 0.5, "eval_a": 0.3}.get(stage)          # eval_b never produces trials
    loop, fake = make(tmp_path, evals, iterations=10, limit=5)
    with pytest.raises(LoopAbort, match="3 consecutive"):
        loop.run()
    rows = summary(loop)
    assert [r["outcome"] for r in rows] == ["baseline", "no_score", "no_score"]
    assert len([c for c in fake.calls if c["stage"] == "eval_b"]) == 9   # 3 stages x 3 attempts


def test_transient_failure_is_retried_not_recorded(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[None, 0.5, 0.4], a=[0.3], b=[0.4]), iterations=1)
    loop.run()
    assert [r["outcome"] for r in summary(loop)] == ["baseline", "val_fail"]
    assert [json.loads(l)["ok"] for l in loop.events_path.read_text().splitlines()][:2] == [
        False, True]


# --------------------------------------------------------------------------- resume
def test_resume_reattaches_in_flight_job(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5, 0.6], a=[0.3], b=[0.5]),
                      iterations=1, pending=2)
    loop.step()                                     # submit baseline
    jid = loop.st["job"]["id"]
    # driver dies; a new driver on the same run dir re-attaches instead of resubmitting
    loop2, _ = make(tmp_path, None, iterations=1, fake=fake)
    assert loop2.st["job"]["id"] == jid
    loop2.run()
    assert [c["stage"] for c in fake.calls].count("val") == 2      # baseline once + candidate
    assert summary(loop2)[-1]["outcome"] == "accepted"


def test_legacy_state_is_refused(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"iteration": 29, "actions": []}))
    with pytest.raises(LoopAbort, match="MHO_LEGACY_LOOP"):
        make(tmp_path, None)


# --------------------------------------------------------------------------- scores
def test_metric_of_keeps_zero():
    assert metric_of({"mean_reward": 0.0, "solved_pass@1": 0.7}) == 0.0
    assert metric_of({"mean_reward": None, "solved_pass@1": 0.7}) == 0.7
    assert metric_of({}) == 0.0
