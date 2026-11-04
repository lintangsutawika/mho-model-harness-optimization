"""State-machine tests for src/mho/loop.py with a fake scheduler (no PBS, no GPUs).

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

from mho.kinds import HarnessKind, ModelKind  # noqa: E402
from mho.loop import Loop, LoopAbort  # noqa: E402
from utils.scores import metric_of  # noqa: E402


class Fake:
    """Scheduler + submitters. `evals(stage, harness, call_no)` -> score or None (no trials);
    `proposes(tag)` -> True (write a harness) / False (fail)."""

    def __init__(self, run_dir: Path, evals, proposes=lambda tag: True, pending: int = 0,
                 trains=lambda tag: True):
        self.run_dir, self.evals, self.proposes, self.pending = run_dir, evals, proposes, pending
        self.trains = trains
        self.calls: list[dict] = []
        self.jobs: dict[str, dict] = {}

    # scheduler
    def host(self, jid):
        return "hnode023"

    def submit_hold(self, sched, host, seconds, logs):
        self.calls.append({"stage": "hold", "host": host, "seconds": seconds})
        return self._job()

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
                    out_dir, logs, job_name, base_model=None):
        stage = ("eval_a" if job_name == "eval_before" else "eval_b" if job_name == "eval_after"
                 else "val")
        n = sum(c["stage"] == stage for c in self.calls)
        score = self.evals(stage, harness, n)
        self.calls.append({"stage": stage, "harness": harness, "model": model,
                           "base_model": base_model, "data": data,
                           "out": str(Path(out_dir) / job_name), "score": score, "log": logs})
        if score == "envfail":               # every sandbox failed to start (bad node)
            for i in range(4):
                tr = Path(out_dir) / job_name / f"t{i}"
                tr.mkdir(parents=True, exist_ok=True)
                (tr / "result.json").write_text(json.dumps({
                    "task_name": f"x{i}", "agent_setup": None,
                    "exception_info": {"exception_type": "RuntimeError"},
                    "verifier_result": None}))
            (Path(out_dir) / job_name / "fake_score.json").write_text("0.0")
            score = 0.0
        elif isinstance(score, dict):        # per-task trials: {task: (reward, output_tokens)}
            for i, (task, (r, tok)) in enumerate(score.items()):
                t = Path(out_dir) / job_name / f"t{i}"
                t.mkdir(parents=True, exist_ok=True)
                (t / "result.json").write_text(json.dumps({
                    "task_name": task, "verifier_result": {"rewards": {"reward": r}},
                    "agent_result": {"n_output_tokens": tok}}))
            score = sum(r for r, _ in score.values()) / len(score)
            (Path(out_dir) / job_name / "fake_score.json").write_text(json.dumps(score))
        elif score is not None:
            t = Path(out_dir) / job_name / "t0"
            t.mkdir(parents=True, exist_ok=True)
            (t / "result.json").write_text("{}")
            (Path(out_dir) / job_name / "fake_score.json").write_text(json.dumps(score))
        return self._job()

    def submit_propose(self, sched, *, run_root, logs, tag, kind):
        ok = self.proposes(tag)
        self.calls.append({"stage": "propose", "tag": tag, "ok": ok, "log": logs})
        self.calls[-1]["kind"] = kind
        if ok and kind == "model":
            c = run_root / "candidates" / tag
            (c / "model_specs").mkdir(parents=True, exist_ok=True)
            (c / "model_specs" / "spec.md").write_text("spec")
            (c / "CANDIDATE.md").write_text(f"# Model Proposal\nreward idea for {tag}\n")
        elif ok:
            h = run_root / "candidates" / tag / "harness"
            h.mkdir(parents=True, exist_ok=True)
            (h / "pyproject.toml").write_text("")
            (h / "CANDIDATE.md").write_text(f"# CANDIDATE.md\nidea for {tag}\n")
        return self._job()

    def submit_train(self, sched, trainer, *, model, harness, out_dir, logs, tag,
                     sigma_reward=None, train_data=None):
        ok = self.trains(tag)
        self.calls.append({"stage": "train", "tag": tag, "model": model, "harness": harness,
                           "sigma": sigma_reward, "ok": ok, "log": logs})
        if ok:
            ck = Path(out_dir) / "checkpoints" / "hf" / "step_7"
            ck.mkdir(parents=True, exist_ok=True)
            (ck / "config.json").write_text("{}")
        return self._job()


def fake_score(d: Path) -> dict:
    return {"mean_reward": json.loads((Path(d) / "fake_score.json").read_text())}


def fake_realize(cand_dir, train_data):
    sr = Path(cand_dir) / "model" / "sigma_rewards"
    if not (Path(cand_dir) / "model_specs").is_dir():
        return False, None
    sr.mkdir(parents=True, exist_ok=True)
    (sr / "spec.json").write_text("{}")
    return True, sr


def make(tmp_path, evals, *, iterations=3, limit=3, proposes=lambda t: True, pending=0,
         fresh=False, fake=None, schedule=(("harness", 1),), trains=lambda t: True):
    train = tmp_path / "train"
    for i in range(20):
        (train / f"p{i}").mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(model="m", trials=1, concurrent=1, train_data=str(train),
                           val_data="VAL", propose_sample=4, iterations=iterations,
                           ts_repropose_limit=limit, fresh=fresh, skip_baseline=False)
    run_dir = tmp_path / "run"
    fake = fake or Fake(run_dir, evals, proposes, pending, trains)
    kinds = {"harness": HarnessKind(),
             "model": ModelKind(train_data=str(train), trainer="miles",
                                submit_train=fake.submit_train, realize=fake_realize)}
    loop = Loop(args, run_dir, fake, kinds=kinds, schedule=list(schedule),
                submit_eval=fake.submit_eval, submit_propose=fake.submit_propose,
                submit_hold=fake.submit_hold,
                score_fn=fake_score, sleep=lambda s: None, poll_seconds=0)
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
    best = json.loads((loop.run_dir / "best" / "pair.json").read_text())
    assert best["candidate"] == "candidate_1_0_1" and best["model"] == "m"
    assert best["harness"].endswith("candidate_1_0_1/harness")
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
    jobs = [e for e in map(json.loads, loop.events_path.read_text().splitlines()) if "rc" in e]
    assert [e["ok"] for e in jobs][:2] == [False, True]


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
    with pytest.raises(LoopAbort, match="older checkout"):
        make(tmp_path, None)


# --------------------------------------------------------------------------- scores
def test_metric_of_keeps_zero():
    assert metric_of({"mean_reward": 0.0, "solved_pass@1": 0.7}) == 0.0
    assert metric_of({"mean_reward": None, "solved_pass@1": 0.7}) == 0.7
    assert metric_of({}) == 0.0


# --------------------------------------------------------------------------- gates (utils.gate)
BASE = [1, 0] * 10                                    # 20 tasks, mean 0.5


def tasks(rewards, tokens=1000):
    return {f"t{i}": (float(r), tokens) for i, r in enumerate(rewards)}


def flip(rewards, up=(), down=()):
    out = list(rewards)
    for i in up:
        out[i] = 1
    for i in down:
        out[i] = 0
    return out


def test_train_gate_rejects_an_improvement_within_noise(tmp_path):
    # B > A on the mean (2 up, 1 down of 20 tasks) but t < 1: not worth a val run
    loop, fake = make(tmp_path, scripted(val=[0.5], a=[tasks(BASE)],
                                         b=[tasks(flip(BASE, up=[1, 3], down=[0]))]),
                      iterations=1)
    loop.run()
    row = summary(loop)[-1]
    assert row["outcome"] == "train_fail" and row["ts_B"] > row["ts_A"]
    assert row["train_gate"]["acc"]["n"] == 20 and row["train_gate"]["acc"]["t"] < 1
    assert [c["stage"] for c in fake.calls].count("val") == 1          # baseline only


def test_clear_improvement_is_accepted_as_better(tmp_path):
    better = flip(BASE, up=[1, 3, 5, 7, 9, 11, 13, 15])
    loop, _ = make(tmp_path, scripted(val=[tasks(BASE), tasks(better)], a=[tasks(BASE)],
                                      b=[tasks(better)]), iterations=1)
    loop.run()
    row = summary(loop)[-1]
    assert row["outcome"] == "accepted" and row["path"] == "better"
    assert row["train_gate"]["path"] == "better" and row["val_gate"]["acc"]["n"] == 20


def test_same_accuracy_fewer_tokens_is_accepted_as_cheaper(tmp_path):
    loop, _ = make(tmp_path, scripted(val=[tasks(BASE, 1000), tasks(BASE, 600)],
                                      a=[tasks(BASE, 1000)], b=[tasks(BASE, 600)]),
                   iterations=1)
    loop.run()
    row = summary(loop)[-1]
    assert row["outcome"] == "accepted" and row["path"] == "cheaper"
    assert row["train_gate"]["path"] == "cheaper"
    fr = json.loads(loop.frontier_path.read_text())
    assert fr["_best"]["candidate"] == "candidate_1_0_1"                # new parent
    assert fr["_anchor"]["candidate"] == "baseline"                     # accuracy bar unchanged
    assert (loop.run_dir / "best" / "harness" / "pyproject.toml").exists()


def test_cheaper_but_worse_is_rejected_on_val(tmp_path):
    worse = flip(BASE, down=[0, 2, 4])
    loop, _ = make(tmp_path, scripted(val=[tasks(BASE, 1000), tasks(worse, 600)],
                                      a=[tasks(BASE, 1000)], b=[tasks(BASE, 600)]),
                   iterations=1)
    loop.run()
    row = summary(loop)[-1]
    assert row["outcome"] == "val_fail" and row["val_gate"]["accept"] is False
    assert json.loads(loop.frontier_path.read_text())["_best"]["candidate"] == "baseline"


# --------------------------------------------------------------------------- model candidates
def test_model_candidate_trains_then_goes_straight_to_val(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5, 0.6, 0.55], a=[0.3, 0.3], b=[]),
                      iterations=2, schedule=[("model", 1)])
    loop.run()
    stages = [c["stage"] for c in fake.calls]
    assert stages[:5] == ["val", "eval_a", "propose", "train", "val"]
    assert "eval_b" not in stages                                  # no train gate for models
    train = next(c for c in fake.calls if c["stage"] == "train")
    assert train["model"] == "m" and train["harness"] is None       # from the current pair
    assert train["sigma"].endswith("candidate_1_0_1/model/sigma_rewards/spec.json")
    ckpt = str(loop.cands / "candidate_1_0_1" / "train" / "checkpoints" / "hf" / "step_7")
    v = fake.calls[4]
    assert v["model"] == ckpt and v["base_model"] == "m"
    assert fake.calls[1]["base_model"] is None                      # base model: no override
    row = summary(loop)[1]
    assert row["kind"] == "model" and row["outcome"] == "accepted" and row["pair"]["model"] == ckpt
    assert row["ts_B"] is None and row["train_gate"] is None
    # accepted -> the checkpoint is the current model: next A and the next training use it
    assert fake.calls[5]["stage"] == "eval_a" and fake.calls[5]["model"] == ckpt
    assert [c["model"] for c in fake.calls if c["stage"] == "train"][1] == ckpt
    assert json.loads((loop.run_dir / "best" / "pair.json").read_text())["model"] == ckpt
    assert "candidate_1_0_1-train.out" in [Path(c["log"]).name for c in fake.calls]


def test_rejected_model_is_never_trained_on(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5, 0.4, 0.45], a=[0.3, 0.3], b=[]),
                      iterations=2, schedule=[("model", 1)])
    loop.run()
    assert [r["outcome"] for r in summary(loop)] == ["baseline", "val_fail", "val_fail"]
    assert [c["model"] for c in fake.calls if c["stage"] == "eval_a"] == ["m", "m"]
    assert [c["model"] for c in fake.calls if c["stage"] == "train"] == ["m", "m"]
    assert loop.st["pair"] == {"harness": None, "model": None}


def test_failed_training_is_build_fail_then_next_try(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5, 0.4], a=[0.3], b=[]), iterations=2,
                      schedule=[("model", 1)], trains=lambda tag: tag != "candidate_1_0_1")
    loop.run()
    rows = summary(loop)
    assert [(r["candidate"], r["outcome"]) for r in rows[1:]] == [
        ("candidate_1_0_1", "build_fail"), ("candidate_1_0_2", "val_fail")]
    assert len([c for c in fake.calls if c["stage"] == "train"]) == 2   # no retry by default
    assert "no HF checkpoint" in rows[1]["reason"]


def test_model_harness_schedule_alternates_and_keeps_the_harness(tmp_path):
    """2 harness then 1 model, repeating. An accepted harness stays in the pair through the
    model candidate, and the accepted model candidate gets a copy of it for the proposer."""
    loop, fake = make(tmp_path, scripted(val=[0.5, 0.6, 0.7], a=[0.3] * 3, b=[0.5, 0.1, 0.1]),
                      iterations=4, limit=5, schedule=[("harness", 2), ("model", 1)])
    loop.run()
    rows = summary(loop)[1:]
    assert [(r["candidate"], r["kind"], r["outcome"]) for r in rows] == [
        ("candidate_1_0_1", "harness", "accepted"), ("candidate_2_1_1", "harness", "train_fail"),
        ("candidate_2_1_2", "model", "accepted"), ("candidate_3_2_1", "harness", "train_fail")]
    h1 = str(loop.cands / "candidate_1_0_1" / "harness")
    train = next(c for c in fake.calls if c["stage"] == "train")
    assert train["harness"] == h1                                      # model trains with it
    assert loop.st["pair"]["harness"] == h1 and loop.st["pair"]["model"].endswith("step_7")
    assert (loop.cands / "candidate_2_1_2" / "harness" / "pyproject.toml").exists()
    assert [c["kind"] for c in fake.calls if c["stage"] == "propose"] == [
        "harness", "harness", "model", "harness"]


def test_schema2_harness_state_is_migrated_on_resume(tmp_path):
    run = tmp_path / "run"
    (run / "candidates" / "candidate_1_0_1" / "harness").mkdir(parents=True)
    parent = str(run / "candidates" / "candidate_1_0_1" / "harness")
    (run / "state.json").write_text(json.dumps({
        "schema": 2, "run_id": "run", "mode": "harness-only", "stage": "eval_a", "iteration": 2,
        "version": 1, "try": 1, "propose_count": 1, "parent": parent, "subset": None,
        "ts_B": None, "job": None, "attempt": 1, "infra_fails": 0, "propose_fails": 0}))
    loop, fake = make(tmp_path, scripted(val=[0.7], a=[0.3], b=[0.5]), iterations=2)
    assert loop.st["schema"] == 3 and loop.st["pair"] == {"harness": parent, "model": None}
    loop.run()
    assert fake.calls[0]["stage"] == "eval_a" and fake.calls[0]["harness"] == parent


# --------------------------------------------------------------------------- bad nodes
def test_sandbox_failures_are_retried_on_another_node_not_scored(tmp_path):
    """Every trial failing before agent setup (a broken node) is infra: the eval is set aside
    and retried after parking the node, never recorded as the candidate's score."""
    loop, fake = make(tmp_path, scripted(val=[0.5, "envfail", 0.6], a=[0.3], b=[0.5]),
                      iterations=1)
    loop.run()
    assert [r["outcome"] for r in summary(loop)] == ["baseline", "accepted"]
    assert summary(loop)[1]["val"] == 0.6
    holds = [c for c in fake.calls if c["stage"] == "hold"]
    assert len(holds) == 1 and holds[0]["host"] == "hnode023"
    stages = [c["stage"] for c in fake.calls]
    assert stages[stages.index("hold") - 1] == "val" and stages[stages.index("hold") + 1] == "val"
    ev = loop.run_dir / "evaluation"
    assert (ev / "candidate_1_0_1.failed-attempt1").is_dir()           # evidence kept
    assert not (ev / "candidate_1_0_1" / "t3").exists()               # retry started clean
    assert list(loop.st["bad_hosts"]) == ["hnode023"]


def test_node_is_parked_once(tmp_path):
    loop, fake = make(tmp_path, scripted(val=[0.5], a=["envfail", "envfail", 0.3], b=[0.1]),
                      iterations=1)
    loop.run()
    assert [c["stage"] for c in fake.calls].count("hold") == 1
    assert [c["stage"] for c in fake.calls].count("eval_a") == 3
    assert summary(loop)[-1]["outcome"] == "train_fail"
