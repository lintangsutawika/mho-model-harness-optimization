"""Adaptive val eval (utils.alloc + the loop's val looks) with a fake scheduler whose evals draw
each trial from a per-task pass rate of the evaluated pair."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mho.kinds import HarnessKind  # noqa: E402
from mho.loop import Loop  # noqa: E402
from utils import alloc  # noqa: E402

N_TASKS = 40


# --------------------------------------------------------------------------- utils.alloc
def test_first_look_floor_cap_and_budget():
    w = {"dead": alloc.spread(0, 15), "coin": alloc.spread(7, 15), "some": alloc.spread(2, 15)}
    k = alloc.first_look(w, budget=7, cap=3)
    assert sum(k.values()) == 7 and min(k.values()) >= 1 and max(k.values()) <= 3
    assert k["coin"] >= k["some"] >= k["dead"] == 1


def test_neyman_adds_on_top_of_existing_trials_and_respects_cap():
    w = {"a": 0.5, "b": 0.5, "c": 0.1}
    add = alloc.neyman(w, 10, start={"a": 9, "b": 1, "c": 1}, cap=10)
    assert add["a"] == 1                      # a hits the cap
    assert add["b"] > add["c"] and sum(add.values()) == 10


def test_groups_and_fill():
    assert alloc.groups({"a": 1, "b": 3, "c": 1, "d": 0}) == {1: ["a", "c"], 3: ["b"]}
    assert alloc.fill_to({"a": 1, "b": 3}, ["a", "b", "c"], 3) == {"a": 2, "b": 0, "c": 3}


def test_task_dirs_maps_harbor_task_names(tmp_path):
    (tmp_path / "2508.1").mkdir()
    (tmp_path / "2508.1" / "task.toml").write_text('[task]\nname = "mho/x-2508.1"\n')
    (tmp_path / "plain").mkdir()
    assert alloc.task_dirs(tmp_path) == {"mho/x-2508.1": tmp_path / "2508.1",
                                         "plain": tmp_path / "plain"}


# --------------------------------------------------------------------------- fake
class Fake:
    """val evals: each trial ~ Bernoulli(rate[pair][task]); train evals: scalar scores."""

    def __init__(self, run_dir, rates, train_a=0.3, train_b=0.5):
        self.run_dir, self.rates = run_dir, rates
        self.train_a, self.train_b = train_a, train_b
        self.calls, self.jobs = [], {}

    def state(self, jid):
        return "DONE", 0

    def host(self, jid):
        return "hnode001"

    def _job(self):
        jid = f"{len(self.jobs) + 1}.pbs"
        self.jobs[jid] = 1
        return jid

    @staticmethod
    def key(harness):
        return "base" if harness is None else Path(harness).parent.name

    def submit_eval(self, sched, task, *, model, harness, data, n_trials, n_concurrent,
                    out_dir, logs, job_name, base_model=None):
        out = Path(out_dir) / job_name
        if job_name in ("eval_before", "eval_after"):
            score = self.train_a if job_name == "eval_before" else self.train_b
            (out / "t0").mkdir(parents=True, exist_ok=True)
            (out / "t0" / "result.json").write_text("{}")
            (out / "fake_score.json").write_text(json.dumps(score))
            self.calls.append({"stage": job_name, "harness": harness})
            return self._job()
        rates = self.rates[self.key(harness)]
        tasks = sorted(p.name for p in Path(data).iterdir() if p.is_dir())
        rng = random.Random(f"{self.key(harness)}/{job_name}")
        rs = []
        for t in tasks:
            for i in range(n_trials):
                r = 1.0 if rng.random() < rates[t] else 0.0
                rs.append(r)
                d = out / f"{t}__{i}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "result.json").write_text(json.dumps({
                    "task_name": t, "verifier_result": {"rewards": {"reward": r}},
                    "agent_result": {"n_output_tokens": 1000}}))
        (out / "fake_score.json").write_text(json.dumps(sum(rs) / len(rs)))
        self.calls.append({"stage": "val", "job": job_name, "harness": harness, "n": n_trials,
                           "tasks": tasks})
        return self._job()

    def submit_propose(self, sched, *, run_root, logs, tag, kind, sibling=(1, 1)):
        h = run_root / "candidates" / tag / "harness"
        h.mkdir(parents=True, exist_ok=True)
        (h / "pyproject.toml").write_text("")
        self.calls.append({"stage": "propose", "tag": tag})
        return self._job()


def fake_score(d):
    return {"mean_reward": json.loads((Path(d) / "fake_score.json").read_text())}


def rates(base, **cands):
    out = {"base": base}
    out.update({f"candidate_{k}": v for k, v in cands.items()})
    return out


def task_names():
    return [f"t{i:02d}" for i in range(N_TASKS)]


def make(tmp_path, rate_table, *, iterations=1, topup=0.5, budget=0.67, fake=None):
    train = tmp_path / "train"
    for i in range(8):
        (train / f"p{i}").mkdir(parents=True, exist_ok=True)
    val = tmp_path / "val"
    for t in task_names():
        (val / t).mkdir(parents=True, exist_ok=True)
        (val / t / "task.toml").write_text(f'[task]\nname = "{t}"\n')
    args = SimpleNamespace(model="m", trials=3, concurrent=1, train_data=str(train),
                           val_data=str(val), propose_sample=4, iterations=iterations,
                           ts_repropose_limit=2, fresh=False, skip_baseline=False, proposals=1,
                           val_adaptive=True, val_budget=budget, val_topup=topup, val_cap=10,
                           gate_early_z=2.5, gate_z=1.645)
    run = tmp_path / "run"
    fake = fake or Fake(run, rate_table)
    loop = Loop(args, run, fake, kinds={"harness": HarnessKind()}, schedule=[("harness", 1)],
                submit_eval=fake.submit_eval, submit_propose=fake.submit_propose,
                score_fn=fake_score, sleep=lambda s: None, poll_seconds=0)
    return loop, fake


def summary(loop):
    return [json.loads(l) for l in loop.summary_path.read_text().splitlines()]


def val_calls(fake, prefix):
    return [c for c in fake.calls if c["stage"] == "val" and c["job"].startswith(prefix)]


# mixed task difficulty: 20 never solved, 10 always, 10 coin flips
BASE = {t: (0.0 if i < 20 else 1.0 if i < 30 else 0.5) for i, t in enumerate(task_names())}


def shifted(delta):
    return {t: min(1.0, p + delta) if 0 < p < 1 else p for t, p in BASE.items()}


# --------------------------------------------------------------------------- the loop
def seed_history(tmp_path, n=12):
    """Earlier val evals of the base pair (what the loop has after a few candidates)."""
    rng = random.Random(0)
    for t, p in BASE.items():
        for i in range(n):
            d = tmp_path / "run" / "evaluation" / "older_eval" / f"{t}__{i}"
            d.mkdir(parents=True, exist_ok=True)
            r = 1.0 if (p == 1.0 or (0 < p < 1 and i % 2 == 0)) else 0.0
            (d / "result.json").write_text(json.dumps({
                "task_name": t, "verifier_result": {"rewards": {"reward": r}}}))


def test_look1_spends_the_budget_where_outcomes_vary(tmp_path):
    seed_history(tmp_path)
    loop, fake = make(tmp_path, rates(BASE, **{"1_0_1": BASE}))
    loop.run()
    look1 = [c for c in fake.calls if c["stage"] == "val"
             and (c["job"] == "candidate_1_0_1" or c["job"].startswith("candidate_1_0_1.a"))]
    per_task = {}
    for c in look1:
        for t in c["tasks"]:
            per_task[t] = per_task.get(t, 0) + c["n"]
    assert set(per_task) == set(task_names())                 # every task at least once
    assert sum(per_task.values()) == round(0.67 * 3 * N_TASKS)
    coin = [per_task[t] for i, t in enumerate(task_names()) if i >= 30]
    dead = [per_task[t] for i, t in enumerate(task_names()) if i < 30]    # never / always solved
    # uncertainty decides: coin flips get the cap first; the settled tasks share what is left
    assert min(coin) == 3 and sum(dead) / len(dead) < 2 and max(dead) <= 2


def test_clear_winner_is_accepted_at_look1_and_filled(tmp_path):
    win = {t: (0.0 if i < 20 else 1.0) for i, t in enumerate(task_names())}   # solves the coins
    loop, fake = make(tmp_path, rates(BASE, **{"1_0_1": win}))
    loop.run()
    row = summary(loop)[-1]
    assert row["outcome"] == "accepted" and row["look"] == 1
    assert not [c for c in fake.calls if ".t-" in c.get("job", "")]           # no look 2
    # filled to a uniform 3 trials on every task -> reported score is the uniform one
    pt = loop._val_pt("candidate_1_0_1")
    assert {r["n"] for r in pt.values()} == {3.0}
    fr = json.loads(loop.frontier_path.read_text())
    assert fr["_best"] == {"candidate": "candidate_1_0_1", "metric": pytest.approx(0.5)}


def test_clear_loser_is_rejected_at_look1(tmp_path):
    lose = {t: (0.0 if i < 20 or i >= 30 else 1.0) for i, t in enumerate(task_names())}
    loop, fake = make(tmp_path, rates(BASE, **{"1_0_1": lose}))
    loop.run()
    row = summary(loop)[-1]
    assert row["outcome"] == "val_fail" and row["look"] == 1
    assert not [c for c in fake.calls if ".t-" in c.get("job", "")]


def test_borderline_gets_look2_for_candidate_and_best(tmp_path):
    loop, fake = make(tmp_path, rates(BASE, **{"1_0_1": shifted(0.1)}))
    loop.run()
    row = summary(loop)[-1]
    assert row["look"] == 2
    best_topups = [c for c in fake.calls if c.get("job", "").startswith("baseline.t-1_0_1")]
    cand_topups = val_calls(fake, "candidate_1_0_1.b")
    assert best_topups and cand_topups
    added = lambda cs: sum(c["n"] * len(c["tasks"]) for c in cs)
    assert added(best_topups) == added(cand_topups) == round(0.5 * 3 * N_TASKS)
    # the best's top-up is part of its record from now on
    assert any(d.name.startswith("baseline.t-") for d in loop._val_dirs("baseline"))
    # look 2 is decided at z + 0.05
    assert "need > 1.695" in row["val_gate"]["reason"] or row["outcome"] == "accepted"


def test_look1_allocation_does_not_depend_on_the_candidate(tmp_path):
    plans = []
    for name, table in (("a", BASE), ("b", shifted(0.4))):
        loop, fake = make(tmp_path / name, rates(BASE, **{"1_0_1": table}))
        loop.run()
        plans.append(sorted((c["n"], tuple(c["tasks"])) for c in val_calls(fake, "candidate_1_0_1.a")))
    assert plans[0] == plans[1]


def test_no_topup_means_one_look_at_gate_z(tmp_path):
    loop, fake = make(tmp_path, rates(BASE, **{"1_0_1": shifted(0.1)}), topup=0)
    loop.run()
    row = summary(loop)[-1]
    assert row["look"] == 1 and not [c for c in fake.calls if ".t-" in c.get("job", "")]
    assert "1.645" in row["val_gate"]["reason"] or row["outcome"] == "accepted"
