import math

import pytest

from harnessforge.bench import FEATURE_NAMES, builder, metrics
from harnessforge.bench.tasks import generate
from harnessforge.records import TrialRecord


def rec(c, t, i, r):
    return TrialRecord(candidate=c, task=t, index=i, finished=True, reward=r)


def test_gap_closed_and_accuracy():
    rs = [rec("a", "t1", 0, 1.0), rec("a", "t1", 1, 0.0), rec("a", "t2", 0, 1.0)]
    assert metrics.accuracy(rs, "a") == pytest.approx(0.75)
    assert metrics.gap_closed(0.2, 0.6, 0.5) == pytest.approx(0.75)
    assert math.isnan(metrics.gap_closed(0.5, 0.5, 0.5))


def test_paired_ablation_effects():
    rs = []
    for k, t in enumerate(["h1", "h2", "c1", "c2"]):
        rs += [rec("gold", t, 0, 1.0), rec("gold", t, 1, 1.0)]
        rs += [rec("ablate_output_truncation", t, 0, 0.0 if t.startswith("h") else 1.0),
               rec("ablate_output_truncation", t, 1, 0.5 if t.startswith("h") else 1.0)]
    fams = {"h1": "huge_log", "h2": "huge_log", "c1": "control", "c2": "control"}
    eff = metrics.ablation_effects(rs, families=fams)
    assert set(eff) == {"output_truncation"}
    assert eff["output_truncation"]["family"].delta == pytest.approx(0.75)
    assert eff["output_truncation"]["all"].delta == pytest.approx(0.375)
    assert eff["output_truncation"]["all"].n_tasks == 4


def test_discovery_timeline_and_curve():
    S = metrics.LineageStep
    steps = [S("c1", frozenset({"loop_warning"}), False, 1, 100),
             S("c2", frozenset({"loop_warning", "context_trim"}), True, 2, 250),
             S("c3", frozenset({"context_trim", "raw_observation"}), True, 3, 400)]
    tl = metrics.discovery_timeline(steps)
    assert tl["loop_warning"].first_proposed.candidate == "c1"
    assert tl["loop_warning"].first_kept.trials == 250 and tl["loop_warning"].lost
    assert tl["raw_observation"].first_kept.proposals == 3 and not tl["raw_observation"].lost
    assert tl["persistent_cwd"].first_proposed is None
    assert metrics.discovery_curve(steps) == [(100, 0), (250, 2), (400, 2)]


def test_experiment_plans_are_runnable_definitions(tmp_path):
    generate(tmp_path / "tasks", per_family=3, seed=0)
    plans = metrics.experiments(tmp_path / "h", tmp_path / "tasks", trials=4, targeted_trials=8)
    by = {p.name: p for p in plans}
    assert set(by) == {"gap_start_vs_gold", "ablation_all", *(f"ablation_{f}" for f in FEATURE_NAMES)}
    assert [c.id for c in by["gap_start_vs_gold"].candidates] == ["gold", "start"]
    assert len(by["ablation_all"].candidates) == 11 and len(by["ablation_all"].tasks) == 30
    t = by["ablation_loop_warning"]
    assert {x.split("-")[0] for x in t.tasks} == {"stuck_poll", "control"} and t.trials == 8
    for p in plans:
        for harness_dir, tasks, trials in p.units():
            assert (harness_dir / "src" / "micro_swe" / "agents" / "default.py").exists()
            assert tasks == p.tasks and trials == p.trials
    req = by["ablation_all"].request("http://pool:8000/v1", policy="qwen3.5-4b")
    assert req.trials == 4 and req.dataset == str(tmp_path / "tasks") and len(req.candidates) == 11
    assert metrics.experiments(tmp_path / "h", tmp_path / "tasks", trials=4)[0].tasks == plans[0].tasks


def test_discovery_report_on_start(tmp_path):
    start = builder.build((), tmp_path / "start")
    rep = metrics.discovery_report(start, features=["raw_observation", "persistent_cwd"])
    assert rep.n_present == 0 and rep.to_dict()["candidate"] == "start"
    assert "0/2" in rep.summary()
