"""utils.gate: per-task parsing, paired statistics, and the better / cheaper decisions."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.gate import GateConfig, decide, decide_dirs, paired, per_task  # noqa: E402

CFG = GateConfig()


def tasks(rewards, cost=1000.0):
    """{task: {"reward", "cost"}} from a list of per-task rewards (cost scalar or list)."""
    costs = cost if isinstance(cost, list) else [cost] * len(rewards)
    return {f"t{i}": {"reward": float(r), "cost": float(c), "n": 3.0}
            for i, (r, c) in enumerate(zip(rewards, costs))}


BASE = [1, 0] * 10                                   # 20 tasks, mean 0.5


def flip(rewards, up=(), down=()):
    out = list(rewards)
    for i in up:
        out[i] = 1
    for i in down:
        out[i] = 0
    return out


# --------------------------------------------------------------------------- statistics
def test_paired_difference_and_se():
    p = paired(tasks(BASE), tasks(flip(BASE, up=[1, 3])))
    assert p.n == 20 and p.mean_a == 0.5 and p.d == pytest.approx(0.1)
    assert p.se == pytest.approx(math.sqrt((2 * 0.9 ** 2 + 18 * 0.1 ** 2) / 19) / math.sqrt(20))


def test_paired_needs_two_shared_tasks():
    assert paired({"a": {"reward": 1.0}}, {"a": {"reward": 0.0}}) is None
    assert paired(tasks(BASE), {}) is None


# --------------------------------------------------------------------------- better path
def test_clear_improvement_is_better():
    d = decide(tasks(BASE), tasks(flip(BASE, up=[1, 3, 5, 7, 9, 11, 13, 15])), z=1.645, cfg=CFG)
    assert d.accept and d.path == "better"


def test_noisy_improvement_is_not_better():
    # B > A on the mean (two up, one down) but well within noise
    d = decide(tasks(BASE), tasks(flip(BASE, up=[1, 3], down=[0])), z=1.0, cfg=CFG)
    assert not d.accept and d.acc.d == pytest.approx(0.05)


def test_z_zero_is_plain_strict_comparison():
    up = tasks(flip(BASE, up=[1, 3], down=[0]))
    assert decide(tasks(BASE), up, z=0, cfg=CFG).accept
    assert not decide(tasks(BASE), tasks(BASE), z=0, cfg=GateConfig(cheaper_delta=0)).accept


# --------------------------------------------------------------------------- cheaper path
def test_same_accuracy_and_clearly_cheaper_is_accepted():
    d = decide(tasks(BASE, 1000), tasks(BASE, 600), z=1.645, cfg=CFG)
    assert d.accept and d.path == "cheaper" and d.extra["cost_gain"] == pytest.approx(0.4)


def test_cheaper_but_less_accurate_is_rejected():
    d = decide(tasks(BASE, 1000), tasks(flip(BASE, down=[0, 2, 4]), 600), z=1.645, cfg=CFG)
    assert not d.accept and "non-inferior" in d.reason


def test_cheaper_by_too_little_is_rejected():
    d = decide(tasks(BASE, 1000), tasks(BASE, 950), z=1.645, cfg=CFG)
    assert not d.accept                                # 5% < min_gain 10%


def test_noisy_cost_saving_is_rejected():
    # the whole saving comes from one task: t = 1.0, not > cheaper_z
    d = decide(tasks(BASE, 1000), tasks(BASE, [1000] * 19 + [0]), z=1.645,
               cfg=GateConfig(cheaper_min_gain=0.0))
    assert not d.accept and d.cost.se > 0


def test_anchor_blocks_drift():
    # B is non-inferior to its (weaker) parent A but clearly worse than the anchor
    anchor = tasks(flip(BASE, up=[1, 3, 5, 7, 9]))
    d = decide(tasks(BASE, 1000), tasks(BASE, 600), z=1.645, cfg=CFG, anchor=anchor)
    assert not d.accept and d.anchor is not None
    assert decide(tasks(BASE, 1000), tasks(BASE, 600), z=1.645, cfg=CFG, anchor=tasks(BASE)).accept


def test_cheaper_path_can_be_disabled():
    assert not decide(tasks(BASE, 1000), tasks(BASE, 600), z=1.645,
                      cfg=GateConfig(cheaper_delta=0)).accept


# --------------------------------------------------------------------------- fallback + io
def test_fallback_to_scalar_without_per_task_data():
    assert decide({}, {}, z=1.645, cfg=CFG, fallback=(0.3, 0.4)).accept
    assert not decide({}, {}, z=1.645, cfg=CFG, fallback=(0.3, 0.3)).accept
    assert decide({}, {}, z=1.645, cfg=CFG, fallback=(None, 0.0)).accept   # nothing to beat


def _trial(d: Path, name: str, task: str, reward, tokens=None, secs=None):
    t = d / name
    t.mkdir(parents=True)
    j = {"task_name": task, "verifier_result": None if reward is None else
         {"rewards": {"reward": reward}},
         "agent_result": {"n_output_tokens": tokens}}
    if secs is not None:
        j["agent_execution"] = {"started_at": "2026-11-04T00:00:00Z",
                                "finished_at": f"2026-11-04T00:00:{secs:02d}Z"}
    (t / "result.json").write_text(json.dumps(j))


def test_per_task_averages_attempts(tmp_path):
    _trial(tmp_path, "a1", "x", 1.0, 100, 10)
    _trial(tmp_path, "a2", "x", None, 300, 30)         # no verifier result -> reward 0
    _trial(tmp_path, "b1", "y", 0.0, None)             # no tokens -> no cost
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "result.json").write_text("{}")
    pt = per_task(tmp_path)
    assert pt["x"] == {"reward": 0.5, "n": 2.0, "cost": 200.0}
    assert pt["y"] == {"reward": 0.0, "n": 1.0}
    assert per_task(tmp_path, "seconds")["x"]["cost"] == 20.0


def test_decide_dirs_ignores_anchor_equal_to_reference(tmp_path):
    for i, r in enumerate(BASE):
        _trial(tmp_path / "A", f"t{i}", f"t{i}", float(r), 1000)
        _trial(tmp_path / "B", f"t{i}", f"t{i}", float(r), 500)
    d = decide_dirs(tmp_path / "A", tmp_path / "B", z=1.645, cfg=CFG, anchor_dir=tmp_path / "A")
    assert d.accept and d.path == "cheaper" and d.anchor is None
