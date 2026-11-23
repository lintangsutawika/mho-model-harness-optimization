"""scripts/optimize/hfloop_wandb.py: reading an hfloop run dir into the frontier stair + proposal points (no W&B)."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hfloop_wandb", ROOT / "scripts" / "optimize" / "hfloop_wandb.py")
hw = importlib.util.module_from_spec(spec)
sys.modules["hfloop_wandb"] = hw
spec.loader.exec_module(hw)

T0 = 1_800_000_000.0
TASKS = [f"t{i}" for i in range(4)]


def _touch(p: Path, t: float, text: str = "{}") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    os.utime(p, (t, t))


def _ref(run: Path, tag: str, rate: float, t: float) -> None:
    n = int(round(rate * 4))
    lines = [json.dumps({"task": task, "reward": 1.0 if i < n else 0.0}) for task in TASKS for i in range(4)]
    _touch(run / "ref" / tag / "records.jsonl", t, "\n".join(lines) + "\n")
    _touch(run / "ref" / tag / "status.json", t)


def _run(tmp: Path) -> Path:
    run = tmp / "hfloop-toy-x"
    _touch(run / "evals" / "val-candidate_0.request.json", T0)
    _ref(run, "candidate_0", 0.25, T0 + 1800)                      # 25 %, done at 0.5 h
    _ref(run, "candidate_1_1", 0.5, T0 + 3 * 3600)                 # 50 %, accept landed at 3 h
    states = {"candidate_1_1": {"verdict": "pass", "look": 2, "acc": {"d": 0.2, "se": 0.02, "t": 10}},
              "candidate_1_2": {"verdict": "fail", "look": 1, "acc": {"d": -0.05, "se": 0.03, "t": -1.6}}}
    race = {"reference": "candidate_0", "started_at": T0 + 3600, "elapsed_s": 3600,
            "decided_at_s": {"candidate_1_2": 900, "candidate_1_1": 1800},
            "decisions": {"states": states, "winner": "candidate_1_1", "done": True}}
    _touch(run / "rounds" / "1" / "race.json", T0 + 7200, json.dumps(race))
    # round 2: started proposing (overlap state), race not yet written
    st = {"round": 2, "reference": "candidate_1_1", "history": [{"accepted": "candidate_1_1"}],
          "rounds": {"1": {"accepted": "candidate_1_1"}, "2": {"propose_started_at": T0 + 2.5 * 3600,
                                                               "superseded": [{"tags": ["candidate_2_1"]}]}}}
    _touch(run / "hfloop.json", T0, json.dumps(st))
    _touch(run / "candidates" / "candidate_1_2" / "CANDIDATE.md", T0, "# candidate_1_2 — a stall detector\n\nbody")
    return run


def test_collect_stair_points_and_bands(tmp_path):
    snap = hw.collect(_run(tmp_path))
    assert snap.frontier == [(0.5, 25.0, "candidate_0"), (3.0, 50.0, "candidate_1_1")]
    p = {r["tag"]: r for r in snap.proposals}
    assert p["candidate_1_2"]["hours"] == 1.25 and p["candidate_1_2"]["est"] == 20.0      # 25 % + (-5) pts
    assert p["candidate_1_2"]["status"] == "rejected" and p["candidate_1_2"]["title"] == "candidate_1_2 — a stall detector"
    assert p["candidate_1_1"]["status"] == "accepted" and p["candidate_1_1"]["accept_val"] == 50.0
    assert snap.rounds[0] == {"round": 1, "start": 0.5, "end": 3.0, "accepted": "candidate_1_1", "done": True}
    assert snap.rounds[1]["start"] == 2.5 and snap.rounds[1]["end"] is None                # overlap: proposing early
    assert snap.superseded == 1


def test_events_ordered_and_stair_riser(tmp_path):
    snap = hw.collect(_run(tmp_path))
    ev = hw.events(snap)
    assert [e["key"] for e in ev] == ["accept:candidate_0", "proposal:candidate_1_2", "proposal:candidate_1_1",
                                      "accept:candidate_1_1"]
    assert ev[-1]["old"] == 25.0 and ev[-1]["val"] == 50.0
    assert hw.frontier_at(snap, 2.9) == 25.0 and hw.frontier_at(snap, 3.0) == 50.0


def test_nothing_before_candidate0_is_done(tmp_path):
    run = tmp_path / "r"
    _touch(run / "evals" / "val-candidate_0.request.json", T0)
    assert hw.collect(run) is None


def test_vega_html_is_valid_json_spec(tmp_path):
    html = hw.vega_html(hw.collect(_run(tmp_path)), "x", ceiling=60.0)
    spec = json.loads(re.search(r"vegaEmbed\('#v', (.*), \{actions", html, re.S).group(1))
    marks = [layer["mark"]["type"] for layer in spec["layer"]]
    assert marks == ["rect", "text", "line", "rule", "point", "rule"]
    assert spec["layer"][2]["mark"]["interpolate"] == "step-after"
    assert "25.0% -> 50.0%" in spec["title"]["text"]


def test_wandb_env_reads_only_wandb_names(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    f = tmp_path / ".env"
    f.write_text("OTHER_SECRET=nope\nexport WANDB_API_KEY='k123'\n")
    assert hw.wandb_env(str(f)) == {"WANDB_API_KEY": "k123"}
