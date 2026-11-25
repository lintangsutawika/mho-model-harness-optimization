"""mho.hfloop: the harnessforge harness-search driver (pure helpers + the accept / top-up path with fake evals)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "harnessforge" / "src"))

from harnessforge.race_run import FileRun  # noqa: E402
from harnessforge.records import TrialRecord, read_jsonl, write_jsonl  # noqa: E402
from harnessforge.stats.reference import topup_plan  # noqa: E402
from mho.hfloop import (START, Config, Loop, evidence_tasks, per_candidate_concurrency, round_summary,  # noqa: E402
                        topup_groups, write_static_side)

TASKS = [f"t{i:03d}" for i in range(8)]


def rec(c, t, i, reward=1.0, **kw):
    return TrialRecord(candidate=c, task=t, index=i, finished=True, reward=reward, **kw)


def cfg(tmp: Path, **kw) -> Config:
    base = dict(run_dir=tmp / "run", start_harness=tmp / "start", dataset="/data", val_tasks=list(TASKS),
                screen_tasks=[f"s{i}" for i in range(20)], trials=4, proposals=2, rounds=1)
    base.update(kw)
    return Config(**base)


def test_concurrency_and_evidence_sampling():
    assert per_candidate_concurrency(1120, 4) == 280
    assert per_candidate_concurrency(10, 0) == 10
    a, b = evidence_tasks(list("abcdefghij"), 4, 1, 1), evidence_tasks(list("abcdefghij"), 4, 1, 2)
    assert len(a) == 4 and a == evidence_tasks(list("abcdefghij"), 4, 1, 1) and a != b
    assert len(evidence_tasks(["x"], 5, 1, 1)) == 1


def test_config_rejects_val_screen_overlap(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"run_dir": str(tmp_path / "r"), "start_harness": str(tmp_path), "dataset": "/d",
                             "val_tasks": ["a", "b"], "screen_tasks": ["b", "c"]}))
    with pytest.raises(ValueError, match="val tasks are in the screen pool"):
        Config.load(p)
    (tmp_path / "val.json").write_text(json.dumps({"tasks": ["a", "b"]}))
    (tmp_path / "screen.txt").write_text("c\nd\n")
    p.write_text(json.dumps({"run_dir": str(tmp_path / "r"), "start_harness": str(tmp_path), "dataset": "/d",
                             "val_tasks": str(tmp_path / "val.json"), "screen_tasks": str(tmp_path / "screen.txt"),
                             "fresh_looks": [1, 2]}))
    c = Config.load(p)
    assert c.val_tasks == ["a", "b"] and c.screen_tasks == ["c", "d"] and c.fresh_looks == (1, 2)


def test_topup_groups_by_missing_count():
    recs = [rec("B", t, i) for t in TASKS[:4] for i in range(4)] + [rec("B", TASKS[4], 0), rec("B", TASKS[5], 0),
                                                                    rec("B", TASKS[5], 1)]
    plan = topup_plan(recs, "B", TASKS, 4)
    g = topup_groups(plan)
    assert g == {2: [TASKS[5]], 3: [TASKS[4]], 4: TASKS[6:]}
    assert sum(k * len(v) for k, v in g.items()) == plan.n_units


def test_static_side_is_a_finished_filerun_side(tmp_path):
    recs = [rec("A", t, i) for t in TASKS for i in range(4)] + [rec("other", TASKS[0], 0)]
    side = write_static_side(tmp_path / "ref", "A", recs, TASKS, 4)
    run = FileRun({"A": side})
    assert run.done() and len(run.records()) == 32
    req = json.loads((side / "request.json").read_text())
    assert req["tasks"] == TASKS and req["trials"] == 4 and req["order"] == "task_major"


def test_round_summary_points_and_invalid():
    race = {"decisions": {"states": {"c1": {"verdict": "pass", "look": 2, "acc": {"d": 0.0612, "se": 0.0151, "t": 4.05},
                                            "reason": "look 2: pass"}}}}
    s = round_summary(1, START, race, ["c2"], "c1")
    assert s["candidates"]["c1"]["d_points"] == 6.12 and s["candidates"]["c1"]["se_points"] == 1.51
    assert s["candidates"]["c2"] == {"verdict": "invalid"} and s["accepted"] == "c1"


class FakeEvals:
    """Stands in for `python -m harnessforge.evals run REQ`: writes every requested unit as a finished trial."""

    def __init__(self, reward=1.0):
        self.calls, self.reward = [], reward

    def __call__(self, cmd, **kw):
        if "harnessforge.evals" in cmd and "run" in cmd:
            req = json.loads(Path(cmd[cmd.index("run") + 1]).read_text())
            self.calls.append(req)
            d = Path(req["out_dir"]) / req["run_id"]
            d.mkdir(parents=True, exist_ok=True)
            write_jsonl(d / "records.jsonl", [rec(c["id"], t, i, self.reward) for c in req["candidates"]
                                              for t in req["tasks"] for i in range(req["trials"])])
            (d / "status.json").write_text(json.dumps({"done": True}))
            (d / "request.json").write_text(json.dumps(req))
        return subprocess.CompletedProcess(cmd, 0, "", "")


def _start(tmp: Path) -> Path:
    h = tmp / "start"
    (h / "src").mkdir(parents=True)
    (h / "pyproject.toml").write_text("[project]\nname='x'\n")
    return h


def test_reference_step0_then_accept_with_topup(tmp_path):
    _start(tmp_path)
    fake = FakeEvals()
    loop = Loop(cfg(tmp_path, fresh="never"), "http://ep", runner=fake)
    side = loop.ensure_reference()
    assert fake.calls[0]["run_id"] == f"val-{START}" and fake.calls[0]["trials"] == 4
    assert len(read_jsonl(side / "records.jsonl")) == 32
    fr = json.loads((loop.cands / "frontier.json").read_text())
    assert fr["_best"]["candidate"] == START and fr["_best"]["metric"] == 1.0

    # a gate in which candidate_1_1 passed at look 2 (tasks 0-3 complete, task 4 half done, the rest never run)
    gdir = loop.evals_dir("gate-r1")
    gdir.mkdir(parents=True)
    write_jsonl(gdir / "records.jsonl", [rec("candidate_1_1", t, i) for t in TASKS[:4] for i in range(4)]
                + [rec("candidate_1_1", TASKS[4], i) for i in range(2)])
    (loop.harness("candidate_1_1")).mkdir(parents=True)
    state = {"verdict": "pass", "look": 2, "acc": {"d": 0.06, "se": 0.015, "t": 4.0}, "efficacy": 2.54,
             "fraction": 0.5, "reason": "look 2: pass"}
    race = {"looks": [{}, {}, {}, {}], "decisions": {"states": {"candidate_1_1": state,
                                                               "candidate_1_2": {**state, "verdict": "fail"}},
                                                    "winner": "candidate_1_1", "ranking": ["candidate_1_1"]}}
    assert loop.accept(1, race) == "candidate_1_1"
    topups = [c for c in fake.calls if c["run_id"].startswith("topup-")]
    assert sorted((c["trials"], len(c["tasks"])) for c in topups) == [(2, 1), (4, 3)]
    merged = read_jsonl(loop.ref_side("candidate_1_1") / "records.jsonl")
    assert {(r.task, r.index) for r in merged} == {(t, i) for t in TASKS for i in range(4)}
    assert loop.state["reference"] == "candidate_1_1"
    assert json.loads((loop.cands / "candidate_1_2" / "status.json").read_text())["status"] == "rejected"
    lin = json.loads(loop.lineage_path.read_text())
    assert lin["current"] == "candidate_1_1" and lin["previous"] == START and lin["history"][0]["fresh"] is False


def test_accept_fresh_policy_reruns_everything(tmp_path):
    _start(tmp_path)
    fake = FakeEvals()
    loop = Loop(cfg(tmp_path, fresh="looks", fresh_looks=(1, 2, 4)), "http://ep", runner=fake)
    loop.ensure_reference()
    gdir = loop.evals_dir("gate-r1")
    gdir.mkdir(parents=True)
    write_jsonl(gdir / "records.jsonl", [rec("candidate_1_1", t, i) for t in TASKS[:2] for i in range(4)])
    loop.harness("candidate_1_1").mkdir(parents=True)
    state = {"verdict": "pass", "look": 1, "acc": {"d": 0.1, "se": 0.02, "t": 5.0}, "efficacy": 3.75, "fraction": 0.25}
    loop.accept(1, {"looks": [{}] * 4, "decisions": {"states": {"candidate_1_1": state}, "winner": "candidate_1_1"}})
    topups = [c for c in fake.calls if c["run_id"].startswith("topup-")]
    assert [(c["trials"], len(c["tasks"])) for c in topups] == [(4, len(TASKS))]   # fresh: every unit re-run
    assert json.loads(loop.lineage_path.read_text())["history"][0]["fresh"] is True


def test_no_winner_keeps_reference(tmp_path):
    _start(tmp_path)
    loop = Loop(cfg(tmp_path), "http://ep", runner=FakeEvals())
    loop.ensure_reference()
    race = {"decisions": {"states": {"candidate_1_1": {"verdict": "fail", "look": 1}}, "winner": None}}
    assert loop.accept(1, race) is None and loop.state["reference"] == START


def test_propose_validity_and_eval_prior_moved_out(tmp_path):
    _start(tmp_path)
    loop = Loop(cfg(tmp_path, proposals=2, propose_cmd=["true"]), "http://ep", runner=FakeEvals())
    loop.ensure_reference()

    class P:
        def __init__(self, cmd, env, **kw):
            tag = env["MHO_TAG"]
            h = loop.harness(tag)
            import shutil
            shutil.copytree(loop.harness(START), h)
            (h / "eval_prior").mkdir()
            (h / "eval_prior" / "x.json").write_text("{}")
            if tag.endswith("_1"):
                (h / "src" / "fix.py").write_text("# a change\n")
            assert env["MHO_PROPOSE_SIBLING"] in ("1/2", "2/2") and env["MICRO_SCAFFOLD_BASE"].endswith("harness")

        def wait(self, timeout=None):
            return 0

    loop.popen = P
    valid, invalid = loop.propose(1)
    assert valid == ["candidate_1_1"] and invalid == ["candidate_1_2"]          # an unchanged copy is not a proposal
    assert not (loop.harness("candidate_1_1") / "eval_prior").exists()
    assert (loop.cands / "candidate_1_1" / "eval_prior" / "x.json").exists()


def test_proposer_diff_tolerates_non_utf8(tmp_path):
    """mho.optimizer.agent._write_diff must not crash on non-UTF-8 bytes (staged trial output in the workspace)."""
    import importlib
    import types
    sys.modules.setdefault("harness", types.ModuleType("harness"))
    sys.modules["harness"].scaffold = types.SimpleNamespace(DEFAULT_BASE=tmp_path, _IGNORE=None)
    agent = importlib.import_module("mho.optimizer.agent")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "x.txt").write_bytes(b"same\n")
    (b / "x.txt").write_bytes(b"changed \x95 byte\n")
    out = tmp_path / "d.patch"
    agent._write_diff(a, b, out)
    assert "changed" in out.read_text()


# ----------------------------------------------------------------------------------------------- overlap
class Recorder:
    """Fake proposer launcher + evals runner that log one shared event list (to check what ran when)."""

    def __init__(self, loop):
        self.loop, self.events, self.evals = loop, [], FakeEvals()

    def run(self, cmd, **kw):
        if "harnessforge.evals" in cmd and "run" in cmd:
            self.events.append(("evals", json.loads(Path(cmd[cmd.index("run") + 1]).read_text())["run_id"]))
        return self.evals(cmd, **kw)

    def popen(self, cmd, env, **kw):
        import shutil
        loop, tag = self.loop, env["MHO_TAG"]
        best = json.loads((loop.cands / "frontier.json").read_text())["_best"]["candidate"]
        self.events.append(("propose", tag, best))
        h = loop.harness(tag)
        if h.exists():
            shutil.rmtree(h)
        shutil.copytree(loop.harness(best), h)
        (h / "src" / f"{tag}.py").write_text("# change\n")
        loop_ = self

        class P:
            def wait(self, timeout=None):
                loop_.events.append(("propose_done", tag))
                return 0
        return P()


def _race(r, winner=None):
    st = {f"candidate_{r}_{k}": {"verdict": "fail", "look": 1, "acc": {"d": -0.01, "se": 0.01, "t": -1.0, "B": 0.5}}
          for k in (1, 2)}
    if winner:
        st[winner] = {"verdict": "pass", "look": 2, "acc": {"d": 0.05, "se": 0.01, "t": 5.0, "B": 0.62},
                      "efficacy": 2.6, "fraction": 0.5}
    return {"looks": [{}] * 4, "decisions": {"states": st, "winner": winner, "done": True}}


def _overlap_loop(tmp_path, rounds, winners):
    _start(tmp_path)
    loop = Loop(cfg(tmp_path, overlap=True, rounds=rounds, proposals=2, propose_cmd=["x"]), "http://ep")
    rec_ = Recorder(loop)
    loop.runner, loop.popen = rec_.run, rec_.popen

    def gate(r, valid):
        rec_.events.append(("gate", r))
        w = winners.get(r)
        if w:
            d = loop.evals_dir(f"gate-r{r}")
            d.mkdir(parents=True, exist_ok=True)
            write_jsonl(d / "records.jsonl", [rec(w, t, i) for t in TASKS[:4] for i in range(4)])
        return _race(r, w)
    loop.gate = gate
    return loop, rec_


def test_overlap_no_accept_uses_proposals_started_during_the_gate(tmp_path):
    loop, r = _overlap_loop(tmp_path, rounds=2, winners={})
    loop.drive()
    ev = r.events
    props = [e for e in ev if e[0] == "propose"]
    assert [e[1] for e in props] == ["candidate_1_1", "candidate_1_2", "candidate_2_1", "candidate_2_2"]
    assert all(e[2] == START for e in props)
    # round 2's proposers were launched before round 1's gate ran, and never relaunched
    assert ev.index(("propose", "candidate_2_1", START)) < ev.index(("gate", 1))
    assert loop.state["rounds"]["2"]["base"] == START and loop.state["rounds"]["2"]["proposals"] == [
        "candidate_2_1", "candidate_2_2"]
    assert not (loop.cands / "_superseded").exists()


def test_overlap_accept_supersedes_and_reproposes_on_winner_during_topup(tmp_path):
    loop, r = _overlap_loop(tmp_path, rounds=2, winners={1: "candidate_1_1"})
    loop.drive()
    ev = r.events
    props2 = [e for e in ev if e[0] == "propose" and e[1].startswith("candidate_2_")]
    assert [e[2] for e in props2] == [START, START, "candidate_1_1", "candidate_1_1"]   # speculative, then redone
    first_topup = next(i for i, e in enumerate(ev) if e[0] == "evals" and e[1].startswith("topup-r1"))
    evid = ev.index(("evals", "evidence-r2-candidate_1_1"))
    relaunch = [i for i, e in enumerate(ev) if e == ("propose", "candidate_2_1", "candidate_1_1")][0]
    assert evid < relaunch < first_topup            # winner's evidence, then proposers, THEN the top-up
    sup = loop.state["rounds"]["2"]["superseded"]
    assert sup[0]["base"] == START and sup[0]["tags"] == ["candidate_2_1", "candidate_2_2"]
    assert all((Path(sup[0]["dir"]) / t / "harness").is_dir() for t in sup[0]["tags"])   # moved aside, not deleted
    assert loop.state["rounds"]["2"]["base"] == "candidate_1_1"
    fr = json.loads((loop.cands / "frontier.json").read_text())
    assert fr["_best"]["candidate"] == "candidate_1_1" and "_provisional" not in fr   # final after the top-up


def test_restart_supersedes_proposals_built_on_a_stale_reference(tmp_path):
    loop, r = _overlap_loop(tmp_path, rounds=1, winners={})
    loop.ensure_reference()
    # a previous job finished round 1's proposals on START, then the reference moved on (accepted elsewhere)
    loop.state["rounds"]["1"] = {"base": START, "proposals": ["candidate_1_1"], "invalid": []}
    (loop.cands / "candidate_1_1").mkdir(parents=True)
    import shutil
    shutil.copytree(loop.harness(START), loop.harness("other"))
    write_static_side(loop.ref_side("other"), "other", [rec("other", t, i) for t in TASKS for i in range(4)], TASKS, 4)
    loop.state["reference"] = "other"
    loop.write_frontier()
    loop.drive()
    assert loop.state["rounds"]["1"]["base"] == "other"
    assert loop.state["rounds"]["1"]["superseded"][0]["base"] == START


def test_no_overlap_by_default(tmp_path):
    _start(tmp_path)
    loop = Loop(cfg(tmp_path, rounds=2, proposals=2, propose_cmd=["x"]), "http://ep")
    r = Recorder(loop)
    loop.runner, loop.popen = r.run, r.popen
    loop.gate = lambda rr, v: (r.events.append(("gate", rr)), _race(rr))[1]
    loop.drive()
    assert r.events.index(("gate", 1)) < r.events.index(("propose", "candidate_2_1", START))
