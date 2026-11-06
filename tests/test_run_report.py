"""utils.run_report (run dir -> flat records) and wandb_sync (with a fake wandb module)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import run_report as rr  # noqa: E402

RUN_OUT = """
===== driver start 2026-11-06 23:50:00 host=h job=1.pbs argv=--mode harness-only --model Qwen/Qwen3.5-9B --trials 3 --concurrent 128 --proposals 2 --val-budget 0.67 --fresh
23:50:01 iter 0 baseline baseline -> submit
  submit[pbs] mho-eval-baseline
  -> 10.pbs
23:55:01 [baseline] job 10.pbs done (rc=0)
23:55:02 iter 1 eval_a candidate_1_0_1 -> submit
  -> 11.pbs
23:59:02 [eval_a] job 11.pbs done (rc=0)
23:59:03 iter 1 propose candidate_1_0_1a -> submit
  [propose 1/4] candidate_1_0_1a (harness)
  submit[pbs] mho-propose-candidate_1_0_1a
  -> 12.pbs
23:59:04 iter 1 propose candidate_1_0_1b -> submit
  -> 13.pbs
00:04:03 [propose] candidate_1_0_1a job 12.pbs done (rc=0)
00:04:04 iter 1 eval_b candidate_1_0_1a -> submit
  -> 14.pbs
00:05:04 [propose] candidate_1_0_1b job 13.pbs done (rc=0)
00:05:05 iter 1 eval_b candidate_1_0_1b -> submit
  -> 15.pbs
00:24:04 [eval_b] candidate_1_0_1a job 14.pbs done (rc=0)
00:30:05 [eval_b] candidate_1_0_1b job 15.pbs done (rc=0)
00:30:06 iter 1 val candidate_1_0_1a.ak2 (x2) -> submit
  -> 16.pbs
00:30:07 iter 1 val candidate_1_0_1a.ak3 (x3) -> submit
  -> 17.pbs
00:40:06 [val] candidate_1_0_1a.ak2 job 16.pbs done (rc=0)
01:00:07 [val] candidate_1_0_1a.ak3 job 17.pbs done (rc=0)
01:00:08 iter 1 val baseline.t-1_0_1a-k2 (x2) -> submit
  -> 18.pbs
01:10:08 [val] baseline.t-1_0_1a-k2 job 18.pbs done (rc=0)
"""


def gate(accept, path, d, se, t, tokA=None, tokB=None, tok_t=None, lb=None, anchor=None):
    g = {"accept": accept, "path": path,
         "reason": f"B-A={d:+.4f}" + (f"; non-inferior (lower bound {lb:+.4f} > -0.03)"
                                      if lb is not None else ""),
         "acc": {"n": 64, "A": 0.1, "B": 0.1 + d, "d": d, "se": se, "t": t}}
    if tokA is not None:
        g["cost"] = {"n": 64, "A": tokA, "B": tokB, "d": tokB - tokA, "se": 1.0, "t": tok_t}
    if anchor is not None:
        g["anchor"] = {"d": anchor, "se": 0.01}
    return g


SUMMARY = [
    {"iteration": 0, "candidate": "baseline", "kind": "baseline", "outcome": "baseline",
     "val": 0.18},
    {"iteration": 1, "candidate": "candidate_1_0_1b", "kind": "harness",
     "outcome": "not_selected", "ts_A": 0.1, "ts_B": 0.12,
     "train_gate": gate(True, "cheaper", 0.02, 0.03, 0.6, 25000, 19000, -1.9, lb=-0.01),
     "hypothesis": "# Candidate 1_0_1b — No-op repetition guard\n\n## hypothesis\n..."},
    {"iteration": 1, "candidate": "candidate_1_0_1a", "kind": "harness", "outcome": "accepted",
     "ts_A": 0.1, "ts_B": 0.14, "path": "cheaper", "look": 2, "val": 0.175, "prev_best": 0.18,
     "val_trials": 402,
     "train_gate": gate(True, "cheaper", 0.04, 0.03, 1.3, 25000, 16000, -3.0, lb=0.008),
     "val_gate": gate(True, "cheaper", -0.005, 0.014, -0.35, 20000, 14000, -3.9, lb=-0.019,
                      anchor=-0.005),
     "hypothesis": "# Candidate 1_0_1a — Final-answer artifact hygiene gate\n"},
]


def write_trials(d: Path, rows):
    for i, (task, reward, exc, minutes, toks) in enumerate(rows):
        t = d / f"{task}__{i}"
        t.mkdir(parents=True, exist_ok=True)
        r = {"task_name": task, "started_at": "2026-11-07T00:00:00Z",
             "finished_at": f"2026-11-07T00:{minutes:02d}:00Z",
             "agent_result": {"n_output_tokens": toks},
             "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
             "exception_info": exc}
        (t / "result.json").write_text(json.dumps(r))
    (d / "result.json").write_text(json.dumps({"id": "job-level"}))   # not a trial


@pytest.fixture
def run(tmp_path):
    rd = tmp_path / "run"
    (rd / "logs").mkdir(parents=True)
    (rd / "logs" / "run.out").write_text(RUN_OUT)
    c = rd / "candidates"
    c.mkdir()
    (c / "summary.json").write_text("".join(json.dumps(r) + "\n" for r in SUMMARY))
    (c / "frontier.json").write_text(json.dumps({
        "_best": {"candidate": "candidate_1_0_1a", "metric": 0.175},
        "_anchor": {"candidate": "baseline", "metric": 0.18}}))
    timeout = {"exception_type": "NonZeroAgentExitCodeError",
               "exception_message": "Command failed (exit 124): mini-swe-agent ..."}
    write_trials(c / "candidate_1_0_1a" / "eval_after",
                 [("p1", 1.0, None, 2, 1000), ("p2", 0.0, timeout, 30, 9000),
                  ("p3", None, {"exception_type": "RuntimeError"}, 1, 10)])
    write_trials(rd / "evaluation" / "candidate_1_0_1a.ak2",
                 [("v1", 1.0, None, 3, 100), ("v1", 0.0, None, 3, 100)])
    write_trials(rd / "evaluation" / "candidate_1_0_1a.ak3", [("v2", 0.0, None, 3, 100)])
    write_trials(rd / "evaluation" / "candidate_1_0_10.ak1", [("v9", 1.0, None, 3, 100)])
    return rd


# --------------------------------------------------------------------------- parsers
def test_tag_title_category():
    assert rr.parse_tag("candidate_4_1_5c") == {"iteration": 4, "parent_iteration": 1,
                                                "round": 5, "sibling": "c"}
    assert rr.parse_tag("candidate_2_1_1") == {"iteration": 2, "parent_iteration": 1,
                                               "round": 1, "sibling": "-"}
    assert rr.parse_tag("baseline") == {}
    assert rr.title_of("# Candidate 2_1_1c — Per-call output-token cap\n## x") == \
        "Per-call output-token cap"
    cats = {t: rr.category_of(t) for t in [
        "Repeated-action loop guard", "Submitted-Answer Whitespace Compaction",
        "Loop-Context Condenser", "Per-call output-token cap (truncation-bloat guard)",
        "Hard dead-end network-lookup guard", "Cheap symbolic-verification tool",
        "Something new"]}
    assert list(cats.values()) == ["loop-guard", "answer-format", "context", "budget",
                                   "lookup", "verification", "other"]


def test_parse_run_log_config_jobs_and_midnight(run):
    config, jobs = rr.parse_run_log(run / "logs" / "run.out")
    assert config["model"] == "Qwen/Qwen3.5-9B" and config["concurrent"] == 128
    assert config["val_budget"] == 0.67 and config["fresh"] is True
    by = {j["job_id"]: j for j in jobs}
    assert by["14.pbs"]["stage"] == "eval_b" and by["14.pbs"]["tag"] == "candidate_1_0_1a"
    assert by["14.pbs"]["minutes"] == pytest.approx(20.0)
    assert by["13.pbs"]["minutes"] == pytest.approx(6.0)        # 23:59:04 -> 00:05:04
    assert by["16.pbs"]["tag"] == "candidate_1_0_1a.ak2"
    assert by["16.pbs"]["done"].day == 7 and by["10.pbs"]["submit"].day == 6
    assert rr.job_owner(by["18.pbs"]) == "candidate_1_0_1a"     # best's top-up -> candidate
    assert rr.job_owner(by["17.pbs"]) == "candidate_1_0_1a"


def test_trial_stats_and_val_dirs(run):
    st = rr.trial_stats([run / "candidates" / "candidate_1_0_1a" / "eval_after"])
    assert st["trials"] == 3 and st["tasks"] == 2
    assert st["timeout_rate"] == pytest.approx(1 / 3)
    assert st["unscored_rate"] == pytest.approx(1 / 3)
    assert st["error_rate"] == pytest.approx(2 / 3)
    assert st["dur_max_min"] == pytest.approx(30)
    vd = rr.val_dirs(run, "candidate_1_0_1a")
    assert [d.name for d in vd] == ["candidate_1_0_1a.ak2", "candidate_1_0_1a.ak3"]
    vs = rr.trial_stats(vd)
    assert vs["acc"] == pytest.approx((0.5 + 0.0) / 2)            # mean of per-task means


# --------------------------------------------------------------------------- rows
def test_candidate_rows(run):
    rows = rr.candidate_rows(run)
    assert [r["step"] for r in rows] == [0, 1, 2]
    base, ns, acc = rows
    assert base["best/val_acc"] == 0.18 and base["cum/proposed"] == 0
    assert ns["outcome"] == "not_selected" and ns["sibling"] == "b"
    assert ns["category"] == "loop-guard"
    assert ns["train/tok_delta_pct"] == pytest.approx(-24.0)
    assert ns["train/tok_cheaper_t"] == pytest.approx(1.9)
    assert ns["train/noninf_lb"] == pytest.approx(-0.01)
    assert ns["best/val_acc"] == 0.18 and ns["cum/train_pass"] == 1
    assert acc["best/val_acc"] == 0.175 and acc["best/candidate"] == "candidate_1_0_1a"
    assert acc["best/anchor_acc"] == 0.18
    assert acc["best/anchor_margin"] == pytest.approx(-0.005)
    assert acc["best/val_tokens"] == 14000
    assert acc["cum/accepted_cheaper"] == 1 and acc["cum/to_val"] == 1
    assert acc["cum/accept_rate"] == pytest.approx(0.5)
    assert acc["val/t"] == -0.35 and acc["val/look"] == 2 and acc["val/anchor_d"] == -0.005
    assert acc["time/propose_min"] == pytest.approx(5.0)
    assert acc["time/val_min"] == pytest.approx(10 + 30 + 10)     # its parts + best's top-up
    assert acc["time/val_wall_min"] == pytest.approx(40 + 2 / 60)  # 00:30:06 -> 01:10:08
    assert acc["eval_b/timeout_rate"] == pytest.approx(1 / 3)
    assert acc["val_eval/trials"] == 3
    assert rr.round_rows(rows) == [{"iteration": 1, "round": 1, "proposed": 2, "train_pass": 2,
                                    "to_val": 1, "accepted": 1, "distinct_categories": 2,
                                    "categories": "answer-format:1,loop-guard:1"}]


def test_rows_tolerate_old_and_partial_runs(tmp_path):
    rd = tmp_path / "r"
    (rd / "candidates").mkdir(parents=True)
    assert rr.candidate_rows(rd) == []
    (rd / "candidates" / "summary.json").write_text(
        json.dumps({"candidate": "candidate_3_2_7", "outcome": "train_fail",
                    "train_gate": None}) + "\nnot json\n")
    rows = rr.candidate_rows(rd)
    assert len(rows) == 1 and rows[0]["cum/train_fail"] == 1 and "best/val_acc" not in rows[0]


# --------------------------------------------------------------------------- wandb_sync
class FakeRun:
    def __init__(self, id):
        self.id, self.logged, self.summary, self.finished = id, [], {}, False

    def log(self, data, step):
        assert not self.logged or step > self.logged[-1][0], "W&B drops non-increasing steps"
        self.logged.append((step, data))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    runs = []
    mod = types.ModuleType("wandb")

    def init(id, **kw):
        run = next((r for r in runs if r.id == id), None) or FakeRun(id)
        if run not in runs:
            runs.append(run)
        run.init_kw = kw
        return run

    mod.init = init
    mod.Table = lambda columns, data: {"columns": columns, "data": data}
    mod.plot = SimpleNamespace(scatter=lambda t, x, y, title: ("scatter", x, y, len(t["data"])))
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return runs


def args(run, **kw):
    a = dict(run_dir=str(run), project="p", entity=None, name=None, group=None, watch=0,
             restart=False, offline=False)
    a.update(kw)
    return SimpleNamespace(**a)


def test_sync_is_incremental_and_resumable(run, fake_wandb):
    import wandb_sync
    s = wandb_sync.Syncer(args(run))
    assert s.sync() == 3 and s.sync() == 0
    r = fake_wandb[0]
    assert [st for st, _ in r.logged] == [0, 1, 2]
    last = r.logged[-1][1]
    assert last["best/val_acc"] == 0.175 and "outcome" not in last      # strings only in tables
    assert len(last["tables/candidates"]["data"]) == 3
    assert last["charts/train_t_vs_val_t"][-1] == 1
    assert r.summary["frontier/best"] == "candidate_1_0_1a"
    assert r.init_kw["config"]["concurrent"] == 128 and r.init_kw["name"] == "run"

    # a new candidate finishes; a NEW process resumes the same W&B run and logs only it
    with (run / "candidates" / "summary.json").open("a") as fh:
        fh.write(json.dumps({"iteration": 2, "candidate": "candidate_2_1_2a",
                             "outcome": "train_fail", "kind": "harness",
                             "train_gate": gate(False, None, -0.01, 0.02, -0.5)}) + "\n")
    s2 = wandb_sync.Syncer(args(run))
    assert s2.run is r and s2.sync() == 1
    assert [st for st, _ in r.logged] == [0, 1, 2, 3]

    # --restart: a fresh W&B run with everything
    s3 = wandb_sync.Syncer(args(run, restart=True))
    assert s3.run is not r and s3.sync() == 4
