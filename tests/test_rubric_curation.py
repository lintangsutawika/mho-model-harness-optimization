"""Rubric curation: rubric.md bullets are yes/no questions, a judge answers them per task, score =
share of yes, and one judge pass yields the threshold and top-N% selections plus its cost."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mho.data import rubric_curation as rc  # noqa: E402

RUBRIC = "# Rubric\nsome prose that is not a question\n- Does it read a CSV?\n- Does it write a report?\n"


def fake_complete(prompt: str):
    """yes to a question when the task text contains its keyword (csv / report)."""
    task = prompt.split("<task>")[1].split("</task>")[0].lower()
    ans = ["yes" if "csv" in task else "no", "yes" if "report" in task else "no"]
    return "```json\n" + json.dumps({"answers": ans}) + "\n```", {
        "prompt_tokens": 100, "completion_tokens": 10, "cost": 0.001}


def make_tasks(d: Path, texts: dict[str, str]) -> list[dict]:
    rows = []
    for i, (name, text) in enumerate(texts.items()):
        (d / name).mkdir(parents=True)
        (d / name / "instruction.md").write_text(text)
        rows.append({"i": i, "task": name, "path": str(d / name)})
    return rows


def test_parse_rubric_reads_bullets_and_numbered_lines():
    assert rc.parse_rubric(RUBRIC) == ["Does it read a CSV?", "Does it write a report?"]
    assert rc.parse_rubric("1. a?\n2) b?\n* c?") == ["a?", "b?", "c?"]
    with pytest.raises(ValueError):
        rc.parse_rubric("- only one?")
    with pytest.raises(ValueError):
        rc.parse_rubric("\n".join(f"- q{k}?" for k in range(rc.MAX_LINES + 1)))


def test_parse_answers():
    assert rc.parse_answers('ok {"answers": ["Yes", "no"]}', 2) == [True, False]
    with pytest.raises(ValueError):
        rc.parse_answers('{"answers": ["yes"]}', 2)
    with pytest.raises(ValueError):
        rc.parse_answers('{"answers": ["maybe", "no"]}', 2)
    with pytest.raises(ValueError):
        rc.parse_answers("no json here", 2)


def test_select_threshold_and_top():
    scores = {0: 1.0, 1: 0.5, 2: 1.0, 3: 0.0, 4: 0.9}
    s = rc.select(scores, threshold=0.85, top_frac=0.3)
    assert s["threshold"] == [0, 2, 4]
    assert s["top"] == [0, 2]                  # ceil(0.3 * 5) = 2, ties in index order


def test_judge_retries_an_unparsable_reply():
    replies = iter([("not json", {"prompt_tokens": 5, "completion_tokens": 1, "cost": 0.1}),
                    ('{"answers": ["yes", "yes"]}', {"prompt_tokens": 5, "completion_tokens": 1, "cost": 0.1})])
    j = rc.judge("task", ["a?", "b?"], complete=lambda p: next(replies))
    assert j["answers"] == [True, True] and j["prompt_tokens"] == 10
    assert j["cost"] == pytest.approx(0.2)


def test_run_rubric_scores_selects_and_costs(tmp_path):
    index = make_tasks(tmp_path / "tasks", {"a": "Read the CSV and write a report.",
                                            "b": "Read the csv file.", "c": "Fix the bug."})
    for r, rw in zip(index, ([1, 0], [1, 1], [0, 0])):
        r["rewards"] = rw
    (tmp_path / "rubric.md").write_text(RUBRIC)
    out = rc.run_rubric(tmp_path / "rubric.md", index, workers=2, complete=fake_complete,
                        threshold=0.85, top_frac=0.5)
    assert [t["score"] for t in out["tasks"]] == [1.0, 0.5, 0.0]
    assert out["selections"] == {"threshold": [0], "top": [0, 1]}
    assert out["yield"] == {"threshold": 1, "top": 2}
    assert out["line_yes"] == [2, 1]
    assert out["cost"]["prompt_tokens"] == 300 and out["cost"]["usd"] == pytest.approx(0.003)
    assert out["rewards"]["threshold"]["kept"] == {"n_tasks": 1, "mean_reward": 0.5, "all_0": 0.0, "all_1": 0.0}
    assert out["rewards"]["threshold"]["dropped"]["n_tasks"] == 2


def test_run_rubric_records_provider_errors(tmp_path):
    index = make_tasks(tmp_path / "tasks", {"a": "csv"})
    (tmp_path / "rubric.md").write_text(RUBRIC)

    def broke(prompt):
        raise RuntimeError("Budget has been exceeded!")
    out = rc.run_rubric(tmp_path / "rubric.md", index, complete=broke)
    assert out["n_scored"] == 0 and "Budget" in out["errors"][0]
    assert out["selections"] == {"threshold": [], "top": []}


def test_init_run_builds_indexes_and_base_candidate(tmp_path):
    root = tmp_path / "root"
    pool = root / "data-harbor" / "pool"
    make_tasks(pool, {f"t{k}": f"task {k}" for k in range(6)})
    ev = tmp_path / "runs" / "r1" / "eval_1"
    for k, (task, reward) in enumerate([("t0", 1.0), ("t0", 0.0), ("t1", 0.0), ("t2", None)]):
        td = ev / f"{task}__{k}"
        td.mkdir(parents=True)
        (td / "config.json").write_text(json.dumps({"task": {"path": f"data-harbor/pool/{task}"}}))
        res = {"verifier_result": {"rewards": {"reward": reward}}} if reward is not None else {}
        (td / "result.json").write_text(json.dumps(res))
    run = tmp_path / "run"
    stats = rc.init_run(run, [tmp_path / "runs" / "r1"], pool, root, n_val=3, seed=0)
    assert stats == {"design_tasks": 2, "trials": 3, "val_tasks": 3, "val_pool": 4}
    design = json.loads((run / "data" / "design.json").read_text())
    assert [(d["task"], d["rewards"]) for d in design] == [("t0", [1.0, 0.0]), ("t1", [0.0])]
    index = json.loads((run / "data" / "index.json").read_text())
    assert {r["task"] for r in index}.isdisjoint({"t0", "t1"}) and [r["i"] for r in index] == [0, 1, 2]
    base = run / "candidates" / rc.BASE
    assert (base / "tasks" / "t0.md").read_text() == "task 0"
    assert sorted(p.name for p in (base / "eval_before").iterdir()) == ["t0__0", "t0__1", "t1__2"]
    with pytest.raises(ValueError):
        rc.init_run(tmp_path / "run2", [tmp_path / "runs" / "r1"], pool, root, n_val=5)


def test_run_cli_writes_curated_and_summary(tmp_path, monkeypatch):
    run = tmp_path / "run"
    index = make_tasks(tmp_path / "tasks", {"a": "csv report", "b": "nothing"})
    (run / "data").mkdir(parents=True)
    (run / "data" / "index.json").write_text(json.dumps(index))
    (run / "candidates" / "candidate_1_0_1").mkdir(parents=True)
    (run / "candidates" / "candidate_1_0_1" / "rubric.md").write_text(RUBRIC)
    monkeypatch.setattr(rc, "_completion", fake_complete)
    assert rc.main(["run", "--run", str(run), "--tag", "candidate_1_0_1", "--repeats", "2"]) == 0
    cur = run / "candidates" / "candidate_1_0_1" / "curated"
    assert sorted(p.name for p in cur.iterdir()) == ["index_r1.json", "index_r2.json"]
    summary = json.loads((run / "candidates" / "summary.json").read_text())
    assert summary[0]["candidate"] == "candidate_1_0_1" and len(summary[0]["runs"]["index"]) == 2
