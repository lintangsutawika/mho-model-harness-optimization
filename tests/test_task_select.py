"""Task selection (realization: select): the spec's rubric is validated, carried into the realized
sigma json, and scores each train task's instruction lexically; tasks at or above the threshold
are copied to a fresh split with seed.jsonl filtered to them."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mho.data import reward_synthesis as rs  # noqa: E402
from mho.data import task_select as ts  # noqa: E402
from mho.optimizer import model_spec  # noqa: E402

RUBRIC = [
    {"criterion": "reads a CSV", "keywords": ["csv", "read_csv"]},
    {"criterion": "handles missing values", "keywords": ["missing values", "NaN"]},
    {"criterion": "writes a report", "keywords": ["report"], "weight": 2},
]


def write_spec(d: Path, realization: str = "select", rubric: str | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    if rubric is None:
        rubric = textwrap.indent(json.dumps(RUBRIC), "  ")
    (d / "clean_csv.md").write_text(textwrap.dedent(f"""\
        ---
        name: clean_csv
        hypothesis: practising CSV cleaning raises reward on data tasks
        spec: load a CSV, fix missing values, write a report
        counterexample: data_science_task_0001
        realization: {realization}
        checks: [ran_python]
        rubric:
        """) + rubric + "\n---\n")
    (d / "clean_csv.py").write_text("def ran_python(traj):\n    return True\n")
    return d / "clean_csv.md"


def make_split(d: Path, instructions: dict[str, str]) -> Path:
    for name, text in instructions.items():
        (d / name / "tests").mkdir(parents=True)
        (d / name / "instruction.md").write_text(text)
        (d / name / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (d / "seed.jsonl").write_text("".join(
        json.dumps({"prompt": t, "metadata": {"instance_id": n}}) + "\n"
        for n, t in instructions.items()))
    (d / "selection_manifest.json").write_text("{}\n")
    return d


def sigma_json(tmp: Path) -> Path:
    p = tmp / "clean_csv.json"
    p.write_text(json.dumps({"sigma": "clean_csv", "checks": ["ran_python"], "rubric": RUBRIC}))
    return p


# --------------------------------------------------------------------------- spec
def test_spec_rubric_reaches_the_realized_config(tmp_path):
    md = write_spec(tmp_path / "model_specs")
    spec, _ = model_spec.parse_spec(md)
    cfg = rs.produce_config(spec, rs.checks_module(md))
    assert cfg["rubric"][2] == {"criterion": "writes a report", "keywords": ["report"], "weight": 2.0}
    assert cfg["rubric"][0]["weight"] == 1.0


def test_select_without_rubric_is_invalid(tmp_path):
    with pytest.raises(ValueError, match="needs a `rubric`"):
        model_spec.parse_spec(write_spec(tmp_path, rubric="  []"))


def test_synthesize_needs_no_rubric_and_adds_none(tmp_path):
    md = write_spec(tmp_path, realization="synthesize", rubric="  []")
    spec, _ = model_spec.parse_spec(md)
    assert "rubric" not in rs.produce_config(spec, rs.checks_module(md))


def test_unknown_realization_is_invalid(tmp_path):
    with pytest.raises(ValueError, match="select or synthesize"):
        model_spec.parse_spec(write_spec(tmp_path, realization="filter"))


def test_blank_keyword_is_invalid(tmp_path):
    rubric = '  - criterion: x\n    keywords: ["  "]'
    with pytest.raises(ValueError, match="blank"):
        model_spec.parse_spec(write_spec(tmp_path, rubric=rubric))


# --------------------------------------------------------------------------- scoring
def test_score_is_weighted_share_of_met_criteria():
    r = ts.compile_rubric(RUBRIC)
    assert ts.score_text("Load data.CSV and print a REPORT.", r) == (0.75, ["reads a CSV", "writes a report"])
    assert ts.score_text("nothing relevant", r) == (0.0, [])


def test_keywords_match_whole_words_and_any_whitespace():
    r = ts.compile_rubric([{"criterion": "c", "keywords": ["missing values"]},
                           {"criterion": "d", "keywords": ["csv"]}])
    assert ts.score_text("Fill the Missing\n  values", r)[1] == ["c"]
    assert ts.score_text("use csvkit", r)[1] == []
    assert ts.score_text("a .csv file", r)[1] == ["d"]


def test_empty_rubric_is_rejected():
    with pytest.raises(ValueError, match="empty rubric"):
        ts.compile_rubric([])


# --------------------------------------------------------------------------- select
INSTRUCTIONS = {
    "t_both": "Read sales.csv, fill missing values, and write a report.",
    "t_report": "Write a short report about the logs.",
    "t_csv": "Convert data.csv to JSON.",
    "t_none": "Fix the segfault in main.c.",
}


def test_select_copies_kept_tasks_and_filters_seed(tmp_path):
    split = make_split(tmp_path / "train", INSTRUCTIONS)
    out = tmp_path / "selected"
    r = ts.select_tasks(split, sigma_json(tmp_path), out, threshold=0.5)
    # t_both 1.0, t_report 0.5 (weight 2 of 4), t_csv 0.25, t_none 0.
    assert r["n_tasks"] == 4 and r["n_kept"] == 2
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["t_both", "t_report"]
    seed = [json.loads(l)["metadata"]["instance_id"] for l in (out / "seed.jsonl").read_text().splitlines()]
    assert seed == ["t_both", "t_report"]
    assert (out / "selection_manifest.json").exists()
    rep = json.loads((out / ts.REPORT).read_text())
    assert rep["tasks"]["t_csv"] == {"score": 0.25, "met": ["reads a CSV"]}
    assert rep["criteria_hits"] == {"reads a CSV": 2, "handles missing values": 1, "writes a report": 2}
    assert (split / "t_none").exists()          # the base split is never modified


def test_select_replaces_a_stale_output(tmp_path):
    split = make_split(tmp_path / "train", INSTRUCTIONS)
    out = tmp_path / "selected"
    (out / "stale_task").mkdir(parents=True)
    ts.select_tasks(split, sigma_json(tmp_path), out)
    assert not (out / "stale_task").exists()


def test_select_nothing_kept_raises_and_dry_run_writes_nothing(tmp_path):
    split = make_split(tmp_path / "train", {"t_none": "Fix the segfault."})
    out = tmp_path / "selected"
    with pytest.raises(ValueError, match="no task"):
        ts.select_tasks(split, sigma_json(tmp_path), out)
    r = ts.select_tasks(split, sigma_json(tmp_path), None)
    assert r["n_kept"] == 0 and not out.exists()


def test_cli_dry_run(tmp_path, capsys):
    split = make_split(tmp_path / "train", INSTRUCTIONS)
    assert ts.main(["--train-dir", str(split), "--sigma", str(sigma_json(tmp_path)), "--dry-run"]) == 0
    assert "kept 2/4" in capsys.readouterr().out


# --------------------------------------------------------------------------- train.sh
@pytest.mark.parametrize("use,last,n", [("reward", "sigma_tasks", 4), ("select", "selected_tasks", 2),
                                        ("both", "sigma_tasks", 2)])
def test_train_sh_sigma_use_variants(tmp_path, use, last, n):
    """Run train.sh's sigma block alone: reward stages all tasks, select keeps the rubric's tasks
    with their own test.sh, both selects then stages."""
    import subprocess
    sh = (ROOT / "scripts" / "optimize" / "train.sh").read_text()
    block = sh[sh.index('if [ -n "${MHO_SIGMA_REWARD:-}" ]'):]
    block = block[:block.index("\nfi\n") + 4]
    sigma = sigma_json(tmp_path)
    (tmp_path / "clean_csv.py").write_text("def ran_python(traj):\n    return True\n")
    split = make_split(tmp_path / "train", INSTRUCTIONS)
    script = (f'set -euo pipefail\nREPO_DIR={ROOT}\nRUN_DIR={tmp_path}/run\nMATH_TRAIN_DIR={split}\n'
              f'MHO_SIGMA_REWARD={sigma}\nMHO_SIGMA_USE={use}\n_PY={sys.executable}\n'
              + block.replace('_PY="${REPO_DIR}/.venv/bin/python"; [ -x "$_PY" ] || _PY=python3', "")
              + '\necho "$MATH_TRAIN_DIR|$HARBOR_TASKS_DIR"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    train_dir, harbor_dir = r.stdout.strip().splitlines()[-1].split("|")
    assert train_dir == harbor_dir == f"{tmp_path}/run/{last}"
    tasks = [p for p in Path(train_dir).iterdir() if p.is_dir()]
    assert len(tasks) == n
    assert all((p / "tests" / "base_test.sh").exists() == (use != "select") for p in tasks)


def test_train_sh_rejects_unknown_sigma_use(tmp_path):
    import subprocess
    sh = (ROOT / "scripts" / "optimize" / "train.sh").read_text()
    block = sh[sh.index('if [ -n "${MHO_SIGMA_REWARD:-}" ]'):]
    block = block[:block.index("\nfi\n") + 4]
    r = subprocess.run(["bash", "-c", f"MHO_SIGMA_REWARD={sigma_json(tmp_path)}\nMHO_SIGMA_USE=curate\n"
                        + block], capture_output=True, text=True)
    assert r.returncode == 2 and "want reward|select|both" in r.stderr


# --------------------------------------------------------------------------- screen
TEMPLATE = "Please solve this issue: {}\n\n## Useful command examples\nwrite a report, read csv\n"


def make_trials(root: Path, tasks: dict[str, tuple[str, list[float]]]) -> Path:
    """Trial dirs <task>__<i>/ with a trajectory whose first user message wraps the instruction
    in a shared template (which itself contains rubric keywords), and the trial's reward."""
    for task, (text, rewards) in tasks.items():
        for i, r in enumerate(rewards):
            d = root / f"{task}__{i}"
            (d / "agent").mkdir(parents=True)
            (d / "agent" / "mini-swe-agent.trajectory.json").write_text(json.dumps(
                {"messages": [{"role": "system", "content": "s"},
                              {"role": "user", "content": TEMPLATE.format(text)}]}))
            (d / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": r}}}))
    return root


SCREEN_TASKS = {
    "t_both": ("Read sales.csv, fill missing values, and write a report.", [0.0, 1.0]),
    "t_report": ("Write a short report about the logs.", [1.0, 1.0]),
    "t_csv": ("Convert data.csv to JSON.", [0.0, 0.0]),
    "t_none": ("Fix the segfault in main.c.", [0.0, 0.0]),
    "t_none2": ("Patch the kernel module.", [0.0, 1.0]),
    "t_none3": ("Harden the ssh config.", [1.0, 1.0]),
    "t_none4": ("Speed up the build.", [0.0, 0.0]),
    "t_none5": ("Rotate the logs nightly.", [0.0, 1.0]),
}


def test_screen_scores_instructions_without_the_template(tmp_path):
    traces = rs.find_traces([make_trials(tmp_path / "trials", SCREEN_TASKS)], limit=1000)
    ok, rep = ts.screen_rubric(RUBRIC, traces)
    # The template mentions "report" and "csv" in every prompt; without stripping it every
    # task would score >= 0.75 and be kept.
    assert ok and rep["status"] == "ok"
    assert (rep["n_tasks"], rep["n_kept"]) == (8, 2)           # t_both, t_report
    assert rep["kept"] == {"n_tasks": 2, "mean_reward": 0.75, "all_0": 0.0, "all_1": 0.5}
    assert rep["dropped"]["n_tasks"] == 6 and rep["dropped"]["all_0"] == 0.5


def test_screen_rejects_a_rubric_that_keeps_everything(tmp_path):
    traces = rs.find_traces([make_trials(tmp_path / "trials", SCREEN_TASKS)], limit=1000)
    ok, rep = ts.screen_rubric([{"criterion": "any", "keywords": ["the", "a", "to"]}], traces)
    assert not ok and rep["status"].startswith("rejected: keeps 8/8")


def test_screen_too_few_tasks_is_unscreened(tmp_path):
    few = dict(list(SCREEN_TASKS.items())[:3])
    traces = rs.find_traces([make_trials(tmp_path / "trials", few)], limit=1000)
    ok, rep = ts.screen_rubric(RUBRIC, traces)
    assert ok and rep["status"].startswith("unscreened")


def test_realize_gates_on_the_rubric_only_when_training_selects(tmp_path, monkeypatch):
    trials = make_trials(tmp_path / "trials", SCREEN_TASKS)
    md = write_spec(tmp_path / "model_specs",
                    rubric='  - criterion: any\n    keywords: [the, a, to]')
    # a check that varies, so only the rubric decides
    md.with_suffix(".py").write_text("from sigma_checklib import *\n"
                                     "def ran_python(traj):\n    return 'csv' in task_prompt(traj)[:40]\n")
    traces = rs.find_traces([trials])
    monkeypatch.delenv("MHO_SIGMA_USE", raising=False)
    ok, rep = rs.realize(md, tmp_path / "out_reward", traces)
    assert ok and rep["rubric"]["status"].startswith("rejected")   # reported, not gated
    monkeypatch.setenv("MHO_SIGMA_USE", "select")
    ok, rep = rs.realize(md, tmp_path / "out_select", traces)
    assert not ok and rep["status"].startswith("rubric rejected: keeps 8/8")
    assert not (tmp_path / "out_select" / "clean_csv.json").exists()


def test_cli_screen(tmp_path, capsys):
    trials = make_trials(tmp_path / "trials", SCREEN_TASKS)
    md = write_spec(tmp_path / "model_specs")
    assert ts.main(["screen", "--spec", str(md), "--traces", str(trials)]) == 0
    out = capsys.readouterr().out
    assert "kept 2/8" in out and "all_0=" in out
