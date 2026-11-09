"""Sigma-reward plumbing: realize (spec -> {sigma, checks}), stage (inject into every train task's
tests/ with that task's own answer), and the reward rule in the sandbox (base verifier's
correctness + sigma checks)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mho.data import reward_synthesis as rs  # noqa: E402
from mho.kinds.model import realize_model_specs  # noqa: E402

BASE_TEST_SH = "#!/bin/bash\necho base\n"
CHECK = "has_final_agent_answer_line(traj)"


def make_split(root: Path, answers: dict[str, str]) -> Path:
    split = root / "train_split"
    for name, ans in answers.items():
        t = split / name
        (t / "tests").mkdir(parents=True)
        (t / "environment").mkdir()
        (t / "environment" / "Dockerfile").write_text("FROM x\n")
        (t / "instruction.md").write_text(f"solve {name}\n")
        # escaped like the real adapters write them: answer = "\\mu+1"
        (t / "task.toml").write_text(f"[metadata]\nanswer = {json.dumps(ans)}\n")
        (t / "tests" / "test.sh").write_text(BASE_TEST_SH)
        (t / "tests" / "test.py").write_text(f"EXPECTED_ANSWER = {json.dumps(ans)}\n")
    (split / "seed.jsonl").write_text("".join(
        json.dumps({"prompt": n, "metadata": {"instance_id": n}}) + "\n" for n in answers))
    return split


def write_spec(cand: Path, checks: list[str]) -> None:
    (cand / "model_specs").mkdir(parents=True)
    body = "\n".join(f"  - {json.dumps(c)}" for c in checks)
    (cand / "model_specs" / "commit_answer.md").write_text(
        "---\nname: commit_answer\nhypothesis: committing a final answer raises reward\n"
        "evidence:\n  - eval_prior/trials/t1\nspec: always commit a final answer\n"
        f"counterexample: stops without answering\nchecks:\n{body}\n---\n")


# --------------------------------------------------------------------------- realize
def test_realized_config_is_task_independent(tmp_path):
    split = make_split(tmp_path, {"a": "2", "b": "7"})
    cand = tmp_path / "cand"
    write_spec(cand, [CHECK])
    ok, sr = realize_model_specs(cand, split)
    assert ok
    cfg = json.loads((sr / "commit_answer.json").read_text())
    assert cfg == {"sigma": "commit_answer", "checks": [CHECK]}   # no single task's answer


def test_unsupported_check_is_not_realized(tmp_path):
    cand = tmp_path / "cand"
    write_spec(cand, ["made_up_predicate(traj)"])
    ok, sr = realize_model_specs(cand, make_split(tmp_path, {"a": "2"}))
    assert ok and not list(sr.glob("*.json"))


# --------------------------------------------------------------------------- stage
def test_stage_injects_per_task_answers_and_keeps_base_split(tmp_path):
    split = make_split(tmp_path, {"a": "2", "b": "\\frac{1}{3}", "c": "7"})
    sigma = tmp_path / "s.json"
    sigma.write_text(json.dumps({"sigma": "commit_answer", "checks": [CHECK]}))
    out = tmp_path / "staged"
    assert rs.stage_tasks(split, sigma, out) == 3
    assert (out / "seed.jsonl").read_text() == (split / "seed.jsonl").read_text()
    for name, ans in {"a": "2", "b": "\\frac{1}{3}", "c": "7"}.items():
        tests = out / name / "tests"
        assert (tests / "base_test.sh").read_text() == BASE_TEST_SH
        assert (tests / "test.sh").read_text() == rs.WRAPPER_TEST_SH
        assert os.access(tests / "test.sh", os.X_OK)
        assert (tests / "reward_rule.py").read_text() == rs.RULE_TEMPLATE
        cfg = json.loads((tests / "sigma_reward.json").read_text())
        assert cfg == {"sigma": "commit_answer", "checks": [CHECK], "expected_answer": ans}
        assert (out / name / "environment" / "Dockerfile").exists()
        # the base split is untouched
        assert (split / name / "tests" / "test.sh").read_text() == BASE_TEST_SH
        assert not (split / name / "tests" / "sigma_reward.json").exists()


def test_restage_starts_from_the_base_split(tmp_path):
    split = make_split(tmp_path, {"a": "2"})
    sigma = tmp_path / "s.json"
    sigma.write_text(json.dumps({"sigma": "x", "checks": []}))
    out = tmp_path / "staged"
    rs.stage_tasks(split, sigma, out)
    rs.stage_tasks(split, sigma, out)                      # e.g. a retried train job
    assert (out / "a" / "tests" / "base_test.sh").read_text() == BASE_TEST_SH


def test_expected_answer_falls_back_to_test_py(tmp_path):
    split = make_split(tmp_path, {"a": "\\mu+1"})
    (split / "a" / "task.toml").write_text("[task]\nname = 'a'\n")
    assert rs._extract_expected_answer(split / "a") == "\\mu+1"


def test_stage_refuses_a_task_without_test_sh(tmp_path):
    split = make_split(tmp_path, {"a": "2", "b": "3"})
    (split / "b" / "tests" / "test.sh").unlink()
    sigma = tmp_path / "s.json"
    sigma.write_text(json.dumps({"sigma": "x", "checks": []}))
    with pytest.raises(ValueError, match="without tests/test.sh"):
        rs.stage_tasks(split, sigma, tmp_path / "staged")


def test_stage_cli(tmp_path):
    split = make_split(tmp_path, {"a": "2"})
    sigma = tmp_path / "s.json"
    sigma.write_text(json.dumps({"sigma": "x", "checks": []}))
    r = subprocess.run([sys.executable, "-m", "mho.data.reward_synthesis", "stage",
                        "--train-dir", str(split), "--sigma", str(sigma), "--out", str(tmp_path / "o")],
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "staged 1 task(s)" in r.stdout and (tmp_path / "o" / "a" / "tests" / "sigma_reward.json").exists()


# --------------------------------------------------------------------------- reward rule
def run_rule(tmp_path, *, base_reward: str | None, output: str, traj: dict | None,
             checks: list[str], expected: str = "2") -> dict:
    vdir = tmp_path / "verifier"
    vdir.mkdir(exist_ok=True)
    if base_reward is not None:
        (vdir / "reward.txt").write_text(base_reward + "\n")
    rule = tmp_path / "reward_rule.py"
    rule.write_text(rs.RULE_TEMPLATE)
    cfg = tmp_path / "sigma_reward.json"
    cfg.write_text(json.dumps({"sigma": "s", "checks": checks, "expected_answer": expected}))
    out = tmp_path / "output.txt"
    out.write_text(output)
    tr = tmp_path / "trajectory.json"
    tr.write_text(json.dumps(traj) if traj is not None else "")
    subprocess.run([sys.executable, str(rule), str(cfg), str(out), str(tr)],
                   env={**os.environ, "MHO_VERIFIER_DIR": str(vdir)}, capture_output=True, text=True)
    info = json.loads((vdir / "sigma_info.json").read_text())
    assert float((vdir / "reward.txt").read_text()) == pytest.approx(info["reward"], abs=1e-4)
    return info


ANSWERED = {"messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "work\nAnswer: 2"}]}


@pytest.mark.parametrize("base,traj,reward", [
    ("1", ANSWERED, 1.0),     # correct (per the task's verifier) + check passes
    ("1", None, 0.5),         # correct, check fails
    ("0", ANSWERED, 0.5),     # wrong, check passes
    ("0", None, 0.0),
])
def test_rule_uses_base_verifier_correctness(tmp_path, base, traj, reward):
    info = run_rule(tmp_path, base_reward=base, output="2", traj=traj, checks=[CHECK])
    assert info["correct_from"] == "base_verifier" and info["reward"] == reward


def test_rule_falls_back_to_expected_answer_on_bare_output(tmp_path):
    # no base reward.txt: the bare answer in output.txt (arxivmath style) is graded directly
    info = run_rule(tmp_path, base_reward=None, output="2\n", traj=None, checks=[], expected="2")
    assert info["correct_from"] == "expected_answer" and info["correct"] == 1.0
    assert info["reward"] == 0.5                       # no checks -> checks fraction 0


# --------------------------------------------------------------------------- train.sh
def test_train_sh_stages_when_sigma_reward_is_set():
    sh = (ROOT / "scripts" / "optimize" / "train.sh").read_text()
    block = sh[sh.index('if [ -n "${MHO_SIGMA_REWARD:-}" ]'):]
    block = block[:block.index("\nfi\n")]
    assert "mho.data.reward_synthesis stage" in block
    assert 'MATH_TRAIN_DIR="$SIGMA_TASKS_DIR"' in block and 'HARBOR_TASKS_DIR="$SIGMA_TASKS_DIR"' in block
    # staging happens before anything reads the task dirs
    assert sh.index("mho.data.reward_synthesis stage") < sh.index('--prompt-data "${MATH_TRAIN_DIR}/seed.jsonl"')
    assert sh.index("mho.data.reward_synthesis stage") < sh.index('--bind "${HARBOR_TASKS_DIR}:')
