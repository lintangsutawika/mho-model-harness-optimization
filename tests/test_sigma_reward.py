"""Sigma-reward plumbing: the proposer writes model_specs/<name>.md + <name>.py (check functions);
realize validates and screens the checks on existing trajectories; stage injects the rule, the
checks module and each task's own answer into every train task's tests/; the rule in the sandbox
combines the base verifier's correctness with the checks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mho.data import reward_synthesis as rs  # noqa: E402
from mho.data import sigma_checklib as lib  # noqa: E402
from mho.kinds.model import realize_model_specs  # noqa: E402
from mho.optimizer import model_spec  # noqa: E402
from mho.optimizer.agent import model_target_note  # noqa: E402

BASE_TEST_SH = "#!/bin/bash\necho base\n"

CHECKS_PY = textwrap.dedent('''
    from sigma_checklib import *

    def ran_python(traj):
        return any("python" in c for c in commands(traj))

    def few_steps(traj):
        return max(0.0, 1.0 - n_steps(traj) / 10)

    def always_true(traj):
        return True

    def boom(traj):
        raise RuntimeError("bad check")

    def quits(traj):
        raise SystemExit(3)

    def slow(traj):
        while True:
            pass

    def wrong_type(traj):
        return "yes"
''')


def traj(cmds: list[str], rcs: list[int] | None = None) -> dict:
    """A trajectory in the real mini-swe-agent shape: tool_calls with JSON-string arguments, tool
    observations as a JSON string with returncode/output."""
    rcs = rcs or [0] * len(cmds)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Problem: q"}]
    for i, (c, rc) in enumerate(zip(cmds, rcs)):
        msgs.append({"role": "assistant", "content": f"step {i}", "reasoning_content": "think",
                     "tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"command": c}),
                                                              "name": "bash"}}]})
        msgs.append({"role": "tool", "tool_call_id": str(i),
                     "content": json.dumps({"returncode": rc, "output": f"out {i}"})})
    return {"info": {}, "messages": msgs, "trajectory_format": "mini-swe-agent-1"}


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


def write_spec(cand: Path, checks: list[str], py: str | None = CHECKS_PY, name: str = "commit") -> Path:
    d = cand / "model_specs"
    d.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"  - {json.dumps(c)}" for c in checks)
    md = d / f"{name}.md"
    md.write_text(f"---\nname: {name}\nhypothesis: running code raises reward\n"
                  "evidence:\n  - c/eval_before/t1\nspec: verify with code\n"
                  f"counterexample: guesses\nchecks:\n{body}\n---\n")
    if py is not None:
        (d / f"{name}.py").write_text(py)
    return md


def make_traces(cands: Path, n: int = 12) -> None:
    """n train-sample trials under candidates/c0/eval_before/: half ran python (and passed)."""
    for i in range(n):
        t = cands / "c0" / "eval_before" / f"task{i}__x"
        (t / "agent").mkdir(parents=True)
        cmds = ["ls", "python3 solve.py"] if i % 2 else ["ls", "cat a", "echo 1 > out"]
        (t / "agent" / "mini-swe-agent.trajectory.json").write_text(json.dumps(traj(cmds)))
        (t / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": float(i % 2)}}}))
    # a val eval must never be screened on
    v = cands / "c0" / "eval-val" / "x__y" / "agent"
    v.mkdir(parents=True)
    (v / "mini-swe-agent.trajectory.json").write_text(json.dumps(traj(["python3 leak"])))


# --------------------------------------------------------------------------- checklib
def test_checklib_reads_the_real_message_shape():
    t = traj(["ls", "python3 x.py"], [0, 1])
    assert lib.commands(t) == ["ls", "python3 x.py"]
    assert lib.tool_results(t) == [{"returncode": 0, "output": "out 0"}, {"returncode": 1, "output": "out 1"}]
    assert lib.n_steps(t) == 2 and lib.assistant_texts(t) == ["step 0", "step 1"]
    assert lib.reasoning_texts(t) == ["think", "think"] and lib.task_prompt(t) == "Problem: q"
    # tool_calls serialized as a string, and garbage, don't crash
    t["messages"][2]["tool_calls"] = json.dumps(t["messages"][2]["tool_calls"])
    assert lib.commands(t) == ["ls", "python3 x.py"]
    assert lib.commands({}) == [] and lib.commands({"messages": "nope"}) == []


# --------------------------------------------------------------------------- spec schema
def test_spec_requires_function_names_defined_in_the_module(tmp_path):
    md = write_spec(tmp_path, ["ran_python"])
    model_spec.validate_file(md)
    with pytest.raises(ValueError, match="identifiers"):
        model_spec.validate_file(write_spec(tmp_path, ["ran_python(traj)"], name="expr"))
    with pytest.raises(ValueError, match="not defined"):
        model_spec.validate_file(write_spec(tmp_path, ["nope"], name="undef"))
    with pytest.raises(ValueError, match="not found"):
        model_spec.validate_file(write_spec(tmp_path, ["ran_python"], py=None, name="nopy"))
    with pytest.raises(ValueError):
        model_spec.validate_file(write_spec(tmp_path, [], name="empty"))


# --------------------------------------------------------------------------- realize + screen
def test_realize_unscreened_without_traces(tmp_path):
    cand = tmp_path / "candidates" / "c1"
    write_spec(cand, ["ran_python"])
    ok, sr = realize_model_specs(cand, make_split(tmp_path, {"a": "2"}))
    assert ok
    assert json.loads((sr / "commit.json").read_text()) == {"sigma": "commit", "checks": ["ran_python"]}
    assert (sr / "commit.py").read_text() == CHECKS_PY
    assert json.loads((sr / "screen" / "commit.json").read_text())["status"].startswith("unscreened")


def test_screen_accepts_varying_checks_and_reports_correlation(tmp_path):
    cands = tmp_path / "candidates"
    make_traces(cands)
    cand = cands / "c1"
    write_spec(cand, ["ran_python", "few_steps"])
    ok, sr = realize_model_specs(cand, "unused")
    rep = json.loads((sr / "screen" / "commit.json").read_text())
    assert rep["status"] == "ok" and rep["n_traces"] == 12          # the eval-val trace is excluded
    assert rep["checks"]["ran_python"]["mean"] == 0.5
    assert rep["checks"]["ran_python"]["corr_with_reward"] == 1.0   # python <=> reward 1 here
    assert rep["checks"]["few_steps"]["distinct_values"] == [0.7, 0.8]
    assert (sr / "commit.json").exists() and (sr / "commit.py").exists()


@pytest.mark.parametrize("check,why", [
    ("always_true", "constant"),
    ("boom", "raises on 12/12"),
    ("quits", "raises on 12/12"),
    ("wrong_type", "raises on 12/12"),
])
def test_screen_rejects_useless_checks(tmp_path, check, why):
    cands = tmp_path / "candidates"
    make_traces(cands)
    cand = cands / "c1"
    write_spec(cand, ["ran_python", check])
    ok, sr = realize_model_specs(cand, "unused")
    assert not (sr / "commit.json").exists() and not (sr / "commit.py").exists()
    rep = json.loads((sr / "screen" / "commit.json").read_text())
    assert rep["status"] == "rejected" and why in rep["checks"][check]["verdict"]
    assert rep["checks"]["ran_python"]["verdict"] == "ok"


def test_screen_rejects_a_module_that_fails_to_import(tmp_path):
    cands = tmp_path / "candidates"
    make_traces(cands)
    write_spec(cands / "c1", ["f"], py="import not_a_real_module\ndef f(traj):\n    return True\n")
    ok, sr = realize_model_specs(cands / "c1", "unused")
    rep = json.loads((sr / "screen" / "commit.json").read_text())
    assert "failed to load" in rep["status"] and not (sr / "commit.json").exists()


def test_screen_cli(tmp_path):
    cands = tmp_path / "candidates"
    make_traces(cands)
    md = write_spec(cands / "c1", ["ran_python", "always_true"])
    r = subprocess.run([sys.executable, "-m", "mho.data.reward_synthesis", "screen", "--spec", str(md),
                        "--traces", str(cands)], env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stderr
    assert "ran_python: ok" in r.stdout and "always_true: rejected: constant" in r.stdout


# --------------------------------------------------------------------------- stage
def sigma_dir(tmp_path: Path, checks: list[str]) -> Path:
    sj = tmp_path / "sr" / "commit.json"
    sj.parent.mkdir(parents=True, exist_ok=True)
    sj.write_text(json.dumps({"sigma": "commit", "checks": checks}))
    sj.with_suffix(".py").write_text(CHECKS_PY)
    return sj


def test_stage_injects_rule_checks_and_per_task_answers(tmp_path):
    split = make_split(tmp_path, {"a": "2", "b": "\\frac{1}{3}", "c": "7"})
    out = tmp_path / "staged"
    assert rs.stage_tasks(split, sigma_dir(tmp_path, ["ran_python"]), out) == 3
    assert (out / "seed.jsonl").read_text() == (split / "seed.jsonl").read_text()
    for name, ans in {"a": "2", "b": "\\frac{1}{3}", "c": "7"}.items():
        tests = out / name / "tests"
        assert (tests / "base_test.sh").read_text() == BASE_TEST_SH
        assert (tests / "test.sh").read_text() == rs.WRAPPER_TEST_SH
        assert os.access(tests / "test.sh", os.X_OK)
        assert (tests / "reward_rule.py").read_text() == rs.RULE_PY.read_text()
        assert (tests / "sigma_checklib.py").read_text() == rs.CHECKLIB_PY.read_text()
        assert (tests / "sigma_checks.py").read_text() == CHECKS_PY
        cfg = json.loads((tests / "sigma_reward.json").read_text())
        assert cfg == {"sigma": "commit", "checks": ["ran_python"], "expected_answer": ans}
        assert (out / name / "environment" / "Dockerfile").exists()
        assert (split / name / "tests" / "test.sh").read_text() == BASE_TEST_SH   # base untouched
        assert not (split / name / "tests" / "sigma_checks.py").exists()


def test_stage_needs_the_checks_module(tmp_path):
    sj = sigma_dir(tmp_path, ["ran_python"])
    sj.with_suffix(".py").unlink()
    with pytest.raises(ValueError, match="checks module"):
        rs.stage_tasks(make_split(tmp_path, {"a": "2"}), sj, tmp_path / "staged")


def test_restage_starts_from_the_base_split(tmp_path):
    split = make_split(tmp_path, {"a": "2"})
    sj, out = sigma_dir(tmp_path, ["ran_python"]), tmp_path / "staged"
    rs.stage_tasks(split, sj, out)
    rs.stage_tasks(split, sj, out)                      # e.g. a retried train job
    assert (out / "a" / "tests" / "base_test.sh").read_text() == BASE_TEST_SH


def test_expected_answer_falls_back_to_test_py(tmp_path):
    split = make_split(tmp_path, {"a": "\\mu+1"})
    (split / "a" / "task.toml").write_text("[task]\nname = 'a'\n")
    assert rs._extract_expected_answer(split / "a") == "\\mu+1"


def test_stage_refuses_a_task_without_test_sh(tmp_path):
    split = make_split(tmp_path, {"a": "2", "b": "3"})
    (split / "b" / "tests" / "test.sh").unlink()
    with pytest.raises(ValueError, match="without tests/test.sh"):
        rs.stage_tasks(split, sigma_dir(tmp_path, ["ran_python"]), tmp_path / "staged")


def test_stage_cli(tmp_path):
    split = make_split(tmp_path, {"a": "2"})
    r = subprocess.run([sys.executable, "-m", "mho.data.reward_synthesis", "stage",
                        "--train-dir", str(split), "--sigma", str(sigma_dir(tmp_path, ["ran_python"])),
                        "--out", str(tmp_path / "o")],
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "staged 1 task(s)" in r.stdout and (tmp_path / "o" / "a" / "tests" / "sigma_checks.py").exists()


# --------------------------------------------------------------------------- reward rule (sandbox)
def run_rule(tmp_path, *, base_reward: str | None, checks: list[str], trajectory: dict | None,
             output: str = "2", expected: str = "2", checks_py: str | None = CHECKS_PY) -> dict:
    """Lay out /tests as stage does (in tmp_path/tests) and run the rule like test.sh does."""
    tests, vdir = tmp_path / "tests", tmp_path / "verifier"
    tests.mkdir(exist_ok=True)
    vdir.mkdir(exist_ok=True)
    if base_reward is not None:
        (vdir / "reward.txt").write_text(base_reward + "\n")
    (tests / "reward_rule.py").write_text(rs.RULE_PY.read_text())
    (tests / "sigma_checklib.py").write_text(rs.CHECKLIB_PY.read_text())
    if checks_py is not None:
        (tests / "sigma_checks.py").write_text(checks_py)
    cfg = tests / "sigma_reward.json"
    cfg.write_text(json.dumps({"sigma": "s", "checks": checks, "expected_answer": expected}))
    (tmp_path / "output.txt").write_text(output)
    tr = tmp_path / "trajectory.json"
    tr.write_text(json.dumps(trajectory) if trajectory is not None else "")
    r = subprocess.run([sys.executable, str(tests / "reward_rule.py"), str(cfg),
                        str(tmp_path / "output.txt"), str(tr)], cwd=tmp_path,
                       env={**os.environ, "MHO_VERIFIER_DIR": str(vdir), "MHO_SIGMA_CHECK_TIMEOUT": "1"},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    info = json.loads((vdir / "sigma_info.json").read_text())
    assert float((vdir / "reward.txt").read_text()) == pytest.approx(info["reward"], abs=1e-4)
    return info


PY_RUN = traj(["python3 solve.py"])          # 1 step: ran_python 1.0, few_steps 0.9
NO_RUN = traj(["ls", "cat x", "echo 2 > out", "ls", "ls"])   # ran_python 0.0, few_steps 0.5


@pytest.mark.parametrize("base,t,reward", [
    ("1", PY_RUN, 0.5 + 0.5 * (1.0 + 0.9) / 2),
    ("1", NO_RUN, 0.5 + 0.5 * (0.0 + 0.5) / 2),
    ("0", PY_RUN, 0.5 * (1.0 + 0.9) / 2),
    ("0", None, 0.5 * (0.0 + 1.0) / 2),      # empty trajectory: 0 steps -> few_steps 1.0
])
def test_rule_combines_base_correctness_and_checks(tmp_path, base, t, reward):
    info = run_rule(tmp_path, base_reward=base, checks=["ran_python", "few_steps"], trajectory=t)
    assert info["correct_from"] == "base_verifier"
    assert info["reward"] == pytest.approx(reward, abs=1e-4)
    assert (info["traj_chars"] > 0) == (t is not None)


def test_rule_survives_bad_checks(tmp_path):
    info = run_rule(tmp_path, base_reward="1", trajectory=PY_RUN,
                    checks=["ran_python", "boom", "quits", "slow", "wrong_type", "missing"])
    by = {c["check"]: c for c in info["checks"]}
    assert by["ran_python"] == {"check": "ran_python", "value": 1.0, "error": None}
    assert "RuntimeError" in by["boom"]["error"] and "SystemExit" in by["quits"]["error"]
    assert "timed out" in by["slow"]["error"] and "TypeError" in by["wrong_type"]["error"]
    assert "NameError" in by["missing"]["error"]
    assert info["reward"] == pytest.approx(0.5 + 0.5 * 1 / 6, abs=1e-4)


def test_rule_survives_a_broken_checks_module(tmp_path):
    info = run_rule(tmp_path, base_reward="1", trajectory=PY_RUN, checks=["f"],
                    checks_py="def f(traj:\n")
    assert "SyntaxError" in info["import_error"] and info["reward"] == 0.5


def test_rule_falls_back_to_expected_answer_on_bare_output(tmp_path):
    # no base reward.txt: the bare answer in output.txt (arxivmath style) is graded directly
    info = run_rule(tmp_path, base_reward=None, checks=["ran_python"], trajectory=PY_RUN,
                    output="2\n", expected="2")
    assert info["correct_from"] == "expected_answer" and info["correct"] == 1.0
    assert info["reward"] == 1.0


# --------------------------------------------------------------------------- proposer + train.sh
def test_model_proposer_is_told_where_to_write():
    note = model_target_note("candidate_3_2_1")
    assert "./candidate_3_2_1/model_specs/<name>.md" in note
    assert "./candidate_3_2_1/model_specs/<name>.py" in note and "reward_synthesis screen" in note


def test_train_sh_stages_when_sigma_reward_is_set():
    sh = (ROOT / "scripts" / "optimize" / "train.sh").read_text()
    block = sh[sh.index('if [ -n "${MHO_SIGMA_REWARD:-}" ]'):]
    block = block[:block.index("\nfi\n")]
    assert "mho.data.reward_synthesis stage" in block
    assert 'MATH_TRAIN_DIR="$SIGMA_TASKS_DIR"' in block and 'HARBOR_TASKS_DIR="$SIGMA_TASKS_DIR"' in block
    # staging happens before anything reads the task dirs
    assert sh.index("mho.data.reward_synthesis stage") < sh.index('--prompt-data "${MATH_TRAIN_DIR}/seed.jsonl"')
    assert sh.index("mho.data.reward_synthesis stage") < sh.index('--bind "${HARBOR_TASKS_DIR}:')
