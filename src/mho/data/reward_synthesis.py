"""Reward synthesis f: turn a model spec sigma into a sigma reward on the train tasks.

Two steps:
  1. realize (per candidate): spec -> <cand>/model/sigma_rewards/<sigma>.json = {sigma, checks}.
     Task-independent: the same checks apply to every train task.
  2. stage (per training run, scripts/optimize/train.sh when MHO_SIGMA_REWARD is set): copy the
     train split to a fresh dir and, in every task's tests/ (the only part of a task harbor
     uploads to the sandbox, at /tests):
       tests/base_test.sh      the task's original test.sh, unchanged
       tests/test.sh           wrapper: run base_test.sh, then reward_rule.py
       tests/reward_rule.py    RULE_TEMPLATE below
       tests/sigma_reward.json {sigma, checks, expected_answer: THIS task's answer}
     The base split is never modified.

In the sandbox, reward_rule.py takes answer correctness from the task's own verifier (the
reward.txt base_test.sh just wrote; falls back to comparing against expected_answer when there
is none) and computes
    reward = max(0, min(1, 0.5 * correctness + 0.5 * checks_passed_fraction))
into /logs/verifier/reward.txt (float [0,1]); harbor parses it via float().

Usage:
  python -m mho.data.reward_synthesis --spec <model_spec.md> --candidate-dir <RUN>/candidates/<tag>
  python -m mho.data.reward_synthesis stage --train-dir <split> --sigma <sigma.json> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# The SHARED, generic executable reward rule. It reads its config (expected answer + sigma-checks)
# from a sigma_reward.json passed as an arg, and takes the artifact paths as arguments, so it can
# be used by ANY harbor task whose test.sh injects a config.
RULE_TEMPLATE = r'''#!/usr/bin/env python3
"""Generic executable reward rule (shared across sigma realizations).

Usage: reward_rule.py <sigma_reward.json> <output_path> <trajectory_path>
Reads expected answer + sigma-checks from the JSON config, scores answer correctness and the
checks over the output/trajectory, and writes /logs/verifier/reward.txt (float [0,1]).
Answer correctness comes from the task's own verifier when it already ran (the staged test.sh
runs base_test.sh first, which writes reward.txt); else from comparing to expected_answer.
"""
import json
import os
import re
import sys
from pathlib import Path

VERIFIER_DIR = Path(os.environ.get("MHO_VERIFIER_DIR", "/logs/verifier"))

ANSWER_RE = re.compile(r"(?:^|\n)\s*Answer:\s*(.+?)\s*$\s*", re.MULTILINE)


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _load_config(manifest: str) -> tuple[str, str, list]:
    data = json.loads(_read(manifest) or "{}")
    return (str(data.get("sigma", "")), str(data.get("expected_answer", "")),
            list(data.get("checks", [])))


def _extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    return matches[-1].lstrip("$").strip() if matches else ""


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _correctness(cand: str, exp: str) -> float:
    if not cand or not exp:
        return 0.0
    try:
        from math_verify import parse, verify
        if verify(parse(exp), parse(cand)):
            return 1.0
    except Exception:
        pass
    try:
        import sympy as sp
        from sympy.parsing.latex import parse_latex
        if sp.simplify(parse_latex(exp) - parse_latex(cand)) == 0:
            return 1.0
    except Exception:
        pass
    return float(_norm(cand) == _norm(exp))

# --- standard sigma-check predicate library -------------------------------
# The proposer emits named trajectory/output predicates in a sigma spec's `checks`.
# These are the IMPLEMENTED vocabulary; reward_rule evaluates checks against it. A check
# with an unknown name -> NameError in the eval below -> that check fails (as before), but
# anything listed here resolves. Each predicate takes (traj: str) and/or (output: str).

def _msgs(traj: str) -> list:
    try:
        d = json.loads(traj)
    except Exception:
        return []
    m = d.get("messages") if isinstance(d, dict) else d
    return m if isinstance(m, list) else []

def _has_answer_line(text: str) -> bool:
    return bool(ANSWER_RE.search(str(text)))

def has_final_agent_answer_line(traj: str) -> bool:
    """An assistant turn in the trajectory carries an 'Answer:' line (the agent committed
    a final answer to the graded artifact / its own reply)."""
    return any(
        isinstance(m, dict) and m.get("role") == "assistant"
        and _has_answer_line(m.get("content", ""))
        for m in _msgs(traj)
    )

def final_agent_answer_present_independent_of_tool_output(traj: str) -> bool:
    """A final-answer line exists in an assistant turn regardless of what tool outputs
    echoed -- i.e. the answer is the agent's own, not merely a tool reflect."""
    return has_final_agent_answer_line(traj)

def traj_has_text(traj: str, needle: str) -> bool:
    """Generic: the raw trajectory text contains `needle` (substring search)."""
    return needle in traj

CHECK_NS = {
    "has_final_agent_answer_line": has_final_agent_answer_line,
    "final_agent_answer_present_independent_of_tool_output": final_agent_answer_present_independent_of_tool_output,
    "traj_has_text": traj_has_text,
}


def _base_correctness():
    """The base verifier's reward (reward.txt it wrote before us), or None if it wrote none."""
    try:
        return max(0.0, min(1.0, float(_read(str(VERIFIER_DIR / "reward.txt")).strip())))
    except ValueError:
        return None


def run(manifest: str, output_path: str, traj_path: str) -> int:
    sigma, expected, checks = _load_config(manifest)
    output = _read(output_path)
    traj = _read(traj_path)
    corr, source = _base_correctness(), "base_verifier"
    if corr is None:
        corr, source = _correctness(_extract_answer(output) or output.strip(), expected), "expected_answer"
    results = []
    for i, expr in enumerate(checks):
        try:
            # The standard predicate namespace (CHECK_NS) is exposed alongside the raw
            # `output`/`traj` strings, so named sigma-checks resolve to real implementations.
            ok = bool(eval(expr, {"__builtins__": {}, **CHECK_NS},
                           {"output": output, "traj": traj}))  # noqa: S307
        except Exception as e:
            ok = False
            expr = f"{expr}  # ERROR: {e}"
        results.append({"i": i, "check": expr, "passed": ok})
    frac = (sum(1 for c in results if c["passed"]) / len(results)) if results else 0.0
    reward = max(0.0, min(1.0, 0.5 * corr + 0.5 * frac))
    info = {"sigma": sigma, "correct": corr, "correct_from": source, "checks_passed": frac,
            "checks": results, "reward": reward,
            # what the rule could see: 0 chars means the checks ran against nothing
            "traj_path": traj_path, "traj_chars": len(traj),
            "output_path": output_path, "output_chars": len(output)}
    out_dir = VERIFIER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reward.txt").write_text(f"{reward:.4f}\n")
    (out_dir / "sigma_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0 if reward >= 0.5 else 1


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: reward_rule.py <sigma_reward.json> <output_path> <trajectory_path>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run(sys.argv[1], sys.argv[2], sys.argv[3]))
'''


# --- realized-check vocabulary + validation ---------------------------------
# The reward runtime (reward_rule.py, rendered from RULE_TEMPLATE above) evaluates a sigma
# config's `checks` against a FIXED namespace of implemented predicates (its CHECK_NS). For the
# realization to emit only RUNNABLE checks, this module mirrors that same vocabulary and
# validates the proposer's checks against it at realization time. These names MUST stay in sync
# with the CHECK_NS emitted in RULE_TEMPLATE.
CHECK_NS = {
    "has_final_agent_answer_line",
    "final_agent_answer_present_independent_of_tool_output",
    "traj_has_text",
}
_FUNC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def unsupported_checks(checks: list[str]) -> list[str]:
    """Return the checks whose leading function name is not in the implemented CHECK_NS."""
    bad = []
    for c in checks:
        m = _FUNC_RE.match(str(c))
        name = m.group(1) if m else ""
        if not name or name not in CHECK_NS:
            bad.append(str(c))
    return bad


def _extract_expected_answer(task: Path) -> str:
    """The task's expected answer: task.toml [metadata] answer, else test.py EXPECTED_ANSWER.
    Both are parsed (TOML / Python literal), not regexed: LaTeX answers are escaped in the files
    ("\\\\frac{1}{3}") and must come back unescaped (\\frac{1}{3})."""
    import ast
    try:
        import tomllib
    except ModuleNotFoundError:          # py < 3.11
        import tomli as tomllib
    toml = task / "task.toml"
    if toml.exists():
        try:
            ans = (tomllib.loads(toml.read_text()).get("metadata") or {}).get("answer")
        except tomllib.TOMLDecodeError:
            ans = None
        if ans is not None:
            return str(ans).strip()
    testpy = task / "tests" / "test.py"
    if testpy.exists():
        try:
            for node in ast.parse(testpy.read_text()).body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and getattr(node.targets[0], "id", None) == "EXPECTED_ANSWER"):
                    return str(ast.literal_eval(node.value))
        except (SyntaxError, ValueError):
            pass
    return ""


def produce_config(spec) -> dict:
    """f(sigma) -> the candidate's sigma config {sigma, checks}. Task-independent; stage_tasks
    adds each task's own expected_answer when it injects the config.

    The checks are validated against the implemented CHECK_NS: an unsupported check name is
    rejected (ValueError), so realization never emits a config the reward runtime cannot run.
    """
    bad = unsupported_checks(spec.checks)
    if bad:
        raise ValueError(
            f"sigma '{spec.name}': unsupported check(s) -> {bad}. Implement the predicate in "
            f"reward_synthesis.CHECK_NS (and the reward_rule template) or constrain the "
            f"proposer to the supported vocabulary: {sorted(CHECK_NS)}")
    return {"sigma": spec.name, "checks": list(spec.checks)}


# The staged tests/test.sh. No `set -e`: the base verifier exits 1 on a wrong answer, and the
# sigma rule must still run. Harbor reads the reward.txt reward_rule.py writes last.
WRAPPER_TEST_SH = """#!/bin/bash
# sigma-reward wrapper (generated by mho.data.reward_synthesis stage). The task's own verifier is
# base_test.sh; it writes /logs/verifier/reward.txt (= answer correctness), then reward_rule.py
# combines that with the sigma checks and overwrites reward.txt.
# The agent writes its trajectory to the agent logs dir (micro-swe-agent:
# /logs/agent/mini-swe-agent.trajectory.json), not /workspace. sigma_info.json records how much
# of it the rule actually saw (traj_chars), so an empty trajectory is visible, not silent.
mkdir -p /logs/verifier
bash /tests/base_test.sh
echo "=== base verifier exit $? ; sigma reward rule ==="
exec python3 /tests/reward_rule.py /tests/sigma_reward.json \\
    "${OUTPUT_ARTIFACT:-/workspace/output.txt}" "${TRAJ_ARTIFACT:-/logs/agent/mini-swe-agent.trajectory.json}"
"""


def stage_tasks(train_dir: Path, sigma_json: Path, out_dir: Path) -> int:
    """Copy the train split `train_dir` to a fresh `out_dir` with the sigma reward injected into
    every task's tests/ (see the module docstring). Top-level files (seed.jsonl) are copied as-is.
    Returns the number of tasks staged; raises ValueError if any task has no tests/test.sh."""
    train_dir, out_dir = Path(train_dir), Path(out_dir)
    cfg = json.loads(Path(sigma_json).read_text())
    tasks = sorted(p for p in train_dir.iterdir() if p.is_dir())
    missing = [p.name for p in tasks if not (p / "tests" / "test.sh").is_file()]
    if missing:
        raise ValueError(f"{len(missing)} task(s) without tests/test.sh, e.g. {missing[:3]}")
    if out_dir.exists():
        shutil.rmtree(out_dir)          # always from the base split: never stack wrappers
    out_dir.mkdir(parents=True)
    for p in train_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, out_dir / p.name)
    for task in tasks:
        dst = out_dir / task.name
        shutil.copytree(task, dst, symlinks=True)
        tests = dst / "tests"
        (tests / "test.sh").rename(tests / "base_test.sh")
        (tests / "test.sh").write_text(WRAPPER_TEST_SH)
        (tests / "test.sh").chmod(0o755)
        (tests / "reward_rule.py").write_text(RULE_TEMPLATE, encoding="utf-8")
        task_cfg = {"sigma": cfg.get("sigma", ""), "checks": list(cfg.get("checks", [])),
                    "expected_answer": _extract_expected_answer(task)}
        (tests / "sigma_reward.json").write_text(json.dumps(task_cfg, indent=2) + "\n")
    return len(tasks)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["stage"]:
        p = argparse.ArgumentParser(prog="reward_synthesis stage",
                                    description="stage a train split with a sigma reward injected")
        p.add_argument("--train-dir", required=True, type=Path)
        p.add_argument("--sigma", required=True, type=Path, help="<cand>/model/sigma_rewards/<s>.json")
        p.add_argument("--out", required=True, type=Path)
        a = p.parse_args(argv[1:])
        n = stage_tasks(a.train_dir, a.sigma, a.out)
        print(f"[reward_synthesis] staged {n} task(s) with sigma {a.sigma.name} -> {a.out}")
        return 0
    from mho.optimizer import model_spec
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True, type=Path, help="model spec (model_specs/<name>.md)")
    p.add_argument("--candidate-dir", required=True, type=Path,
                   help="candidate unit dir (<RUN>/candidates/<tag>/); writes the sigma config to "
                        "<candidate>/model/sigma_rewards/<sigma>.json")
    args = p.parse_args(argv)
    spec, _body = model_spec.parse_spec(args.spec)  # validate + load
    dest = args.candidate_dir / "model" / "sigma_rewards" / f"{spec.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(produce_config(spec), indent=2) + "\n", encoding="utf-8")
    print(f"[reward_synthesis] sigma config -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())