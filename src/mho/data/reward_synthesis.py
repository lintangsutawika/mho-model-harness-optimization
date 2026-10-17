"""Reward synthesis f: given a train sample x and a model spec sigma, produce the sigma config.

The realized artifact is a single small `sigma_reward.json` (the sigma config: expected answer +
checks) designed to be INJECTED into a base task's verifier just-in-time. The base task's verifier
is the shared generic `reward_rule.py`, which reads this config and applies the sigma reward.
Each sigma-sample = one small config file (keyed by task dir name); the base data is reused
untouched.

  reward_rule.py (shared, generic)  +  sigma_reward.json (per-sigma config)  ->  sigma reward

reward_rule.py (tests/reward_rule.py, generic) reads sigma_reward.json:
  - expected_answer  (to score answer correctness)
  - checks           (the explicit, checkable sigma conditions over output + trajectory)
and computes:  reward = max(0, min(1, 0.5 * answer_correctness + 0.5 * checks_passed_fraction))
writing /logs/verifier/reward.txt (float [0,1]); harbor parses it via float().

Usage:
  # emit the sigma config into a candidate unit's sigma_rewards map (for JIT injection):
  python -m mho.data.reward_synthesis --task <x dir> --spec <model_spec.md> --candidate-dir <RUN>/candidates/<tag>
  #   -> writes <candidate>/sigma_rewards/<task dir name>.json
  # or emit a single config to a dir:
  python -m mho.data.reward_synthesis --task <x dir> --spec <model_spec.md> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
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
"""
import json
import re
import sys
from pathlib import Path

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


def run(manifest: str, output_path: str, traj_path: str) -> int:
    sigma, expected, checks = _load_config(manifest)
    output = _read(output_path)
    traj = _read(traj_path)
    cand = _extract_answer(output)
    corr = _correctness(cand, expected)
    results = []
    for i, expr in enumerate(checks):
        try:
            ok = bool(eval(expr, {"__builtins__": {}}, {"output": output, "traj": traj}))  # noqa: S307
        except Exception as e:
            ok = False
            expr = f"{expr}  # ERROR: {e}"
        results.append({"i": i, "check": expr, "passed": ok})
    frac = (sum(1 for c in results if c["passed"]) / len(results)) if results else 0.0
    reward = max(0.0, min(1.0, 0.5 * corr + 0.5 * frac))
    info = {"sigma": sigma, "correct": corr, "checks_passed": frac, "checks": results, "reward": reward}
    out_dir = Path("/logs/verifier")
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


def _extract_expected_answer(task: Path) -> str:
    """Grab the expected answer from x: task.toml [metadata] answer, else test.py EXPECTED_ANSWER."""
    toml = task / "task.toml"
    if toml.exists():
        m = re.search(r"answer\s*=\s*[\"']?([^\"'\\n]+)", toml.read_text())
        if m:
            return m.group(1).strip()
    testpy = task / "tests" / "test.py"
    if testpy.exists():
        m = re.search(r"EXPECTED_ANSWER\s*=\s*[\"']([^\"']*)", testpy.read_text())
        if m:
            return m.group(1)
    return ""


def produce_config(task: Path, spec) -> dict:
    """f(x, sigma) -> the sigma config (sigma_reward.json dict) to inject into a base task."""
    return {"sigma": spec.name,
            "from_task": str(task),
            "expected_answer": _extract_expected_answer(task),
            "checks": spec.checks}


def materialize_rule(out: Path) -> Path:
    """Write the shared generic reward rule (tests/reward_rule.py) if not present."""
    tests = out / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    rule_path = tests / "reward_rule.py"
    if not rule_path.exists():
        rule_path.write_text(RULE_TEMPLATE, encoding="utf-8")
    return rule_path


def main(argv: list[str] | None = None) -> int:
    from mho.optimizer import model_spec
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True, type=Path, help="source task dir x (instruction.md + task.toml)")
    p.add_argument("--spec", required=True, type=Path, help="model spec (model_specs/<name>.md)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--out", type=Path, help="output dir (writes sigma_reward.json into it)")
    g.add_argument("--candidate-dir", type=Path,
                   help="candidate unit dir (<RUN>/candidates/<tag>/); writes the sigma config to "
                        "<candidate>/model/sigma_rewards/<sigma>.json")
    args = p.parse_args(argv)
    spec, _body = model_spec.parse_spec(args.spec)  # validate + load
    config = produce_config(args.task, spec)
    if args.candidate_dir:
        # One sigma config per candidate, keyed by the sigma (check) name, co-located with the
        # model's training artifacts: candidates/<tag>/model/sigma_rewards/<sigma>.json
        dest_dir = args.candidate_dir / "model" / "sigma_rewards"
        dest = dest_dir / f"{spec.name}.json"
    else:
        out = args.out or (args.task / "tests")
        dest_dir = out if out.is_dir() else out.parent
        dest = dest_dir / "sigma_reward.json"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[reward_synthesis] sigma config -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())