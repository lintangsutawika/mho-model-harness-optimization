#!/usr/bin/env python3
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
