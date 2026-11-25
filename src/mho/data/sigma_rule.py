#!/usr/bin/env python3
"""The sigma reward rule. Stdlib only: stage_tasks copies this file into every task's tests/ as
reward_rule.py, next to sigma_checks.py (the proposer's check functions), sigma_checklib.py and
sigma_reward.json, and the staged test.sh runs it after the task's own verifier.

  reward_rule.py <sigma_reward.json> <output_path> <trajectory_path>
      Score one rollout: correctness from the task's own verifier (the reward.txt base_test.sh
      just wrote; falls back to comparing output against expected_answer when there is none),
      checks = mean of the check functions on the trajectory, and
          reward = max(0, min(1, 0.5 * correctness + 0.5 * checks))
      into $MHO_VERIFIER_DIR/reward.txt (default /logs/verifier), with sigma_info.json beside it.

  reward_rule.py screen <checks.py> <name,name,...> <trajectory_list.txt>
      Run the named checks over many trajectories (one path per line) and print
      {name: [[value | null, error | null], ...one per trajectory, in order]} as JSON. Realization uses this to reject checks
      that crash or never vary, with exactly the code path the sandbox runs.

The combination weights are fixed here, not by the proposer: correctness keeps half the reward
whatever the checks say.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import signal
import sys
import threading
from pathlib import Path

VERIFIER_DIR = Path(os.environ.get("MHO_VERIFIER_DIR", "/logs/verifier"))
CHECK_TIMEOUT_S = int(os.environ.get("MHO_SIGMA_CHECK_TIMEOUT", "10"))
W_CORRECT = 0.5

ANSWER_RE = re.compile(r"(?:^|\n)\s*Answer:\s*(.+?)\s*$\s*", re.MULTILINE)


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _load_traj(text: str):
    try:
        return json.loads(text) if text.strip() else {}
    except ValueError:
        return {}


# --------------------------------------------------------------------------- checks
class CheckTimeout(Exception):
    pass


def load_checks(path: str):
    """Import the proposer's checks module. Its dir and this rule's dir go on sys.path so
    `from sigma_checklib import *` resolves to the copy shipped beside them (explicitly: under
    `python -I` the script's own dir is not on sys.path)."""
    for d in (str(Path(__file__).resolve().parent), str(Path(path).resolve().parent)):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("sigma_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _coerce(v) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(v):
        return max(0.0, min(1.0, float(v)))
    raise TypeError(f"returned {type(v).__name__} {v!r:.40}; a check must return bool or a float in [0, 1]")


def call_check(fn, traj, timeout: int = CHECK_TIMEOUT_S) -> float:
    """fn(traj) as a float in [0, 1]. Raises on an exception, a bad return type, or a timeout."""
    use_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if use_alarm:
        def _raise(_sig, _frm):
            raise CheckTimeout(f"timed out after {timeout}s")
        old = signal.signal(signal.SIGALRM, _raise)
        signal.alarm(timeout)
    try:
        return _coerce(fn(traj))
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def run_checks(mod, names: list, traj) -> list:
    out = []
    for name in names:
        fn = getattr(mod, name, None) if mod is not None else None
        try:
            if not callable(fn):
                raise NameError(f"{name} is not defined in the checks module")
            out.append({"check": name, "value": call_check(fn, traj), "error": None})
        except KeyboardInterrupt:
            raise
        except BaseException as e:            # incl. SystemExit: a check never ends the verifier
            out.append({"check": name, "value": 0.0, "error": f"{type(e).__name__}: {e}"[:300]})
    return out


# --------------------------------------------------------------------------- correctness
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


def _base_correctness():
    """The base verifier's reward (the reward.txt it wrote before us), or None if it wrote none."""
    try:
        return max(0.0, min(1.0, float(_read(str(VERIFIER_DIR / "reward.txt")).strip())))
    except ValueError:
        return None


# --------------------------------------------------------------------------- entry points
def run(manifest: str, output_path: str, traj_path: str) -> int:
    cfg = json.loads(_read(manifest) or "{}")
    names = [str(n) for n in cfg.get("checks", [])]
    output, traj_text = _read(output_path), _read(traj_path)
    corr, source = _base_correctness(), "base_verifier"
    if corr is None:
        corr = _correctness(_extract_answer(output) or output.strip(), str(cfg.get("expected_answer", "")))
        source = "expected_answer"
    checks_py = Path(manifest).resolve().parent / "sigma_checks.py"
    mod, import_error = None, None
    try:
        mod = load_checks(str(checks_py))
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        import_error = f"{type(e).__name__}: {e}"[:300]
    results = run_checks(mod, names, _load_traj(traj_text))
    frac = sum(r["value"] for r in results) / len(results) if results else 0.0
    reward = max(0.0, min(1.0, W_CORRECT * corr + (1 - W_CORRECT) * frac))
    info = {"sigma": cfg.get("sigma", ""), "correct": corr, "correct_from": source,
            "checks_score": frac, "checks": results, "import_error": import_error, "reward": reward,
            # what the rule could see: 0 chars means the checks ran against nothing
            "traj_path": traj_path, "traj_chars": len(traj_text),
            "output_path": output_path, "output_chars": len(output)}
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "reward.txt").write_text(f"{reward:.4f}\n")
    (VERIFIER_DIR / "sigma_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "checks"}, indent=2))
    return 0


def screen(checks_py: str, names_csv: str, traj_list: str) -> int:
    names = [n for n in names_csv.split(",") if n]
    mod = load_checks(checks_py)          # an import error propagates: the whole config is bad
    res = {n: [] for n in names}           # per check: one [value | null, error | null] per trajectory
    for line in _read(traj_list).splitlines():
        if not line.strip():
            continue
        traj = _load_traj(_read(line.strip()))
        for r in run_checks(mod, names, traj):
            res[r["check"]].append([None, r["error"]] if r["error"] else [r["value"], None])
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["screen"] and len(sys.argv) == 5:
        raise SystemExit(screen(*sys.argv[2:5]))
    if len(sys.argv) != 4:
        print("usage: reward_rule.py <sigma_reward.json> <output_path> <trajectory_path>\n"
              "       reward_rule.py screen <checks.py> <name,...> <trajectory_list.txt>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run(*sys.argv[1:4]))
