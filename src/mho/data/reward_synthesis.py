"""Reward synthesis f: turn a model spec sigma into a sigma reward on the train tasks.

The proposer writes two files per capability:
  model_specs/<name>.md   the spec (YAML frontmatter); `checks:` lists function NAMES
  model_specs/<name>.py   those functions: def <check>(traj: dict) -> bool | float in [0, 1]
                          (helpers: `from sigma_checklib import *`, see sigma_checklib.py)

Two steps:
  1. realize (per candidate): validate the spec and its checks module, SCREEN the checks on the
     train-sample trajectories already in the run (reject a check that crashes or returns the
     same value on every trajectory: it can't change any advantage), then write
       <cand>/model/sigma_rewards/<name>.json   {sigma, checks}
       <cand>/model/sigma_rewards/<name>.py     the checks module
       <cand>/model/sigma_rewards/screen/<name>.json   the screen report
  2. stage (per training run, scripts/optimize/train.sh when MHO_SIGMA_REWARD is set): copy the
     train split to a fresh dir and, in every task's tests/ (what harbor uploads to /tests):
       tests/base_test.sh       the task's original test.sh, unchanged
       tests/test.sh            wrapper: run base_test.sh, then reward_rule.py
       tests/reward_rule.py     sigma_rule.py (fixed: the combination formula is ours)
       tests/sigma_checks.py    the proposer's checks module
       tests/sigma_checklib.py  trajectory helpers
       tests/sigma_reward.json  {sigma, checks, expected_answer: THIS task's answer}
     The base split is never modified.

In the sandbox: reward = max(0, min(1, 0.5 * correctness + 0.5 * mean(checks))), correctness
from the task's own verifier. See sigma_rule.py.

Usage:
  python -m mho.data.reward_synthesis screen --spec model_specs/<name>.md [--traces DIR ...]
  python -m mho.data.reward_synthesis --spec <spec.md> --candidate-dir <RUN>/candidates/<tag>
  python -m mho.data.reward_synthesis stage --train-dir <split> --sigma <sigma.json> --out <dir>
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULE_PY = HERE / "sigma_rule.py"          # -> tests/reward_rule.py
CHECKLIB_PY = HERE / "sigma_checklib.py"  # -> tests/sigma_checklib.py
TRAJ = "agent/mini-swe-agent.trajectory.json"

# Screening thresholds. Below MIN_TRACES the checks can't be judged and pass unscreened (first
# iteration of a run); above it a check must vary and must not raise on more than MAX_ERR_FRAC.
MIN_TRACES = int(os.environ.get("MHO_SIGMA_MIN_TRACES", "8"))
MAX_TRACES = int(os.environ.get("MHO_SIGMA_MAX_TRACES", "64"))
MAX_ERR_FRAC = 0.1
SCREEN_TIMEOUT_S = 900


# --------------------------------------------------------------------------- the checks module
def checks_module(spec_md: Path) -> Path:
    """The checks module that belongs to a spec: model_specs/<name>.py beside <name>.md."""
    return Path(spec_md).with_suffix(".py")


def defined_functions(py: Path) -> set[str]:
    """Top-level function names in a module, by parsing it (never importing it)."""
    try:
        tree = ast.parse(Path(py).read_text(encoding="utf-8"), filename=str(py))
    except SyntaxError as e:
        raise ValueError(f"{py}: does not parse: {e}") from e
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def produce_config(spec, checks_py: Path) -> dict:
    """f(sigma) -> {sigma, checks}: every name in spec.checks must be a top-level function in
    checks_py. Task-independent; stage_tasks adds each task's expected_answer. Raises ValueError."""
    if not spec.checks:
        raise ValueError(f"sigma '{spec.name}': `checks` is empty; list the check functions "
                         f"defined in {Path(checks_py).name}")
    if not Path(checks_py).is_file():
        raise ValueError(f"sigma '{spec.name}': checks module {checks_py} not found")
    missing = [c for c in spec.checks if c not in defined_functions(checks_py)]
    if missing:
        raise ValueError(f"sigma '{spec.name}': {missing} not defined in {Path(checks_py).name}")
    cfg = {"sigma": spec.name, "checks": list(spec.checks)}
    if spec.rubric:     # read by mho.data.task_select (MHO_SIGMA_USE=select|both)
        cfg["rubric"] = [r.model_dump() for r in spec.rubric]
    return cfg


# --------------------------------------------------------------------------- screening
def find_traces(roots, limit: int = MAX_TRACES) -> list[Path]:
    """Train-sample trajectories under each root, newest first: a candidates/ dir
    (<tag>/eval_before|eval_after/<trial>/...) or a dir of trial dirs (<trial>/...). Val-split
    evals are never used (they must stay hidden from the optimizer)."""
    found: dict[Path, float] = {}
    for root in roots:
        root = Path(root)
        for pat in (f"*/{TRAJ}", f"*/eval_before/*/{TRAJ}", f"*/eval_after/*/{TRAJ}"):
            for p in root.glob(pat):
                if "eval-val" in p.parts or "evaluation" in p.parts:
                    continue
                try:
                    found[p.resolve()] = p.stat().st_mtime
                except OSError:
                    pass
    return sorted(found, key=lambda p: -found[p])[:limit]


def _trial_reward(traj: Path) -> float | None:
    try:
        d = json.loads((traj.parents[1] / "result.json").read_text())
        return float(((d.get("verifier_result") or {}).get("rewards") or {}).get("reward"))
    except (OSError, ValueError, TypeError):
        return None


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    try:
        return round(statistics.correlation(xs, ys), 3)
    except (AttributeError, statistics.StatisticsError):    # py < 3.10 / degenerate
        return None


def screen(checks_py: Path, names: list[str], traces: list[Path]) -> tuple[bool, dict]:
    """Run each check over `traces` in a subprocess (the sandbox's own code path, sigma_rule.py)
    and judge it. Returns (ok, report); ok is False if any check is rejected."""
    report: dict = {"n_traces": len(traces), "checks": {}}
    if len(traces) < MIN_TRACES:
        report["status"] = f"unscreened: {len(traces)} trajectories < {MIN_TRACES}"
        return True, report
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "traces.txt"
        lst.write_text("\n".join(str(t) for t in traces) + "\n")
        try:
            r = subprocess.run([sys.executable, "-I", str(RULE_PY), "screen", str(checks_py),
                                ",".join(names), str(lst)],
                               cwd=td, capture_output=True, text=True, timeout=SCREEN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            report["status"] = f"rejected: screening took over {SCREEN_TIMEOUT_S}s"
            return False, report
    if r.returncode != 0:
        report["status"] = "rejected: the checks module failed to load: " + r.stderr.strip()[-600:]
        return False, report
    raw = json.loads(r.stdout.strip().splitlines()[-1])
    rewards = [_trial_reward(t) for t in traces]
    ok = True
    for name in names:
        rows = raw[name]                       # one [value, error] per trace, in trace order
        vals = [v for v, e in rows if e is None]
        errs = [e for v, e in rows if e is not None]
        n = len(rows)
        rep = {"n": n, "n_errors": len(errs),
               "mean": round(statistics.fmean(vals), 4) if vals else None,
               "distinct_values": sorted(set(vals))[:10]}
        if errs:
            rep["first_error"] = errs[0]
        # Informational, not gated: does the check agree with the trial's recorded reward?
        pairs = [(v, w) for (v, e), w in zip(rows, rewards) if e is None and w is not None]
        rep["corr_with_reward"] = _corr([p[0] for p in pairs], [p[1] for p in pairs])
        if n and len(errs) / n > MAX_ERR_FRAC:
            rep["verdict"] = f"rejected: raises on {len(errs)}/{n} trajectories"
        elif len(set(vals)) < 2:
            const = vals[0] if vals else None
            rep["verdict"] = (f"rejected: constant ({const}) on all {len(vals)} trajectories, "
                              "so it adds the same reward to every rollout and changes no advantage")
        else:
            rep["verdict"] = "ok"
        ok &= rep["verdict"] == "ok"
        report["checks"][name] = rep
    report["status"] = "ok" if ok else "rejected"
    return ok, report


# --------------------------------------------------------------------------- realize
def realize(spec_md: Path, out_dir: Path, traces: list[Path]) -> tuple[bool, dict]:
    """Realize one spec into out_dir (<cand>/model/sigma_rewards): validate, screen, and only if
    every check passes write <name>.json + <name>.py. The screen report is always written to
    out_dir/screen/<name>.json. Returns (ok, report)."""
    from mho.optimizer import model_spec
    spec, _ = model_spec.parse_spec(Path(spec_md))
    py = checks_module(spec_md)
    cfg = produce_config(spec, py)
    ok, report = screen(py, cfg["checks"], traces)
    if spec.rubric:     # realization: select -- the rubric must also separate tasks
        from mho.data import task_select
        r_ok, report["rubric"] = task_select.screen_rubric(cfg["rubric"], traces)
        # Gate only when training will select on it (MHO_SIGMA_USE, see train.sh).
        if not r_ok and os.environ.get("MHO_SIGMA_USE", "reward") != "reward":
            ok, report["status"] = False, "rubric " + report["rubric"]["status"]
    (out_dir / "screen").mkdir(parents=True, exist_ok=True)
    (out_dir / "screen" / f"{spec.name}.json").write_text(json.dumps(report, indent=2) + "\n")
    if ok:
        shutil.copy2(py, out_dir / f"{spec.name}.py")
        (out_dir / f"{spec.name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return ok, report


# --------------------------------------------------------------------------- stage
def _extract_expected_answer(task: Path) -> str:
    """The task's expected answer: task.toml [metadata] answer, else test.py EXPECTED_ANSWER.
    Both are parsed (TOML / Python literal), not regexed: LaTeX answers are escaped in the files
    ("\\\\frac{1}{3}") and must come back unescaped (\\frac{1}{3})."""
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


# The staged tests/test.sh. No `set -e`: the base verifier exits 1 on a wrong answer, and the
# sigma rule must still run. Harbor reads the reward.txt reward_rule.py writes last.
WRAPPER_TEST_SH = """#!/bin/bash
# sigma-reward wrapper (generated by mho.data.reward_synthesis stage). The task's own verifier is
# base_test.sh; it writes /logs/verifier/reward.txt (= answer correctness), then reward_rule.py
# combines that with the sigma checks (sigma_checks.py) and overwrites reward.txt.
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
    every task's tests/ (see the module docstring). The checks module is the .py beside
    sigma_json. Top-level files (seed.jsonl) are copied as-is. Returns the number of tasks staged;
    raises ValueError if the checks module is missing or any task has no tests/test.sh."""
    train_dir, out_dir, sigma_json = Path(train_dir), Path(out_dir), Path(sigma_json)
    cfg = json.loads(sigma_json.read_text())
    checks_py = sigma_json.with_suffix(".py")
    if not checks_py.is_file():
        raise ValueError(f"checks module {checks_py} not found (realize writes it beside {sigma_json.name})")
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
        shutil.copyfile(RULE_PY, tests / "reward_rule.py")
        shutil.copyfile(CHECKLIB_PY, tests / "sigma_checklib.py")
        shutil.copyfile(checks_py, tests / "sigma_checks.py")
        task_cfg = {"sigma": cfg.get("sigma", ""), "checks": list(cfg.get("checks", [])),
                    "expected_answer": _extract_expected_answer(task)}
        (tests / "sigma_reward.json").write_text(json.dumps(task_cfg, indent=2) + "\n")
    return len(tasks)


# --------------------------------------------------------------------------- CLI
def _print_report(report: dict) -> None:
    print(f"[screen] {report['status']}  ({report['n_traces']} trajectories)")
    for name, rep in report["checks"].items():
        print(f"  {name}: {rep['verdict']}  mean={rep['mean']} distinct={rep['distinct_values']}"
              f" errors={rep['n_errors']}/{rep['n']} corr_with_reward={rep.get('corr_with_reward')}")
        if rep.get("first_error"):
            print(f"    first error: {rep['first_error']}")


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
    if argv[:1] == ["screen"]:
        p = argparse.ArgumentParser(prog="reward_synthesis screen",
                                    description="run a spec's checks over existing trajectories")
        p.add_argument("--spec", required=True, type=Path, help="model_specs/<name>.md")
        p.add_argument("--traces", nargs="*", type=Path, default=[Path(".")],
                       help="candidates/ dir(s) or dir(s) of trial dirs (default: cwd)")
        a = p.parse_args(argv[1:])
        try:
            spec, _ = model_spec.parse_spec(a.spec)
            cfg = produce_config(spec, checks_module(a.spec))
        except ValueError as e:
            print(f"[screen] INVALID: {e}", file=sys.stderr)
            return 2
        ok, report = screen(checks_module(a.spec), cfg["checks"], find_traces(a.traces))
        _print_report(report)
        return 0 if ok else 2
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True, type=Path, help="model spec (model_specs/<name>.md)")
    p.add_argument("--candidate-dir", required=True, type=Path,
                   help="candidate unit dir (<RUN>/candidates/<tag>/); writes the sigma reward to "
                        "<candidate>/model/sigma_rewards/; screens on <RUN>/candidates/")
    p.add_argument("--traces", nargs="*", type=Path, default=None,
                   help="where to find trajectories to screen on (default: the candidate's parent)")
    a = p.parse_args(argv)
    ok, report = realize(a.spec, a.candidate_dir / "model" / "sigma_rewards",
                         find_traces(a.traces or [a.candidate_dir.parent]))
    _print_report(report)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
