"""Task selection (realization: select): keep the train tasks that a model spec's rubric says
exercise its capability.

The rubric lives in the spec (model_spec.RubricItem) and is carried into the realized
<cand>/model/sigma_rewards/<name>.json by reward_synthesis.produce_config:
    "rubric": [{"criterion": str, "keywords": [str, ...], "weight": float}, ...]

Scoring is lexical, so it costs nothing per task: a criterion is met when any of its keywords
occurs in the task's instruction.md (case-insensitive, whole words; spaces in a keyword match
any whitespace). score = weight of the met criteria / total weight, in [0, 1]. Tasks with
score >= threshold are copied to the output split; the base split is never modified.

The output split holds the kept task dirs, the base split's top-level files (seed.jsonl
filtered to the kept tasks), and select_report.json (every task's score and matched criteria,
plus the wall time per task, to cost a scorer before scaling it up).

Screening (realize, and the proposer while it writes the rubric): the train split itself is not
in the proposer's workspace, so the rubric is tried on the tasks of the run's train-sample
trials instead. Each trial's task text is its first user message with the harness template
(the prefix and suffix every trial shares) cut off. A rubric that keeps none or all of those
tasks selects nothing and is rejected. The report also gives, for kept vs dropped tasks, the
mean reward and the share of tasks whose trials all failed or all passed: the curation target
is tasks the model sometimes solves, so a good rubric keeps fewer all-0 / all-1 tasks.

Usage:
  python -m mho.data.task_select --train-dir <split> --sigma <sigma.json> --out <dir>
                                 [--threshold 0.5] [--dry-run]
  python -m mho.data.task_select screen --spec model_specs/<name>.md [--traces DIR ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

THRESHOLD = float(os.environ.get("MHO_SELECT_THRESHOLD", "0.5"))
REPORT = "select_report.json"


def _pattern(keyword: str) -> re.Pattern:
    words = [re.escape(w) for w in keyword.lower().split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)")


def compile_rubric(rubric: list[dict]) -> list[tuple[str, float, list[re.Pattern]]]:
    if not rubric:
        raise ValueError("empty rubric: nothing to select on")
    return [(r["criterion"], float(r.get("weight", 1.0)), [_pattern(k) for k in r["keywords"]])
            for r in rubric]


def score_text(text: str, rubric) -> tuple[float, list[str]]:
    """(score in [0, 1], criteria met) for one instruction under a compiled rubric."""
    text = text.lower()
    met = [c for c, _, pats in rubric if any(p.search(text) for p in pats)]
    total = sum(w for _, w, _ in rubric)
    return sum(w for c, w, _ in rubric if c in met) / total, met


def _instance_id(row: dict) -> str | None:
    return (row.get("metadata") or {}).get("instance_id")


def select_tasks(train_dir: Path, sigma_json: Path, out_dir: Path | None,
                 threshold: float = THRESHOLD) -> dict:
    """Score every task in train_dir under the sigma's rubric and, unless out_dir is None (dry
    run), write the kept tasks to a fresh out_dir. Returns the report. Raises ValueError if the
    sigma has no rubric or nothing is kept."""
    train_dir, sigma_json = Path(train_dir), Path(sigma_json)
    cfg = json.loads(sigma_json.read_text())
    rubric = compile_rubric(cfg.get("rubric") or [])
    t0 = time.perf_counter()
    tasks = {}
    for task in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        ins = task / "instruction.md"
        text = ins.read_text(encoding="utf-8", errors="replace") if ins.is_file() else ""
        s, met = score_text(text, rubric)
        tasks[task.name] = {"score": round(s, 4), "met": met}
    secs = time.perf_counter() - t0
    kept = [t for t, r in tasks.items() if r["score"] >= threshold]
    report = {"sigma": cfg.get("sigma", ""), "train_dir": str(train_dir), "scorer": "lexical",
              "threshold": threshold, "n_tasks": len(tasks), "n_kept": len(kept),
              "seconds": round(secs, 3),
              "seconds_per_task": round(secs / len(tasks), 6) if tasks else None,
              "criteria_hits": {c: sum(c in r["met"] for r in tasks.values()) for c, _, _ in rubric},
              "tasks": tasks}
    if out_dir is None:
        return report
    if not kept:
        raise ValueError(f"no task in {train_dir} scores >= {threshold} under sigma "
                         f"'{report['sigma']}'; see the criteria_hits in a --dry-run")
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)          # always from the base split
    out_dir.mkdir(parents=True)
    keep = set(kept)
    for p in train_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == "seed.jsonl":
            rows = [l for l in p.read_text().splitlines()
                    if l.strip() and _instance_id(json.loads(l)) in keep]
            (out_dir / p.name).write_text("".join(r + "\n" for r in rows))
        else:
            shutil.copy2(p, out_dir / p.name)
    for t in kept:
        shutil.copytree(train_dir / t, out_dir / t, symlinks=True)
    (out_dir / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _strip_template(texts: list[str]) -> list[str]:
    """Cut the prefix and suffix that every text shares (the harness prompt template)."""
    if len(set(texts)) < 2:
        return texts
    pre = os.path.commonprefix(texts)
    suf = os.path.commonprefix([t[::-1] for t in texts])[::-1]
    return [t[len(pre):len(t) - len(suf)] for t in texts]


def _extremes(rewards: list[list[float]]) -> dict:
    n = len(rewards)
    return {"n_tasks": n,
            "mean_reward": round(sum(sum(r) / len(r) for r in rewards) / n, 4) if n else None,
            "all_0": round(sum(max(r) <= 0 for r in rewards) / n, 4) if n else None,
            "all_1": round(sum(min(r) >= 1 for r in rewards) / n, 4) if n else None}


def screen_rubric(rubric: list[dict], traces: list[Path],
                  threshold: float = THRESHOLD) -> tuple[bool, dict]:
    """Score the tasks behind `traces` (trial trajectories, as reward_synthesis.find_traces
    returns them) and judge the rubric. Returns (ok, report); ok is False if it keeps none or all
    of them. Too few tasks to judge -> unscreened, ok."""
    from mho.data import reward_synthesis as rs
    from mho.data.sigma_checklib import task_prompt
    compiled = compile_rubric(rubric)
    prompts, rewards = {}, {}
    for t in traces:
        task = t.parents[1].name.split("__")[0]
        try:
            prompts.setdefault(task, task_prompt(json.loads(t.read_text())))
        except (OSError, ValueError):
            continue
        r = rs._trial_reward(t)
        if r is not None:
            rewards.setdefault(task, []).append(r)
    names = sorted(prompts)
    texts = _strip_template([prompts[n] for n in names])
    scores = {n: score_text(x, compiled) for n, x in zip(names, texts)}
    kept = [n for n in names if scores[n][0] >= threshold]
    report = {"threshold": threshold, "n_tasks": len(names), "n_kept": len(kept),
              "criteria_hits": {c: sum(c in m for _, m in scores.values()) for c, _, _ in compiled},
              "kept": _extremes([rewards[n] for n in kept if n in rewards]),
              "dropped": _extremes([rewards[n] for n in names if n not in kept and n in rewards])}
    if len(names) < rs.MIN_TRACES:
        report["status"] = f"unscreened: {len(names)} tasks < {rs.MIN_TRACES}"
        return True, report
    if not kept or len(kept) == len(names):
        report["status"] = (f"rejected: keeps {len(kept)}/{len(names)} tasks, so it selects nothing; "
                            "make the criteria / keywords separate the tasks")
        return False, report
    report["status"] = "ok"
    return True, report


def _print_screen(report: dict) -> None:
    print(f"[select screen] {report['status']}  kept {report['n_kept']}/{report['n_tasks']} tasks "
          f"at score >= {report['threshold']}")
    for c, n in report["criteria_hits"].items():
        print(f"  {n:>6}  {c}")
    for side in ("kept", "dropped"):
        e = report[side]
        print(f"  {side:<8} tasks={e['n_tasks']} mean_reward={e['mean_reward']} "
              f"all_0={e['all_0']} all_1={e['all_1']}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["screen"]:
        from mho.data import reward_synthesis as rs
        from mho.optimizer import model_spec
        p = argparse.ArgumentParser(prog="task_select screen",
                                    description="try a spec's rubric on the run's train-sample tasks")
        p.add_argument("--spec", required=True, type=Path, help="model_specs/<name>.md")
        p.add_argument("--traces", nargs="*", type=Path, default=[Path(".")],
                       help="candidates/ dir(s) or dir(s) of trial dirs (default: cwd)")
        p.add_argument("--threshold", type=float, default=THRESHOLD)
        a = p.parse_args(argv[1:])
        try:
            spec, _ = model_spec.parse_spec(a.spec)
            ok, report = screen_rubric([r.model_dump() for r in spec.rubric],
                                       rs.find_traces(a.traces, limit=10**6), a.threshold)
        except ValueError as e:
            print(f"[select screen] INVALID: {e}", file=sys.stderr)
            return 2
        _print_screen(report)
        return 0 if ok else 2
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument("--sigma", required=True, type=Path, help="<cand>/model/sigma_rewards/<s>.json")
    p.add_argument("--out", type=Path, help="output split (required unless --dry-run)")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--dry-run", action="store_true", help="score and print, write nothing")
    a = p.parse_args(argv)
    if not a.dry_run and a.out is None:
        p.error("--out is required unless --dry-run")
    try:
        r = select_tasks(a.train_dir, a.sigma, None if a.dry_run else a.out, a.threshold)
    except ValueError as e:
        print(f"[select] {e}", file=sys.stderr)
        return 2
    print(f"[select] kept {r['n_kept']}/{r['n_tasks']} task(s) at score >= {r['threshold']} "
          f"({r['seconds_per_task']}s/task)" + ("" if a.dry_run else f" -> {a.out}"))
    for c, n in r["criteria_hits"].items():
        print(f"  {n:>6}  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
