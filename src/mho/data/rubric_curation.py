"""Rubric curation: an optimizer agent reads the train-sample trials and writes a yes/no rubric;
an LLM then answers every rubric line for each task in a data index, and the tasks with the
largest share of yes answers are kept.

Run layout (same candidates/ archive as the harness loop; tasks are referenced, not copied):
  <RUN>/data/index.json        tasks to curate: [{"i", "task", "path"}]
  <RUN>/data/design.json       the evaluated tasks, same rows plus "rewards"
  <RUN>/candidates/candidate_0_0_0/eval_before/<trial>   the evaluated trials (symlinks)
  <RUN>/candidates/candidate_0_0_0/tasks/<task>.md       their instructions
  <RUN>/candidates/<tag>/rubric.md     written by the agent: one yes/no question per bullet
  <RUN>/candidates/<tag>/CANDIDATE.md  the hypothesis behind it
  <RUN>/candidates/<tag>/propose.json  agent tokens / cost
  <RUN>/candidates/<tag>/curated/<index>_r<k>.json   one judge pass (scores, selections, cost)
  <RUN>/candidates/summary.json        one row per candidate

score = yes answers / rubric lines. One judge pass gives two selections:
  threshold: score >= MHO_RUBRIC_THRESHOLD (0.85)
  top:       the top MHO_RUBRIC_TOP_FRAC (0.10) of the index by score, ties in index order

Usage:
  python -m mho.data.rubric_curation init --run R --trials <runs dir> ... --val-tasks <dir>
                                          [--task-root DIR] [--n-val 250] [--seed 0]
  python -m mho.data.rubric_curation generate --run R --tag candidate_1_0_1
  python -m mho.data.rubric_curation run --run R --tag T --index data/index.json [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

THRESHOLD = float(os.environ.get("MHO_RUBRIC_THRESHOLD", "0.85"))
TOP_FRAC = float(os.environ.get("MHO_RUBRIC_TOP_FRAC", "0.10"))
MAX_LINES = 10
MAX_TASK_CHARS = 12000
BASE = "candidate_0_0_0"
_BULLET = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$")

GUIDELINE = """You screen tasks for training a coding agent. Read the task below and answer each
rubric question about it with yes or no. Judge the task as written, not how hard it would be
for you.

Rubric:
{rubric}

Task:
<task>
{task}
</task>

Reply with only a JSON object: {{"answers": ["yes" or "no", ...]}}, one answer per rubric
question, in order ({n} answers)."""


# ---- rubric and scoring -------------------------------------------------------------------

def parse_rubric(text: str) -> list[str]:
    """The rubric questions: every bullet or numbered line of rubric.md."""
    lines = [m.group(1) for m in map(_BULLET.match, text.splitlines()) if m]
    if not 2 <= len(lines) <= MAX_LINES:
        raise ValueError(f"rubric needs 2-{MAX_LINES} bullet lines, found {len(lines)}")
    return lines


def parse_answers(reply: str, n: int) -> list[bool]:
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    answers = json.loads(m.group(0))["answers"]
    if len(answers) != n:
        raise ValueError(f"{len(answers)} answers for {n} questions")
    out = []
    for a in answers:
        a = str(a).strip().lower()
        if a not in ("yes", "no"):
            raise ValueError(f"answer {a!r} is not yes/no")
        out.append(a == "yes")
    return out


def select(scores: dict[int, float], threshold: float = THRESHOLD,
           top_frac: float = TOP_FRAC) -> dict[str, list[int]]:
    """Both selections over the scored tasks (index -> score; unscored tasks are left out)."""
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    k = math.ceil(top_frac * len(scores))
    return {"threshold": sorted(i for i in scores if scores[i] >= threshold),
            "top": sorted(ranked[:k])}


def _extremes(rewards: list[list[float]]) -> dict:
    n = len(rewards)
    if not n:
        return {"n_tasks": 0}
    return {"n_tasks": n,
            "mean_reward": round(sum(sum(r) / len(r) for r in rewards) / n, 4),
            "all_0": round(sum(max(r) <= 0 for r in rewards) / n, 4),
            "all_1": round(sum(min(r) >= 1 for r in rewards) / n, 4)}


# ---- judge --------------------------------------------------------------------------------

def _completion(prompt: str) -> tuple[str, dict]:
    import litellm
    resp = litellm.completion(model=os.environ["OPTIMIZER_MODEL"],
                              api_key=os.environ["OPTIMIZER_LLM_API_KEY"],
                              api_base=os.environ["OPTIMIZER_LLM_BASE_URL"],
                              messages=[{"role": "user", "content": prompt}],
                              temperature=0, timeout=300)
    u = resp.usage
    cost = (getattr(resp, "_hidden_params", None) or {}).get("response_cost")
    return resp.choices[0].message.content, {
        "prompt_tokens": u.prompt_tokens if u else 0,
        "completion_tokens": u.completion_tokens if u else 0,
        "cost": cost}


def judge(task_text: str, questions: list[str], complete=None) -> dict:
    """Answer every question for one task. Two tries; on failure answers is None."""
    complete = complete or _completion
    prompt = GUIDELINE.format(rubric="\n".join(f"{k}. {q}" for k, q in enumerate(questions, 1)),
                              task=task_text[:MAX_TASK_CHARS], n=len(questions))
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    err = None
    for _ in range(2):
        try:
            reply, u = complete(prompt)
        except Exception as e:      # provider errors (rate limit, budget, timeout)
            err = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        usage["prompt_tokens"] += u["prompt_tokens"]
        usage["completion_tokens"] += u["completion_tokens"]
        usage["cost"] = None if u["cost"] is None or usage["cost"] is None else usage["cost"] + u["cost"]
        try:
            return {"answers": parse_answers(reply, len(questions)), "error": None, **usage}
        except (ValueError, KeyError, TypeError) as e:
            err = f"unparsable reply: {e}"
    return {"answers": None, "error": err, **usage}


def run_rubric(rubric_md: Path, index: list[dict], workers: int = 8, complete=None,
               threshold: float = THRESHOLD, top_frac: float = TOP_FRAC) -> dict:
    """One judge pass of the rubric over the index. Returns the curated record."""
    questions = parse_rubric(Path(rubric_md).read_text())
    texts = [(Path(r["path"]) / "instruction.md").read_text(encoding="utf-8", errors="replace")
             for r in index]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda t: judge(t, questions, complete), texts))
    secs = time.perf_counter() - t0
    scores = {r["i"]: sum(j["answers"]) / len(questions)
              for r, j in zip(index, results) if j["answers"] is not None}
    sel = select(scores, threshold, top_frac)
    costs = [j["cost"] for j in results]
    out = {"rubric": questions, "n_tasks": len(index), "n_scored": len(scores),
           "threshold": threshold, "top_frac": top_frac,
           "selections": sel,
           "yield": {k: len(v) for k, v in sel.items()},
           "line_yes": [sum(j["answers"][q] for j in results if j["answers"]) for q in range(len(questions))],
           "cost": {"prompt_tokens": sum(j["prompt_tokens"] for j in results),
                    "completion_tokens": sum(j["completion_tokens"] for j in results),
                    "usd": None if any(c is None for c in costs) else round(sum(costs), 6),
                    "seconds": round(secs, 1)},
           "errors": {r["i"]: j["error"] for r, j in zip(index, results) if j["answers"] is None},
           "tasks": [{"i": r["i"], "task": r["task"], "score": scores.get(r["i"]),
                      "answers": j["answers"]} for r, j in zip(index, results)]}
    if all("rewards" in r for r in index):       # design set: what the selection keeps
        rw = {r["i"]: r["rewards"] for r in index if r["rewards"]}
        out["rewards"] = {k: {"kept": _extremes([rw[i] for i in v if i in rw]),
                              "dropped": _extremes([rw[i] for i in rw if i not in set(v)])}
                          for k, v in sel.items()}
    return out


# ---- init: indexes and the base candidate -----------------------------------------------

def _trial_rows(runs: list[Path], task_root: Path) -> dict[str, dict]:
    """task -> {"path", "rewards", "trials"} from harbor runs dirs (eval_*/<trial>/)."""
    tasks: dict[str, dict] = {}
    for rd in runs:
        for td in sorted(Path(rd).glob("eval_*/*__*")):
            try:
                path = json.loads((td / "config.json").read_text())["task"]["path"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            p = Path(path) if Path(path).is_absolute() else task_root / path
            row = tasks.setdefault(p.name, {"path": str(p), "rewards": [], "trials": []})
            row["trials"].append(td)
            try:
                r = json.loads((td / "result.json").read_text())["verifier_result"]["rewards"]["reward"]
                row["rewards"].append(float(r))
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return tasks


def init_run(run: Path, runs: list[Path], val_tasks: Path, task_root: Path,
             n_val: int = 250, seed: int = 0) -> dict:
    run = Path(run)
    trials = {t: r for t, r in _trial_rows(runs, task_root).items() if r["rewards"]}
    design = [{"i": i, "task": t, "path": trials[t]["path"], "rewards": trials[t]["rewards"]}
              for i, t in enumerate(sorted(trials))]
    pool = sorted(p.name for p in Path(val_tasks).iterdir()
                  if (p / "instruction.md").is_file() and p.name not in trials)
    if len(pool) < n_val:
        raise ValueError(f"{val_tasks} has {len(pool)} tasks outside the design set, need {n_val}")
    val = sorted(random.Random(seed).sample(pool, n_val))
    index = [{"i": i, "task": t, "path": str(Path(val_tasks) / t)} for i, t in enumerate(val)]
    (run / "data").mkdir(parents=True, exist_ok=True)
    (run / "data" / "design.json").write_text(json.dumps(design, indent=1) + "\n")
    (run / "data" / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    base = run / "candidates" / BASE
    (base / "tasks").mkdir(parents=True, exist_ok=True)
    (base / "eval_before").mkdir(exist_ok=True)
    for t, r in trials.items():
        (base / "tasks" / f"{t}.md").write_text(
            (Path(r["path"]) / "instruction.md").read_text(encoding="utf-8", errors="replace"))
        for td in r["trials"]:
            link = base / "eval_before" / td.name
            if not link.exists():
                link.symlink_to(td.resolve(), target_is_directory=True)
    (base / "task_rewards.json").write_text(json.dumps(
        {t: r["rewards"] for t, r in sorted(trials.items())}, indent=1) + "\n")
    return {"design_tasks": len(design), "trials": sum(len(r["trials"]) for r in trials.values()),
            "val_tasks": len(index), "val_pool": len(pool)}


# ---- generate: the optimizer agent writes rubric.md ----------------------------------------

def generate(run: Path, tag: str) -> dict:
    from mho.optimizer import agent
    cand_root = Path(run) / "candidates"
    cand = cand_root / tag
    cand.mkdir(parents=True, exist_ok=True)
    note = (f"\n\n## Your candidate\nThis session's candidate is `{tag}`: write the rubric to "
            f"`./{tag}/rubric.md` and the hypothesis behind it to `./{tag}/CANDIDATE.md`.\n")
    t0 = time.perf_counter()
    llm = agent._run_openhands(cand_root, agent._build_instructions(
        agent._instructions_md("rubric"), note), agent._skill_dir("rubric"))
    m = getattr(llm, "metrics", None)
    usage = getattr(m, "accumulated_token_usage", None)
    rec = {"seconds": round(time.perf_counter() - t0, 1),
           "usd": getattr(m, "accumulated_cost", None),
           "prompt_tokens": getattr(usage, "prompt_tokens", None),
           "completion_tokens": getattr(usage, "completion_tokens", None)}
    try:
        rec["rubric"] = parse_rubric((cand / "rubric.md").read_text())
        rec["ok"] = True
    except (OSError, ValueError) as e:
        rec["ok"], rec["reason"] = False, str(e)
    (cand / "propose.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def _update_summary(run: Path, tag: str, index_name: str, rec: dict) -> None:
    p = Path(run) / "candidates" / "summary.json"
    rows = json.loads(p.read_text()) if p.is_file() else []
    row = next((r for r in rows if r["candidate"] == tag), None)
    if row is None:
        row = {"candidate": tag}
        rows.append(row)
    cand = Path(run) / "candidates" / tag
    if (cand / "propose.json").is_file():
        row["propose"] = json.loads((cand / "propose.json").read_text())
    row.setdefault("runs", {}).setdefault(index_name, []).append(
        {k: rec[k] for k in ("yield", "cost", "n_scored") if k in rec} | (
            {"rewards": rec["rewards"]} if "rewards" in rec else {}))
    p.write_text(json.dumps(rows, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a_init = sub.add_parser("init")
    a_init.add_argument("--run", required=True, type=Path)
    a_init.add_argument("--trials", required=True, nargs="+", type=Path, help="harbor runs dirs")
    a_init.add_argument("--val-tasks", required=True, type=Path, help="dir of task dirs")
    a_init.add_argument("--task-root", type=Path, default=Path("."),
                        help="what relative task paths in config.json are relative to")
    a_init.add_argument("--n-val", type=int, default=250)
    a_init.add_argument("--seed", type=int, default=0)
    a_gen = sub.add_parser("generate")
    a_gen.add_argument("--run", required=True, type=Path)
    a_gen.add_argument("--tag", required=True)
    a_run = sub.add_parser("run")
    a_run.add_argument("--run", required=True, type=Path)
    a_run.add_argument("--tag", required=True)
    a_run.add_argument("--index", default="data/index.json", help="relative to --run")
    a_run.add_argument("--repeats", type=int, default=1)
    a_run.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    if a.cmd == "init":
        print(json.dumps(init_run(a.run, a.trials, a.val_tasks, a.task_root, a.n_val, a.seed)))
        return 0
    if a.cmd == "generate":
        rec = generate(a.run, a.tag)
        print(json.dumps(rec))
        return 0 if rec["ok"] else 3
    index = json.loads((a.run / a.index).read_text())
    name = Path(a.index).stem
    out_dir = a.run / "candidates" / a.tag / "curated"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(1, a.repeats + 1):
        rec = run_rubric(a.run / "candidates" / a.tag / "rubric.md", index, a.workers)
        (out_dir / f"{name}_r{k}.json").write_text(json.dumps(rec, indent=1) + "\n")
        _update_summary(a.run, a.tag, name, rec)
        print(f"[rubric] {a.tag} {name} r{k}: scored {rec['n_scored']}/{rec['n_tasks']} "
              f"yield {rec['yield']} cost {rec['cost']}")
        if rec["n_scored"] == 0:
            print(f"[rubric] nothing scored; first error: {next(iter(rec['errors'].values()))}",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
