"""Sigma-reward smoke test: proposer-written check functions, end to end through Miles training.

  submit: write a spec + checks module the way the model proposer does (model_specs/<name>.md +
          <name>.py), realize it through the real path (validate + screen on existing
          trajectories, --traces), build a small train split, and submit ONE short Miles GRPO
          job (scripts/optimize/train.sh via mho.jobs.submit_train with MHO_SIGMA_REWARD set):
          the same path a --model-source propose candidate takes.
  report: read every finished trial's verifier/sigma_info.json: did the rule see the
          trajectory, did the checks module import and run without errors, and did each
          check vary across rollouts (a constant check changes no advantage)?

Usage (repo root, .env loaded):
  python scripts/smoke/sigma_smoke.py submit --out <dir> [--data <split>] [--n 16] [--traces <dir>]
  python scripts/smoke/sigma_smoke.py report --out <dir>
"""
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

NAME = "smoke_sigma"
SPEC_MD = f"""---
name: {NAME}
hypothesis: |
  Rollouts that compute the answer with code, write it once, and finish in few steps are
  more often correct.
evidence:
  - smoke test (no real evidence)
spec: |
  Verify a derived answer by running code before committing it, commit it once, and stop.
counterexample: |
  The agent writes an answer from reasoning alone, rewrites it several times, and keeps going.
realization: synthesize
checks:
  - ran_python
  - wrote_output_once
  - few_steps
---
Smoke-test spec: the checks only need to vary across rollouts.
"""
CHECKS_PY = '''from sigma_checklib import *


def ran_python(traj):
    """The agent executed Python at least once."""
    return any("python" in c for c in commands(traj))


def wrote_output_once(traj):
    """Exactly one command writes the answer file."""
    return sum(1 for c in commands(traj) if "output.txt" in c and ">" in c) == 1


def few_steps(traj):
    """Partial credit for finishing in fewer assistant turns (1.0 at 0, 0.0 at 20+)."""
    return max(0.0, 1.0 - n_steps(traj) / 20)
'''
SMOKE_ENV = {"NUM_ROLLOUT": "1", "TRAIN_BATCH_SIZE": "4", "N_SAMPLES_PER_PROMPT": "4",
             "GLOBAL_BATCH_SIZE": "16", "MHO_ASYNC_MAX_CONCURRENT": "16", "SAVE_HF": ""}
PREV = "/groups/gai51740/aci18914wh/mho/smoke/sigma-smoke-1/train/harbor_trials"


def submit(a):
    from mho.data import reward_synthesis as rs
    from mho.jobs import Scheduler, submit_train
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"{out} exists; pick a fresh --out")
    specs = out / "model_specs"
    specs.mkdir(parents=True)
    (specs / f"{NAME}.md").write_text(SPEC_MD)
    (specs / f"{NAME}.py").write_text(CHECKS_PY)
    ok, report = rs.realize(specs / f"{NAME}.md", out / "sigma_rewards", rs.find_traces([a.traces]))
    rs._print_report(report)
    if not ok:
        sys.exit("realize rejected the checks; not submitting")
    src = Path(a.data)
    tasks = sorted(p for p in src.iterdir() if p.is_dir())
    pick = sorted(random.Random(a.seed).sample(tasks, min(a.n, len(tasks))), key=lambda p: p.name)
    split = out / "train_split"
    split.mkdir(parents=True)
    for t in pick:
        (split / t.name).symlink_to(t)
    keep = {t.name for t in pick}
    rows = [line for line in (src / "seed.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line)["metadata"]["instance_id"] in keep]
    (split / "seed.jsonl").write_text("\n".join(rows) + "\n")
    print(f"split: {len(pick)} tasks, {len(rows)} seed rows -> {split}")
    os.environ.update(SMOKE_ENV)          # qsub -V carries these into the train job
    j = submit_train(Scheduler(), a.trainer, model=a.model, harness=None, out_dir=out / "train",
                     logs=out / "train.out", tag="sigma_smoke",
                     sigma_reward=str(out / "sigma_rewards" / f"{NAME}.json"),
                     train_data=str(split), objective="grpo", steps=1)
    print("train job:", j)


def report(a):
    out = Path(a.out)
    infos = sorted(out.glob("train/harbor_trials/*/verifier/sigma_info.json"))
    trials = list(out.glob("train/harbor_trials/*/result.json"))
    tests = out / "train" / "sigma_tasks"
    staged = [p for p in tests.iterdir() if p.is_dir()] if tests.is_dir() else []
    print(f"staged sigma tasks: {len(staged)}"
          + (f" (sigma_checks.py in each: {all((p / 'tests' / 'sigma_checks.py').is_file() for p in staged)})"
             if staged else ""))
    print(f"trials with result.json: {len(trials)}; with verifier/sigma_info.json: {len(infos)}")
    if not infos:
        print("VERDICT: no sigma_info.json yet: the wrapper has not run, or no trial has finished")
        return 1
    seen, src, imp = collections.Counter(), collections.Counter(), collections.Counter()
    vals, errs = collections.defaultdict(list), collections.defaultdict(list)
    for f in infos:
        d = json.loads(f.read_text())
        seen["traj seen" if d.get("traj_chars", 0) > 0 else "traj EMPTY"] += 1
        src[d.get("correct_from")] += 1
        imp["import ok" if not d.get("import_error") else d["import_error"][:80]] += 1
        for c in d["checks"]:
            (errs if c.get("error") else vals)[c["check"]].append(c.get("error") or c["value"])
    print("trajectory:", dict(seen))
    print("correctness from:", dict(src))
    print("checks module:", dict(imp))
    varied = True
    for name in sorted(set(vals) | set(errs)):
        v = vals[name]
        mean = sum(v) / len(v) if v else None
        print(f"  {name}: n={len(v)} errors={len(errs[name])} mean={mean and round(mean, 3)} "
              f"distinct={sorted(set(v))[:8]}" + (f" first error: {errs[name][0]}" if errs[name] else ""))
        varied &= len(set(v)) >= 2 and not errs[name]
    ex = json.loads(infos[0].read_text())
    print("example:", json.dumps({k: ex.get(k) for k in ("traj_chars", "correct", "checks_score", "reward")}))
    ok = seen["traj seen"] == len(infos) and imp["import ok"] == len(infos) and varied
    print("VERDICT:", "PASS: checks staged, imported, ran on the trajectory and varied across rollouts"
          if ok else "FAIL: see above (empty trajectory, import error, check error, or a constant check)")
    return 0 if ok else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit")
    s.add_argument("--out", required=True)
    s.add_argument("--data", default=str(REPO.parent / "data-harbor" / "arxivmath_training_train_split"))
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--traces", default=PREV, help="trial dirs to screen the checks on")
    s.add_argument("--model", default="Qwen/Qwen3.5-4B")
    s.add_argument("--trainer", default=os.environ.get("MHO_TRAINER", "miles"))
    r = sp.add_parser("report")
    r.add_argument("--out", required=True)
    a = p.parse_args()
    sys.exit(submit(a) if a.cmd == "submit" else report(a))
