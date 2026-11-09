"""Sigma-reward smoke test: does reward_rule.py, run by the verifier in the rollout sandbox,
actually see the rollout's trajectory during Miles training?

  submit: build a small train split (N tasks + their seed.jsonl rows) and a sigma config whose
          first check passes only if the trajectory is readable, then submit ONE short Miles
          GRPO job (scripts/optimize/train.sh via mho.jobs.submit_train with MHO_SIGMA_REWARD
          set): the same path a --model-source propose candidate takes.
  report: read every finished trial's verifier/sigma_info.json and say whether the checks saw
          a trajectory (traj_chars > 0) and where correctness came from.

Usage (repo root, .env loaded):
  python scripts/smoke/sigma_smoke.py submit --out <dir> [--data <split>] [--n 16]
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

# traj_has_text(traj, '"role"') is true for any non-empty chat trajectory: if it fails, the rule
# saw no trajectory at all. has_final_agent_answer_line is a realistic sigma check.
CHECKS = ["traj_has_text(traj, '\"role\"')", "has_final_agent_answer_line(traj)"]
SMOKE_ENV = {"NUM_ROLLOUT": "1", "TRAIN_BATCH_SIZE": "4", "N_SAMPLES_PER_PROMPT": "4",
             "GLOBAL_BATCH_SIZE": "16", "MHO_ASYNC_MAX_CONCURRENT": "16"}


def submit(a):
    from mho.jobs import Scheduler, submit_train
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"{out} exists; pick a fresh --out")
    src = Path(a.data)
    tasks = sorted(p for p in src.iterdir() if p.is_dir())
    pick = sorted(random.Random(0).sample(tasks, min(a.n, len(tasks))), key=lambda p: p.name)
    split = out / "train_split"
    split.mkdir(parents=True)
    for t in pick:
        (split / t.name).symlink_to(t)
    keep = {t.name for t in pick}
    rows = [line for line in (src / "seed.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line)["metadata"]["instance_id"] in keep]
    (split / "seed.jsonl").write_text("\n".join(rows) + "\n")
    sigma = out / "sigma_smoke.json"
    sigma.write_text(json.dumps({"sigma": "sigma_smoke", "checks": CHECKS}, indent=2) + "\n")
    print(f"split: {len(pick)} tasks, {len(rows)} seed rows -> {split}")
    os.environ.update(SMOKE_ENV)          # qsub -V carries these into the train job
    j = submit_train(Scheduler(), a.trainer, model=a.model, harness=None, out_dir=out / "train",
                     logs=out / "train.out", tag="sigma_smoke", sigma_reward=str(sigma),
                     train_data=str(split), objective="grpo", steps=1)
    print("train job:", j)


def report(a):
    out = Path(a.out)
    infos = sorted(out.glob("train/harbor_trials/*/verifier/sigma_info.json"))
    trials = list(out.glob("train/harbor_trials/*/result.json"))
    staged = out / "train" / "sigma_tasks"
    n_staged = len([p for p in staged.iterdir() if p.is_dir()]) if staged.is_dir() else 0
    print(f"staged sigma tasks: {n_staged}")
    print(f"trials with result.json: {len(trials)}; with verifier/sigma_info.json: {len(infos)}")
    if not infos:
        print("VERDICT: no sigma_info.json yet: the wrapper has not run, or no trial has finished")
        return 1
    seen, chk, src = collections.Counter(), collections.Counter(), collections.Counter()
    for f in infos:
        d = json.loads(f.read_text())
        seen["traj seen" if d.get("traj_chars", 0) > 0 else "traj EMPTY"] += 1
        src[d.get("correct_from")] += 1
        for c in d["checks"]:
            chk[(c["check"].split("(")[0], c["passed"])] += 1
    ex = json.loads(infos[0].read_text())
    print("trajectory:", dict(seen))
    print("correctness from:", dict(src))
    print("checks (name, passed):", dict(chk))
    print("example:", json.dumps({k: ex.get(k) for k in ("traj_path", "traj_chars", "output_chars",
                                                         "correct", "checks_passed", "reward")}))
    ok = seen["traj seen"] == len(infos)
    print("VERDICT:", "PASS: the rule sees the trajectory in every trial" if ok else
          "FAIL: some or all trials scored the checks against an empty trajectory")
    return 0 if ok else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit")
    s.add_argument("--out", required=True)
    s.add_argument("--data", default=str(REPO.parent / "data-harbor" / "arxivmath_training_train_split"))
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--model", default="Qwen/Qwen3.5-4B")
    s.add_argument("--trainer", default=os.environ.get("MHO_TRAINER", "miles"))
    r = sp.add_parser("report")
    r.add_argument("--out", required=True)
    a = p.parse_args()
    sys.exit(submit(a) if a.cmd == "submit" else report(a))
