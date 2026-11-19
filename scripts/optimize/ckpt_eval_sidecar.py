"""Evaluate a training run's intermediate checkpoints while it trains (async eval).

Watches <hf_dir> for miles HF exports (step_N/.complete, written every --save-interval) and, for
each new one, completes it (scripts/optimize/complete_hf_export.py: config/tokenizer + the
untrained vision/MTP tensors, so vLLM can serve it) and submits a val eval of it as its own
PBS/SLURM job, exactly as the loop evaluates a model candidate. Training never pauses: the eval
runs on another node against the saved checkpoint.

Scores land in <out_dir>/ckpt_evals.tsv (step = number of updates, i.e. rollout_id + 1).
Exits once the final step (--num-steps) has been evaluated, or after --max-hours.

Run via scripts/run/ckpt-eval.sh (loads .env for the scheduler/queue settings), e.g. as a small
CPU job next to the train job.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mho.jobs import Scheduler, submit_eval  # noqa: E402
from utils.scores import metric_of, score_trials  # noqa: E402

COMPLETE_MARKER = ".complete"  # miles hf_export: written after the export finished


def complete_export(hf_dir: Path, base: str, sif: str, hf_cache: str) -> None:
    subprocess.run(
        ["singularity", "exec", "--bind", f"{hf_cache}:/root/.cache/huggingface",
         "--bind", f"{hf_dir}:{hf_dir}", "--env", "HF_HOME=/root/.cache/huggingface",
         sif, "env", "-u", "PYTHONPATH", "/opt/sglang/bin/python", "-I",
         str(REPO / "scripts/optimize/complete_hf_export.py"), str(hf_dir), base],
        check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="train/checkpoints/hf of the run")
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--base-model", required=True, help="hub id or dir the run started from")
    ap.add_argument("--out-dir", required=True, help="evaluation dir (one subdir per checkpoint)")
    ap.add_argument("--num-steps", type=int, required=True, help="the run's --num-rollout")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--concurrent", type=int, default=128)
    ap.add_argument("--skip-final", action="store_true",
                    help="the loop already evaluates the final checkpoint")
    ap.add_argument("--sif", required=True, help="miles.sif (for completing the export)")
    ap.add_argument("--hf-cache", required=True, help="host HF cache dir")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=48)
    a = ap.parse_args()

    hf_dir, out = Path(a.hf_dir), Path(a.out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_f = out / "ckpt_evals.state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else {"submitted": {}, "scored": {}}
    tsv = out / "ckpt_evals.tsv"
    if not tsv.exists():
        tsv.write_text("step\tval\tn_trials\tcheckpoint\n")
    sched, final = Scheduler(), a.num_steps
    deadline = time.time() + a.max_hours * 3600

    def save():
        state_f.write_text(json.dumps(st, indent=2))

    while time.time() < deadline:
        # 1. new checkpoints -> complete + submit
        for d in sorted(hf_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
            step = int(d.name.split("_")[1]) + 1
            if str(step) in st["submitted"] or not (d / COMPLETE_MARKER).exists():
                continue
            if a.skip_final and step == final:
                st["submitted"][str(step)] = None
                save()
                continue
            complete_export(hf_dir, a.base_model, a.sif, a.hf_cache)
            name = f"ckpt-step{step:04d}"
            jid = submit_eval(sched, "", model=str(d), harness=None, data=a.val_data,
                              n_trials=a.trials, n_concurrent=a.concurrent, out_dir=out,
                              logs=out / "logs" / f"{name}.out", job_name=name,
                              base_model=a.base_model)
            st["submitted"][str(step)] = jid
            save()
            print(f"[ckpt-eval] step {step}: {d} -> eval job {jid}", flush=True)
        # 2. finished evals -> score
        for step, jid in st["submitted"].items():
            if jid is None or step in st["scored"]:
                continue
            state, rc = sched.state(jid)
            if state != "DONE":
                continue
            name = f"ckpt-step{int(step):04d}"
            try:
                score = score_trials(out / name)
                val, n = metric_of(score), score.get("n_trials", score.get("n", ""))
            except Exception as e:  # noqa: BLE001
                val, n = float("nan"), f"error rc={rc}: {e}"
            st["scored"][step] = val
            save()
            with tsv.open("a") as f:
                f.write(f"{step}\t{val:.4f}\t{n}\t{hf_dir / f'step_{int(step) - 1}'}\n")
            print(f"[ckpt-eval] step {step}: val={val:.4f} (rc={rc})", flush=True)
        done_final = str(final) in st["submitted"] and (
            st["submitted"][str(final)] is None or str(final) in st["scored"])
        if done_final and all(s in st["scored"] for s, j in st["submitted"].items() if j):
            print("[ckpt-eval] final checkpoint handled; done", flush=True)
            return 0
        time.sleep(a.poll)
    print("[ckpt-eval] max-hours reached", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
