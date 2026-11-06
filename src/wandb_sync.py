#!/usr/bin/env python3
"""Mirror an optimization run dir into Weights & Biases (a sidecar: it only reads the run dir).

One W&B step per finished candidate (summary.json row, in finish order), so the x-axis is
"proposals used". Each step logs the candidate's train/val gate numbers, the cumulative run
state (best so far, outcome counts), stage wall times from logs/run.out, and trial health
(timeouts, durations, tokens) of its evals. Tables (candidates, rounds, jobs) and the
train-t vs val-t scatter are refreshed every sync.

It is idempotent and resumable: the W&B run id and the number of rows already logged are kept
in <run_dir>/logs/wandb_sync.json, so it can be re-run (or run on a cron / --watch) at any time,
including on finished runs (backfill). It never touches the driver; run it wherever W&B is
reachable (e.g. the login node), or with --offline and `wandb sync` later.

    uv run --with wandb python src/wandb_sync.py --run-dir <mho-runs/RUN> --watch 600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import REPO_DIR  # noqa: E402
from utils import run_report as rr  # noqa: E402

TABLE_COLS = [
    "step", "candidate", "outcome", "category", "title", "iteration", "round", "sibling",
    "ts_A", "ts_B", "train/path", "train/d", "train/se", "train/t", "train/tok_delta_pct",
    "train/tok_cheaper_t", "val/acc", "val/prev_best", "val/look", "val/val_trials", "val/path",
    "val/d", "val/se", "val/t", "val/tok_delta_pct", "val/anchor_d", "time/propose_min",
    "time/eval_after_min", "time/val_wall_min", "eval_b/timeout_rate", "eval_b/dur_p95_min",
    "val_eval/timeout_rate", "val_eval/trials",
]
JOB_COLS = ["job_id", "stage", "tag", "owner", "submit", "done", "minutes", "rc"]
ROUND_COLS = ["iteration", "round", "proposed", "train_pass", "to_val", "accepted",
              "distinct_categories", "categories"]


def load_env(path: Path) -> None:
    """WANDB_* from the repo .env (values never printed)."""
    if not path.exists():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for k, v in dotenv_values(path).items():
        if k.startswith("WANDB_") and v and k not in os.environ:
            os.environ[k] = v


def scalars(row: dict) -> dict:
    """The numeric part of a row (W&B charts); strings go to the tables."""
    return {k: v for k, v in row.items() if rr.finite(v) and k != "step"}


class Syncer:
    def __init__(self, args):
        import wandb
        self.wandb, self.args = wandb, args
        self.run_dir = Path(args.run_dir).resolve()
        self.state_path = self.run_dir / "logs" / "wandb_sync.json"
        self.state = {}
        if self.state_path.exists() and not args.restart:
            self.state = json.loads(self.state_path.read_text())
        self.cache: dict = {}
        config, _ = rr.parse_run_log(self.run_dir / "logs" / "run.out")
        run_id = self.state.get("id") or uuid.uuid4().hex[:12]
        self.run = wandb.init(
            project=args.project, entity=args.entity, id=run_id, resume="allow",
            name=args.name or self.run_dir.name, group=args.group, job_type="mho-run",
            config={**config, "run_dir": str(self.run_dir)},
            mode="offline" if args.offline else None)
        if self.state.get("id") != run_id:
            self.state = {"id": run_id, "logged": 0}
            self._save()

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state))

    def sync(self) -> int:
        """Log rows not logged yet (one step each; step = proposals used). Tables and the
        scatter ride along on the last new step, since W&B only accepts increasing steps."""
        rows = rr.candidate_rows(self.run_dir, cache=self.cache)
        new = rows[self.state["logged"]:]
        if not new:
            return 0
        for r in new[:-1]:
            self.run.log(scalars(r), step=r["step"])
        self.run.log({**scalars(new[-1]), **self.tables(rows)}, step=new[-1]["step"])
        fr = rr.frontier(self.run_dir)
        s = self.run.summary
        s["frontier/best"] = (fr.get("_best") or {}).get("candidate")
        s["frontier/best_val_acc"] = (fr.get("_best") or {}).get("metric")
        s["frontier/anchor"] = (fr.get("_anchor") or {}).get("candidate")
        self.state["logged"] = len(rows)
        self._save()
        return len(new)

    def tables(self, rows) -> dict:
        W = self.wandb
        _, jobs = rr.parse_run_log(self.run_dir / "logs" / "run.out")
        tv = W.Table(columns=["candidate", "outcome", "train_t", "val_t"],
                     data=[[r["candidate"], r["outcome"], r["train/t"], r["val/t"]]
                           for r in rows if "train/t" in r and "val/t" in r])
        return {
            "tables/candidates": W.Table(columns=TABLE_COLS,
                                         data=[[r.get(c) for c in TABLE_COLS] for r in rows]),
            "tables/rounds": W.Table(columns=ROUND_COLS, data=[
                [g.get(c) for c in ROUND_COLS] for g in rr.round_rows(rows)]),
            "tables/jobs": W.Table(columns=JOB_COLS, data=[
                [j["job_id"], j["stage"], j["tag"], rr.job_owner(j), str(j["submit"]),
                 str(j["done"] or ""), j["minutes"], j["rc"]] for j in jobs]),
            "charts/train_t_vs_val_t": W.plot.scatter(
                tv, "train_t", "val_t", title="train-gate t vs val-gate t"),
        }

    def finish(self):
        self.run.finish()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "mho"))
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--name", default=None, help="W&B run name (default: the run dir name)")
    p.add_argument("--group", default=None, help="W&B group, e.g. the model or dataset")
    p.add_argument("--watch", type=float, default=0,
                   help="re-sync every N seconds until interrupted (0 = once)")
    p.add_argument("--restart", action="store_true",
                   help="ignore logs/wandb_sync.json and log everything into a NEW W&B run")
    p.add_argument("--offline", action="store_true", help="WANDB_MODE=offline (wandb sync later)")
    p.add_argument("--env-file", default=str(REPO_DIR / ".env"))
    args = p.parse_args(argv)
    load_env(Path(args.env_file))
    s = Syncer(args)
    try:
        while True:
            n = s.sync()
            print(f"{time.strftime('%H:%M:%S')} synced {n} new candidate(s)", flush=True)
            if not args.watch:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
    finally:
        s.finish()


if __name__ == "__main__":
    main()
