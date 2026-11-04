"""MHO optimization loop driver: co-optimizes the harness and the model (RL) on PBS/SLURM.

This is the long-running *driver*. It does no heavy work itself: every stage -- eval
(harbor), propose (the proposer agent), train (Miles RL) -- is its own batch job
(mho/jobs.py), and the driver polls until it finishes before advancing.

The loop itself is mho/loop.py: one state machine over a (harness, model) pair, where each
candidate changes one side of it (mho/kinds/harness.py, mho/kinds/model.py) and is accepted by
the shared paired per-task gates (utils/gate.py). --mode only picks the schedule of candidate
kinds:

    harness-only  : harness candidates (fixed model)
    model-only    : model candidates (fixed harness)
    model-harness : --harness-proposals harness, then --model-proposals model, repeating

Data: --train-data is the local train task dir (each iteration evaluates a fresh random
--propose-sample subset of it: the proposer's evidence and the train gate), --val-data the
held-out val split (the accept gate).

This file is the CLI: arguments, the run dir and its logs/run.out, and the crash log.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from mho.jobs import (AGENT_IMPORT, EVAL_SCRIPT, LOG_VERBOSE, POLL_SECONDS,  # noqa: F401
                      PROPOSE_JOB, TRAIN_SCRIPT, Scheduler, submit_eval, submit_propose,
                      submit_train)
from mho.kinds import HarnessKind, ModelKind
from mho.loop import Loop
from utils.gate import GateConfig

REPO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("MHO_RUNS_DIR", REPO_DIR / "runs_output" / "optimize"))


def schedule_for(args: argparse.Namespace) -> list[tuple[str, int]]:
    return {"harness-only": [("harness", 1)],
            "model-only": [("model", 1)],
            "model-harness": [("harness", max(1, args.harness_proposals)),
                              ("model", max(1, args.model_proposals))]}[args.mode]


class _Tee:
    """Write to the original stream and to a log file (line-flushed, so `tail -f` works)."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _tee_driver_log(path: Path) -> None:
    """Mirror the driver's stdout/stderr into <run_dir>/logs/run.out (appended across
    restarts), so the log lives with the run and is readable while the job is running --
    PBS only copies its -o file back when the job ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "a", buffering=1)
    log.write(f"\n===== driver start {datetime.now():%Y-%m-%d %H:%M:%S} "
              f"host={os.uname().nodename} job={os.environ.get('PBS_JOBID') or os.environ.get('SLURM_JOB_ID') or '-'} "
              f"argv={' '.join(sys.argv[1:])}\n")
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["model-only", "harness-only", "model-harness"],
                   default="model-harness")

    p.add_argument("--trainer", choices=["miles"], default="miles",
                   help="model candidates' trainer (scripts/optimize/train.sh runs Miles)")
    p.add_argument("--model", default=os.environ.get("MHO_MODEL", ""),
                   help="starting model: HF name/local dir (self-served) or a litellm id (API). "
                        "In model modes, each iteration's eval uses the freshly-trained checkpoint instead.")
    p.add_argument("--train-data", default=os.environ.get("MHO_TRAIN_DATA", ""),
                   help="local train task dir: each iteration evaluates a fresh random "
                        "--propose-sample subset of it (the proposer's evidence + the train gate); "
                        "model candidates also train on it. Env: MHO_TRAIN_DATA.")
    p.add_argument("--val-data", default=os.environ.get("MHO_VAL_DATA", ""),
                   help="held-out val task/dataset dir (e.g. the --val-size val split): the accept "
                        "gate. Never shown to the proposer. Env: MHO_VAL_DATA.")
    p.add_argument("--propose-sample", type=int,
                   default=int(os.environ.get("MHO_PROPOSE_SAMPLE", 64)),
                   help="how many problems to draw (random, fresh each iteration) "
                        "from the FULL train task dir (--train-data) for the proposer's improvement signal. "
                        "Keeps a 1000s-problem train set from being eval'd in full. Env: MHO_PROPOSE_SAMPLE.")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--harness-proposals", type=int, default=int(os.environ.get("MHO_HARNESS_PROPOSALS", 10)),
                   help="model-harness: # harness proposals in a harness phase before switching to model (M_harness)")
    p.add_argument("--model-proposals", type=int, default=int(os.environ.get("MHO_MODEL_PROPOSALS", 2)),
                   help="model-harness: # model proposals in a model phase before switching to harness")
    p.add_argument("--trials", type=int, default=int(os.environ.get("EVAL_N_ATTEMPTS", 5)),
                   help="attempts per problem (harbor --n-attempts): each eval task is sampled "
                        "this many times and pass@1 is the mean solve rate. NOT concurrency. "
                        "Total trials = <#dataset tasks> * this. Env override: EVAL_N_ATTEMPTS.")
    p.add_argument("--concurrent", type=int, default=int(os.environ.get("EVAL_CONCURRENCY", 16)),
                   help="trials run in parallel (harbor -n). Pure throughput; does not affect the "
                        "score. Bounded by GPU serving capacity. Env override: EVAL_CONCURRENCY.")
    p.add_argument("--ts-repropose-limit", type=int,
                   default=int(os.environ.get("MHO_TS_REPROPOSE_LIMIT", 3)),
                   help="Max consecutive train-sample failures (T_B < T_A on the same subset) "
                        "before giving up on this harness/subset and moving to a fresh subset.",
                   metavar="N")
    _g = GateConfig.env_defaults()
    p.add_argument("--gate-train-z", dest="gate_train_z", type=float, default=_g["train_z"],
                   help="train gate, 'better' path: accept B iff mean_i(B_i - A_i) > z * SE over "
                        "the subset's tasks (paired; attempts averaged per task). 0 = plain B > A. "
                        "Env: MHO_GATE_TRAIN_Z.")
    p.add_argument("--gate-val-z", dest="gate_val_z", type=float, default=_g["val_z"],
                   help="val gate, 'better' path: same test vs the frontier best's val trials "
                        "(1.645 = one-sided 95%%). Applies to harness and model candidates. "
                        "Env: MHO_GATE_VAL_Z.")
    p.add_argument("--gate-cheaper-delta", dest="gate_cheaper_delta", type=float,
                   default=_g["cheaper_delta"],
                   help="'cheaper' path (both gates): also accept a candidate whose accuracy is not "
                        "worse by more than this (lower bound of the paired difference, vs the "
                        "reference and the most accurate accepted entry) if it is clearly cheaper. "
                        "0 disables the path. Env: MHO_GATE_CHEAPER_DELTA.")
    p.add_argument("--gate-cheaper-z", dest="gate_cheaper_z", type=float, default=_g["cheaper_z"],
                   help="z for the cheaper path's non-inferiority and cost tests. "
                        "Env: MHO_GATE_CHEAPER_Z.")
    p.add_argument("--gate-cheaper-min-gain", dest="gate_cheaper_min_gain", type=float,
                   default=_g["cheaper_min_gain"],
                   help="cheaper path: minimum relative cost reduction (0.10 = 10%%). "
                        "Env: MHO_GATE_CHEAPER_MIN_GAIN.")
    p.add_argument("--gate-cost", dest="gate_cost_metric", choices=["tokens", "seconds"],
                   default=_g["cost_metric"],
                   help="cost for the cheaper path: agent output tokens per trial (default; "
                        "independent of node load) or agent wall seconds. Env: MHO_GATE_COST.")
    p.add_argument("--run-name", default=None, help="stable name so a killed run can resume its state.json")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--fresh", action="store_true", help="ignore any existing state.json and restart the run")
    args = p.parse_args(argv)
    # Every eval needs a model (the starting point of model candidates, the fixed model of
    # harness candidates). Without one, eval.sh falls back to a bogus litellm id and every
    # trial silently scores 0 ("eval produced no score"). Fail fast at launch instead.
    if not args.model:
        p.error("pass --model <HF name/dir or litellm id> (or set MHO_MODEL), "
                "e.g. --model Qwen/Qwen3.5-4B")
    if not args.val_data:
        p.error("pass --val-data <held-out val split>: it is the accept gate (selection on the "
                "train data alone is not supported)")
    if not Path(args.train_data or "").is_dir():
        p.error("--train-data must be a LOCAL train task dir (a path), because each iteration "
                "evaluates a random subset of it; got: " + repr(args.train_data))
    if args.propose_sample <= 0:
        p.error(f"--propose-sample must be > 0 (got {args.propose_sample})")
    args.ts_repropose_limit = max(1, args.ts_repropose_limit)
    # Fix the run name once so the driver log and the loop agree on the run dir.
    args.run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / args.run_name
    _tee_driver_log(run_dir / "logs" / "run.out")
    try:
        kinds = {"harness": HarnessKind(),
                 "model": ModelKind(train_data=args.train_data, trainer=args.trainer,
                                    submit_train=submit_train)}
        Loop(args, run_dir, Scheduler(), kinds=kinds, schedule=schedule_for(args),
             submit_eval=submit_eval, submit_propose=submit_propose,
             poll_seconds=POLL_SECONDS).run()
    except Exception:
        import traceback as _tb
        _msg = _tb.format_exc()
        # stderr is tee'd into logs/run.out; also keep a standalone crash file in the run dir
        print("MHO LOOP CRASH:", file=sys.stderr)
        print(_msg, file=sys.stderr)
        try:
            (run_dir / "logs" / "loop_crash.log").write_text(_msg)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
