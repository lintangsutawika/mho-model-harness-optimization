"""MHO optimization loop: a PBS job manager that co-optimizes model (RL) + harness.

This is the long-running *driver*. It does no heavy work itself: each phase --
propose (harness mutation), train (RL), eval (harbor) -- is submitted to PBS as its own
`qsub` job, and the driver polls `qstat` until it finishes before advancing. Per
iteration the state machine is:

    [propose] -> [train] -> eval -> score -> frontier

which phases run is set by --mode:
    model-only    : train -> eval          (fixed harness)
    harness-only  : propose -> eval         (fixed model)
    model-harness : propose -> train -> eval

Only the *fitting* bits are borrowed from stanford-iris-lab/meta-harness: scoring a
harbor trials dir (via scripts/eval/aggregate_results.py), a JSONL frontier + evolution
summary, and the propose->eval sequencing. Everything else here is PBS orchestration.

The phase jobs are the existing scripts, submitted directly (`qsub -V <script>` with
per-phase `-l` resources); they already read their config from env vars and resolve
their own repo dir, so no PBS wrapper scripts are needed. A harness candidate is a
scaffold snapshot dir passed through MICRO_SCAFFOLD_DIR.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("MHO_RUNS_DIR", REPO_DIR / "runs_output" / "optimize"))
AGGREGATE_PY = REPO_DIR / "scripts" / "eval" / "aggregate_results.py"

# Phase job scripts submitted by the loop (one per phase). Each dispatches to the concrete
# entrypoint from env (eval.sh on MHO_TASK; train.sh on MHO_TRAINER). Overridable via env.
EVAL_SCRIPT = Path(os.environ.get("MHO_EVAL_SCRIPT", REPO_DIR / "scripts" / "optimize" / "eval.sh"))
TRAIN_SCRIPT = Path(os.environ.get("MHO_TRAIN_SCRIPT", REPO_DIR / "scripts" / "optimize" / "train.sh"))
# Propose job: the analysis/mutation step. Set MHO_PROPOSE_JOB (the run/*.sh front-ends point
# it at scripts/optimize/propose.sh). Absent -> the harness phase is skipped (model-only runs).
PROPOSE_JOB = os.environ.get("MHO_PROPOSE_JOB", "")

AGENT_IMPORT = os.environ.get("AGENT_IMPORT", "harness.agent_harness:AgentHarness")
POLL_SECONDS = int(os.environ.get("MHO_POLL_SECONDS", 60))


# --------------------------------------------------------------------------- #
# Scoring -- reuse scripts/eval/aggregate_results.py (no duplication).
# --------------------------------------------------------------------------- #
_AGG = None


def _agg():
    global _AGG
    if _AGG is None:
        spec = importlib.util.spec_from_file_location("aggregate_results", AGGREGATE_PY)
        _AGG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_AGG)  # type: ignore[union-attr]
    return _AGG


def score_trials(trials_dir: str | Path) -> dict[str, Any]:
    trials_dir = Path(trials_dir)
    return _agg().aggregate(str(trials_dir), str(trials_dir / "aggregated.jsonl"))


# --------------------------------------------------------------------------- #
# Scheduler abstraction -- PBS (qsub/qstat) or SLURM (sbatch/squeue+sacct).
# scripts/hpc/ holds the per-scheduler shell wrappers; this is the Python driver side,
# so the loop is scheduler-agnostic. Force one with MHO_SCHEDULER=pbs|slurm.
# --------------------------------------------------------------------------- #
def detect_scheduler() -> str:
    kind = os.environ.get("MHO_SCHEDULER", "").lower()
    if kind in ("pbs", "slurm"):
        return kind
    if shutil.which("sbatch"):
        return "slurm"
    if shutil.which("qsub"):
        return "pbs"
    raise RuntimeError("no batch scheduler found (need sbatch or qsub); set MHO_SCHEDULER")


class Scheduler:
    """Submit/poll batch jobs on PBS or SLURM behind one interface.

    Resources are given abstractly (ngpus/cpus/walltime/queue/account); each backend
    renders its own flags. Jobs export the full env (PBS -V / SLURM --export=ALL) so the
    phase config rides along without fragile per-key quoting.
    """

    def __init__(self, kind: str | None = None):
        self.kind = kind or detect_scheduler()

    def submit(self, script: Path, env: dict[str, str], *, name: str = "", ngpus: int = 0,
               cpus: int = 0, walltime: str = "", queue: str = "", account: str = "",
               select: str = "", resource_type: str = "", out_dir: Path | None = None) -> str:
        # `select` (raw PBS -l select string, e.g. ABCI's "1:ncpus=192:mem=1920gb:ngpus=8") wins
        # over the auto-built ngpus/cpus select -- clusters with full-node/reservation shapes need
        # the exact string. `resource_type` -> PBS `-v RTYPE=...`, required by ABCI reservation
        # (RESERVE) queues (e.g. rt_HF / rt_HC). On SLURM both are ignored (use gres/cpus).
        job_env = {**os.environ, **{k: str(v) for k, v in env.items()}}
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        if self.kind == "pbs":
            cmd = ["qsub", "-V"]
            if resource_type: cmd += ["-v", f"RTYPE={resource_type}"]
            if name: cmd += ["-N", name]
            if queue: cmd += ["-q", queue]
            if account: cmd += ["-P", account]
            sel = select or ("select=1" + (f":ngpus={ngpus}" if ngpus else "") + (f":ncpus={cpus}" if cpus else ""))
            cmd += ["-l", sel]
            if walltime: cmd += ["-l", f"walltime={walltime}"]
            if out_dir: cmd += ["-o", str(out_dir), "-j", "oe"]
            cmd.append(str(script))
        else:  # slurm
            cmd = ["sbatch", "--parsable", "--export=ALL"]
            if name: cmd += ["-J", name]
            if queue: cmd += ["-p", queue]
            if account: cmd += ["-A", account]
            if ngpus: cmd += [f"--gres=gpu:{ngpus}"]
            if cpus: cmd += [f"--cpus-per-task={cpus}"]
            if walltime: cmd += ["-t", walltime]
            if out_dir: cmd += ["-o", str(out_dir / "%x-%j.out")]
            cmd.append(str(script))
        print(f"  submit[{self.kind}]: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=str(REPO_DIR), env=job_env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"submit failed rc={r.returncode}: {r.stderr.strip() or r.stdout.strip()}")
        out = r.stdout.strip()
        # SLURM --parsable: "<id>[;cluster]"; PBS: "<id>.<server>".
        job_id = (out.split()[-1].split(";")[0]) if self.kind == "slurm" else out.split()[0]
        print(f"  submitted job {job_id}", flush=True)
        return job_id

    def state(self, job_id: str) -> tuple[str, int | None]:
        """Return (normalized_state, exit_status); normalized in PENDING/RUNNING/DONE."""
        if self.kind == "pbs":
            r = subprocess.run(["qstat", "-f", job_id], capture_output=True, text=True)
            if r.returncode != 0:
                return "DONE", None  # purged from the queue => finished
            st, exit_status = "?", None
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith("job_state"):
                    st = line.split("=", 1)[1].strip()
                elif line.startswith("Exit_status"):
                    try:
                        exit_status = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
            norm = {"R": "RUNNING", "E": "RUNNING", "Q": "PENDING", "H": "PENDING",
                    "F": "DONE"}.get(st, "DONE" if exit_status is not None else "PENDING")
            return norm, exit_status
        # slurm: squeue for liveness, sacct for the final exit code
        r = subprocess.run(["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                           capture_output=True, text=True)
        live = r.stdout.strip()
        if live:
            return ("RUNNING" if live.startswith("RUN") else "PENDING"), None
        s = subprocess.run(["sacct", "-j", str(job_id), "-n", "-P", "-o", "State,ExitCode"],
                           capture_output=True, text=True)
        exit_status = None
        for line in s.stdout.splitlines():
            parts = line.split("|")
            if len(parts) >= 2 and parts[1].strip():
                try:
                    exit_status = int(parts[1].split(":")[0])
                except ValueError:
                    pass
                break
        return "DONE", exit_status

    def wait(self, job_id: str) -> int:
        """Block until `job_id` finishes; return its exit status (0 if unreported)."""
        while True:
            norm, exit_status = self.state(job_id)
            if norm == "DONE":
                return exit_status if exit_status is not None else 0
            time.sleep(POLL_SECONDS)


# --------------------------------------------------------------------------- #
# Frontier + summary (JSONL) -- the fitting bits from the reference.
# --------------------------------------------------------------------------- #
def update_frontier(frontier_path: Path, candidate: str, score: dict[str, Any]) -> bool:
    frontier = json.loads(frontier_path.read_text()) if frontier_path.exists() else {}
    metric = score.get("mean_reward") or score.get("solved_pass@1") or 0.0
    improved = metric > frontier.get("_best", {}).get("metric", -1)
    if improved:
        frontier["_best"] = {"candidate": candidate, "metric": metric, "score": score}
    frontier.setdefault("history", []).append({"candidate": candidate, "metric": metric})
    frontier_path.write_text(json.dumps(frontier, indent=2))
    return improved


def append_summary(summary_path: Path, row: dict[str, Any]) -> None:
    with summary_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- #
# Phase submitters -- each returns a job id (or None if the phase is skipped).
# --------------------------------------------------------------------------- #
def _phase_env(extra: dict[str, str], scaffold_dir: Path | None) -> dict[str, str]:
    env = dict(extra)
    # PBS spools the job script to /var/spool/pbs/..., so the phase scripts' BASH_SOURCE-based
    # repo-dir resolution is wrong in the job. Pass the real repo dir explicitly.
    env["MHO_REPO_DIR"] = str(REPO_DIR)
    env["PYTHONPATH"] = f"{REPO_DIR / 'src'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    if scaffold_dir is not None:
        env["MICRO_SCAFFOLD_DIR"] = str(scaffold_dir)
        env["MINI_FORK_LOCAL"] = str(scaffold_dir)
    return env


_ACCOUNT = os.environ.get("MHO_ACCOUNT", "")  # PBS -P / SLURM -A


def _data_env(data: str) -> dict[str, str]:
    """Map a data spec to the phase script's -p/-d env: a path on disk -> MHO_DATA_PATH (-p),
    else a registry name -> MHO_DATA (-d)."""
    if data and Path(data).exists():
        return {"MHO_DATA_PATH": data}
    return {"MHO_DATA": data} if data else {}


def submit_eval(sched: Scheduler, task: str, *, model: str | None, harness: str | None,
                data: str, n_trials: int, n_concurrent: int, out_dir: Path, logs: Path,
                tag: str) -> str:
    # The loop hands eval.sh the (model, harness, data) explicitly via MHO_* -- exactly the
    # manual form `MHO_MODEL=<model> MHO_DATA=<data> MHO_HARNESS=<harness> bash .../eval.sh`.
    # Name + locate the output by the iteration tag (baseline, iter1, ...), not a timestamp, so
    # each iteration's eval is findable/scoreable at MHO_OUT/<tag>.
    extra = {
        "MHO_TASK": task,
        "AGENT_IMPORT": AGENT_IMPORT,
        "TASK_SET": "full",
        "N_ATTEMPTS": str(n_trials),
        "N_CONCURRENT": str(n_concurrent),
        "MHO_OUT": str(out_dir),
        "EVAL_JOB_NAME": tag,
        **_data_env(data),
    }
    if model:
        extra["MHO_MODEL"] = model
    if harness:
        extra["MHO_HARNESS"] = harness
    # eval self-serves the model with vLLM on the node -> needs GPU(s). Default 1 (Qwen3.5-4B
    # fits on one H200); set MHO_EVAL_NGPUS=0 for an API/litellm model (no self-serve).
    return sched.submit(EVAL_SCRIPT, _phase_env(extra, None), name=f"mho-eval-{tag}",
                        queue=os.environ.get("MHO_EVAL_QUEUE", ""), account=_ACCOUNT,
                        select=os.environ.get("MHO_EVAL_SELECT", ""),
                        resource_type=os.environ.get("MHO_EVAL_RTYPE", ""),
                        ngpus=int(os.environ.get("MHO_EVAL_NGPUS", 1)),
                        cpus=int(os.environ.get("MHO_EVAL_CPUS", 0) or 0),
                        walltime=os.environ.get("MHO_EVAL_WALLTIME", "08:00:00"), out_dir=logs)


def submit_train(sched: Scheduler, trainer: str, *, model: str | None, harness: str | None,
                 out_dir: Path, logs: Path, tag: str) -> str:
    extra = {"MHO_TRAINER": trainer, "MHO_OUT": str(out_dir)}
    if model:
        extra["MHO_MODEL"] = model
    if harness:
        extra["MHO_HARNESS"] = harness
    return sched.submit(TRAIN_SCRIPT, _phase_env(extra, None), name=f"mho-train-{tag}",
                        queue=os.environ.get("MHO_TRAIN_QUEUE", ""), account=_ACCOUNT,
                        select=os.environ.get("MHO_TRAIN_SELECT", ""),
                        resource_type=os.environ.get("MHO_TRAIN_RTYPE", ""),
                        ngpus=int(os.environ.get("MHO_TRAIN_NGPUS", 8)),
                        walltime=os.environ.get("MHO_TRAIN_WALLTIME", "24:00:00"), out_dir=logs)


def submit_propose(sched: Scheduler, parent_scaffold: str | None, *, candidate_dir: Path,
                   frontier_path: Path, summary_path: Path, diff_dir: Path, trials_dir: str | None,
                   logs: Path, tag: str) -> str | None:
    """Submit the harness-mutation job (MHO_PROPOSE_JOB). It builds the new candidate ON TOP of
    `parent_scaffold` (the frontier-best, so changes accumulate; None -> the base) and writes it
    to $MHO_CANDIDATE_DIR (a valid installable scaffold) + two provenance diffs. Returns None if
    unconfigured."""
    if not PROPOSE_JOB:
        print("  MHO_PROPOSE_JOB unset -> skipping harness proposal this iteration")
        return None
    env = _phase_env({
        "MHO_CANDIDATE_DIR": str(candidate_dir),
        "MHO_PARENT_SCAFFOLD": parent_scaffold or "",   # accumulate on the frontier-best
        "MHO_DIFF_DIR": str(diff_dir),
        "MHO_FRONTIER": str(frontier_path),
        "MHO_SUMMARY": str(summary_path),
        # the previous eval's trials, for the proposer to analyze (empty before any eval)
        "MHO_TRIALS_DIR": trials_dir or "",
    }, None)
    return sched.submit(Path(PROPOSE_JOB), env, name=f"mho-propose-{tag}",
                        queue=os.environ.get("MHO_PROPOSE_QUEUE", ""), account=_ACCOUNT,
                        select=os.environ.get("MHO_PROPOSE_SELECT", ""),
                        resource_type=os.environ.get("MHO_PROPOSE_RTYPE", ""),
                        ngpus=int(os.environ.get("MHO_PROPOSE_NGPUS", 0) or 0),
                        walltime=os.environ.get("MHO_PROPOSE_WALLTIME", "02:00:00"), out_dir=logs)


def latest_trials_dir() -> str | None:
    return os.environ.get("MHO_TRIALS_DIR") or _agg()._newest_trials_dir()


def _candidate_hypothesis(candidate_dir: Path) -> str | None:
    """The candidate's own description of what it changed (CANDIDATE.md), for the summary so the
    next proposer sees what was tried. None if the agent didn't write one."""
    md = candidate_dir / "CANDIDATE.md"
    if md.exists():
        return md.read_text()[:1000].strip()
    return None


def _latest_hf_ckpt(hf_dir: Path) -> Path | None:
    """Newest loadable HF checkpoint under a miles --save-hf dir (step_N subdirs), or the dir
    itself if it's already a checkpoint, else None."""
    if not hf_dir.is_dir():
        return None
    steps = sorted((d for d in hf_dir.glob("step_*") if (d / "config.json").exists()),
                   key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else -1)
    if steps:
        return steps[-1]
    return hf_dir if (hf_dir / "config.json").exists() else None


# --------------------------------------------------------------------------- #
# Main loop.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# State cursor -- the loop's single source of truth for "which stage are we on".
# Persisted after every transition so the driver can die and resume (re-attaching
# to an in-flight job rather than resubmitting).
# --------------------------------------------------------------------------- #
def actions_for(iteration: int, do_propose: bool, do_train: bool) -> list[str]:
    """The ordered stages for one iteration. Iteration 0 is the baseline (measure the
    starting model+harness); later iterations run the mode's full cycle."""
    if iteration == 0:
        return ["eval"]
    acts: list[str] = []
    if do_propose:
        acts.append("propose")
    if do_train:
        acts.append("train")
    acts.append("eval")
    return acts


def _load_state(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def _save_state(path: Path, st: dict) -> None:
    path.write_text(json.dumps(st, indent=2))


def optimize(args: argparse.Namespace) -> None:
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    frontier_path = run_dir / "frontier.json"
    summary_path = run_dir / "evolution_summary.jsonl"
    state_path = run_dir / "state.json"

    do_train = "model" in args.mode
    do_propose = "harness" in args.mode
    sched = Scheduler()

    st = None if args.fresh else _load_state(state_path)
    if st is None:
        start = 0 if not args.skip_baseline else 1
        st = {"run_id": run_id, "mode": args.mode, "iteration": start, "action_index": 0,
              "actions": actions_for(start, do_propose, do_train), "job_id": None,
              "scaffold_dir": None,        # the candidate currently being trained/evaled
              "parent_scaffold": None,     # frontier-best so far; next propose accumulates on it
              "model_dir": None,           # latest trained checkpoint (HF dir); None -> args.model
              "last_eval_dir": None,       # most-recent eval's trials dir, for the next propose
              "version": 0,                # # of ACCEPTED (improving) candidates so far
              "try": 1}                    # attempts at the current version; resets on acceptance
        _save_state(state_path, st)
        print(f"{datetime.now():%H:%M:%S} MHO optimize  run={run_id}  mode={args.mode}  task={args.task}  "
              f"trainer={args.trainer}  scheduler={sched.kind}  iters={args.iterations}")
    else:
        print(f"{datetime.now():%H:%M:%S} MHO optimize  RESUME run={run_id}  iter={st['iteration']}  "
              f"action={st['actions'][st['action_index']]}  job={st['job_id']}  scheduler={sched.kind}")

    def cur_tag() -> str:
        # candidate_{iteration}_{version}_{try}: iteration=total attempts, version=accepted
        # improvements so far, try=attempts at this version (resets on acceptance). Harness modes
        # only; model-only has no candidates -> iter{n}. Iteration 0 is the base-harness baseline.
        if st["iteration"] == 0:
            return "baseline"
        if do_propose:
            return f"candidate_{st['iteration']}_{st['version']}_{st['try']}"
        return f"iter{st['iteration']}"

    def advance(skip_rest: bool = False) -> None:
        """Move to the next stage, or (past the last, or on skip_rest) the next iteration."""
        if skip_rest or st["action_index"] + 1 >= len(st["actions"]):
            st["iteration"] += 1
            st["actions"] = actions_for(st["iteration"], do_propose, do_train)
            st["action_index"] = 0
        else:
            st["action_index"] += 1

    def submit_current() -> str | None:
        # The loop initiates each stage; the model + harness + data are threaded in
        # automatically from state. harness = the current candidate (st["scaffold_dir"], the
        # freshly-proposed one, else the frontier-best parent, else the base). model = the
        # latest trained checkpoint (st["model_dir"]) if any, else args.model / the HF default.
        action = st["actions"][st["action_index"]]
        tag = cur_tag()
        harness = st["scaffold_dir"] or st["parent_scaffold"]           # latest_harness
        model = st["model_dir"] or args.model                          # latest_model (or default)
        if action == "propose":
            return submit_propose(sched, st["parent_scaffold"], candidate_dir=run_dir / "candidates" / tag,
                                  frontier_path=frontier_path, summary_path=summary_path,
                                  diff_dir=run_dir / "diffs" / tag, trials_dir=st["last_eval_dir"],
                                  logs=logs_dir, tag=tag)
        if action == "train":
            return submit_train(sched, args.trainer, model=model, harness=harness,
                                out_dir=run_dir / "train" / tag, logs=logs_dir, tag=tag)
        # out_dir is the harbor jobs-dir; EVAL_JOB_NAME=tag makes the per-iteration subdir,
        # so trials land at run_dir/eval/<tag>/.
        return submit_eval(sched, args.task, model=model, harness=harness, data=args.data,
                           n_trials=args.trials, n_concurrent=args.concurrent,
                           out_dir=run_dir / "eval", logs=logs_dir, tag=tag)

    def on_done(rc: int) -> None:
        """Handle the finished stage's result and advance the cursor."""
        action, tag, it = st["actions"][st["action_index"]], cur_tag(), st["iteration"]
        if action == "propose":
            cand_dir = run_dir / "candidates" / tag
            if rc == 0 and (cand_dir / "pyproject.toml").exists():
                st["scaffold_dir"] = str(cand_dir)
                advance()
            else:
                append_summary(summary_path, {"iteration": it, "candidate": tag, "phase": "propose",
                                              "status": "no_candidate", "rc": rc})
                print(f"  propose failed (rc={rc}); skipping to next iteration")
                advance(skip_rest=True)
        elif action == "train":
            if rc == 0:
                # the freshly-trained checkpoint becomes the model for this iteration's eval.
                # miles saves HF checkpoints as <out>/checkpoints/hf/step_N (--save-hf); pick the
                # latest step. If HF export is off/absent, keep the prior model.
                ckpt = _latest_hf_ckpt(run_dir / "train" / tag / "checkpoints" / "hf")
                if ckpt is not None:
                    st["model_dir"] = str(ckpt)
                advance()
            else:
                append_summary(summary_path, {"iteration": it, "candidate": tag, "phase": "train",
                                              "status": f"rc={rc}"})
                print(f"  train failed (rc={rc}); skipping to next iteration")
                advance(skip_rest=True)
        else:  # eval (last stage of the iteration)
            # eval.sh wrote trials to the deterministic MHO_OUT/<tag> = run_dir/eval/<tag>
            # (job name = tag). Score that dir directly -- not the Ray-session glob.
            trials_dir = run_dir / "eval" / tag
            has_trials = trials_dir.is_dir() and any(trials_dir.glob("*/result.json"))
            if rc != 0 or not has_trials:
                append_summary(summary_path, {"iteration": it, "candidate": tag, "phase": "eval",
                                              "status": f"rc={rc}" if rc else "no_trials",
                                              "trials_dir": str(trials_dir)})
                print(f"  eval produced no score (rc={rc}, trials={trials_dir})")
            else:
                score = score_trials(trials_dir)
                st["last_eval_dir"] = str(trials_dir)   # next propose analyzes this eval
                improved = update_frontier(frontier_path, tag, score)
                if improved and st["scaffold_dir"]:
                    # ACCEPTED: accumulate (next propose builds ON this new best), bump the
                    # harness version, reset the retry counter.
                    st["parent_scaffold"] = st["scaffold_dir"]
                    st["version"] += 1
                    st["try"] = 1
                elif do_propose and st["scaffold_dir"]:
                    # rejected harness attempt (a real candidate) -> another try at same version.
                    # Guard on scaffold_dir so the baseline eval (scaffold_dir=None, improved=True
                    # only because the frontier was empty) does NOT bump try -- otherwise the first
                    # real candidate is misnamed candidate_1_0_2 instead of candidate_1_0_1.
                    st["try"] += 1
                # record WHAT this candidate changed (its CANDIDATE.md), so the next proposer
                # can learn from the outcome (build on winners, avoid repeating regressions).
                hypothesis = _candidate_hypothesis(run_dir / "candidates" / tag)
                append_summary(summary_path, {"iteration": it, "candidate": tag, "phase": "eval",
                                              "trials_dir": str(trials_dir), "score": score,
                                              "scaffold_dir": st["scaffold_dir"], "improved": improved,
                                              "hypothesis": hypothesis})
                print(f"  {'NEW BEST' if improved else 'no improvement'}  "
                      f"pass@1={score.get('solved_pass@1')} mean_reward={score.get('mean_reward')}")
            advance()

    # ── The loop: wait on the in-flight job, else submit the current stage. ──
    while st["iteration"] <= args.iterations:
        action = st["actions"][st["action_index"]]
        if st["job_id"]:  # a job is in flight -> check and wait
            state, rc = sched.state(st["job_id"])
            if state != "DONE":
                time.sleep(POLL_SECONDS)
                continue
            print(f"{datetime.now():%H:%M:%S} [{action}] job {st['job_id']} done (rc={rc})")
            st["job_id"] = None
            on_done(rc if rc is not None else 0)
            _save_state(state_path, st)
            continue
        # nothing in flight -> submit the current stage (or skip it if unconfigured)
        print(f"{datetime.now():%H:%M:%S} iter {st['iteration']} stage={action} -> submit")
        jid = submit_current()
        if jid is None:  # e.g. propose with MHO_PROPOSE_JOB unset
            advance()
        else:
            st["job_id"] = jid
        _save_state(state_path, st)

    best = (json.loads(frontier_path.read_text()) if frontier_path.exists() else {}).get("_best", {})
    print(f"\n{datetime.now():%H:%M:%S} Optimize complete. frontier={best.get('candidate')} "
          f"@ {best.get('metric')}  run_dir={run_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["model-only", "harness-only", "model-harness"],
                   default="model-harness")
    p.add_argument("--task", choices=["math", "aime", "tb2"], default="math")
    p.add_argument("--trainer", choices=["skyrl", "miles"], default="miles")
    p.add_argument("--model", default=os.environ.get("MHO_MODEL", ""),
                   help="starting model: HF name/local dir (self-served) or a litellm id (API). "
                        "In model modes, each iteration's eval uses the freshly-trained checkpoint instead.")
    p.add_argument("--data", default=os.environ.get("MHO_DATA", "aime/aime"),
                   help="eval data: a registry name (e.g. aime/aime -> -d) or a local task/dataset dir (-> -p).")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--trials", type=int, default=int(os.environ.get("EVAL_N_ATTEMPTS", 5)),
                   help="attempts per problem (harbor --n-attempts): each eval task is sampled "
                        "this many times and pass@1 is the mean solve rate. NOT concurrency. "
                        "Total trials = <#dataset tasks> * this. Env override: EVAL_N_ATTEMPTS.")
    p.add_argument("--concurrent", type=int, default=int(os.environ.get("EVAL_CONCURRENCY", 16)),
                   help="trials run in parallel (harbor -n). Pure throughput; does not affect the "
                        "score. Bounded by GPU serving capacity. Env override: EVAL_CONCURRENCY.")
    p.add_argument("--run-name", default=None, help="stable name so a killed run can resume its state.json")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--fresh", action="store_true", help="ignore any existing state.json and restart the run")
    args = p.parse_args(argv)
    # harness-only never trains, so no checkpoint is ever produced -- the harness must be
    # evaluated against a FIXED model. Without one, eval.sh falls back to a bogus litellm id and
    # every trial silently scores 0 ("eval produced no score"). Fail fast at launch instead.
    if args.mode == "harness-only" and not args.model:
        p.error("harness-only mode needs a fixed model to evaluate the harness against; pass "
                "--model <HF name/dir or litellm id> (or set MHO_MODEL), e.g. --model Qwen/Qwen3.5-4B")
    optimize(args)


if __name__ == "__main__":
    main()
