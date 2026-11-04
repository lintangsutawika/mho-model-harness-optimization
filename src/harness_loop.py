"""Harness-only optimization loop (dual mode: --train-data + --val-data) as an explicit state machine.

One candidate cycle, over a train subset S that is fixed until a new iteration starts:

    eval_a  : evaluate the PARENT harness (frontier best) on S          -> ts_A
    propose : the proposer writes candidates/<tag>/harness from the parent
    eval_b  : evaluate the candidate on the SAME S                       -> ts_B
              train gate: utils.gate, B vs A paired per task (--gate-train-z)
    eval_val: evaluate the candidate on the val split                    -> val
              val gate: utils.gate, candidate vs the frontier best's val trials, paired per
              task (--gate-val-z); or the "cheaper" path: not worse by more than
              --gate-cheaper-delta (vs the best AND the most accurate accepted, `_anchor`)
              and clearly cheaper (agent output tokens). Without per-task data both gates
              fall back to the plain strict B > A.

Transitions:
    baseline --ok--> eval_a                          (iteration 0 = base harness on val)
    eval_a   --ok--> propose
    propose  --ok--> eval_b      --fail--> next_try   (outcome no_proposal)
    eval_b   --pass-> eval_val   --fail--> next_try   (outcome train_fail | no_score)
    eval_val -------> new iteration                   (outcome accepted | val_fail | no_score)
    next_try: try += 1; re-propose on the same S, or after --ts-repropose-limit tries on S
              start a new iteration (fresh S, fresh A).

Every finished candidate gets exactly ONE terminal record, written by `_finish`: its
candidates/<tag>/status.json and one candidates/summary.json row, with the same fields.
Outcomes: baseline | no_proposal | no_score | train_fail | val_fail | accepted.
Non-terminal events go to <run>/logs/events.jsonl (not visible to the proposer).
Logs: <run>/logs/run.out (driver), <run>/logs/<activity>.out per job (+ <activity>-vllm.out
for evals), e.g. baseline.out, candidate_1_0_1-eval_before.out, candidate_1_0_1-propose.out.

Infra failures (a job that produced no score / no harness) are retried up to
MAX_ATTEMPTS. After that the stage is treated as failed, and the loop ABORTS on a failed
baseline or after MAX_INFRA_FAILS consecutive failed stages, so a broken node/vLLM/API never
silently burns the iteration budget.

The state is a JSON dict persisted after every transition (schema 2), so the driver can be
killed and resumed: an in-flight job is re-attached, never resubmitted.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from utils.gate import GateConfig, decide_dirs
from utils.scores import metric_of, score_trials
from utils.state import _candidate_hypothesis, _sample_subset

SCHEMA = 2
MAX_ATTEMPTS = int(os.environ.get("MHO_MAX_ATTEMPTS", "3"))          # submits per stage
MAX_INFRA_FAILS = int(os.environ.get("MHO_MAX_EVAL_INFRA_FAILS", "3"))  # consecutive, then abort

# Generic proposal-title headers to skip when logging what a candidate changed.
_GENERIC_TITLES = {"candidate.md", "candidate", "hypothesis", "candidate proposal",
                   "proposal", "candidate overview"}


class LoopAbort(RuntimeError):
    """Raised when the loop must stop for a human (persistent infra failure)."""


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _has_trials(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*/result.json"))


def _title(hyp: str | None) -> str:
    for ln in (hyp or "").splitlines():
        t = ln.lstrip("# ").strip()
        if t and t.lower().rstrip(":") not in _GENERIC_TITLES:
            return t[:120]
    return ""


class HarnessLoop:
    """Drives the harness-only dual-mode loop. All side effects go through injectable
    callables (submit_eval / submit_propose / scheduler / score_fn / sleep) so the whole
    state machine is unit-testable with a fake scheduler."""

    def __init__(self, args: Any, run_dir: Path, sched: Any, *,
                 submit_eval: Callable[..., str], submit_propose: Callable[..., str | None],
                 score_fn: Callable[[Path], dict] = score_trials,
                 sleep: Callable[[float], None] = time.sleep, poll_seconds: float = 60):
        self.args, self.run_dir, self.sched = args, run_dir, sched
        self._submit_eval, self._submit_propose = submit_eval, submit_propose
        self.score_fn, self.sleep, self.poll_seconds = score_fn, sleep, poll_seconds
        self.gate_cfg = GateConfig.from_args(args)
        self.cands = run_dir / "candidates"
        self.frontier_path = self.cands / "frontier.json"
        self.summary_path = self.cands / "summary.json"
        self.state_path = run_dir / "state.json"
        self.events_path = run_dir / "logs" / "events.jsonl"
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.cands.mkdir(parents=True, exist_ok=True)
        self.st = self._load_or_init()

    # ------------------------------------------------------------------ state
    def _load_or_init(self) -> dict:
        if not self.args.fresh and self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            if st.get("schema") != SCHEMA:
                raise LoopAbort(
                    f"{self.state_path} was written by the legacy loop (no schema={SCHEMA}). "
                    f"Resume it with MHO_LEGACY_LOOP=1, or start a new --run-name / --fresh.")
            self.st = st
            print(f"{_now()} RESUME {self.run_dir.name}: stage={st['stage']} "
                  f"tag={self.tag} job={(st.get('job') or {}).get('id')}")
            return st
        # Fresh start: this run seeds its OWN frontier from its own baseline.
        if self.frontier_path.exists():
            self.frontier_path.unlink()
        shutil.rmtree(self.run_dir / "best", ignore_errors=True)
        skip = bool(getattr(self.args, "skip_baseline", False))
        st = {
            "schema": SCHEMA, "run_id": self.run_dir.name, "mode": "harness-only",
            "stage": "eval_a" if skip else "baseline",
            "iteration": 1 if skip else 0,
            "version": 0,              # accepted candidates so far
            "try": 1,                  # attempts at the current version (resets on accept)
            "propose_count": 0,        # --iterations budget = proposals that produced a harness
            "parent": None,            # parent harness dir (frontier best); None = base scaffold
            "subset": None,            # {dir, seed, a_tag, ts_A, tries} for the current iteration
            "ts_B": None,              # current candidate's train-sample score
            "job": None,               # in-flight {id, stage, attempt}
            "attempt": 1,              # submit attempt for the current stage
            "infra_fails": 0,          # consecutive stages that failed after all attempts
            "propose_fails": 0,        # consecutive no_proposal outcomes
        }
        self._save(st)
        print(f"{_now()} MHO harness-only  run={self.run_dir.name}  budget={self.args.iterations}")
        return st

    def _save(self, st: dict | None = None) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st or self.st, indent=2))
        tmp.replace(self.state_path)           # atomic: a crash never leaves half a state.json

    @property
    def tag(self) -> str:
        st = self.st
        return "baseline" if st["iteration"] == 0 else \
            f"candidate_{st['iteration']}_{st['version']}_{st['try']}"

    def tag_dir(self, tag: str | None = None) -> Path:
        return self.cands / (tag or self.tag)

    # ------------------------------------------------------------- records
    def _event(self, **row: Any) -> None:
        row = {"time": datetime.now().isoformat(timespec="seconds"), **row}
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def _finish(self, outcome: str, **fields: Any) -> None:
        """The ONE terminal record per candidate: status.json + one summary.json row."""
        st, tag = self.st, self.tag
        sub = st["subset"] or {}
        rec = {
            "iteration": st["iteration"], "candidate": tag, "outcome": outcome,
            "accepted": outcome in ("accepted", "baseline"),
            "ts_A": sub.get("ts_A"), "ts_B": st.get("ts_B"), "subset_seed": sub.get("seed"),
            "train_gate": st.get("train_gate"),
            "hypothesis": _candidate_hypothesis(self.tag_dir(tag)),
            **fields,
        }
        self.tag_dir(tag).mkdir(parents=True, exist_ok=True)
        (self.tag_dir(tag) / "status.json").write_text(json.dumps(rec, indent=2))
        with self.summary_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  => {tag}: {outcome.upper()}"
              + "".join(f" {k}={v}" for k, v in fields.items()
                        if k != "reason" and not isinstance(v, dict)))

    # ------------------------------------------------------------- transitions
    def _goto(self, stage: str) -> None:
        self.st.update(stage=stage, job=None, attempt=1)

    def _new_iteration(self) -> None:
        self.st["iteration"] += 1
        self.st["subset"] = None
        self.st["ts_B"] = None
        self.st["train_gate"] = None
        self._goto("eval_a")

    def _next_try(self) -> None:
        """Same version, next attempt: re-propose on the same subset, or give up on it."""
        st = self.st
        st["try"] += 1
        st["ts_B"] = None
        st["train_gate"] = None
        st["subset"]["tries"] += 1
        if st["subset"]["tries"] >= self.args.ts_repropose_limit:
            print(f"  {st['subset']['tries']} tries on this subset; fresh subset next iteration")
            self._new_iteration()
        else:
            self._goto("propose")

    def _infra_fail(self, what: str) -> None:
        """A stage failed after all attempts. Abort on baseline or too many in a row."""
        st = self.st
        st["infra_fails"] += 1
        if st["stage"] == "baseline" or st["infra_fails"] >= MAX_INFRA_FAILS:
            why = "baseline" if st["stage"] == "baseline" else f"{st['infra_fails']} consecutive stages"
            st.update(job=None, attempt=1, infra_fails=0)   # a restart retries this stage
            self._save()
            raise LoopAbort(f"ABORT: {why} failed after {MAX_ATTEMPTS} attempts each "
                            f"(last: {what}). Likely infra (no GPU / vLLM / proposer API); "
                            f"check {self.log_path()}. Restart the driver to retry.")

    # ------------------------------------------------------------- submit
    def _eval(self, *, harness: str | None, data: str, out_dir: Path, job_name: str) -> str:
        return self._submit_eval(self.sched, "", model=self.args.model, harness=harness,
                                 data=data, n_trials=self.args.trials,
                                 n_concurrent=self.args.concurrent, out_dir=out_dir,
                                 logs=self.log_path(), job_name=job_name)

    _ACTIVITY = {"baseline": "", "eval_a": "eval_before", "propose": "propose",
                 "eval_b": "eval_after", "eval_val": "val"}

    def log_path(self) -> Path:
        """<run>/logs/<activity>.out for the current stage; retries get a -attemptN suffix."""
        st = self.st
        tag = st["subset"]["a_tag"] if st["stage"] == "eval_a" and st["subset"] else self.tag
        act = self._ACTIVITY[st["stage"]]
        name = f"{tag}-{act}" if act else tag
        if st.get("attempt", 1) > 1:
            name += f"-attempt{st['attempt']}"
        return self.run_dir / "logs" / f"{name}.out"

    def _out_dir(self, stage: str) -> Path:
        """Where this stage's trials land (submit and result must agree)."""
        st = self.st
        if stage in ("baseline", "eval_val"):
            return self.run_dir / "evaluation" / self.tag
        if stage == "eval_a":
            return self.tag_dir(st["subset"]["a_tag"]) / "eval_before"
        return self.tag_dir() / "eval_after"            # eval_b

    def _submit(self) -> str | None:
        st, stage, tag = self.st, self.st["stage"], self.tag
        if stage in ("baseline", "eval_val"):
            harness = None if stage == "baseline" else str(self.tag_dir() / "harness")
            return self._eval(harness=harness, data=self.args.val_data,
                              out_dir=self.run_dir / "evaluation", job_name=tag)
        if stage == "eval_a":
            if st["subset"] is None:
                seed = int.from_bytes(os.urandom(4), "big") >> 1
                sdir = _sample_subset(self.args.train_data, self.args.propose_sample,
                                      self.run_dir / "samples" / f"{tag}-train_sample", seed)
                st["subset"] = {"dir": str(sdir), "seed": seed, "a_tag": tag,
                                "ts_A": None, "tries": 0}
                self._save()
            return self._eval(harness=st["parent"], data=st["subset"]["dir"],
                              out_dir=self.tag_dir(st["subset"]["a_tag"]), job_name="eval_before")
        if stage == "propose":
            # Each try is its own candidate dir; give it the subset's A traces so every
            # candidate is self-contained for the proposer (eval_before + eval_after).
            a_tag = st["subset"]["a_tag"]
            link = self.tag_dir() / "eval_before"
            if a_tag != tag and not (link.is_symlink() or link.exists()):
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(Path("..") / a_tag / "eval_before")
            st["propose_count"] += 1
            print(f"  [propose {st['propose_count']}/{self.args.iterations}] {tag}", flush=True)
            jid = self._submit_propose(self.sched, run_root=self.run_dir,
                                       logs=self.log_path(), tag=tag, kind="harness")
            if jid is None:
                raise LoopAbort("propose job not configured (MHO_PROPOSE_JOB unset)")
            return jid
        if stage == "eval_b":
            return self._eval(harness=str(self.tag_dir() / "harness"), data=st["subset"]["dir"],
                              out_dir=self.tag_dir(), job_name="eval_after")
        raise AssertionError(f"no submit for stage {stage!r}")

    # ------------------------------------------------------------- results
    def _job_ok(self, stage: str, rc: int) -> bool:
        if rc != 0:
            return False
        if stage == "propose":
            return (self.tag_dir() / "harness" / "pyproject.toml").exists()
        return _has_trials(self._out_dir(stage))

    def _on_done(self, stage: str, rc: int) -> None:
        st = self.st
        ok = self._job_ok(stage, rc)
        self._event(stage=stage, tag=self.tag, rc=rc, ok=ok, attempt=st["attempt"])
        # Evals are retried (same inputs, transient infra). A failed propose is NOT: each
        # propose burns budget, so it is recorded as no_proposal and the next try proposes
        # afresh; repeated failures trip the propose_fails abort instead.
        if not ok and stage != "propose" and st["attempt"] < MAX_ATTEMPTS:
            st["attempt"] += 1
            st["job"] = None
            why = f"rc={rc}" if rc else "no trials"
            print(f"  {stage} {self.tag} failed ({why}); attempt {st['attempt']}/{MAX_ATTEMPTS}")
            return
        getattr(self, f"_done_{stage}")(ok, rc)

    def _score(self, stage: str) -> float:
        self.st["infra_fails"] = 0
        return metric_of(self.score_fn(self._out_dir(stage)))

    def _done_baseline(self, ok: bool, rc: int) -> None:
        if not ok:
            self._infra_fail(f"baseline eval rc={rc}")
        val = self._score("baseline")
        self._update_frontier(val)
        self._finish("baseline", val=val)
        self._new_iteration()

    def _done_eval_a(self, ok: bool, rc: int) -> None:
        st = self.st
        if not ok:
            self._infra_fail(f"eval_a {self.tag} rc={rc}")
            print("  eval_a produced no score; drawing a fresh subset")
            st["subset"] = None
            self._goto("eval_a")
            return
        st["subset"]["ts_A"] = self._score("eval_a")
        print(f"  A (parent) on subset = {st['subset']['ts_A']:.4f}")
        self._goto("propose")

    def _done_propose(self, ok: bool, rc: int) -> None:
        st = self.st
        if not ok:
            # Infra-type failure (proposer env / API): refund the budget and the subset slot,
            # move to a new tag. MAX_INFRA_FAILS consecutive failures abort the run instead.
            st["propose_fails"] += 1
            st["propose_count"] -= 1
            st["subset"]["tries"] -= 1
            self._finish("no_proposal", reason=f"rc={rc} or no harness/pyproject.toml")
            if st["propose_fails"] >= MAX_INFRA_FAILS:
                st["propose_fails"] = 0
                self._next_try()
                self._save()
                raise LoopAbort(f"ABORT: {MAX_INFRA_FAILS} consecutive proposals produced no "
                                f"harness; check {self.run_dir}/logs/*-propose.out.")
            self._next_try()
            return
        st["propose_fails"] = 0
        title = _title(_candidate_hypothesis(self.tag_dir()))
        if title:
            print(f"  [+] {self.tag}: {title}")
        self._goto("eval_b")

    def _done_eval_b(self, ok: bool, rc: int) -> None:
        st = self.st
        if not ok:
            self._infra_fail(f"eval_b {self.tag} rc={rc}")
            self._finish("no_score", reason=f"train-sample eval rc={rc}")
            self._next_try()
            return
        st["ts_B"] = self._score("eval_b")
        ts_A = st["subset"]["ts_A"]
        dec = decide_dirs(self._out_dir("eval_a"), self._out_dir("eval_b"),
                          z=self.gate_cfg.train_z, cfg=self.gate_cfg, fallback=(ts_A, st["ts_B"]))
        st["train_gate"] = dec.fields()
        self._event(stage="train_gate", tag=self.tag, **dec.fields())
        if dec.accept:
            print(f"  train gate PASS [{dec.path}] (B {st['ts_B']:.4f} vs A {ts_A:.4f}): "
                  f"{dec.reason} -> val")
            self._goto("eval_val")
        else:
            print(f"  train gate FAIL (B {st['ts_B']:.4f} vs A {ts_A:.4f}): {dec.reason}; "
                  f"not run on val")
            self._finish("train_fail")
            self._next_try()

    def _done_eval_val(self, ok: bool, rc: int) -> None:
        st = self.st
        if not ok:
            self._infra_fail(f"eval_val {self.tag} rc={rc}")
            self._finish("no_score", reason=f"val eval rc={rc}")
            st["try"] += 1
            self._new_iteration()
            return
        val = self._score("eval_val")
        best = self._best_metric()
        fr = self._frontier()
        ref = fr.get("_best", {}).get("candidate")
        anc = (fr.get("_anchor") or fr.get("_best") or {}).get("candidate")
        dec = decide_dirs(self._val_dir(ref) if ref else None, self._out_dir("eval_val"),
                          z=self.gate_cfg.val_z, cfg=self.gate_cfg,
                          anchor_dir=self._val_dir(anc) if anc else None, fallback=(best, val))
        self._event(stage="val_gate", tag=self.tag, ref=ref, anchor_tag=anc, **dec.fields())
        print(f"  val gate {'PASS [' + str(dec.path) + ']' if dec.accept else 'FAIL'} "
              f"(val {val:.4f} vs {ref} {best}): {dec.reason}")
        if self._update_frontier(val, accept=dec.accept):
            harness = self.tag_dir() / "harness"
            shutil.rmtree(self.run_dir / "best", ignore_errors=True)
            shutil.copytree(harness, self.run_dir / "best" / "harness")
            self._finish("accepted", val=val, prev_best=best, path=dec.path,
                         val_gate=dec.fields())
            st["parent"] = str(harness)
            st["version"] += 1
            st["try"] = 1
        else:
            self._finish("val_fail", val=val, best=best, val_gate=dec.fields())
            st["try"] += 1
        self._new_iteration()

    # ------------------------------------------------------------- frontier
    def _frontier(self) -> dict:
        return json.loads(self.frontier_path.read_text()) if self.frontier_path.exists() else {}

    def _best_metric(self) -> float | None:
        return self._frontier().get("_best", {}).get("metric")

    def _val_dir(self, tag: str) -> Path:
        return self.run_dir / "evaluation" / tag

    def _update_frontier(self, val: float, accept: bool | None = None) -> bool:
        """Same frontier.json format the proposer reads (_best.candidate -> parent harness).
        `accept` is the gate's verdict (None = first entry / plain val > best). `_anchor` is
        the most accurate accepted entry: the cheaper path must stay non-inferior to it."""
        fr = self._frontier()
        best = fr.get("_best", {}).get("metric")
        improved = best is None or (val > best if accept is None else accept)
        if improved:
            fr["_best"] = {"candidate": self.tag, "metric": val}
            anc = fr.get("_anchor")
            if anc is None or val > anc.get("metric", float("-inf")):
                fr["_anchor"] = {"candidate": self.tag, "metric": val}
        fr.setdefault("history", []).append({"candidate": self.tag, "metric": val,
                                             "accepted": bool(improved)})
        self.frontier_path.write_text(json.dumps(fr, indent=2))
        return improved

    # ------------------------------------------------------------- driver
    def _budget_left(self) -> bool:
        return self.st["propose_count"] < self.args.iterations

    def step(self) -> bool:
        """Advance by at most one job event. Returns False when the run is finished."""
        st = self.st
        job = st["job"]
        if job:
            norm, rc = self.sched.state(job["id"])
            if norm != "DONE":
                self.sleep(self.poll_seconds)
                return True
            rc = 0 if rc is None else rc
            print(f"{_now()} [{job['stage']}] job {job['id']} done (rc={rc})")
            self._on_done(job["stage"], rc)
            self._save()
            return True
        # Nothing in flight. Stop before starting a new candidate once the budget is spent;
        # a candidate already proposed always runs to its terminal outcome.
        if st["stage"] in ("eval_a", "propose") and not self._budget_left():
            return False
        print(f"{_now()} iter {st['iteration']} {st['stage']} {self.tag} -> submit"
              + (f" (attempt {st['attempt']}/{MAX_ATTEMPTS})" if st["attempt"] > 1 else ""))
        jid = self._submit()
        st["job"] = {"id": jid, "stage": st["stage"], "attempt": st["attempt"]}
        self._save()
        return True

    def run(self) -> None:
        while self.step():
            pass
        fr = json.loads(self.frontier_path.read_text()) if self.frontier_path.exists() else {}
        best = fr.get("_best", {})
        print(f"\n{_now()} Optimize complete. proposals={self.st['propose_count']} "
              f"best={best.get('candidate')} @ {best.get('metric')}  run_dir={self.run_dir}")
