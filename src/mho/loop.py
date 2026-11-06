"""The optimization loop (dual mode: --train-data + --val-data) as an explicit state machine.

Every eval is of a (harness, model) PAIR; the state holds the current best pair. Each candidate
changes one side of it, as decided by its kind (mho/kinds/): a harness candidate edits the
harness, a model candidate trains a new checkpoint. The schedule picks the kind of each new
candidate (harness-only, model-only, or N harness then M model, repeating).

One candidate cycle, over a train subset S that is fixed until a new iteration starts:

    eval_a  : evaluate the CURRENT pair on S                             -> ts_A
    round   : M sibling candidates (--proposals; 1 for kinds that are not `parallel`), each
              independently and concurrently, tags candidate_<i>_<v>_<t>{a,b,c,...}:
      propose : the proposer writes candidates/<tag>/ (harness/ or model_specs/); sibling k of
                M is told so (MHO_PROPOSE_SIBLING=k/M) to steer it to a different idea
      build   : the kind's job, if any (model: train the current model) [kind.build_stage]
      eval_b  : evaluate the candidate pair on the SAME S                -> ts_B
                train gate: utils.gate, B vs A paired per task (--gate-z)
                [only kinds with train_gate: a model candidate was trained on the train data,
                 so it goes from build straight to selection]
      select  : of the siblings that passed, the one with the largest paired t (train gate)
                goes to val; the other passers are recorded as not_selected. Only ONE goes to
                val, so the val gate's false-accept rate does not grow with M.
    eval_val: evaluate the selected candidate pair on the val split      -> val
              val gate: utils.gate, vs the current best's val trials, paired per task
              (--gate-z, the same test as the train gate); or the "cheaper" path: not worse by more than
              --gate-cheaper-delta (vs the best AND the most accurate accepted, `_anchor`)
              and clearly cheaper (agent output tokens). Without per-task data both gates
              fall back to the plain strict B > A.
    accept  : the candidate pair becomes the current pair (a rejected one changes nothing,
              so a rejected model is never trained on).

Transitions:
    baseline --ok--> eval_a                          (iteration 0 = base pair on val)
    eval_a   --ok--> round
    round    --a sibling passed--> eval_val
             --none passed-------> next round: try += 1; re-propose on the same S, or after
                                   --ts-repropose-limit rounds on S start a new iteration
    eval_val -------> new iteration                       (outcome accepted | val_fail | no_score)
  per sibling:
    propose  --ok--> build | eval_b | selection  --fail--> out (outcome no_proposal)
    build    --ok--> eval_b | selection          --fail--> out (outcome build_fail)
    eval_b   --pass-> selection                  --fail--> out (outcome train_fail | no_score)

Every finished candidate gets exactly ONE terminal record, written by `_finish`: its
candidates/<tag>/status.json and one candidates/summary.json row, with the same fields.
Outcomes: baseline | no_proposal | build_fail | no_score | train_fail | not_selected |
val_fail | accepted.
Non-terminal events go to <run>/logs/events.jsonl (not visible to the proposer).
Logs: <run>/logs/run.out (driver), <run>/logs/<activity>.out per job (+ <activity>-vllm.out
for evals), e.g. baseline.out, candidate_1_0_1-eval_before.out, candidate_1_0_1-propose.out,
candidate_1_0_1-train.out.

Infra failures (a job that produced no score / no harness) are retried up to MAX_ATTEMPTS
(builds: MAX_BUILD_ATTEMPTS). After that the stage is treated as failed, and the loop ABORTS on a
failed baseline or after MAX_INFRA_FAILS consecutive failed stages, so a broken node/vLLM/API
never silently burns the iteration budget.

The state is a JSON dict persisted after every transition (schema 4; schema-2/3 runs are
migrated on resume), so the driver can be killed and resumed: in-flight jobs are re-attached,
never resubmitted.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from mho.kinds import Kind, Pair
from utils.gate import GateConfig, decide_dirs, env_failures
from utils.scores import metric_of, score_trials
from utils.state import _candidate_hypothesis, _sample_subset

SCHEMA = 4
MAX_ATTEMPTS = int(os.environ.get("MHO_MAX_ATTEMPTS", "3"))             # submits per eval stage
MAX_BUILD_ATTEMPTS = int(os.environ.get("MHO_MAX_BUILD_ATTEMPTS", "1"))  # e.g. a training run
MAX_INFRA_FAILS = int(os.environ.get("MHO_MAX_EVAL_INFRA_FAILS", "3"))  # consecutive, then abort
# An eval where more than this fraction of trials never got a sandbox is an infra failure (bad
# node), retried -- not a score of 0 for the candidate.
ENV_FAIL_MAX = float(os.environ.get("MHO_ENV_FAIL_MAX", "0.1"))
# Before that retry, park a sleeping job on the bad node for this long so the retry lands
# elsewhere (0 = don't).
BAD_NODE_HOLD = int(os.environ.get("MHO_BAD_NODE_HOLD", "7200"))

# Generic proposal-title headers to skip when logging what a candidate changed.
_GENERIC_TITLES = {"candidate.md", "candidate", "hypothesis", "candidate proposal",
                   "proposal", "candidate overview"}
_EVAL_STAGES = ("baseline", "eval_a", "eval_b", "eval_val")


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


class Loop:
    """Drives the dual-mode loop. All side effects go through injectable callables
    (submit_eval / submit_propose / the kinds' builds / scheduler / score_fn / sleep), so the
    whole state machine is unit-testable with a fake scheduler.

    kinds:    {"harness": HarnessKind(), "model": ModelKind(...)} (only the ones scheduled)
    schedule: [(kind, n), ...] cycled: n candidates of each kind in turn, e.g.
              [("harness", 1)] (harness-only) or [("harness", 10), ("model", 2)].
    """

    def __init__(self, args: Any, run_dir: Path, sched: Any, *, kinds: dict[str, Kind],
                 schedule: list[tuple[str, int]],
                 submit_eval: Callable[..., str], submit_propose: Callable[..., str | None],
                 submit_hold: Callable[..., str] | None = None,
                 score_fn: Callable[[Path], dict] = score_trials,
                 sleep: Callable[[float], None] = time.sleep, poll_seconds: float = 60):
        unknown = {k for k, _ in schedule} - set(kinds)
        if not schedule or unknown or any(n < 1 for _, n in schedule):
            raise ValueError(f"bad schedule {schedule!r} for kinds {sorted(kinds)}")
        self.args, self.run_dir, self.sched = args, run_dir, sched
        self.kinds, self.schedule = kinds, schedule
        self._submit_eval, self._submit_propose = submit_eval, submit_propose
        self._submit_hold = submit_hold
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
            if st.get("schema") == 2:
                st = self._migrate_v2(st)
            if st.get("schema") == 3:
                st = self._migrate_v3(st)
            if st.get("schema") != SCHEMA:
                raise LoopAbort(
                    f"{self.state_path} was written by the old optimize() loop (no schema). "
                    f"Finish it with an older checkout, or start a new --run-name / --fresh.")
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
            "schema": SCHEMA, "run_id": self.run_dir.name, "mode": getattr(self.args, "mode", ""),
            "stage": "eval_a" if skip else "baseline",
            "iteration": 1 if skip else 0,
            "version": 0,              # accepted candidates so far
            "try": 1,                  # attempts at the current version (resets on accept)
            "propose_count": 0,        # --iterations budget = proposals that produced a candidate
            "pair": Pair().as_dict(),  # current best (harness, model); None = base / --model
            "kind": None,              # kind of the round / candidate in flight
            "schedule": {"i": 0, "done": 0},   # position in self.schedule
            "subset": None,            # {dir, seed, a_tag, ts_A, tries} for the current iteration
            "round": None,             # {members: [sibling, ...]} while stage == "round"
            "cur": None,               # the selected sibling's tag (eval_val)
            "ts_B": None,              # selected candidate's train-sample score
            "train_gate": None,        # selected candidate's train-gate stats
            "job": None,               # in-flight {id, stage, attempt} (not used by "round")
            "attempt": 1,              # submit attempt for the current stage
            "infra_fails": 0,          # consecutive stages that failed after all attempts
            "propose_fails": 0,        # consecutive no_proposal outcomes
        }
        self._save(st)
        print(f"{_now()} MHO {st['mode'] or 'loop'}  run={self.run_dir.name}  "
              f"budget={self.args.iterations}  schedule={self.schedule}")
        return st

    @staticmethod
    def _migrate_v2(st: dict) -> dict:
        """A schema-2 run (the harness-only HarnessLoop) is a schema-3 run whose pair has the
        base model and whose every candidate was a harness candidate."""
        st = dict(st)
        st["pair"] = {"harness": st.pop("parent", None), "model": None}
        st["kind"] = "harness" if st.get("stage") in ("propose", "eval_b", "eval_val") else None
        st.setdefault("schedule", {"i": 0, "done": 0})
        st.setdefault("train_gate", None)
        st["schema"] = 3                       # then _migrate_v3
        print("  migrated state.json schema 2 -> 3 (pair.harness = previous parent)")
        return st

    def _migrate_v3(self, st: dict) -> dict:
        """Schema 3 ran one candidate at a time through stages propose/<build>/eval_b. A run
        stopped in one of those becomes a round with that candidate as its only sibling (its
        in-flight job is kept)."""
        st = dict(st)
        st.setdefault("round", None)
        st.setdefault("cur", None)
        if st["stage"] not in ("baseline", "eval_a", "eval_val"):
            st["kind"] = st.get("kind") or self.schedule[st["schedule"]["i"] % len(self.schedule)][0]
            tag = f"candidate_{st['iteration']}_{st['version']}_{st['try']}"
            st["round"] = {"members": [{
                "tag": tag, "i": 0, "n": 1, "stage": st["stage"],
                "job": (st.get("job") or {}).get("id"), "attempt": st.get("attempt", 1),
                "ts_B": st.get("ts_B"), "train_gate": st.get("train_gate"), "status": "running"}]}
            st.update(stage="round", job=None, attempt=1)
        st["schema"] = SCHEMA
        print("  migrated state.json schema 3 -> 4 (one-candidate round)")
        return st

    def _save(self, st: dict | None = None) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st or self.st, indent=2))
        tmp.replace(self.state_path)           # atomic: a crash never leaves half a state.json

    @property
    def tag(self) -> str:
        """The candidate in focus: the selected sibling during eval_val, else the round's base
        tag (siblings append a letter when there is more than one)."""
        st = self.st
        if st.get("cur"):
            return st["cur"]
        return "baseline" if st["iteration"] == 0 else \
            f"candidate_{st['iteration']}_{st['version']}_{st['try']}"

    def tag_dir(self, tag: str | None = None) -> Path:
        return self.cands / (tag or self.tag)

    @property
    def pair(self) -> Pair:
        return Pair.from_dict(self.st["pair"])

    @property
    def kind(self) -> Kind:
        return self.kinds[self.st["kind"]]

    def _scheduled_kind(self) -> str:
        return self.schedule[self.st["schedule"]["i"] % len(self.schedule)][0]

    def _schedule_step(self) -> None:
        """A candidate of the scheduled kind finished: count it, switch kind after n of them."""
        sch = self.st["schedule"]
        sch["done"] += 1
        name, n = self.schedule[sch["i"] % len(self.schedule)]
        if sch["done"] >= n:
            sch["i"], sch["done"] = (sch["i"] + 1) % len(self.schedule), 0
            nxt = self._scheduled_kind()
            if nxt != name:
                print(f"  [schedule] {n} {name} candidate(s) done -> {nxt} candidates")

    # ------------------------------------------------------------- records
    def _event(self, **row: Any) -> None:
        row = {"time": datetime.now().isoformat(timespec="seconds"), **row}
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def _finish(self, outcome: str, m: dict | None = None, **fields: Any) -> None:
        """The ONE terminal record per candidate: status.json + one summary.json row.
        `m` = the round sibling it is about (default: the candidate in focus)."""
        st = self.st
        src = m if m is not None else st
        tag = m["tag"] if m is not None else self.tag
        sub = st["subset"] or {}
        kind = "baseline" if outcome == "baseline" else st.get("kind")
        rec = {
            "iteration": st["iteration"], "candidate": tag, "kind": kind, "outcome": outcome,
            "accepted": outcome in ("accepted", "baseline"),
            "ts_A": sub.get("ts_A"), "ts_B": src.get("ts_B"), "subset_seed": sub.get("seed"),
            "train_gate": src.get("train_gate"),
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
        if outcome not in ("baseline", "no_proposal"):
            self._schedule_step()

    # ------------------------------------------------------------- transitions
    def _goto(self, stage: str) -> None:
        self.st.update(stage=stage, job=None, attempt=1)

    def _reset_candidate(self) -> None:
        self.st.update(ts_B=None, train_gate=None, kind=None, round=None, cur=None)

    def _new_iteration(self) -> None:
        self.st["iteration"] += 1
        self.st["subset"] = None
        self._reset_candidate()
        self._goto("eval_a")

    def _next_round(self, *, count_try: bool = True) -> None:
        """No sibling of the round passed: re-propose on the same subset, or give up on it.
        A round where every sibling failed to propose (infra) does not use up the subset."""
        st = self.st
        st["try"] += 1
        self._reset_candidate()
        if count_try:
            st["subset"]["tries"] += 1
        if st["subset"]["tries"] >= self.args.ts_repropose_limit:
            print(f"  {st['subset']['tries']} rounds on this subset; fresh subset next iteration")
            self._new_iteration()
        else:
            self._goto("round")

    def _infra_fail(self, what: str) -> None:
        """A stage failed after all attempts. Abort on baseline or too many in a row."""
        st = self.st
        st["infra_fails"] += 1
        if st["stage"] == "baseline" or st["infra_fails"] >= MAX_INFRA_FAILS:
            why = "baseline" if st["stage"] == "baseline" else f"{st['infra_fails']} consecutive stages"
            st.update(job=None, attempt=1, infra_fails=0)   # a restart retries this stage
            self._save()
            raise LoopAbort(f"ABORT: {why} failed after all attempts "
                            f"(last: {what}). Likely infra (no GPU / vLLM / proposer API / "
                            f"trainer); check {self.log_path()}. Restart the driver to retry.")

    # ------------------------------------------------------------- submit
    def _eval(self, pair: Pair, *, data: str, out_dir: Path, job_name: str,
              logs: Path | None = None) -> str:
        # A trained checkpoint is served/sampled with its base model's configs (MHO_BASE_MODEL).
        return self._submit_eval(self.sched, "", model=pair.model or self.args.model,
                                 harness=pair.harness, data=data, n_trials=self.args.trials,
                                 n_concurrent=self.args.concurrent, out_dir=out_dir,
                                 logs=logs or self.log_path(), job_name=job_name,
                                 base_model=self.args.model if pair.model else None)

    _ACTIVITY = {"baseline": "", "eval_a": "eval_before", "propose": "propose",
                 "eval_b": "eval_after", "eval_val": "val"}

    def log_path(self, tag: str | None = None, stage: str | None = None,
                 attempt: int | None = None) -> Path:
        """<run>/logs/<tag>-<activity>.out (default: the current stage); retries get a
        -attemptN suffix."""
        st = self.st
        stage = stage or st["stage"]
        attempt = attempt or st.get("attempt", 1)
        if tag is None:
            tag = st["subset"]["a_tag"] if stage == "eval_a" and st["subset"] else self.tag
        act = self._ACTIVITY.get(stage, stage)     # build stages: their own name
        name = f"{tag}-{act}" if act else tag
        if attempt > 1:
            name += f"-attempt{attempt}"
        return self.run_dir / "logs" / f"{name}.out"

    def _out_dir(self, stage: str, tag: str | None = None) -> Path:
        """Where this stage's trials land (submit and result must agree)."""
        st = self.st
        if stage in ("baseline", "eval_val"):
            return self.run_dir / "evaluation" / (tag or self.tag)
        if stage == "eval_a":
            return self.tag_dir(st["subset"]["a_tag"]) / "eval_before"
        return self.tag_dir(tag) / "eval_after"            # eval_b

    def _cand_pair(self, tag: str | None = None) -> Pair:
        return self.kind.candidate_pair(self.tag_dir(tag), self.pair)

    def _submit(self) -> str | None:
        st, stage, tag = self.st, self.st["stage"], self.tag
        if stage == "baseline":
            return self._eval(self.pair, data=self.args.val_data,
                              out_dir=self.run_dir / "evaluation", job_name=tag)
        if stage == "eval_val":
            return self._eval(self._cand_pair(), data=self.args.val_data,
                              out_dir=self.run_dir / "evaluation", job_name=tag)
        if stage == "eval_a":
            if st["subset"] is None:
                seed = int.from_bytes(os.urandom(4), "big") >> 1
                sdir = _sample_subset(self.args.train_data, self.args.propose_sample,
                                      self.run_dir / "samples" / f"{tag}-train_sample", seed)
                st["subset"] = {"dir": str(sdir), "seed": seed, "a_tag": tag,
                                "ts_A": None, "tries": 0}
                self._save()
            return self._eval(self.pair, data=st["subset"]["dir"],
                              out_dir=self.tag_dir(st["subset"]["a_tag"]), job_name="eval_before")
        raise AssertionError(f"no submit for stage {stage!r}")

    def _submit_member(self, m: dict) -> str:
        """Submit the current stage of round sibling `m`."""
        st, stage, tag = self.st, m["stage"], m["tag"]
        logs = self.log_path(tag, stage, m["attempt"])
        if stage == "propose":
            # Each sibling is its own candidate dir; give it the subset's A traces so every
            # candidate is self-contained for the proposer (eval_before + eval_after).
            a_tag = st["subset"]["a_tag"]
            link = self.tag_dir(tag) / "eval_before"
            if a_tag != tag and not (link.is_symlink() or link.exists()):
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(Path("..") / a_tag / "eval_before")
            st["propose_count"] += 1
            print(f"  [propose {st['propose_count']}/{self.args.iterations}] {tag} "
                  f"({st['kind']})", flush=True)
            jid = self._submit_propose(self.sched, run_root=self.run_dir, logs=logs, tag=tag,
                                       kind=st["kind"], sibling=(m["i"] + 1, m["n"]))
            if jid is None:
                raise LoopAbort("propose job not configured (MHO_PROPOSE_JOB unset)")
            return jid
        if stage == "eval_b":
            return self._eval(self._cand_pair(tag), data=st["subset"]["dir"],
                              out_dir=self.tag_dir(tag), job_name="eval_after", logs=logs)
        if stage == self.kind.build_stage:
            return self.kind.submit_build(self, self.tag_dir(tag), self.pair, logs)
        raise AssertionError(f"no submit for sibling stage {stage!r}")

    # ------------------------------------------------------------- results
    def _job_ok(self, stage: str, rc: int, tag: str | None = None,
                job_id: str | None = None) -> tuple[bool, str]:
        if rc != 0:
            return False, f"rc={rc}"
        if stage == "propose":
            return self.kind.proposal_ok(self.tag_dir(tag))
        if stage in _EVAL_STAGES:
            out = self._out_dir(stage, tag)
            if not _has_trials(out):
                return False, "no trials"
            bad, n = env_failures(out)
            if n and bad / n > ENV_FAIL_MAX:
                host = self._job_host(job_id or (self.st.get("job") or {}).get("id"))
                self._park(host)
                return False, (f"{bad}/{n} trials never got a sandbox (node {host or '?'}); "
                               f"infra, not the candidate")
            return True, ""
        return self.kind.build_ok(self.tag_dir(tag))

    def _job_host(self, job_id: str | None) -> str | None:
        get = getattr(self.sched, "host", None)
        try:
            return get(job_id) if get and job_id else None
        except Exception:
            return None

    def _park(self, host: str | None) -> None:
        """Hold a node whose sandboxes do not start, so retries go elsewhere (once per host)."""
        held = self.st.setdefault("bad_hosts", {})
        if not host or host in held or not self._submit_hold or BAD_NODE_HOLD <= 0:
            return
        log = self.run_dir / "logs" / f"hold-{host}.out"
        try:
            held[host] = self._submit_hold(self.sched, host, BAD_NODE_HOLD, log)
        except Exception as e:                     # parking is best effort
            held[host] = None
            print(f"  could not park bad node {host}: {e}")
            return
        self._event(stage="bad_node", host=host, hold_job=held[host], seconds=BAD_NODE_HOLD)
        print(f"  BAD NODE {host}: sandboxes do not start there; parked job {held[host]} on it "
              f"for {BAD_NODE_HOLD}s so the retry runs elsewhere. Report {host} to the admins.")

    @staticmethod
    def _attempt_limit(stage: str) -> int:
        # Evals (and builds, up to MAX_BUILD_ATTEMPTS) are retried: same inputs, transient
        # infra. A failed propose is NOT: each propose burns budget, so it is recorded as
        # no_proposal; repeated failures abort instead.
        return MAX_ATTEMPTS if stage in _EVAL_STAGES else \
            0 if stage == "propose" else MAX_BUILD_ATTEMPTS

    def _on_done(self, stage: str, rc: int) -> None:
        """A top-level job (baseline / eval_a / eval_val) finished."""
        st = self.st
        ok, why = self._job_ok(stage, rc)
        self._why = why
        self._event(stage=stage, tag=self.tag, rc=rc, ok=ok, attempt=st["attempt"],
                    **({"why": why} if why else {}))
        limit = self._attempt_limit(stage)
        if not ok and st["attempt"] < limit:
            self._set_aside(stage, st["attempt"])
            st["attempt"] += 1
            st["job"] = None
            print(f"  {stage} {self.tag} failed ({why}); attempt {st['attempt']}/{limit}")
            return
        getattr(self, f"_done_{stage}")(ok, rc)

    def _set_aside(self, stage: str, attempt: int, tag: str | None = None) -> None:
        """Move a failed eval's trials out of the way, so the retry starts clean (the same
        --job-name dir would otherwise be resumed) and the evidence is kept."""
        d = self._out_dir(stage, tag)
        if d.is_dir():
            dst = d.with_name(f"{d.name}.failed-attempt{attempt}")
            if dst.exists():
                shutil.rmtree(dst)
            d.rename(dst)

    def _score(self, stage: str, tag: str | None = None) -> float:
        self.st["infra_fails"] = 0
        return metric_of(self.score_fn(self._out_dir(stage, tag)))

    def _done_baseline(self, ok: bool, rc: int) -> None:
        if not ok:
            self._infra_fail(f"baseline eval rc={rc}")
        val = self._score("baseline")
        self._update_frontier(val)
        self._write_best_pair()
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
        print(f"  A (current pair) on subset = {st['subset']['ts_A']:.4f}")
        self._goto("round")

    # ------------------------------------------------------------- round (M siblings)
    def _round_size(self) -> int:
        """--proposals siblings (1 for kinds that are not parallel), capped by what is left of
        the budget and, when the schedule alternates kinds, of its current slot (so the kinds
        still alternate exactly: N harness then M model)."""
        st = self.st
        m = max(1, int(getattr(self.args, "proposals", 1) or 1)) if self.kind.parallel else 1
        m = min(m, self.args.iterations - st["propose_count"])
        if len({k for k, _ in self.schedule}) > 1:
            _, n = self.schedule[st["schedule"]["i"] % len(self.schedule)]
            m = min(m, n - st["schedule"]["done"])
        return max(1, m)

    def _start_round(self) -> None:
        st = self.st
        st["kind"] = st.get("kind") or self._scheduled_kind()
        n = self._round_size()
        base = self.tag
        tags = [base] if n == 1 else [base + chr(ord("a") + k) for k in range(n)]
        st["round"] = {"members": [
            {"tag": t, "i": k, "n": n, "stage": "propose", "job": None, "attempt": 1,
             "ts_B": None, "train_gate": None, "status": "running"}
            for k, t in enumerate(tags)]}
        if n > 1:
            print(f"  round of {n} {st['kind']} proposals: {', '.join(tags)}")

    def _step_round(self) -> bool:
        """Advance every running sibling by at most one job event; select when all are done."""
        st = self.st
        if st["round"] is None:
            if not self._budget_left():
                return False
            self._start_round()
            self._save()
        progressed = False
        for m in st["round"]["members"]:
            if m["status"] != "running":
                continue
            if m["job"] is None:
                print(f"{_now()} iter {st['iteration']} {m['stage']} {m['tag']} -> submit"
                      + (f" (attempt {m['attempt']})" if m["attempt"] > 1 else ""))
                m["job"] = self._submit_member(m)
                self._save()
                progressed = True
                continue
            norm, rc = self.sched.state(m["job"])
            if norm != "DONE":
                continue
            rc = 0 if rc is None else rc
            print(f"{_now()} [{m['stage']}] {m['tag']} job {m['job']} done (rc={rc})")
            self._member_done(m, rc)
            self._save()
            progressed = True
        if all(m["status"] != "running" for m in st["round"]["members"]):
            self._end_round()
            self._save()
            return True
        if not progressed:
            self.sleep(self.poll_seconds)
        return True

    def _member_done(self, m: dict, rc: int) -> None:
        st, stage, tag = self.st, m["stage"], m["tag"]
        ok, why = self._job_ok(stage, rc, tag, m["job"])
        self._event(stage=stage, tag=tag, rc=rc, ok=ok, attempt=m["attempt"],
                    **({"why": why} if why else {}))
        limit = self._attempt_limit(stage)
        if not ok and m["attempt"] < limit:
            if stage in _EVAL_STAGES:
                self._set_aside(stage, m["attempt"], tag)
            m["attempt"] += 1
            m["job"] = None
            print(f"  {stage} {tag} failed ({why}); attempt {m['attempt']}/{limit}")
            return
        m["job"], m["attempt"] = None, 1     # an abort below retries this stage on restart
        if stage == "propose":
            if not ok:
                # Infra-type failure (proposer env / API): refund the budget.
                # MAX_INFRA_FAILS consecutive failures abort the run instead.
                st["propose_fails"] += 1
                st["propose_count"] -= 1
                m.update(status="out", outcome="no_proposal")
                self._finish("no_proposal", m, reason=why)
                if st["propose_fails"] >= MAX_INFRA_FAILS:
                    st["propose_fails"] = 0
                    raise LoopAbort(f"ABORT: {MAX_INFRA_FAILS} consecutive proposals produced "
                                    f"no candidate; check {self.run_dir}/logs/*-propose.out.")
                return
            st["propose_fails"] = 0
            title = _title(_candidate_hypothesis(self.tag_dir(tag)))
            if title:
                print(f"  [+] {tag} ({st['kind']}): {title}")
            if self.kind.build_stage:
                m["stage"] = self.kind.build_stage
            else:
                self._after_build(m)
            return
        if stage == "eval_b":
            self._member_eval_b(m, ok, rc)
            return
        # the kind's build job (e.g. model training)
        if not ok:
            self._infra_fail(f"{stage} {tag}: {why}")
            m.update(status="out", outcome="build_fail")
            self._finish("build_fail", m, reason=f"{stage}: {why}")
            return
        st["infra_fails"] = 0
        built = self._cand_pair(tag)
        print(f"  built {tag}: {built.model or built.harness}")
        self._after_build(m)

    def _after_build(self, m: dict) -> None:
        if self.kind.train_gate:
            m["stage"] = "eval_b"
        else:
            print(f"  {self.st['kind']} candidate {m['tag']}: no train gate -> selection")
            m["status"] = "pass"

    def _member_eval_b(self, m: dict, ok: bool, rc: int) -> None:
        st, tag = self.st, m["tag"]
        if not ok:
            self._infra_fail(f"eval_b {tag} rc={rc}")
            m.update(status="out", outcome="no_score")
            self._finish("no_score", m, reason=f"train-sample eval rc={rc}")
            return
        m["ts_B"] = self._score("eval_b", tag)
        ts_A = st["subset"]["ts_A"]
        dec = decide_dirs(self._out_dir("eval_a"), self._out_dir("eval_b", tag),
                          z=self.gate_cfg.z, cfg=self.gate_cfg, fallback=(ts_A, m["ts_B"]))
        m["train_gate"] = dec.fields()
        self._event(stage="train_gate", tag=tag, **dec.fields())
        if dec.accept:
            print(f"  train gate PASS [{dec.path}] {tag} (B {m['ts_B']:.4f} vs A {ts_A:.4f}): "
                  f"{dec.reason}")
            m["status"] = "pass"
        else:
            print(f"  train gate FAIL {tag} (B {m['ts_B']:.4f} vs A {ts_A:.4f}): {dec.reason}; "
                  f"not run on val")
            m.update(status="out", outcome="train_fail")
            self._finish("train_fail", m)

    @staticmethod
    def _rank(m: dict) -> tuple:
        """Selection order among siblings that passed: the train gate's paired t (strength of
        the evidence), then the raw train score. Siblings without stats rank by order."""
        acc = (m.get("train_gate") or {}).get("acc") or {}
        t = acc.get("t")
        if t is None:      # no per-task data, or zero variance (gate stores an infinite t as None)
            d = acc.get("d")
            t = float("-inf") if d is None else math.copysign(float("inf"), d) if d else 0.0
        return (t,
                m["ts_B"] if m.get("ts_B") is not None else float("-inf"), -m["i"])

    def _end_round(self) -> None:
        st = self.st
        members = st["round"]["members"]
        passed = [m for m in members if m["status"] == "pass"]
        if not passed:
            infra_only = all(m.get("outcome") == "no_proposal" for m in members)
            self._next_round(count_try=not infra_only)
            return
        win = max(passed, key=self._rank)
        for m in passed:
            if m is not win:
                self._finish("not_selected", m, selected=win["tag"])
        if len(members) > 1:
            print(f"  selected {win['tag']} for val ({len(passed)}/{len(members)} passed the "
                  f"train gate)")
        st.update(cur=win["tag"], ts_B=win["ts_B"], train_gate=win["train_gate"], round=None)
        self._goto("eval_val")

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
                          z=self.gate_cfg.z, cfg=self.gate_cfg,
                          anchor_dir=self._val_dir(anc) if anc else None, fallback=(best, val))
        self._event(stage="val_gate", tag=self.tag, ref=ref, anchor_tag=anc, **dec.fields())
        print(f"  val gate {'PASS [' + str(dec.path) + ']' if dec.accept else 'FAIL'} "
              f"(val {val:.4f} vs {ref} {best}): {dec.reason}")
        cand = self._cand_pair()
        if self._update_frontier(val, accept=dec.accept):
            self.kind.on_accept(self.tag_dir(), self.run_dir, cand)
            st["pair"] = cand.as_dict()
            self._write_best_pair()
            self._finish("accepted", val=val, prev_best=best, path=dec.path,
                         val_gate=dec.fields(), pair=cand.as_dict())
            st["version"] += 1
            st["try"] = 1
        else:
            self._finish("val_fail", val=val, best=best, val_gate=dec.fields(),
                         pair=cand.as_dict())
            st["try"] += 1
        self._new_iteration()

    # ------------------------------------------------------------- frontier
    def _frontier(self) -> dict:
        return json.loads(self.frontier_path.read_text()) if self.frontier_path.exists() else {}

    def _best_metric(self) -> float | None:
        return self._frontier().get("_best", {}).get("metric")

    def _val_dir(self, tag: str) -> Path:
        return self.run_dir / "evaluation" / tag

    def _write_best_pair(self) -> None:
        """<run>/best/pair.json: the current best (harness, model) -- what to deploy."""
        p = self.pair
        d = self.run_dir / "best"
        d.mkdir(parents=True, exist_ok=True)
        (d / "pair.json").write_text(json.dumps({
            "candidate": self._frontier().get("_best", {}).get("candidate"),
            "harness": p.harness, "model": p.model or self.args.model}, indent=2))

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
                                             "kind": self.st.get("kind") or "baseline",
                                             "accepted": bool(improved)})
        self.frontier_path.write_text(json.dumps(fr, indent=2))
        return improved

    # ------------------------------------------------------------- driver
    def _budget_left(self) -> bool:
        return self.st["propose_count"] < self.args.iterations

    def step(self) -> bool:
        """Advance by at most one job event. Returns False when the run is finished."""
        st = self.st
        if st["stage"] == "round":
            return self._step_round()
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
        if st["stage"] == "eval_a" and not self._budget_left():
            return False
        print(f"{_now()} iter {st['iteration']} {st['stage']} {self.tag} -> submit"
              + (f" (attempt {st['attempt']})" if st["attempt"] > 1 else ""))
        jid = self._submit()
        st["job"] = {"id": jid, "stage": st["stage"], "attempt": st["attempt"]}
        self._save()
        return True

    def run(self) -> None:
        while self.step():
            pass
        best = self._frontier().get("_best", {})
        p = self.pair
        print(f"\n{_now()} Optimize complete. proposals={self.st['propose_count']} "
              f"best={best.get('candidate')} @ {best.get('metric')}  "
              f"harness={p.harness or 'base'} model={p.model or self.args.model}  "
              f"run_dir={self.run_dir}")
