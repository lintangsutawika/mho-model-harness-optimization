"""Harness search on harnessforge: evidence -> K proposals -> val gate (race) -> accept, one round at a time.

The harbor-era loop (mho/loop.py) evaluates with harbor jobs and gates with utils/gate.py. This driver runs the same
proposer (mho.optimizer.agent via scripts/optimize/propose.sh) on the harnessforge stack instead:

  evidence   the current reference harness on `evidence_tasks` screen-pool tasks (never the val set), exported in
             harbor's layout to candidates/<ref>/eval_after/ (what the proposer reads, `--harbor-export`)
  propose    K proposer jobs in parallel (MHO_PROPOSE_SIBLING k/K), each writing candidates/<tag>/harness
  gate       the K candidates on the val tasks x `trials`, TASK_MAJOR, raced against the reference's val records
             (reused, a static side) with `race_run drive --mode all --design spending --alpha alpha/K`
             (Bonferroni over the round's candidates; non-binding futility drops nulls early)
  accept     the race winner (max-t passer): `Lineage.accept` with the fresh-reference policy; the new reference's
             val records = its gate records + a top-up of the missing units (or a fresh run), written as a static
             side for the next round; frontier.json / summary.json updated so the next proposals build on it

Two proposals that both pass in one round: only the winner is accepted (their combination was never tested); the
other's idea can be re-proposed on top of the new reference.

Overlap (`overlap: true`). The proposer runs on its own endpoint, so it need not wait for the pool:
  - when round r's gate starts, round r+1's proposers start too, on the current reference (speculative: they do not
    see round r's verdicts). If round r accepts nothing, they ARE round r+1's proposals (no proposer wait at all);
  - if round r accepts a winner, they are killed and moved to candidates/_superseded/, and round r+1's proposers
    restart on the winner right after its evidence run, WHILE the winner's top-up runs (instead of after it). The
    proposer reads a provisional frontier.json (_best = winner, metric = its gate estimate) until the top-up lands.
  rounds/<r>.base in hfloop.json records which reference a round's proposals were built on; on restart, proposals
  built on a reference that is no longer current are superseded the same way.

State: <run>/hfloop.json (resumable: a finished step is skipped on restart). Layout:

    <run>/candidates/frontier.json, summary.json        proposer inputs (`_best.candidate` = current reference)
    <run>/candidates/<tag>/harness/                       a harness (candidate_0 = the start harness)
    <run>/candidates/<tag>/eval_after/                    evidence trials (harbor layout) of a reference
    <run>/candidates/<tag>/status.json                    accepted / rejected / invalid, with the gate's verdict
    <run>/ref/<tag>/{records.jsonl,request.json,status.json}   a reference's val records (static race side)
    <run>/rounds/<r>/{gate.request.json,race.json,...}   per-round gate
    <run>/evals/<run_id>/                                 every `harnessforge.evals run` output
    <run>/lineage.json

    python -m mho.hfloop run --config hfloop.json --endpoint http://ctxcap:port
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harnessforge.contracts import Order
from harnessforge.records import TrialRecord, read_jsonl, write_jsonl
from harnessforge.stats.reference import Lineage, TopUp, apply_topup

START = "candidate_0"


# ----------------------------------------------------------------------------------------------- config / state
@dataclass
class Config:
    run_dir: Path
    start_harness: Path
    dataset: str                       # task dir root (val and screen tasks both live here)
    val_tasks: list[str]               # the gate's launch order
    screen_tasks: list[str]            # evidence pool (disjoint from val_tasks)
    trials: int = 4
    proposals: int = 4
    rounds: int = 10
    alpha: float = 0.05
    fresh: str = "looks"
    fresh_looks: tuple[int, ...] = (1, 2, 4)
    evidence_tasks: int = 150
    evidence_trials: int = 1
    total_concurrency: int = 1120
    policy: str = "Qwen/Qwen3.5-4B"
    platoon: str = "platoon"
    site: str | None = None
    request_options: dict[str, Any] = field(default_factory=dict)
    propose_cmd: list[str] = field(default_factory=list)
    propose_env: dict[str, str] = field(default_factory=dict)
    propose_timeout_s: float = 5400.0
    probe_cmd: list[str] = field(default_factory=list)
    race_poll_s: float = 60.0
    seed: int = 20261123
    overlap: bool = False              # propose round r+1 while round r's gate / top-up runs (see module docstring)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())

        def tasks(v: Any) -> list[str]:
            if isinstance(v, list):
                return [str(t) for t in v]
            p = Path(v)
            if p.suffix == ".json":
                d = json.loads(p.read_text())
                return [str(t) for t in (d["tasks"] if isinstance(d, dict) else d)]
            return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

        raw["run_dir"] = Path(raw["run_dir"])
        raw["start_harness"] = Path(raw["start_harness"])
        raw["val_tasks"] = tasks(raw["val_tasks"])
        raw["screen_tasks"] = tasks(raw["screen_tasks"])
        if "fresh_looks" in raw:
            raw["fresh_looks"] = tuple(int(x) for x in raw["fresh_looks"])
        cfg = cls(**raw)
        overlap = set(cfg.val_tasks) & set(cfg.screen_tasks)
        if overlap:
            raise ValueError(f"{len(overlap)} val tasks are in the screen pool: the evidence must not see the val set")
        return cfg


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def log(msg: str) -> None:
    print(f"[hfloop {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- pure helpers
def per_candidate_concurrency(total: int, n_candidates: int) -> int:
    return max(1, total // max(1, n_candidates))


def evidence_tasks(screen: Sequence[str], n: int, seed: int, round_: int) -> list[str]:
    """A fresh random screen sample per round (seeded), so the proposer sees new tasks every round."""
    rng = random.Random(f"{seed}:{round_}")
    return rng.sample(list(screen), min(n, len(screen)))


def topup_groups(plan: TopUp) -> dict[int, list[str]]:
    """Tasks of a top-up plan grouped by how many trials each still needs (an evals request has one trial count),
    in the gate's task order."""
    out: dict[int, list[str]] = {}
    for t in plan.tasks:
        k = len(plan.missing.get(t, ()))
        if k:
            out.setdefault(k, []).append(t)
    return dict(sorted(out.items()))


def write_static_side(dest: Path, label: str, records: Sequence[TrialRecord], tasks: Sequence[str], trials: int) -> Path:
    """A finished race side for `label` from records already on file (the reused reference)."""
    dest.mkdir(parents=True, exist_ok=True)
    write_jsonl(dest / "records.jsonl", [r for r in records if r.candidate == label])
    _atomic_json(dest / "request.json", {"run_id": f"ref-{label}", "candidates": [{"id": label}],
                                         "tasks": list(tasks), "trials": trials, "order": Order.TASK_MAJOR.value})
    _atomic_json(dest / "status.json", {"done": True, "static": True, "candidate": label})
    return dest


def frontier_doc(best: str, metric: float | None, history: list[dict[str, Any]]) -> dict[str, Any]:
    return {"_best": {"candidate": best, "metric": metric}, "_anchor": {"candidate": START},
            "accepted": [h["accepted"] for h in history]}


def round_summary(round_: int, ref: str, race: Mapping[str, Any] | None, invalid: Sequence[str],
                  accepted: str | None) -> dict[str, Any]:
    """One entry of candidates/summary.json: what each proposal of the round scored at the gate."""
    states = ((race or {}).get("decisions") or {}).get("states") or {}
    cands = {}
    for c, s in states.items():
        acc = s.get("acc") or {}
        cands[c] = {"verdict": s.get("verdict"), "look": s.get("look"), "d_points": _pts(acc.get("d")),
                    "se_points": _pts(acc.get("se")), "t": acc.get("t"), "reason": s.get("reason")}
    for c in invalid:
        cands[c] = {"verdict": "invalid"}
    return {"round": round_, "reference": ref, "accepted": accepted, "candidates": cands}


def _pts(x: float | None) -> float | None:
    return None if x is None else round(100 * x, 2)


# ----------------------------------------------------------------------------------------------- the loop
Runner = Callable[..., subprocess.CompletedProcess]


class Loop:
    def __init__(self, cfg: Config, endpoint: str, *, py: str = sys.executable, runner: Runner = subprocess.run,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> None:
        self.cfg, self.endpoint, self.py, self.runner, self.popen = cfg, endpoint, py, runner, popen
        self.run = cfg.run_dir
        self.cands = self.run / "candidates"
        self.state_path = self.run / "hfloop.json"
        self.lineage_path = self.run / "lineage.json"
        self.state: dict[str, Any] = _read_json(self.state_path, {}) or {"round": 0, "reference": START,
                                                                         "history": [], "rounds": {}}
        self.pending: dict[int, dict[str, Any]] = {}     # round -> proposers started ahead (overlap)

    # -- state
    def save(self) -> None:
        _atomic_json(self.state_path, self.state)

    def harness(self, tag: str) -> Path:
        return self.cands / tag / "harness"

    def ref_side(self, tag: str) -> Path:
        return self.run / "ref" / tag

    # -- evals
    def request(self, run_id: str, cands: Mapping[str, Path], tasks: Sequence[str], trials: int) -> Path:
        out = self.run / "evals"
        req = {"run_id": run_id, "candidates": [{"id": c, "harness_dir": str(h)} for c, h in cands.items()],
               "tasks": list(tasks), "dataset": self.cfg.dataset, "trials": trials, "endpoint": self.endpoint,
               "policy": self.cfg.policy,
               "concurrency": per_candidate_concurrency(self.cfg.total_concurrency, len(cands)),
               "out_dir": str(out), "platoon": self.cfg.platoon, "site": self.cfg.site,
               "options": dict(self.cfg.request_options)}
        p = out / f"{run_id}.request.json"
        _atomic_json(p, req)
        return p

    def evals_dir(self, run_id: str) -> Path:
        return self.run / "evals" / run_id

    def evals_done(self, run_id: str) -> bool:
        return bool((_read_json(self.evals_dir(run_id) / "status.json", {}) or {}).get("done"))

    def run_evals(self, run_id: str, cands: Mapping[str, Path], tasks: Sequence[str], trials: int,
                  harbor_export: Path | None = None) -> Path:
        if self.evals_done(run_id):
            log(f"evals {run_id}: done earlier")
            return self.evals_dir(run_id)
        req = self.request(run_id, cands, tasks, trials)
        cmd = [self.py, "-m", "harnessforge.evals", "run", str(req), "--poll", "30"]
        if harbor_export is not None:
            cmd += ["--harbor-export", str(harbor_export)]
        log(f"evals {run_id}: {len(cands)} candidate(s) x {len(tasks)} tasks x {trials}")
        with open(self.run / "evals" / f"{run_id}.log", "ab") as out:
            self.runner(cmd, check=True, stdout=out, stderr=subprocess.STDOUT)
        return self.evals_dir(run_id)

    # -- steps
    def ensure_reference(self) -> Path:
        """The reference's val records as a static side (step 0: a full run of the start harness)."""
        ref = self.state["reference"]
        side = self.ref_side(ref)
        if (side / "status.json").exists():
            return side
        if ref != START:
            raise RuntimeError(f"reference {ref} has no val records at {side} (accept step incomplete)")
        h = self.harness(START)
        if not (h / "pyproject.toml").exists():
            h.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.cfg.start_harness, h, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        d = self.run_evals(f"val-{START}", {START: h}, self.cfg.val_tasks, self.cfg.trials)
        recs = read_jsonl(d / "records.jsonl")
        write_static_side(side, START, recs, self.cfg.val_tasks, self.cfg.trials)
        lin = Lineage.load(self.lineage_path) if self.lineage_path.exists() else Lineage(START)
        lin.add_records(START, str(side / "records.jsonl"))
        lin.save(self.lineage_path)
        self.write_frontier()
        return side

    def ensure_evidence(self, round_: int, ref: str | None = None) -> None:
        ref = ref or self.state["reference"]
        dest = self.cands / ref
        if any((dest / "eval_after").glob("*/result.json")):
            return
        tasks = evidence_tasks(self.cfg.screen_tasks, self.cfg.evidence_tasks, self.cfg.seed, round_)
        rid = f"evidence-r{round_}-{ref}"
        # candidate id "eval_after": harbor export writes <dest>/eval_after/<task>__r<k>/ (the proposer's input)
        self.run_evals(rid, {"eval_after": self.harness(ref)}, tasks, self.cfg.evidence_trials, harbor_export=dest)

    def tags(self, round_: int) -> list[str]:
        return [f"candidate_{round_}_{k}" for k in range(1, self.cfg.proposals + 1)]

    def propose(self, round_: int) -> tuple[list[str], list[str]]:
        """K proposer jobs in parallel on the current reference; returns (valid, invalid) tags."""
        return self.finish_proposers(round_, self.start_proposers(round_, self.state["reference"]))

    def start_proposers(self, round_: int, base: str) -> dict[str, Any]:
        """Launch round `round_`'s K proposers on reference `base` (frontier.json must already name it); a tag whose
        proposal finished earlier is not relaunched. Proposals built on another base are superseded first."""
        rs = self.state["rounds"].setdefault(str(round_), {})
        if rs.get("base") not in (None, base):
            self.supersede(round_, f"built on {rs['base']}, reference is now {base}")
        rs["base"] = base
        rs.setdefault("propose_started_at", time.time())
        self.save()
        K = self.cfg.proposals
        procs = {}
        for k, tag in enumerate(self.tags(round_), 1):
            if (self.harness(tag) / "pyproject.toml").exists() and (self.cands / tag / "proposed.json").exists():
                continue
            env = {**os.environ, **self.cfg.propose_env, "MHO_RUN_DIR": str(self.run), "MHO_TAG": tag,
                   "MHO_PROPOSE_KIND": "harness", "MHO_PROPOSE_SIBLING": f"{k}/{K}",
                   "MICRO_SCAFFOLD_BASE": str(self.harness(START))}
            (self.cands / tag).mkdir(parents=True, exist_ok=True)
            out = open(self.cands / tag / "propose.log", "ab")
            procs[tag] = (self.popen(self.cfg.propose_cmd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True), out)
            log(f"propose {tag} started (on {base})")
        return {"round": round_, "base": base, "deadline": time.monotonic() + self.cfg.propose_timeout_s,
                "procs": procs}

    def finish_proposers(self, round_: int, pend: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        """Wait for proposers started by `start_proposers`; returns (valid, invalid) tags."""
        for tag, (p, out) in pend["procs"].items():
            try:
                rc = p.wait(timeout=max(1.0, pend["deadline"] - time.monotonic()))
            except subprocess.TimeoutExpired:
                _kill(p)
                rc = "timeout"
            out.close()
            _atomic_json(self.cands / tag / "proposed.json", {"rc": rc, "at": time.time()})
            log(f"propose {tag}: rc {rc}")
        rs = self.state["rounds"].setdefault(str(round_), {})
        rs["propose_finished_at"] = time.time()
        valid, invalid = [], []
        parent = self.harness(pend["base"])
        for tag in self.tags(round_):
            h = self.harness(tag)
            prior = h / "eval_prior"
            if prior.is_dir():            # the proposer's staged evidence: keep it out of the installed harness
                os.replace(prior, self.cands / tag / "eval_prior")
            ok = (h / "pyproject.toml").exists() and _differs(h, parent)
            (valid if ok else invalid).append(tag)
            if not ok:
                _atomic_json(self.cands / tag / "status.json", {"status": "invalid", "round": round_})
        return valid, invalid

    def supersede(self, round_: int, why: str) -> None:
        """Drop round `round_`'s proposals (speculative ones built on a reference that changed): kill any still
        running, move their dirs to candidates/_superseded/r<round>-<ts>/ (kept, never deleted), reset the round."""
        pend = self.pending.pop(round_, None)
        for p, out in (pend or {}).get("procs", {}).values():
            _kill(p)
            out.close()
        dest = self.cands / "_superseded" / f"r{round_}-{int(time.time())}"
        moved = []
        for tag in self.tags(round_):
            if (self.cands / tag).exists():
                dest.mkdir(parents=True, exist_ok=True)
                os.replace(self.cands / tag, dest / tag)
                moved.append(tag)
        rs = self.state["rounds"].setdefault(str(round_), {})
        rs.setdefault("superseded", []).append({"base": rs.get("base"), "why": why, "at": time.time(),
                                                "dir": str(dest) if moved else None, "tags": moved})
        for k in ("base", "proposals", "invalid", "propose_started_at", "propose_finished_at"):
            rs.pop(k, None)
        self.save()
        log(f"round {round_} proposals superseded ({why}): {moved or 'none started'}")

    def start_ahead(self, round_: int, base: str) -> None:
        """Overlap: start round `round_`'s proposers now, on `base`, unless they already exist on it."""
        if not self.cfg.overlap or round_ > self.cfg.rounds or round_ in self.pending:
            return
        rs = self.state["rounds"].get(str(round_), {})
        if "proposals" in rs and rs.get("base") == base:
            return
        log(f"overlap: round {round_} proposers start now on {base}")
        self.pending[round_] = self.start_proposers(round_, base)

    def write_summary(self, round_: int, race: Mapping[str, Any] | None, accepted: str | None) -> None:
        rs = self.state["rounds"].get(str(round_), {})
        summ = _read_json(self.cands / "summary.json", []) or []
        summ = [s for s in summ if s.get("round") != round_] + [round_summary(
            round_, rs.get("reference_before", ""), race, rs.get("invalid", []), accepted)]
        _atomic_json(self.cands / "summary.json", summ)

    def gate(self, round_: int, valid: Sequence[str]) -> dict[str, Any]:
        rdir = self.run / "rounds" / str(round_)
        race_path = rdir / "race.json"
        race = _read_json(race_path)
        if race and (race.get("decisions") or {}).get("done") and self.evals_done(f"gate-r{round_}"):
            return race
        ref = self.state["reference"]
        rid = f"gate-r{round_}"
        req = self.request(rid, {t: self.harness(t) for t in valid}, self.cfg.val_tasks, self.cfg.trials)
        rdir.mkdir(parents=True, exist_ok=True)
        ev_log = open(rdir / "gate.evals.log", "ab")
        ev = self.popen([self.py, "-m", "harnessforge.evals", "run", str(req), "--poll", "30"],
                        stdout=ev_log, stderr=subprocess.STDOUT)
        gdir = self.evals_dir(rid)
        while not (gdir / "request.json").exists():
            if ev.poll() is not None:
                raise RuntimeError(f"gate evals exited rc {ev.returncode} before writing {gdir}/request.json")
            time.sleep(5)
        alpha = self.cfg.alpha / max(1, len(valid))
        cmd = [self.py, "-m", "harnessforge.race_run", "drive", "--reference", ref,
               "--side", f"{ref}={self.ref_side(ref)}", *[x for t in valid for x in ("--side", f"{t}={gdir}")],
               "--mode", "all", "--design", "spending", "--alpha", f"{alpha:.6g}",
               "--poll", str(self.cfg.race_poll_s), "--out", str(race_path)]
        log(f"gate r{round_}: {len(valid)} candidate(s) vs {ref}, alpha {alpha:.4g} each")
        with open(rdir / "race.log", "ab") as out:
            self.runner(cmd, check=True, stdout=out, stderr=subprocess.STDOUT)
        ev.wait()
        ev_log.close()
        return json.loads(race_path.read_text())

    def accept(self, round_: int, race: Mapping[str, Any]) -> str | None:
        st = race["decisions"]
        winner = st.get("winner")
        for c, s in st["states"].items():
            status = "accepted" if c == winner else "rejected"
            _atomic_json(self.cands / c / "status.json", {"status": status, "round": round_, "gate": s})
        if not winner:
            return None
        rid = f"gate-r{round_}"
        gate_recs_path = self.evals_dir(rid) / "records.jsonl"
        gate_recs = read_jsonl(gate_recs_path)
        lin = Lineage.load(self.lineage_path)
        new, plan = lin.accept(winner, st["states"][winner], gate_recs, self.cfg.val_tasks, self.cfg.trials,
                               fresh=self.cfg.fresh, fresh_looks=self.cfg.fresh_looks,
                               n_looks=len(race.get("looks") or []) or None, record_files=[str(gate_recs_path)])
        log(f"accept {winner} (look {new.history[-1]['look']}, t {new.history[-1]['t']}): "
            f"{'fresh' if plan.fresh else 'top-up'} {plan.n_units} units")
        nxt = round_ + 1
        if self.cfg.overlap and nxt <= self.cfg.rounds:
            # the speculative round-r+1 proposals were built on the old reference: redo them on the winner, now,
            # so they run during the top-up (evidence first: it is what the proposer reads)
            # (start_proposers supersedes the ones on file; on a restart nothing of them is running)
            if nxt in self.pending and self.pending[nxt]["base"] != winner:
                self.supersede(nxt, f"round {round_} accepted {winner}")
            self.ensure_evidence(nxt, winner)
            est = (st["states"][winner].get("acc") or {}).get("B")
            _atomic_json(self.cands / "frontier.json",
                         {**frontier_doc(winner, est, [*self.state["history"], {"accepted": winner}]),
                          "_provisional": "gate estimate; the full val records land after the top-up"})
            self.write_summary(round_, race, winner)
            self.start_ahead(nxt, winner)
        top: list[TrialRecord] = []
        for k, tasks in topup_groups(plan).items():
            d = self.run_evals(f"topup-r{round_}-{winner}-x{k}", {winner: self.harness(winner)}, tasks, k)
            top += [r for r in read_jsonl(d / "records.jsonl") if r.candidate == winner]
        merged = apply_topup(gate_recs, top, plan)
        side = write_static_side(self.ref_side(winner), winner, merged, self.cfg.val_tasks, self.cfg.trials)
        new.records[winner] = [str(side / "records.jsonl")]
        new.save(self.lineage_path)
        self.state["reference"] = winner
        self.state["history"].append(new.history[-1])
        self.save()
        self.write_frontier()
        self.probe(winner)
        return winner

    def probe(self, tag: str) -> None:
        if not self.cfg.probe_cmd:
            return
        p = self.runner([*self.cfg.probe_cmd, str(self.harness(tag))], capture_output=True, text=True)
        (self.cands / tag / "probe.json").write_text(p.stdout or json.dumps({"error": p.stderr[-2000:]}))
        log(f"probe {tag}: {p.stdout.strip()[:300]}")

    def write_frontier(self) -> None:
        ref = self.state["reference"]
        recs = read_jsonl(self.ref_side(ref) / "records.jsonl") if (self.ref_side(ref) / "records.jsonl").exists() else []
        scored = [r.reward or 0.0 for r in recs if r.finished and r.scored]
        metric = sum(scored) / len(scored) if scored else None
        _atomic_json(self.cands / "frontier.json", frontier_doc(ref, metric, self.state["history"]))

    # -- driver
    def round(self, r: int) -> None:
        rs = self.state["rounds"].setdefault(str(r), {})
        self.ensure_reference()
        self.ensure_evidence(r)
        ref = self.state["reference"]
        if "proposals" not in rs or rs.get("base", ref) != ref:
            pend = self.pending.pop(r, None)
            if pend is None or pend["base"] != ref:
                pend = self.start_proposers(r, ref)
            else:
                log(f"round {r}: using the proposals started ahead on {ref}")
            valid, invalid = self.finish_proposers(r, pend)
            rs.update(proposals=valid, invalid=invalid)
            self.save()
        if "accepted" not in rs:
            self.start_ahead(r + 1, ref)     # overlap: next round's proposers run during this gate
        race = self.gate(r, rs["proposals"]) if rs["proposals"] else None
        if "accepted" not in rs:
            rs["accepted"] = self.accept(r, race) if race else None
            self.save()
        self.write_summary(r, race, rs["accepted"])
        log(f"round {r} done: accepted {rs['accepted']}")

    def drive(self) -> None:
        self.run.mkdir(parents=True, exist_ok=True)
        while self.state["round"] < self.cfg.rounds:
            r = self.state["round"] + 1
            self.state["rounds"].setdefault(str(r), {}).setdefault("reference_before", self.state["reference"])
            self.round(r)
            self.state["round"] = r
            self.save()


def _kill(p: Any) -> None:
    """Stop a proposer and its children (it runs in its own session: uv -> python -> tools)."""
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except (AttributeError, TypeError, OSError):
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        p.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass


def _differs(a: Path, b: Path) -> bool:
    p = subprocess.run(["diff", "-rq", "-x", "__pycache__", "-x", ".git", "-x", "eval_prior", str(a), str(b)],
                       capture_output=True, text=True)
    return p.returncode == 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--endpoint", required=True, help="model endpoint for every evaluation (e.g. the ctxcap proxy)")
    a = ap.parse_args(argv)
    Loop(Config.load(a.config), a.endpoint).drive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
