"""Drive a sequential race on a running evaluation: the core loop of the harness search.

    run = backend.start(EvalRequest(..., candidates=[A, B1, B2], tasks=shuffled, trials=4))
    race = Race("A", ["B1", "B2"], LookSchedule(run.order, request.tasks, request.trials))
    summary = RaceDriver(run, race, poll_s=60, summary_path=out / "race.json").drive()

Every `poll_s` the driver takes a records snapshot (`run.records()`: finished and in-flight trials), feeds it to the
`Race` state machine, stops each newly decided candidate (`run.stop(c)`, its remaining trials are waste) and, once
no candidate is alive, the reference. It then keeps polling until every side has settled (no trial in flight) or
`drain_timeout_s` passed, so the summary has the real stop latencies and final trial counts.

What it logs and writes (`summary_path`, rewritten after every poll):

  timeline     one entry per (candidate, look) the first time both sides' units of that look are launched and
               finished: tasks, paired D +- SE, t, the efficacy and futility bounds, the design's verdict there,
               wall clock. Looks are evaluated in order up to the candidate's decision (the race's own rule), so
               the deciding entry's verdict is the race's verdict.
  decisions    the race's final `CandidateState`s, winner and ranking
  sides        per candidate: units planned / launched / finished / scored / env_failed / cancelled ("stopped"
               or "cancelled") / never launched, and whether it settled
  stops        when each side was stopped, at which look, when it settled (no trial in flight) and the latency
  polls        a compact progress log (time, launched, finished, in flight per side)

`replay(records, ...)` re-runs a race offline on a finished evaluation's records, as the live driver would have seen
them (snapshots at every finish time, by `started_at`/`finished_at`), e.g. with the roles reversed.

`FileRun` is an `EvalRun` over the file protocol of `python -m harnessforge.evals run` (one such job per side, each
on its own node): it reads each job's `records.jsonl`, stops a side by creating `<run_dir>/<candidate>.stop`, and is
done when every job wrote `status.json` with `done: true`. So one race can span several eval nodes.

CLI (inside a job; see docs/race.md):

    python -m harnessforge.race_run drive --side A=RUN_DIR_A --side B=RUN_DIR_B --reference A --out race.json
    python -m harnessforge.race_run drive --request REQUEST.json --reference A --out race.json   # in-process
    python -m harnessforge.race_run replay records.jsonl ... --tasks request.json --reference B --out rev.json

val1000 gate (task-major, LD-OBF spending on observed information; docs/stats.md section 8):

    python -m harnessforge.race_run drive --side REF=... --side B=... --reference REF --mode all --design spending \
        [--sampling sampling.json]                       # frontier-weighted order from `python -m harnessforge.stats.sampling`
    python -m harnessforge.race_run accept --lineage lineage.json --race race.json --candidate B \
        --records B.records.jsonl --tasks request.json --fresh marginal --out-topup topup.json
    python -m harnessforge.race_run topup --records REF.records.jsonl --reference REF --tasks request.json --out topup.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from harnessforge.contracts import EvalRun, Order
from harnessforge.records import TrialRecord, read_jsonl
from harnessforge.stats.boundaries import Design, Verdict
from harnessforge.stats.looks import LookSchedule, UnitIndex, collect
from harnessforge.stats.paired import paired
from harnessforge.stats.information import VAL200, Information
from harnessforge.stats.race import Race, RaceStatus, ReferenceMode, recommended_c
from harnessforge.stats.reference import FRESH_POLICIES, Lineage, topup_plan
from harnessforge.stats.sampling import TaskSampling
from harnessforge.stats.spending import SpendingDesign

__all__ = ["FileRun", "RaceDriver", "replay", "side_counts", "snapshot_at"]

CANCELLED = frozenset({"stopped", "cancelled"})


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None or not math.isfinite(x) else round(x, nd)


def side_counts(records: Iterable[TrialRecord], candidate: str, planned: int) -> dict[str, int]:
    """Trial counts of one side: launched (has a record), finished, scored, env_failed (not cancelled), cancelled,
    in flight, never launched."""
    rs = [r for r in records if r.candidate == candidate]
    fin = [r for r in rs if r.finished]
    cancelled = sum(1 for r in fin if r.env_failed and r.error in CANCELLED)
    return {"planned": planned, "launched": len(rs), "finished": len(fin),
            "scored": sum(1 for r in fin if r.scored),
            "env_failed": sum(1 for r in fin if r.env_failed) - cancelled, "cancelled": cancelled,
            "in_flight": len(rs) - len(fin), "never_launched": planned - len(rs),
            "successes": sum(1 for r in fin if r.scored and (r.reward or 0) > 0)}


class RaceDriver:
    """Runs the race loop on `run` (see the module docstring)."""

    def __init__(self, run: EvalRun, race: Race, *, poll_s: float = 60.0, drain_timeout_s: float = 900.0,
                 drain_poll_s: float = 10.0,
                 summary_path: str | Path | None = None, records_path: str | Path | None = None,
                 log: Callable[[str], None] | None = None, clock: Callable[[], float] = time.time,
                 meta: Mapping[str, Any] | None = None) -> None:
        self.run = run
        self.race = race
        self.poll_s = poll_s
        self.drain_timeout_s = drain_timeout_s
        self.drain_poll_s = drain_poll_s   # poll faster once something was stopped, to time the stop latency
        self.summary_path = Path(summary_path) if summary_path else None
        self.records_path = Path(records_path) if records_path else None
        self.log = log or (lambda s: print(s, flush=True))
        self.clock = clock
        self.meta = dict(meta or {})
        self.sides = (race.reference, *race.candidates)
        self.planned = len(race.schedule.tasks) * race.schedule.trials
        self.t0: float | None = None
        self.timeline: list[dict[str, Any]] = []
        self._logged: dict[str, int] = {c: 0 for c in race.candidates}   # last look logged per candidate
        self.stops: dict[str, dict[str, Any]] = {}
        self.polls: list[dict[str, Any]] = []
        self.decided_at: dict[str, float] = {}
        self.status: RaceStatus | None = None
        self.records: list[TrialRecord] = []

    # ------------------------------------------------------------------ loop
    def drive(self) -> dict[str, Any]:
        self.t0 = self.clock()
        self.log(f"race: reference {self.race.reference}, candidates {list(self.race.candidates)}, "
                 f"{len(self.race.schedule.tasks)} tasks x {self.race.schedule.trials} trials, looks at "
                 f"{[len(lk.tasks) for lk in self.race.schedule.looks]} tasks; design {self.race.design.as_dict()}")
        drain_started: float | None = None
        while True:
            self.step()
            if self.status.reference_can_stop:
                if drain_started is None:
                    drain_started = self.clock()
                if self._settled() or self.run.done():
                    break
                if self.clock() - drain_started > self.drain_timeout_s:
                    self.log(f"drain timeout after {self.drain_timeout_s:.0f} s; sides still in flight: "
                             f"{[c for c in self.sides if not self._side_settled(c)]}")
                    break
            elif self.run.done():
                self.log("evaluation ended before the race was decided")
                break
            self.run.wait(timeout=min(self.poll_s, self.drain_poll_s) if self.stops else self.poll_s)
        self.step(final=True)
        summary = self.summary()
        self._write(summary)
        return summary

    def step(self, final: bool = False) -> RaceStatus:
        now = self.clock()
        records = self.run.records()
        self.records = records
        idx = UnitIndex(records)
        before = {c: s.verdict for c, s in (self.status.states.items() if self.status else ())}
        st = self.race.update(idx)
        self.status = st
        self._log_looks(idx, st, before, now)
        for c in st.newly_decided:
            self.decided_at[c] = now
            self._stop(c, now, f"decided {st.states[c].verdict.value} at look {st.states[c].look}")
        if st.reference_can_stop and self.race.reference not in self.stops:
            self._stop(self.race.reference, now, "no candidate alive")
        for c, s in self.stops.items():
            if s.get("settled_at") is None and self._side_settled(c):
                s["settled_at"] = now
                s["latency_s"] = _r(now - s["at"], 1)
                self.log(f"[{self._el(now)}] {c} settled {s['latency_s']} s after its stop")
        poll = {"t": _r(now - self.t0, 1)}
        for c in self.sides:
            k = side_counts(records, c, self.planned)
            poll[c] = [k["launched"], k["finished"], k["in_flight"]]
        self.polls.append(poll)
        nxt = {c: (st.states[c].look or 0) + 1 for c in st.alive}
        self.log(f"[{self._el(now)}] " + "  ".join(f"{c}: {poll[c][1]}/{poll[c][0]} fin/launched "
                                                     f"({poll[c][2]} in flight)" for c in self.sides)
                 + (f"  next looks {nxt}" if nxt else ""))
        if self.records_path is not None and not final:
            from harnessforge.records import write_jsonl
            write_jsonl(self.records_path, records)
        if not final:
            self._write(self.summary())
        return st

    # ------------------------------------------------------------------ helpers
    def _el(self, now: float) -> str:
        s = int(now - (self.t0 or now))
        return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    def _stop(self, c: str, now: float, why: str) -> None:
        self.run.stop(c)
        self.stops[c] = {"at": now, "elapsed_s": _r(now - self.t0, 1), "why": why,
                         "in_flight_at_stop": sum(1 for r in self.records if r.candidate == c and not r.finished),
                         "never_launched_at_stop": self.planned - sum(1 for r in self.records if r.candidate == c),
                         "settled_at": None, "latency_s": None}
        self.log(f"[{self._el(now)}] stop {c}: {why} ({self.stops[c]['in_flight_at_stop']} in flight, "
                 f"{self.stops[c]['never_launched_at_stop']} never launched)")

    def _side_settled(self, c: str) -> bool:
        return not any(r.candidate == c and not r.finished for r in self.records)

    def _settled(self) -> bool:
        return all(self._side_settled(c) for c in self.sides)

    def _log_looks(self, idx: UnitIndex, st: RaceStatus, before: Mapping[str, Verdict], now: float) -> None:
        """A timeline entry for every look a candidate newly had evaluated by the race (in order, up to its
        decision), recomputed from the same units so intermediate looks of a multi-look jump are not lost."""
        for c in self.race.candidates:
            if before.get(c, Verdict.CONTINUE) is not Verdict.CONTINUE:
                continue
            upto = st.states[c].look or 0
            for j in range(self._logged[c] + 1, upto + 1):
                entry = look_entry(idx, self.race, c, j)
                entry["elapsed_s"] = _r(now - self.t0, 1)
                entry["wall"] = time.strftime("%H:%M:%S", time.localtime(now))
                self.timeline.append(entry)
                self._logged[c] = j
                self.log(f"[{self._el(now)}] look {j} {c} vs {self.race.reference}: {entry['tasks']} tasks "
                         f"({entry['paired_tasks']} paired), D={_fmt(entry['d'])}+-{_fmt(entry['se'], sign=False)} "
                         f"t={_fmt(entry['t'], 2)}; pass if t > {_fmt(entry['efficacy'], 2)}, stop if t <= "
                         f"{_fmt(entry['futility'], 2)} -> {entry['verdict']}"
                         + (" (skipped: infra-suspect)" if entry.get("skipped") else ""))

    def summary(self) -> dict[str, Any]:
        st = self.status or self.race.status()
        now = self.clock()
        sched = self.race.schedule
        return {
            **self.meta,
            "reference": self.race.reference, "candidates": list(self.race.candidates),
            "reference_mode": self.race.reference_mode.value, "order": sched.order.value,
            "tasks": len(sched.tasks), "trials": sched.trials,
            "looks": [{"look": lk.number, "tasks": len(lk.tasks), "fraction": _r(lk.fraction)} for lk in sched.looks],
            "design": self.race.design.as_dict(),
            "decisions": st.as_dict(),
            "decided_at_s": {c: _r(t - self.t0, 1) for c, t in self.decided_at.items()},
            "timeline": self.timeline,
            "sides": {c: side_counts(self.records, c, self.planned) for c in self.sides},
            "stops": {c: {k: v for k, v in s.items() if k not in ("at", "settled_at")} |
                      {"settled_elapsed_s": _r(s["settled_at"] - self.t0, 1) if s.get("settled_at") else None}
                      for c, s in self.stops.items()},
            "started_at": self.t0, "elapsed_s": _r(now - self.t0, 1) if self.t0 else None,
            "polls": self.polls,
        }

    def _write(self, summary: Mapping[str, Any]) -> None:
        if self.summary_path is None:
            return
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=1))
        tmp.replace(self.summary_path)


def _fmt(x: float | None, nd: int = 4, sign: bool = True) -> str:
    return "-" if x is None else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def look_entry(idx: UnitIndex, race: Race, c: str, j: int) -> dict[str, Any]:
    """Candidate `c`'s look j against the race's reference: the units, the paired statistic and the bounds."""
    sched = race.schedule
    look = sched.look(j)
    b = collect(idx, c, look)
    ref_look = sched.full(look) if race.reference_mode is ReferenceMode.ALL_TRIALS else look
    a = collect(idx, race.reference, ref_look)
    acc = (paired(a.stats, b.stats, weights=sched.weights(look, race.reference_mode is ReferenceMode.ALL_TRIALS),
                  strata=sched.strata(look))
           if a.ready and b.ready else None)
    t = acc.t if acc else None
    env = max(a.env_fail_frac, b.env_fail_frac)
    lb = race.look_bounds(c, j)
    skipped = race.max_env_fail is not None and env > race.max_env_fail and not lb.final
    verdict = Verdict.CONTINUE if skipped else lb.verdict(t)
    return {"candidate": c, "look": j, "tasks": len(look.tasks), "trials": look.trials, "fraction": _r(look.fraction),
            "info_fraction": _r(lb.fraction), "units": look.size, "paired_tasks": acc.n if acc else 0,
            "acc_ref": _r(acc.mean_a) if acc else None, "acc_cand": _r(acc.mean_b) if acc else None,
            "d": _r(acc.d) if acc else None, "se": _r(acc.se) if acc else None, "t": _r(t, 3),
            "efficacy": _r(lb.efficacy, 3), "futility": _r(lb.futility, 3),
            "verdict": verdict.value, "env_fail_frac": _r(env), "skipped": skipped}


# ---------------------------------------------------------------------------------------------------- offline replay
def snapshot_at(records: Sequence[TrialRecord], t: float, end: float) -> list[TrialRecord]:
    """The records as a poll at time `t` would have seen them: launched ones only, unfinished ones in flight.
    Trials cancelled by a stop (error "stopped"/"cancelled") stay in flight forever: they never completed, so a look
    that needs them was never ready (counting them as finished env-failures would make the looks after a stop
    "ready" on truncated data). Other records with no `finished_at` count as finished at `end`."""
    out = []
    for r in records:
        started = r.started_at if r.started_at is not None else -math.inf
        if started > t:
            continue
        if r.env_failed and r.error in CANCELLED:
            fin = math.inf
        else:
            fin = r.finished_at if r.finished_at is not None else (end if r.finished else math.inf)
        out.append(r if fin <= t else replace(r, finished=False, reward=None, env_failed=False, error=None))
    return out


def replay(records: Sequence[TrialRecord], race: Race) -> dict[str, Any]:
    """Run `race` offline on a finished evaluation's records, with a snapshot at every finish time, as a live
    driver polling continuously would have. Returns the timeline (with each look's finish time relative to the
    first launch), the decisions and the decision times. Looks that needed trials a stop cancelled are never ready,
    so a race replayed on a stopped evaluation can end undecided (`truncated_at_look`: the last look it took)."""
    recs = [r for r in records if r.candidate in (race.reference, *race.candidates)]
    if not recs:
        raise ValueError("no records of the race's sides")
    starts = [r.started_at for r in recs if r.started_at is not None]
    fins = [r.finished_at for r in recs if r.finished_at is not None]
    t0 = min(starts) if starts else 0.0
    end = max(fins + starts) if fins or starts else 0.0
    times = sorted({*fins, end})
    timeline: list[dict[str, Any]] = []
    logged = {c: 0 for c in race.candidates}
    decided: dict[str, float] = {}
    st = race.status()
    for t in times:
        idx = UnitIndex(snapshot_at(recs, t, end))
        before = {c: s.verdict for c, s in st.states.items()}
        st = race.update(idx)
        for c in race.candidates:
            if before[c] is not Verdict.CONTINUE:
                continue
            for j in range(logged[c] + 1, (st.states[c].look or 0) + 1):
                e = look_entry(idx, race, c, j)
                e["elapsed_s"] = _r(t - t0, 1)
                timeline.append(e)
                logged[c] = j
        for c in st.newly_decided:
            decided[c] = t - t0
        if st.done:
            break
    return {"reference": race.reference, "candidates": list(race.candidates),
            "design": race.design.as_dict(), "decisions": st.as_dict(),
            "decided_at_s": {c: _r(v, 1) for c, v in decided.items()}, "timeline": timeline,
            "undecided": [c for c in race.candidates if st.states[c].alive]}


# ---------------------------------------------------------------------------------------------------- FileRun
class FileRun:
    """EvalRun over `python -m harnessforge.evals run` jobs (one per side, possibly on different nodes). The launch
    order is the sides' request.json "order" (written by the evals CLI; default task_major)."""

    def __init__(self, sides: Mapping[str, str | Path], *, poll_s: float = 5.0, order: Order | str | None = None) -> None:
        self.sides = {c: Path(d) for c, d in sides.items()}
        self.poll_s = poll_s
        if order is None:
            orders = set()
            for d in self.sides.values():
                try:
                    orders.add(json.loads((d / "request.json").read_text()).get("order") or Order.TASK_MAJOR.value)
                except (OSError, ValueError):
                    orders.add(Order.TASK_MAJOR.value)
            if len(orders) > 1:
                raise ValueError(f"the sides were launched in different orders: {sorted(orders)}")
            order = orders.pop() if orders else Order.TASK_MAJOR
        self.order = Order(order)

    def records(self) -> list[TrialRecord]:
        out: list[TrialRecord] = []
        for c, d in self.sides.items():
            try:
                out.extend(r for r in read_jsonl(d / "records.jsonl") if r.candidate == c)
            except (OSError, ValueError, TypeError):  # being replaced: the next poll reads it
                pass
        return out

    def stop(self, candidate: str) -> None:
        (self.sides[candidate] / f"{candidate}.stop").touch()

    def stop_all(self) -> None:
        for c in self.sides:
            self.stop(c)

    def side_done(self, candidate: str) -> bool:
        try:
            return bool(json.loads((self.sides[candidate] / "status.json").read_text()).get("done"))
        except (OSError, ValueError):
            return False

    def done(self) -> bool:
        return all(self.side_done(c) for c in self.sides)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.done():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(self.poll_s if deadline is None else max(0.0, min(self.poll_s, deadline - time.monotonic())))
        return True


# ---------------------------------------------------------------------------------------------------- CLI
def _schedule_from(request_json: Path, a: argparse.Namespace, order: Order | None = None) -> LookSchedule:
    raw = json.loads(request_json.read_text())
    order = Order(order or getattr(a, "order", None) or raw.get("order") or Order.TASK_MAJOR)
    return _schedule(order, tuple(raw["tasks"]), int(raw["trials"]), a, raw)


def _schedule(order: Order, tasks: Sequence[str], trials: int, a: argparse.Namespace,
              raw: Mapping[str, Any] | None = None) -> LookSchedule:
    """The race's schedule: --trial-looks / --plan, else the request's own (options trial_looks / plan)."""
    checkpoints = tuple(float(x) for x in a.checkpoints.split(","))
    raw = raw or {}
    raw = {**raw, **raw.get("options", {})}
    if order is not Order.ATTEMPT_MAJOR:
        sampling = None
        if getattr(a, "sampling", None):
            sampling = TaskSampling.from_dict(json.loads(Path(a.sampling).read_text()))
        elif isinstance(raw.get("sampling"), dict):
            sampling = TaskSampling.from_dict(raw["sampling"])
        # task-major with planning information: spending on observed information (the val1000 gate)
        info = (Information(a.tau2, a.sigma2) if (sampling is not None or a.design == "spending") and a.sigma2 > 0
                else None)
        return LookSchedule(order, tuple(tasks), trials, checkpoints, information=info, sampling=sampling)
    looks = (tuple(int(x) for x in a.trial_looks.split(",")) if a.trial_looks
             else tuple(raw["trial_looks"]) if raw.get("trial_looks") and not a.plan else None)
    plan = (parse_plan(a.plan) if a.plan else tuple(tuple(tuple(seg) for seg in lk) for lk in raw["plan"])
            if raw.get("plan") and not a.trial_looks else None)
    info = Information(a.tau2, a.sigma2) if (looks or plan) and a.sigma2 > 0 else None
    return LookSchedule(order, tuple(tasks), trials, checkpoints, looks, info, plan)


def parse_plan(text: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    """'50x2;50x2,100x4;50x2,100x4,150x8;50x2,100x4,150x8,200x24': looks separated by ';', each a list of
    END x TRIALS segments over the ordered tasks (see LookSchedule.plan)."""
    return tuple(tuple(tuple(int(v) for v in seg.split("x")) for seg in look.split(","))  # type: ignore[misc]
                 for look in text.split(";") if look.strip())


def _design(schedule: LookSchedule, a: argparse.Namespace) -> Any:
    """None: the race picks its default (`race.default_design`) for the schedule and mode."""
    if a.design == "obf" or a.c is not None:
        return Design.for_schedule(schedule, c=a.c if a.c is not None else recommended_c(schedule, a.mode),
                                   first_efficacy=1 if schedule.planned else 2)
    if a.design == "spending":
        return SpendingDesign.for_schedule(schedule, alpha=a.alpha, spending=a.spending, observed=not a.planned,
                                           all_trials_reference=ReferenceMode(a.mode) is ReferenceMode.ALL_TRIALS)
    return None


def cmd_drive(a: argparse.Namespace) -> int:
    mode = ReferenceMode(a.mode)
    if a.request:
        from harnessforge.evals.__main__ import load_request
        from harnessforge.evals.backend import PlatoonBackend
        request, raw = load_request(a.request)
        backend = PlatoonBackend(platoon=raw.get("platoon", "platoon"), site=raw.get("site"))
        run: Any = backend.start(request)
        schedule = _schedule(Order(run.order), request.tasks, request.trials, a, raw)
        ids = [c.id for c in request.candidates]
    else:
        sides = dict(s.split("=", 1) for s in a.side)
        run = FileRun(sides)
        schedules = {c: _schedule_from(Path(d) / "request.json", a, run.order) for c, d in sides.items()}
        schedule = next(iter(schedules.values()))
        if any(s != schedule for s in schedules.values()):
            raise SystemExit("the sides' requests differ in tasks or trials")
        ids = list(sides)
    cands = a.candidates.split(",") if a.candidates else [c for c in ids if c != a.reference]
    race = Race(a.reference, cands, schedule, _design(schedule, a), reference_mode=mode,
                max_env_fail=a.max_env_fail)
    out = Path(a.out)
    driver = RaceDriver(run, race, poll_s=a.poll, drain_timeout_s=a.drain_timeout,
                        summary_path=out, records_path=out.with_name(out.stem + ".records.jsonl"))
    summary = driver.drive()
    if a.reverse:
        rev = Race(cands[0], [a.reference], schedule, _design(schedule, a), reference_mode=mode,
                   max_env_fail=a.max_env_fail)
        summary["reverse"] = replay(driver.records, rev)
        driver._write(summary)
    st = summary["decisions"]
    print(json.dumps({"winner": st["winner"], "states": {c: s["reason"] for c, s in st["states"].items()},
                      "stops": summary["stops"]}, indent=1), flush=True)
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    mode = ReferenceMode(a.mode)
    records = [r for p in a.records for r in read_jsonl(p)]
    schedule = _schedule_from(Path(a.tasks), a)
    cands = a.candidates.split(",") if a.candidates else sorted({r.candidate for r in records} - {a.reference})
    race = Race(a.reference, cands, schedule, _design(schedule, a), reference_mode=mode,
                max_env_fail=a.max_env_fail)
    result = replay(records, race)
    text = json.dumps(result, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    for e in result["timeline"]:
        print(f"look {e['look']} {e['candidate']}: {e['tasks']} tasks D={_fmt(e['d'])}+-{_fmt(e['se'], sign=False)} "
              f"t={_fmt(e['t'], 2)} eff {_fmt(e['efficacy'], 2)} fut {_fmt(e['futility'], 2)} -> {e['verdict']} "
              f"@{e['elapsed_s']} s")
    print(json.dumps({c: s["reason"] for c, s in result["decisions"]["states"].items()}, indent=1))
    return 0


def _request(path: str) -> tuple[list[str], int, dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    return list(raw["tasks"]), int(raw["trials"]), raw


def cmd_topup(a: argparse.Namespace) -> int:
    tasks, trials, _ = _request(a.tasks)
    records = [r for p in a.records for r in read_jsonl(p)]
    plan = topup_plan(records, a.reference, tasks, trials, fresh=a.fresh)
    Path(a.out).write_text(json.dumps(plan.as_dict(), indent=1))
    print(json.dumps({k: v for k, v in plan.as_dict().items() if k != "missing"}, indent=1))
    return 0


def cmd_accept(a: argparse.Namespace) -> int:
    tasks, trials, _ = _request(a.tasks)
    lp = Path(a.lineage)
    if lp.exists():
        lineage = Lineage.load(lp)
    elif a.reference:
        lineage = Lineage(a.reference, step0_labels=a.step0_labels)
    else:
        raise SystemExit("no lineage file: pass --reference (and --step0-labels) to create it")
    race = json.loads(Path(a.race).read_text())
    state = race["decisions"]["states"][a.candidate]
    if state["verdict"] != "pass":
        raise SystemExit(f"{a.candidate} did not pass ({state['verdict']})")
    records = [r for p in a.records for r in read_jsonl(p)]
    new, plan = lineage.accept(a.candidate, state, records, tasks, trials, fresh=a.fresh, margin=a.margin,
                               n_looks=len(race.get("looks") or []) or None, record_files=list(a.records))
    new.save(lp)
    Path(a.out_topup).write_text(json.dumps(plan.as_dict(), indent=1))
    print(json.dumps({"lineage": {"current": new.current, "previous": new.previous,
                                  "labels_next": new.labels_source()["kind"]},
                      "accept": new.history[-1]}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m harnessforge.race_run")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("drive", "replay"):
        s = sub.add_parser(name)
        s.add_argument("--reference", required=True)
        s.add_argument("--candidates", help="comma-separated (default: every other side)")
        s.add_argument("--checkpoints", default="0.25,0.5,0.75,1")
        s.add_argument("--mode", default=ReferenceMode.SAME_UNITS.value, choices=[m.value for m in ReferenceMode])
        s.add_argument("--c", type=float, help="efficacy constant of the c/sqrt(tau) design (implies --design obf)")
        s.add_argument("--max-env-fail", type=float)
        s.add_argument("--trial-looks", help="ATTEMPT_MAJOR: trials per task at each look, e.g. 2,4,8,24")
        s.add_argument("--plan", help="ATTEMPT_MAJOR look plan, e.g. '100x1;200x2;200x8;200x24' (see parse_plan)")
        s.add_argument("--tau2", type=float, default=VAL200.tau2,
                       help="planning effect heterogeneity for information fractions (default: val200)")
        s.add_argument("--sigma2", type=float, default=VAL200.sigma2,
                       help="planning per-trial variance, both sides (default: val200; 0 = fractions k/K)")
        s.add_argument("--design", default="auto", choices=["auto", "obf", "spending"],
                       help="auto: race.default_design (spending for --trial-looks, else c/sqrt(tau))")
        s.add_argument("--alpha", type=float, default=0.05, help="spending design: one-sided alpha")
        s.add_argument("--spending", default="obf", help="spending design: obf, pocock or power:R")
        s.add_argument("--planned", action="store_true",
                       help="spending design: planned information fractions instead of observed")
        s.add_argument("--sampling", help="TASK_MAJOR: a frontier-weighted look order (TaskSampling JSON from "
                                          "`python -m harnessforge.stats.sampling`; default: the request's "
                                          "options.sampling, else uniform prefixes)")
        if name == "drive":
            g = s.add_mutually_exclusive_group(required=True)
            g.add_argument("--request", help="start PlatoonBackend on this request in-process")
            g.add_argument("--side", action="append", help="CANDIDATE=RUN_DIR of a `harnessforge.evals run` job")
            s.add_argument("--out", required=True, help="summary JSON (rewritten every poll)")
            s.add_argument("--poll", type=float, default=60.0)
            s.add_argument("--drain-timeout", type=float, default=900.0)
            s.add_argument("--reverse", action="store_true",
                           help="also replay the race with the first candidate as reference")
        else:
            s.add_argument("records", nargs="+")
            s.add_argument("--tasks", required=True, help="a request.json (tasks, trials)")
            s.add_argument("--order", choices=[o.value for o in Order],
                           help="the evaluation's launch order (default: the request's \"order\", else task_major)")
            s.add_argument("--out")
    t = sub.add_parser("topup", help="the units a reference still needs for a full evaluation")
    t.add_argument("--records", nargs="+", required=True)
    t.add_argument("--reference", required=True)
    t.add_argument("--tasks", required=True, help="the gate's request.json (tasks, trials)")
    t.add_argument("--fresh", action="store_true", help="plan a full fresh evaluation instead")
    t.add_argument("--out", required=True)
    c = sub.add_parser("accept", help="a candidate passed the gate: update the lineage, write its top-up plan")
    c.add_argument("--lineage", required=True, help="lineage JSON (created with --reference when missing)")
    c.add_argument("--reference", help="the current reference when creating the lineage")
    c.add_argument("--step0-labels", help="labels for gates before any accept (an independent pass)")
    c.add_argument("--race", required=True, help="the gate's race summary JSON (race_run drive --out)")
    c.add_argument("--candidate", required=True)
    c.add_argument("--records", nargs="+", required=True, help="the candidate's gate records (JSONL)")
    c.add_argument("--tasks", required=True, help="the gate's request.json (tasks, trials)")
    c.add_argument("--fresh", default="never", choices=FRESH_POLICIES,
                   help="re-evaluate the new reference from scratch: never, after a marginal accept, always")
    c.add_argument("--margin", type=float, default=0.5, help="marginal: t - efficacy below this, or the final look")
    c.add_argument("--out-topup", required=True)
    a = p.parse_args(argv)
    return {"drive": cmd_drive, "replay": cmd_replay, "topup": cmd_topup, "accept": cmd_accept}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
