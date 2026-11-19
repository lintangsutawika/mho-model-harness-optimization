"""Trial records: the one result format every evaluation backend produces and every gate reads.

A *trial* is one rollout of one candidate harness (under one policy) on one task. Backends
(platoon eval, harbor, a fake in tests) convert their own artifacts into `TrialRecord`s; the
statistics (harnessforge.stats) and the optimizer only ever see records, never backend dirs.

`index` is the trial's position in its candidate's launch order for that task (0-based), so a
sequential test can take "the first r trials of every task" or "every trial of the first n tasks
of a fixed order" without depending on completion time (slow trials are mostly failures, so
"whatever finished first" is biased).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = 1


@dataclass(frozen=True)
class TrialRecord:
    candidate: str                 # candidate id (harness [+ policy]) this trial evaluated
    task: str                      # task id, identical across candidates for pairing
    index: int                     # launch order of this trial within (candidate, task), 0-based
    finished: bool                 # False: started, not finished (no reward yet)
    reward: float | None = None    # primary reward in [0, 1] when finished; None if not
    env_failed: bool = False       # sandbox never started: no evidence about the candidate
    error: str | None = None       # exception type / outcome status when the trial errored
    output_tokens: int | None = None
    input_tokens: int | None = None
    wall_seconds: float | None = None
    started_at: float | None = None    # unix seconds
    finished_at: float | None = None
    policy: str | None = None      # policy (model) id that served the trial
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        """Counts as evidence: finished and the sandbox did start (a harness that crashes its
        own agent setup still counts, as a 0)."""
        return self.finished and not self.env_failed

    def to_json(self) -> str:
        return json.dumps({"schema": SCHEMA, **asdict(self)}, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrialRecord":
        d = {k: v for k, v in d.items() if k != "schema"}
        return cls(**d)


def write_jsonl(path: str | Path, records: Iterable[TrialRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(r.to_json() + "\n" for r in records))
    tmp.replace(p)


def read_jsonl(path: str | Path) -> list[TrialRecord]:
    p = Path(path)
    if not p.exists():
        return []
    return [TrialRecord.from_dict(json.loads(l)) for l in p.read_text().splitlines() if l.strip()]


@dataclass(frozen=True)
class TaskStats:
    """One candidate's evidence on one task: mean reward over its scored trials."""
    reward: float
    n: int
    cost: float | None = None      # mean output tokens over scored trials that reported them


def per_task(records: Iterable[TrialRecord], candidate: str | None = None, *,
             max_index: int | None = None, tasks: set[str] | None = None) -> dict[str, TaskStats]:
    """{task: TaskStats} over scored trials (optionally only one candidate, only trials with
    index < max_index, only `tasks`). A missing reward on a finished trial counts as 0."""
    acc: dict[str, list[float]] = {}
    cost: dict[str, list[float]] = {}
    for r in records:
        if candidate is not None and r.candidate != candidate:
            continue
        if not r.scored or (max_index is not None and r.index >= max_index):
            continue
        if tasks is not None and r.task not in tasks:
            continue
        acc.setdefault(r.task, []).append(float(r.reward or 0.0))
        if r.output_tokens is not None:
            cost.setdefault(r.task, []).append(float(r.output_tokens))
    return {t: TaskStats(sum(v) / len(v), len(v),
                         (sum(cost[t]) / len(cost[t])) if cost.get(t) else None)
            for t, v in acc.items()}


def by_candidate(records: Iterable[TrialRecord]) -> dict[str, list[TrialRecord]]:
    out: dict[str, list[TrialRecord]] = {}
    for r in records:
        out.setdefault(r.candidate, []).append(r)
    return out


def iter_finished(records: Iterable[TrialRecord]) -> Iterator[TrialRecord]:
    return (r for r in records if r.finished)
