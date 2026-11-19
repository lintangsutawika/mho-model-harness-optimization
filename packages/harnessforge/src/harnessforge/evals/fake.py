"""A deterministic EvalBackend for tests: scripted rewards, simulated time, task-major launches, per-candidate stop.

    backend = FakeBackend(rewards={"A": {"t1": 1.0, "t2": [0, 1]}, "B": {"t1": 0.5}}, durations={"t2": 3})
    run = backend.start(EvalRequest(run_id="r", candidates=[a, b], tasks=["t1", "t2"], trials=2, ...))
    while not run.done():
        records = run.records()      # each call advances the clock one tick (auto_advance=True)

Time is a tick counter. Each candidate launches its trials in TASK_MAJOR order (every trial of task 1, then task 2,
...) with at most `request.concurrency` in flight; a trial launched at tick t finishes at t + duration(task).
`records()` returns the snapshot at the current tick (finished trials, and in-flight ones as finished=False), then
advances one tick unless `auto_advance=False` (call `step()` yourself). `wait()` runs the clock to the end.

A reward spec per (candidate, task) is a probability p (each trial is 1 with probability p, drawn from a hash of
seed/candidate/task/index, so runs are reproducible) or a sequence of per-trial rewards (indexed by trial index,
cycled). Tasks without a spec get `default`. `env_failures` names (candidate, task, index) triples whose trial ends
env_failed. `stop(candidate)` launches nothing more for it and ends its in-flight trials as env_failed "stopped".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Collection, Mapping, Sequence

from harnessforge.contracts import EvalRequest, Order
from harnessforge.records import TrialRecord

RewardSpec = float | Sequence[float]


def _uniform(*parts: object) -> float:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass
class _Trial:
    task: str
    index: int
    launched: int
    finishes: int
    stopped: bool = False


@dataclass
class _Lane:
    """One candidate's schedule."""
    candidate: str
    queue: list[tuple[str, int]]              # (task, index) not launched yet, in launch order
    running: list[_Trial] = field(default_factory=list)
    done: list[_Trial] = field(default_factory=list)
    stopped: bool = False


class FakeRun:
    order = Order.TASK_MAJOR

    def __init__(self, backend: "FakeBackend", request: EvalRequest, *, auto_advance: bool) -> None:
        self._backend = backend
        self._request = request
        self._auto = auto_advance
        self.tick = 0
        queue = [(t, i) for t in request.tasks for i in range(request.trials)]
        self._lanes = {c.id: _Lane(c.id, list(queue)) for c in request.candidates}
        self._launch()

    # ---------------------------------------------------------------- clock
    def _launch(self) -> None:
        for lane in self._lanes.values():
            while not lane.stopped and lane.queue and len(lane.running) < max(1, self._request.concurrency):
                task, index = lane.queue.pop(0)
                lane.running.append(_Trial(task, index, self.tick, self.tick + self._backend.duration(task)))

    def step(self, n: int = 1) -> None:
        """Advance the clock n ticks: trials due finish, freed slots launch the next trials."""
        for _ in range(n):
            self.tick += 1
            for lane in self._lanes.values():
                due = [t for t in lane.running if t.finishes <= self.tick]
                lane.running = [t for t in lane.running if t.finishes > self.tick]
                lane.done.extend(due)
            self._launch()

    # ---------------------------------------------------------------- EvalRun
    def records(self) -> list[TrialRecord]:
        out: list[TrialRecord] = []
        for lane in self._lanes.values():
            out.extend(self._record(lane.candidate, t, True) for t in lane.done)
            out.extend(self._record(lane.candidate, t, False) for t in lane.running)
        out.sort(key=lambda r: (r.candidate, r.task, r.index))
        if self._auto and not self.done():
            self.step()
        return out

    def stop(self, candidate: str) -> None:
        lane = self._lanes[candidate]
        lane.stopped = True
        lane.queue.clear()
        for trial in lane.running:
            trial.stopped = True
            trial.finishes = self.tick
        lane.done.extend(lane.running)
        lane.running = []

    def stop_all(self) -> None:
        for candidate in self._lanes:
            self.stop(candidate)

    def done(self) -> bool:
        return all(not lane.queue and not lane.running for lane in self._lanes.values())

    def wait(self, timeout: float | None = None) -> bool:
        """Run the clock until every trial finished (`timeout` counts ticks here); True when done."""
        limit = None if timeout is None else self.tick + int(timeout)
        while not self.done() and (limit is None or self.tick < limit):
            self.step()
        return self.done()

    # ---------------------------------------------------------------- helpers
    def _record(self, candidate: str, trial: _Trial, finished: bool) -> TrialRecord:
        policy = self._request.policy
        common = dict(candidate=candidate, task=trial.task, index=trial.index, started_at=float(trial.launched),
                      policy=policy)
        if not finished:
            return TrialRecord(finished=False, **common)
        if trial.stopped:
            return TrialRecord(finished=True, env_failed=True, error="stopped", finished_at=float(trial.finishes),
                               **common)
        wall = float(trial.finishes - trial.launched)
        if (candidate, trial.task, trial.index) in self._backend.env_failures:
            return TrialRecord(finished=True, env_failed=True, error="sandbox_start",
                               finished_at=float(trial.finishes), wall_seconds=wall, **common)
        reward = self._backend.reward(candidate, trial.task, trial.index)
        tokens = self._backend.tokens(candidate, trial.task)
        return TrialRecord(finished=True, reward=reward, output_tokens=tokens, input_tokens=10 * tokens,
                           wall_seconds=wall, finished_at=float(trial.finishes), **common)


class FakeBackend:
    """EvalBackend with scripted rewards (see the module docstring)."""

    def __init__(self, rewards: Mapping[str, Mapping[str, RewardSpec]] | None = None, *, default: RewardSpec = 0.0,
                 durations: Mapping[str, int] | None = None, default_duration: int = 1,
                 output_tokens: Mapping[str, int] | None = None, default_tokens: int = 1000,
                 env_failures: Collection[tuple[str, str, int]] = (), seed: int = 0,
                 auto_advance: bool = True) -> None:
        self._rewards = {c: dict(t) for c, t in (rewards or {}).items()}
        self._default = default
        self._durations = dict(durations or {})
        self._default_duration = max(1, int(default_duration))
        self._tokens = dict(output_tokens or {})
        self._default_tokens = default_tokens
        self.env_failures = frozenset(env_failures)
        self.seed = seed
        self.auto_advance = auto_advance
        self.runs: list[FakeRun] = []

    def start(self, request: EvalRequest) -> FakeRun:
        run = FakeRun(self, request, auto_advance=self.auto_advance)
        self.runs.append(run)
        return run

    def duration(self, task: str) -> int:
        return max(1, int(self._durations.get(task, self._default_duration)))

    def tokens(self, candidate: str, task: str) -> int:
        return int(self._tokens.get(f"{candidate}/{task}", self._tokens.get(task, self._default_tokens)))

    def reward(self, candidate: str, task: str, index: int) -> float:
        spec = self._rewards.get(candidate, {}).get(task, self._default)
        if isinstance(spec, (int, float)):
            return 1.0 if _uniform(self.seed, candidate, task, index) < float(spec) else 0.0
        values = list(spec)
        return float(values[index % len(values)]) if values else 0.0
