"""The interfaces the workstreams build against. Implementations live in their own modules:

    harnessforge.stats      sequential paired tests (pure)                       [workstream A]
    harnessforge.pool       ModelPool on SGLang, N nodes, policy swap            [workstream B]
    harnessforge.evals      EvalBackend on platoon eval (+ a fake for tests)     [workstream C]
    harnessforge.bench      gold-harness toy benchmark                           [workstream D]
    harnessforge.launch     Launcher: PBS / SLURM / local, from Python           [shared]

Everything crossing a module boundary is one of the types below or a TrialRecord
(harnessforge.records). Keep these small; change them only in a PR of their own.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from harnessforge.records import TrialRecord


# ----------------------------------------------------------------------------- candidates
@dataclass(frozen=True)
class Candidate:
    """A harness to evaluate: a directory holding an installable agent (e.g. a micro-swe-agent
    fork), identified by `id`. `parent` is the candidate it was derived from (None = seed)."""
    id: str
    harness_dir: Path
    parent: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- policies / pool
@dataclass(frozen=True)
class PolicyRef:
    """A model checkpoint the pool can serve. `path` is an HF-format directory (or hub id);
    `base_model` gives the tokenizer/chat template/parsers when `path` is a fine-tune."""
    id: str
    path: str
    base_model: str | None = None


@dataclass(frozen=True)
class PoolSpec:
    nodes: int = 4                      # configurable; one launcher job per node
    gpus_per_node: int = 8
    engine: str = "sglang"
    tp: int = 1                          # tensor parallel per engine (dp = gpus_per_node // tp)
    served_name: str = "policy"          # model name clients send
    max_context: int | None = None
    engine_args: dict[str, Any] = field(default_factory=dict)
    launcher: dict[str, Any] = field(default_factory=dict)   # queue/account/walltime/...


@runtime_checkable
class ModelPool(Protocol):
    """A long-lived set of inference engines behind one OpenAI-compatible endpoint.

    Serves harness evaluations and, during training, acts as the rollout engine: `load` swaps
    every engine to a new policy (e.g. after a GRPO run, or each RL step). Trials record which
    policy served them (TrialRecord.policy), so a swap mid-evaluation is detectable."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def endpoint(self) -> str: ...                      # OpenAI base URL, e.g. http://host:port/v1
    def policy(self) -> PolicyRef: ...
    def load(self, policy: PolicyRef, *, drain: bool = True) -> None: ...
    def health(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------------------- evaluation
class Order(str, enum.Enum):
    """How a backend launches a candidate's trials; decides what a sequential look is.
    TASK_MAJOR: every trial of task 1, then task 2, ... (platoon eval); looks = task prefixes
    of `EvalRequest.tasks` (shuffle that list once, up front). ATTEMPT_MAJOR: every task once,
    then every task again, ... (harbor); looks = the first r trials of every task."""
    TASK_MAJOR = "task_major"
    ATTEMPT_MAJOR = "attempt_major"


@dataclass(frozen=True)
class EvalRequest:
    run_id: str
    candidates: Sequence[Candidate]
    tasks: Sequence[str]                 # ordered: the look order for TASK_MAJOR backends
    dataset: str                         # backend-specific dataset ref (e.g. harbor task dir)
    trials: int                          # trials per task per candidate
    endpoint: str                        # model endpoint (ModelPool.endpoint())
    policy: str | None = None            # policy id recorded on every trial
    concurrency: int = 32                # concurrent trials per candidate
    out_dir: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EvalRun(Protocol):
    """A running evaluation of one or more candidates on the same tasks."""

    order: Order

    def records(self) -> list[TrialRecord]: ...       # snapshot: finished AND in-flight trials
    def stop(self, candidate: str) -> None: ...         # stop one candidate's remaining trials
    def stop_all(self) -> None: ...
    def done(self) -> bool: ...
    def wait(self, timeout: float | None = None) -> bool: ...


@runtime_checkable
class EvalBackend(Protocol):
    def start(self, request: EvalRequest) -> EvalRun: ...


# ----------------------------------------------------------------------------- launching
class JobState(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


@dataclass(frozen=True)
class JobSpec:
    """A batch job described in Python (no hand-written job scripts)."""
    name: str
    command: Sequence[str]
    nodes: int = 1
    gpus: int = 0                        # per node
    cpus: int = 0                        # per node
    walltime: str = "4:00:00"
    env: dict[str, str] = field(default_factory=dict)
    log: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)   # queue, account, resource type, host


@dataclass(frozen=True)
class JobStatus:
    state: JobState
    exit_code: int | None = None
    hosts: tuple[str, ...] = ()


@runtime_checkable
class Launcher(Protocol):
    def submit(self, spec: JobSpec) -> str: ...
    def status(self, job_id: str) -> JobStatus: ...
    def cancel(self, job_id: str) -> None: ...
