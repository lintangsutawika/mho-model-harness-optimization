"""What a candidate kind must provide to the optimization loop.

The loop always evaluates a (harness, model) Pair. Its state holds the current best pair; a
candidate changes ONE side of it:

    harness candidate: the proposer edits the harness  -> (candidate harness, current model)
    model candidate:   the proposer writes a model spec, which is trained into a checkpoint
                       -> (current harness, candidate checkpoint)

Everything else (stages, retries, resume, records, the train/val gates, the frontier) is the
loop's and is identical for every kind. A kind only knows how to check its proposal, build it
(optional job between propose and eval_b), which pair it evaluates, and what to persist when
it is accepted.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Pair:
    harness: str | None = None    # harness dir; None = the base scaffold
    model: str | None = None      # HF id / checkpoint dir; None = the run's --model

    def as_dict(self) -> dict[str, str | None]:
        return {"harness": self.harness, "model": self.model}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Pair":
        d = d or {}
        return cls(harness=d.get("harness"), model=d.get("model"))


class Kind(ABC):
    name: str = ""
    # Job run between propose and eval_b (its stage name), or None. The loop submits it via
    # submit_build() and checks it with build_ok(); its log is <run>/logs/<tag>-<build_stage>.out.
    build_stage: str | None = None
    # Run eval_b + the train gate on the train subset before val? Off for kinds whose candidate
    # has seen the train data (a model trained on it), where that comparison is meaningless.
    train_gate: bool = True
    # May the loop run several sibling candidates of this kind at once (--proposals)? Off for
    # kinds whose build is heavy (model: a training run per sibling).
    parallel: bool = True
    # Does a candidate start with a proposer job? Off for kinds whose candidate is fully
    # determined by the current pair (rsft: train the current model with the current harness).
    # Such a kind skips eval_a (no subset evidence is needed) and starts at its build stage.
    proposes: bool = True
    # Accept every candidate that gets a val score, without the val gate (WHALE's weight
    # updates are always kept). Its val eval is a plain uniform one (no adaptive looks).
    always_accept: bool = False

    @abstractmethod
    def proposal_ok(self, cand_dir: Path) -> tuple[bool, str]:
        """After the propose job: did it produce a usable proposal? (ok, reason-if-not)."""

    def submit_build(self, loop: Any, cand_dir: Path, current: Pair, logs: Path) -> str:
        raise NotImplementedError(f"{self.name} has no build stage")

    def build_ok(self, cand_dir: Path) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def candidate_pair(self, cand_dir: Path, current: Pair) -> Pair:
        """The pair eval_b and eval_val evaluate for this candidate."""

    def on_accept(self, cand_dir: Path, run_dir: Path, pair: Pair) -> None:
        """Persist an accepted candidate (the loop then makes `pair` the current best)."""
