"""RSFT candidates (WHALE's weight update): no proposer. A candidate is a short rejection-
sampling fine-tuning phase -- `steps` rounds of (roll out the current model with the current
harness on --train-data, keep the successful trajectories, one SFT step on them) -- run by
scripts/optimize/train.sh with MHO_TRAIN_OBJECTIVE=rsft (filters: mho/train/rsft.py). The phase
starts from the CURRENT best model, so phases chain; its newest HF checkpoint is the candidate
model, judged on the val split (it trained on --train-data, so there is no train gate).

  --mode rsft-only     : RSFT phases only (weight-only RSFT, WHALE's baseline)
  --mode harness-rsft  : --harness-proposals harness candidates, then one RSFT phase, repeating
                         (WHALE's alternation, with this loop's harness search)

With `always_accept` (--rsft-accept always, WHALE's choice) every phase that produces a
checkpoint becomes the current model; with --rsft-accept gate it must pass the val gate.
Each phase restarts the optimizer from the checkpoint (Adam state is not carried over).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .base import Pair
from .model import ModelKind


class RSFTKind(ModelKind):
    name = "rsft"
    proposes = False

    def __init__(self, *, train_data: str, trainer: str, submit_train: Callable[..., str],
                 steps: int, always_accept: bool = True):
        super().__init__(train_data=train_data, trainer=trainer, submit_train=submit_train)
        self.steps, self.always_accept = steps, always_accept

    def proposal_ok(self, cand_dir: Path) -> tuple[bool, str]:      # never called: no proposer
        return True, ""

    def submit_build(self, loop: Any, cand_dir: Path, current: Pair, logs: Path) -> str:
        model = current.model or loop.args.model
        cand_dir.mkdir(parents=True, exist_ok=True)
        # the candidate's record (summary.json / status.json hypothesis) says what it is
        (cand_dir / "CANDIDATE.md").write_text(
            f"# RSFT phase: {self.steps} rejection-sampling fine-tuning steps\n\n"
            f"From model `{model}` with harness `{current.harness or '(base scaffold)'}`: each "
            f"step rolls out the train data, keeps the successful trajectories, and takes one "
            f"SFT step on them.\n")
        return self._submit_train(loop.sched, self.trainer, model=model,
                                  harness=current.harness, out_dir=cand_dir / "train",
                                  logs=logs, tag=cand_dir.name, train_data=self.train_data,
                                  objective="rsft", steps=self.steps)
