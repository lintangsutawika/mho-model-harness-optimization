"""Model candidates: train the CURRENT model with the current harness on --train-data
(scripts/optimize/train.sh, Miles); the newest HF checkpoint under
candidates/<tag>/train/checkpoints/hf/ is the candidate model. It trained on --train-data, so it
skips eval_b and the train gate and is judged on the val split directly.

A model candidate is configured on two independent axes:

  source     where the training signal comes from
    propose  the proposer writes model specs (candidates/<tag>/model_specs/*.md: what the model
             should learn), realized into sigma-reward configs
             (candidates/<tag>/model/sigma_rewards/*.json) that the train job stages into the
             train tasks (MHO_SIGMA_REWARD). Needs the proposer and eval_a's train-sample traces.
    sample   no proposer: roll out the train tasks and use their own verifier reward (WHALE).
  objective  how the rollouts update the weights (MHO_TRAIN_OBJECTIVE; mho/train/objectives.py)
    grpo     policy gradient with group-relative advantages
    rsft     rejection-sampling fine-tuning: SFT on the successful rollouts only

  accept     gate (the val gate, like every candidate) | always (WHALE keeps every update; its
             val score is still recorded)
  steps      rollout+update steps per candidate (NUM_ROLLOUT); None = train.sh's default

    propose x grpo   the sigma method (default)       sample x grpo   RL on the task reward
    propose x rsft   SFT on sigma-successful rollouts sample x rsft   WHALE's weight update
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from utils.state import _latest_hf_ckpt

from .base import Kind, Pair


def realize_model_specs(cand_dir: Path, train_data: str | Path) -> tuple[bool, Path | None]:
    """Realize a MODEL proposal's model_specs/*.md into model/sigma_rewards/<name>.json.
    Idempotent + re-runnable: on resume we call this again for any spec file that has no
    realized config yet, so a crash between "specs written" and "sigma_rewards written" is
    recovered instead of re-running the whole propose job.
    Returns (ok, sigma_rewards_dir | None). Already-realized configs are left untouched."""
    spec_dir = cand_dir / "model_specs"
    sr_dir = cand_dir / "model" / "sigma_rewards"
    if not spec_dir.is_dir():
        return False, None
    from mho.data import reward_synthesis
    from mho.optimizer import model_spec as _ms
    train_root = Path(train_data)
    src_task = next((p for p in sorted(train_root.iterdir()) if p.is_dir()), None)
    sr_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for md in sorted(spec_dir.glob("*.md")):
        if md.name.startswith("eval_prior") or md.name == "harness.md":
            continue
        out = sr_dir / f"{md.stem}.json"
        if out.exists():           # already realized -> skip, don't clobber
            continue
        try:
            spec, _ = _ms.parse_spec(md)
            cfg = reward_synthesis.produce_config(src_task, spec) if src_task else \
                {"sigma": spec.name, "expected_answer": "", "checks": spec.checks}
            out.write_text(json.dumps(cfg, indent=2) + "\n")
            n += 1
        except Exception as e:
            print(f"  [realize] {md.name} failed: {e}")
    if n:
        print(f"  [realize] realized {n} sigma-reward config(s) -> {sr_dir}")
    # Uniform candidate record: a MODEL proposal gets a CANDIDATE.md too (synthesized from its
    # model_spec's hypothesis + evidence), so _candidate_hypothesis / evolution_summary record
    # model candidates exactly like harness candidates. Idempotent: won't overwrite an existing
    # CANDIDATE.md (e.g. one the proposer already wrote).
    if not (cand_dir / "CANDIDATE.md").exists():
        src = None
        for md in sorted(spec_dir.glob("*.md")):
            if md.name.startswith("eval_prior") or md.name == "harness.md":
                continue
            try:
                src = _ms.parse_spec(md)[0]
                break
            except Exception:
                continue
        if src is not None:
            ev = "\n".join(f"- {e}" for e in (src.evidence or [])[:5])
            _cand_md = (f"# Model Proposal: {src.name}\n\n"
                        f"A sigma-reward capability proposal realized for training. "
                        f"See model_specs/{src.name}.md for the full spec.\n\n"
                        f"## Hypothesis\n{src.hypothesis}\n\n"
                        f"## Evidence\n{ev or '- (none)'}\n")
            (cand_dir / "CANDIDATE.md").write_text(_cand_md)
    return True, sr_dir


SOURCES = ("propose", "sample")
OBJECTIVES = ("grpo", "rsft")


class ModelKind(Kind):
    name = "model"
    build_stage = "train"
    train_gate = False      # trained on --train-data: judged on the val split only
    parallel = False        # one training run at a time

    def __init__(self, *, train_data: str, trainer: str, submit_train: Callable[..., str],
                 realize: Callable[[Path, str], tuple[bool, Path | None]] = realize_model_specs,
                 source: str = "propose", objective: str = "grpo", accept: str = "gate",
                 steps: int | None = None):
        if source not in SOURCES or objective not in OBJECTIVES or accept not in ("gate", "always"):
            raise ValueError(f"bad model kind config: source={source!r} objective={objective!r} "
                             f"accept={accept!r}")
        self.train_data, self.trainer = train_data, trainer
        self._submit_train, self._realize = submit_train, realize
        self.source, self.objective, self.steps = source, objective, steps
        self.proposes = source == "propose"         # loop: proposer + eval_a, else straight to train
        self.always_accept = accept == "always"

    def describe(self) -> str:
        return (f"source={self.source} objective={self.objective} "
                f"accept={'always' if self.always_accept else 'gate'} steps={self.steps or 'default'}")

    @staticmethod
    def sigma_reward(cand_dir: Path) -> Path | None:
        # One training run per candidate: the first realized sigma-reward config.
        sr = cand_dir / "model" / "sigma_rewards"
        return next(iter(sorted(sr.glob("*.json"))), None) if sr.is_dir() else None

    @staticmethod
    def checkpoint(cand_dir: Path) -> Path | None:
        return _latest_hf_ckpt(cand_dir / "train" / "checkpoints" / "hf")

    def proposal_ok(self, cand_dir: Path) -> tuple[bool, str]:
        ok, _ = self._realize(cand_dir, self.train_data)
        if not ok:
            return False, "no model_specs/"
        if self.sigma_reward(cand_dir) is None:
            return False, "no sigma-reward config realized from model_specs/"
        return True, ""

    def submit_build(self, loop: Any, cand_dir: Path, current: Pair, logs: Path) -> str:
        # Train from the CURRENT best model (a rejected checkpoint is never trained on) with the
        # current best harness.
        model = current.model or loop.args.model
        sigma = None
        if self.proposes:
            sigma = str(self.sigma_reward(cand_dir))
        else:
            # no proposer wrote a CANDIDATE.md: say what this candidate is, for its record
            cand_dir.mkdir(parents=True, exist_ok=True)
            (cand_dir / "CANDIDATE.md").write_text(
                f"# Model update: {self.objective.upper()} on the task reward, "
                f"{self.steps or 'default'} steps\n\n"
                f"From model `{model}` with harness `{current.harness or '(base scaffold)'}`: roll "
                f"out the train tasks with the current weights and "
                + ("keep the successful trajectories for an SFT step." if self.objective == "rsft"
                   else "take a GRPO step on their rewards.") + "\n")
        return self._submit_train(loop.sched, self.trainer, model=model, harness=current.harness,
                                  out_dir=cand_dir / "train", logs=logs, tag=cand_dir.name,
                                  sigma_reward=sigma, train_data=self.train_data,
                                  objective=self.objective, steps=self.steps)

    def build_ok(self, cand_dir: Path) -> tuple[bool, str]:
        ok = self.checkpoint(cand_dir) is not None
        return ok, "" if ok else "no HF checkpoint under train/checkpoints/hf"

    def candidate_pair(self, cand_dir: Path, current: Pair) -> Pair:
        ckpt = self.checkpoint(cand_dir)
        return Pair(harness=current.harness, model=str(ckpt) if ckpt else None)

    def on_accept(self, cand_dir: Path, run_dir: Path, pair: Pair) -> None:
        # The proposer takes its parent harness from candidates/<frontier best>/harness. When the
        # best is a model candidate, give it a copy of the harness it was evaluated with, so the
        # next harness proposal keeps building on it (not on the base scaffold).
        if pair.harness and not (cand_dir / "harness").exists():
            shutil.copytree(pair.harness, cand_dir / "harness", symlinks=True)
