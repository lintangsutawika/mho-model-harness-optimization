"""Model candidates: the proposer writes model specs (candidates/<tag>/model_specs/*.md: what
the model should learn); they are realized into sigma-reward configs
(candidates/<tag>/model/sigma_rewards/*.json), and the build step trains the CURRENT model with
the current harness on that reward (scripts/optimize/train.sh, Miles). The newest HF checkpoint
under candidates/<tag>/train/checkpoints/hf/ is the candidate model. It is trained on
--train-data, so it skips eval_b and the train gate and is judged on the val split directly.
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
            # {sigma, checks}; each train task's own expected answer is added when train.sh
            # stages the split (reward_synthesis.stage_tasks).
            cfg = reward_synthesis.produce_config(spec)
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


class ModelKind(Kind):
    name = "model"
    build_stage = "train"
    train_gate = False      # trained on --train-data: judged on the val split only
    parallel = False        # one training run at a time

    def __init__(self, *, train_data: str, trainer: str, submit_train: Callable[..., str],
                 realize: Callable[[Path, str], tuple[bool, Path | None]] = realize_model_specs):
        self.train_data, self.trainer = train_data, trainer
        self._submit_train, self._realize = submit_train, realize

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
        return self._submit_train(loop.sched, self.trainer,
                                  model=current.model or loop.args.model, harness=current.harness,
                                  out_dir=cand_dir / "train", logs=logs,
                                  tag=cand_dir.name, sigma_reward=str(self.sigma_reward(cand_dir)),
                                  train_data=self.train_data)

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
