"""Rejection-sampling fine-tuning (RSFT) filters for the Miles trainer (WHALE's weight update).

One RSFT step, all inside Miles (scripts/optimize/train.sh with MHO_TRAIN_OBJECTIVE=rsft):
  1. roll out --n-samples-per-prompt trajectories per prompt with the CURRENT weights and the
     current harness (on-policy, like GRPO),
  2. keep only the successful ones (reward > MHO_RSFT_THRESHOLD [0.5], strictly, as WHALE;
     binary verifier -> correct),
  3. SFT on the kept trajectories, one pass in minibatches of --global-batch-size of them
     (--loss-type sft_loss, token-level mean): negative log-likelihood of the model's own tokens
     only -- tool output / observations are already masked out by the agentic loss mask.
Unlike GRPO, a prompt whose rollouts ALL succeed is still training data; a prompt with no
success gives none.

Miles hooks (wired by mho/train/objectives.py):
  keep_successful --rollout-sample-filter-path: REMOVES the failed trajectories from the batch
                  (in place), so the SFT minibatches hold successful trajectories only and the
                  number of optimizer steps is floor(kept / global batch size), as WHALE's
                  ceil(accepted / 64). Masking them instead would still spend a forward/backward
                  on each and, with a per-sample mean, scale the update down by the accept rate.
                  If fewer than one global batch succeed, failed trajectories are added back
                  MASKED (remove_sample=True: no gradient) to fill one batch, since Miles needs at
                  least one; that step's update is then over the few successes only.
  any_success     --dynamic-sampling-filter-path, only with MHO_RSFT_GROUP_FILTER=any_success:
                  drop prompt groups with no success and sample more prompts until the batch is
                  full. Off by default: WHALE rolls out a fixed batch of prompts per step.

Pure Python on purpose (no miles import): Miles accepts a bool from a dynamic filter, and
samples are duck-typed (`get_reward_value(args)`, `remove_sample`).
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Iterator


def threshold() -> float:
    return float(os.environ.get("MHO_RSFT_THRESHOLD", "0.5"))


def _samples(group: Iterable[Any]) -> Iterator[Any]:
    for s in group:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def _reward(args: Any, sample: Any) -> float | None:
    try:
        v = sample.get_reward_value(args)
    except Exception:
        v = getattr(sample, "reward", None)
    if isinstance(v, dict):                     # a reward dict: its "reward" field
        v = v.get("reward")
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def is_success(args: Any, sample: Any) -> bool:
    r = _reward(args, sample)
    return r is not None and r > threshold()


def any_success(args: Any, samples: list, **kwargs: Any) -> bool:
    """Group filter: keep a prompt's group only if at least one rollout succeeded."""
    return any(is_success(args, s) for s in _samples(samples))


def _units_ok(args: Any, unit: Any) -> bool:
    """A trajectory is one Sample, or a list of Samples (its segments) sharing one reward."""
    return all(is_success(args, s) for s in _samples([unit]))


def _count(units: Iterable[Any]) -> int:
    return sum(1 for _ in _samples(list(units)))


def keep_successful(args: Any, data: list) -> None:
    """Sample filter: keep only successful trajectories (in place). Groups left empty are
    dropped. If fewer than --global-batch-size samples remain, failed ones are added back
    masked (remove_sample=True) to fill one batch."""
    gbs = int(getattr(args, "global_batch_size", 0) or 0)
    kept_groups, failed = [], []
    for group in data:
        kept = [u for u in group if _units_ok(args, u)]
        failed.extend(u for u in group if not _units_ok(args, u))
        if kept:
            kept_groups.append(kept)
    short = gbs - _count(u for g in kept_groups for u in g)
    if short > 0 and failed:
        pad = []
        for u in failed:
            if _count(pad) >= short:
                break
            for s in _samples([u]):
                s.remove_sample = True
            pad.append(u)
        kept_groups.append(pad)
    data[:] = kept_groups
