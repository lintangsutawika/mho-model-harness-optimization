"""Rejection-sampling fine-tuning (RSFT) filters for the Miles trainer (WHALE's weight update).

One RSFT step, all inside Miles (scripts/optimize/train.sh with MHO_TRAIN_OBJECTIVE=rsft):
  1. roll out --n-samples-per-prompt trajectories per prompt with the CURRENT weights and the
     current harness (on-policy, like GRPO),
  2. keep only the successful ones (reward >= MHO_RSFT_THRESHOLD; binary verifier -> correct),
  3. one SFT pass over the kept trajectories (--loss-type sft_loss): negative log-likelihood of
     the model's own tokens only -- tool output / observations are already masked out by the
     agentic loss mask.
Unlike GRPO, a prompt whose rollouts ALL succeed is still training data; a prompt with no
success gives none.

Miles hooks (wired in run_training.py):
  any_success     --dynamic-sampling-filter-path: drop a prompt group with no successful
                  rollout, so Miles samples more prompts until the batch holds
                  --rollout-batch-size groups that each have >= 1 success
                  (MHO_RSFT_GROUP_FILTER=none keeps every group, as WHALE does).
  keep_successful --rollout-sample-filter-path: mark failed samples remove_sample=True, which
                  zeroes their loss mask (they cost compute but carry no gradient).

Pure Python on purpose (no miles import): Miles accepts a bool from a dynamic filter, and
samples are duck-typed (`get_reward_value(args)`, `remove_sample`).
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Iterator


# ----------------------------------------------------------------------------- launcher flags
GRPO_GROUP_FILTER = "miles.rollout.filter_hub.dynamic_sampling_filters.apply_reward_nonzero_std_filter"
RSFT_GROUP_FILTER = "mho.train.rsft.any_success"
RSFT_SAMPLE_FILTER = "mho.train.rsft.keep_successful"


def objective_train_args(objective: str) -> str:
    """Loss / advantage flags for the objective."""
    if objective == "grpo":
        return (
            "--advantage-estimator grpo "
            "--use-kl-loss "
            "--kl-loss-coef 0.01 "
            "--kl-loss-type low_var_kl "
            "--entropy-coef 0.0 "
            "--eps-clip 0.2 "
            "--eps-clip-high 0.28 "
        )
    if objective == "rsft":
        # SFT loss on the kept rollouts' own tokens; no advantages, no KL, no clipping.
        return (
            "--loss-type sft_loss "
            "--disable-compute-advantages-and-returns "
            f"--rollout-sample-filter-path {RSFT_SAMPLE_FILTER} "
        )
    raise ValueError(f"unknown objective {objective!r}")


def objective_group_filter(agent_args: str, objective: str) -> str:
    """GRPO drops prompt groups whose rewards do not vary (all right / all wrong: zero
    advantage). RSFT must keep all-right groups (they are its training data): it drops only
    groups with no success, or none at all (MHO_RSFT_GROUP_FILTER=none)."""
    if objective == "grpo":
        return agent_args
    flag = f"--dynamic-sampling-filter-path {GRPO_GROUP_FILTER} "
    assert flag in agent_args, "launch_common no longer sets the GRPO group filter"
    keep = os.environ.get("MHO_RSFT_GROUP_FILTER", "any_success").strip().lower()
    repl = "" if keep == "none" else f"--dynamic-sampling-filter-path {RSFT_GROUP_FILTER} "
    return agent_args.replace(flag, repl)


# ----------------------------------------------------------------------------- filters
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
    return r is not None and r >= threshold()


def any_success(args: Any, samples: list, **kwargs: Any) -> bool:
    """Group filter: keep a prompt's group only if at least one rollout succeeded."""
    return any(is_success(args, s) for s in _samples(samples))


def keep_successful(args: Any, data: list) -> None:
    """Sample filter: failed rollouts take no part in the SFT loss."""
    for group in data:
        for s in _samples(group):
            if not is_success(args, s):
                s.remove_sample = True
