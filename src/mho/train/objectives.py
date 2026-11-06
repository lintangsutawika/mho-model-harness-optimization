"""Training objectives: the Miles flags that define each objective's loss and sampling.

run_training.py (the Miles launcher) asks this module, for MHO_TRAIN_OBJECTIVE:
  objective_train_args    loss / advantage flags
  objective_group_filter  which prompt groups a rollout batch keeps (rewritten in the agentic
                          args from Miles' launch_common, which sets GRPO's filter)
  rollout_temperature     sampling temperature of the rollouts

  grpo  policy gradient with group-relative advantages, KL to the reference, PPO clipping.
        Keeps only groups whose rewards vary (all right / all wrong carry no advantage).
  rsft  rejection-sampling fine-tuning (WHALE): SFT loss on the successful rollouts only.
        Keeps every prompt group; failed trajectories are removed by its sample filter
        (mho/train/rsft.py). Batch shape: train.sh's RSFT defaults (WHALE: 256 prompts x 8).

Plain string builders (no miles import), so they are unit-testable outside the container.
"""
from __future__ import annotations

import os

GRPO_GROUP_FILTER = "miles.rollout.filter_hub.dynamic_sampling_filters.apply_reward_nonzero_std_filter"
RSFT_GROUP_FILTER = "mho.train.rsft.any_success"
RSFT_SAMPLE_FILTER = "mho.train.rsft.keep_successful"


# Rollout sampling temperature per objective (MHO_ROLLOUT_TEMPERATURE overrides): RSFT samples
# at 1.0 as WHALE; GRPO keeps the launcher's 0.8.
ROLLOUT_TEMPERATURE = {"grpo": 0.8, "rsft": 1.0}


def rollout_temperature(objective: str) -> float:
    v = os.environ.get("MHO_ROLLOUT_TEMPERATURE", "").strip()
    return float(v) if v else ROLLOUT_TEMPERATURE[objective]


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
        # SFT loss on the kept rollouts' own tokens, averaged over the minibatch's tokens (as
        # WHALE: a long trajectory weighs more than a short one); no advantages, KL or clipping.
        return (
            "--loss-type sft_loss "
            "--calculate-per-token-loss "
            "--disable-compute-advantages-and-returns "
            f"--rollout-sample-filter-path {RSFT_SAMPLE_FILTER} "
        )
    raise ValueError(f"unknown objective {objective!r}")


def objective_group_filter(agent_args: str, objective: str) -> str:
    """GRPO drops prompt groups whose rewards do not vary (all right / all wrong: zero
    advantage). RSFT must keep all-right groups (they are its training data). By default it
    keeps every group, as WHALE (a fixed batch of prompts per step; failed trajectories are
    removed by the sample filter); MHO_RSFT_GROUP_FILTER=any_success drops groups with no
    success and samples more prompts instead."""
    if objective == "grpo":
        return agent_args
    flag = f"--dynamic-sampling-filter-path {GRPO_GROUP_FILTER} "
    assert flag in agent_args, "launch_common no longer sets the GRPO group filter"
    keep = os.environ.get("MHO_RSFT_GROUP_FILTER", "none").strip().lower()
    repl = "" if keep == "none" else f"--dynamic-sampling-filter-path {RSFT_GROUP_FILTER} "
    return agent_args.replace(flag, repl)
