"""Training objectives: the Miles flags that define each objective's loss and sampling.

run_training.py (the Miles launcher) asks this module, for MHO_TRAIN_OBJECTIVE:
  objective_train_args    loss / advantage flags
  objective_group_filter  which prompt groups a rollout batch keeps (rewritten in the agentic
                          args from Miles' launch_common, which sets GRPO's filter)

  grpo  policy gradient with group-relative advantages, KL to the reference, PPO clipping.
        Keeps only groups whose rewards vary (all right / all wrong carry no advantage).
  rsft  rejection-sampling fine-tuning (WHALE): SFT loss on the successful rollouts only.
        Keeps groups with at least one success. Its filters are in mho/train/rsft.py.

Plain string builders (no miles import), so they are unit-testable outside the container.
"""
from __future__ import annotations

import os

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
