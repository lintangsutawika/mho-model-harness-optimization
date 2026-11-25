"""Training objectives: the Miles flags that define each objective's loss and sampling.

run_training.py (the Miles launcher) asks this module, for MHO_TRAIN_OBJECTIVE:
  objective_train_args    loss / advantage flags
  objective_group_filter  which prompt groups a rollout batch keeps (rewritten in the agentic
                          args from Miles' launch_common, which sets GRPO's filter)
  rollout_temperature     sampling temperature of the rollouts
  async_train_args        Miles fully-async rollout (MHO_FULLY_ASYNC, on by default)

  grpo  policy gradient with group-relative advantages, KL to the reference, PPO clipping.
        Keeps only groups whose rewards vary (all right / all wrong carry no advantage).
        Dr. GRPO by default (MHO_DR_GRPO=0: plain GRPO): no std normalization of the
        advantages and a constant, not per-trajectory-length, loss divisor (mho/train/drgrpo.py).
  rsft  rejection-sampling fine-tuning (WHALE): SFT loss on the successful rollouts only.
        Keeps every prompt group; failed trajectories are removed by its sample filter
        (mho/train/rsft.py). Batch shape: train.sh's defaults (WHALE: rsft 256 x 8, grpo 64 x 8).

Plain string builders (no miles import), so they are unit-testable outside the container.
"""
from __future__ import annotations

import os

GRPO_GROUP_FILTER = "miles.rollout.filter_hub.dynamic_sampling_filters.apply_reward_nonzero_std_filter"
RSFT_GROUP_FILTER = "mho.train.rsft.any_success"
RSFT_SAMPLE_FILTER = "mho.train.rsft.keep_successful"
DRGRPO_REDUCER = "mho.train.drgrpo.get_pg_loss_reducer"
DPPO_TIS = "mho.train.dppo.dppo_tis_function"


def dr_grpo_enabled() -> bool:
    return os.environ.get("MHO_DR_GRPO", "1").strip().lower() not in ("0", "false", "no", "")


def pg_loss_kind() -> str:
    """MHO_PG_LOSS: ppo (default; PPO clip + KL 0.01 + TIS) or dppo (TMax's DPPO: divergence
    trust region against the rollout policy, ratio cap 10, no KL; mho/train/dppo.py)."""
    v = os.environ.get("MHO_PG_LOSS", "ppo").strip().lower() or "ppo"
    if v not in ("ppo", "dppo"):
        raise ValueError(f"MHO_PG_LOSS must be ppo or dppo, got {v!r}")
    return v


# Rollout sampling temperature per objective (MHO_ROLLOUT_TEMPERATURE overrides): 1.0 for both,
# as WHALE's verl script, so the objectives sample the same distribution.
ROLLOUT_TEMPERATURE = {"grpo": 1.0, "rsft": 1.0}


def rollout_temperature(objective: str) -> float:
    v = os.environ.get("MHO_ROLLOUT_TEMPERATURE", "").strip()
    return float(v) if v else ROLLOUT_TEMPERATURE[objective]


def objective_train_args(objective: str) -> str:
    """Loss / advantage flags for the objective."""
    if objective == "grpo" and pg_loss_kind() == "dppo":
        # DPPO needs the TIS hook, so it requires fully-async (which adds --use-tis).
        assert fully_async_enabled(), "MHO_PG_LOSS=dppo needs MHO_FULLY_ASYNC=1 (--use-tis)"
        out = (
            "--advantage-estimator grpo "
            "--entropy-coef 0.0 "
            "--eps-clip 1e9 "
            "--eps-clip-high 1e9 "
            "--skip-actor-forward-only "
            f"--custom-tis-function-path {DPPO_TIS} "
        )
        if dr_grpo_enabled():
            out += ("--disable-grpo-std-normalization "
                    f"--custom-pg-loss-reducer-function-path {DRGRPO_REDUCER} ")
        return out
    if objective == "grpo":
        out = (
            "--advantage-estimator grpo "
            "--use-kl-loss "
            "--kl-loss-coef 0.01 "
            "--kl-loss-type low_var_kl "
            "--entropy-coef 0.0 "
            "--eps-clip 0.2 "
            "--eps-clip-high 0.28 "
        )
        if dr_grpo_enabled():
            out += ("--disable-grpo-std-normalization "
                    f"--custom-pg-loss-reducer-function-path {DRGRPO_REDUCER} ")
        return out
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


ROLLOUT_FN_FLAG = "--rollout-function-path generate.RolloutFn "


def fully_async_enabled() -> bool:
    return os.environ.get("MHO_FULLY_ASYNC", "1").strip().lower() not in ("0", "false", "no", "")


def async_train_args(agent_args: str, objective: str) -> str:
    """Miles fully-async rollout: a persistent worker keeps generating while the trainer drains
    finished groups, so a step no longer waits on the slowest trajectory of its batch, and the
    groups the filter drops are replaced continuously instead of in rounds. Agentic rollouts
    here are long-tailed (8k mean, 30k max tokens) and most GRPO groups are all-wrong, so the
    synchronous step was ~90% waiting.

    Returns agent_args with the async flags (MHO_FULLY_ASYNC=0: unchanged).
      MHO_ASYNC_MAX_STALENESS   groups more than this many weight versions behind are recycled
                                (default 3, the upper end of FrogNano's policy lag 1-3)
      MHO_ASYNC_MAX_CONCURRENT  trajectories generating at once (default 128). Each one holds a
                                sandbox from the single host executor, which cannot provision
                                a full 64 x 8 batch at once: at 512, 45% of trials timed out
                                waiting for a sandbox and no group survived the filter.
    GRPO also gets truncated importance sampling (--use-tis): samples can come from an older
    policy than the one being updated. RSFT's SFT loss has no policy ratio to correct.

    --fully-async selects Miles' FullyAsyncRolloutFn, so the harbor example's
    generate.RolloutFn (which only adds agent/* metrics) is dropped; Miles rejects both.
    """
    if not fully_async_enabled():
        return agent_args
    assert ROLLOUT_FN_FLAG in agent_args, "launch_common no longer sets generate.RolloutFn"
    out = agent_args.replace(ROLLOUT_FN_FLAG, "") + "--fully-async "
    staleness = os.environ.get("MHO_ASYNC_MAX_STALENESS", "3").strip()
    if staleness:
        out += f"--max-weight-staleness {int(staleness)} "
    concurrent = os.environ.get("MHO_ASYNC_MAX_CONCURRENT", "128").strip()
    if concurrent:
        out += f"--async-max-concurrent-samples {int(concurrent)} "
    if objective == "grpo":
        out += "--use-tis "
    return out


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
