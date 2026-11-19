"""Dr. GRPO's length-unbiased policy-gradient reduction (Liu et al. 2025, arXiv 2503.20783).

GRPO as Miles runs it by default has two normalizations that Dr. GRPO removes:
  1. advantages divided by the group's reward std -- weights groups by how unanimous they are
     (a nearly-all-wrong group gets a huge advantage on its one success). Removed with Miles'
     --disable-grpo-std-normalization (wired in mho/train/objectives.py).
  2. each trajectory's token losses averaged over ITS OWN length -- a token in a long wrong
     trajectory is penalized less than one in a short wrong trajectory, which lets wrong answers
     grow long. Removed here: every trajectory's token-loss SUM is divided by one constant.

Miles hook: --custom-pg-loss-reducer-function-path mho.train.drgrpo.get_pg_loss_reducer. It
replaces only pg_loss's reduction; KL, clipfrac and the other metrics keep Miles' per-sample mean.

The constant (MHO_DRGRPO_DIVISOR, default 8192 = --rollout-max-response-len) only rescales the
gradient, which Adam largely normalizes away; it still matters for gradient clipping (1.0), so it
is set near our typical trajectory length (~8k tokens), where the update size matches GRPO's.
"""
from __future__ import annotations

import os
from collections.abc import Callable

DEFAULT_DIVISOR = 8192.0


def divisor() -> float:
    v = os.environ.get("MHO_DRGRPO_DIVISOR", "").strip()
    return float(v) if v else DEFAULT_DIVISOR


def get_pg_loss_reducer(total_lengths, response_lengths, loss_masks,
                        calculate_per_token_loss: bool = False) -> Callable:
    """Same signature as Miles' cp_utils.get_sum_of_sample_mean (called by loss_hub/losses.py).
    Sum over the minibatch's trajectories of (token-loss sum / constant). CP must be 1: with
    context parallelism each rank holds a slice of every sequence (we run TP=2, CP=1)."""
    import torch  # noqa: F401  (container only; the module stays importable for tests)

    try:
        from miles.backends.training_utils.parallel import get_parallel_state
        cp = get_parallel_state().cp.size
    except Exception:  # noqa: BLE001  (outside Miles: tests)
        cp = 1
    assert cp == 1, "mho.train.drgrpo supports context-parallel size 1 only"

    d = divisor()

    def sum_of_token_over_constant(x):
        return sum((x_i * m_i).sum() / d
                   for x_i, m_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=True))

    return sum_of_token_over_constant
