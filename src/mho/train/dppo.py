"""DPPO (divergence-based trust region; arXiv 2602.04879) as used by TMax (allenai/tmax-4b).

Reference: hamishivi/tmax training/open-instruct/open_instruct/grpo_utils.py
(compute_binary_divergence, compute_dppo_mask, compute_dppo_policy_loss), recipe flags
--loss_fn dppo --dppo_divergence_type tv --dppo_divergence_threshold 0.1 --dppo_ratio_cap 10.

Per token, with mu = rollout (inference-engine) policy and pi = trainer policy:
  ratio = pi / mu,  D = |mu(y_t) - pi(y_t)|  (binary TV; or binary KL)
  masked if D > delta AND the update pushes further away (A > 0 & ratio > 1, or A < 0 & ratio < 1)
  loss  = -A * min(ratio, cap).detach() * log pi      (gradient: -A * min(ratio, cap) * grad log pi)

Miles wiring (mho/train/objectives.py, MHO_PG_LOSS=dppo):
  --skip-actor-forward-only    the "old" log-probs are the detached current ones, so Miles'
                               ratio is exp(log pi - log pi.detach()) = 1 carrying grad log pi,
                               and pg_loss = -A * that (needs one optimizer step per rollout
                               batch: our global batch = 8 x 32). PPO clipping never binds at 1;
                               --eps-clip 1e9 makes that explicit.
  --use-tis --custom-tis-function-path mho.train.dppo.dppo_tis_function   (this hook)
  (Miles forbids --use-rollout-logprobs together with --use-tis, hence this route.)
The hook gets pg_loss = -A * 1 (sign(A) = -sign(pg_loss)), train_log_probs = detached current
pi and rollout_log_probs = mu, and multiplies pg_loss by the detached weight
mask * min(pi / mu, cap): exactly TMax's -A * min(ratio, cap).detach() * log pi.

Env: MHO_DPPO_DELTA (0.1), MHO_DPPO_CAP (10; <= 0 = no cap), MHO_DPPO_DIVERGENCE (tv | kl).
"""
from __future__ import annotations

import os
from typing import Any

LOG_RATIO_LIMIT = 20.0


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def binary_divergence(mu_lp, pi_lp, kind: str):
    import torch

    mu = torch.exp(mu_lp.clamp(min=-30.0, max=0.0))
    pi = torch.exp(pi_lp.clamp(min=-30.0, max=0.0))
    if kind == "tv":
        return (mu - pi).abs()
    if kind == "kl":
        eps = 1e-9
        m, p = mu.clamp(eps, 1 - eps), pi.clamp(eps, 1 - eps)
        return m * (m.log() - p.log()) + (1 - m) * ((1 - m).log() - (1 - p).log())
    raise ValueError(f"MHO_DPPO_DIVERGENCE must be tv or kl, got {kind!r}")


def dppo_weights(pg_loss, pi_lp, mu_lp, delta: float, cap: float, kind: str):
    """Detached per-token multiplier for pg_loss (= -A * 1 with grad log pi), plus metrics."""
    import torch

    with torch.no_grad():
        ratio = torch.exp((pi_lp - mu_lp).clamp(-LOG_RATIO_LIMIT, LOG_RATIO_LIMIT))
        div = binary_divergence(mu_lp, pi_lp, kind)
        outside = div > delta
        adv_pos = pg_loss.detach() < 0
        adv_neg = pg_loss.detach() > 0
        bad = (adv_pos & (ratio > 1.0) & outside) | (adv_neg & (ratio < 1.0) & outside)
        keep = (~bad).to(pg_loss.dtype)
        if cap > 0:
            weight = ratio.clamp(max=cap).to(pg_loss.dtype)
            capped = (ratio > cap).float()
        else:
            weight = ratio.to(pg_loss.dtype)
            capped = torch.zeros_like(ratio)
    metrics = {
        "dppo_maskfrac": bad.float(),
        "dppo_div": div.float(),
        "dppo_ratio": ratio.float(),
        "dppo_capfrac": capped,
    }
    return keep * weight, metrics


def dppo_tis_function(
    args: Any,
    *,
    pg_loss,
    train_log_probs: list,
    rollout_log_probs: list,
    loss_masks: list,
    **kwargs: Any,
):
    """Miles --custom-tis-function-path hook (same contract as vanilla_tis_function)."""
    import torch

    try:
        from miles.backends.training_utils.parallel import get_parallel_state
        cp = get_parallel_state().cp.size
    except Exception:  # noqa: BLE001  (outside Miles: tests)
        cp = 1
    assert cp == 1, "mho.train.dppo supports context-parallel size 1 only"
    if args is not None:
        assert not args.use_rollout_logprobs, "DPPO hook expects trainer log-probs as the old policy"
        assert args.skip_actor_forward_only, "DPPO hook needs --skip-actor-forward-only (ratio 1)"

    pi_lp = torch.cat(train_log_probs, dim=0)
    mu_lp = torch.cat(rollout_log_probs, dim=0)
    w, metrics = dppo_weights(
        pg_loss, pi_lp, mu_lp,
        delta=_env_float("MHO_DPPO_DELTA", 0.1),
        cap=_env_float("MHO_DPPO_CAP", 10.0),
        kind=os.environ.get("MHO_DPPO_DIVERGENCE", "tv").strip() or "tv",
    )
    return pg_loss * w, loss_masks, metrics
