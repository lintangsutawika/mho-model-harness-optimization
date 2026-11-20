"""DPPO (divergence-based trust region; arXiv 2602.04879) as TMax 4B trains with it (allenai/tmax-4b: "Loss fn:
DPPO", "Divergence: binary TV", "TV threshold: 0.1"), as a miles-tinker custom loss.

Reference: hamishivi/tmax open-instruct `grpo_utils.py` (compute_binary_divergence, compute_dppo_mask,
compute_dppo_policy_loss), flags --loss_fn dppo --dppo_divergence_type tv --dppo_divergence_threshold 0.1
--dppo_ratio_cap 10. Ported from mho-model-harness-optimization `src/mho/train/dppo.py` (the Miles-path port that
was validated against TMax); tests/test_train_dppo.py checks parity with it.

Per token, with mu = the sampler's probability of the sampled token (Tinker `loss_fn_inputs["logprobs"]`) and
pi = the trainer's:
    ratio = pi / mu,  D = |mu - pi|  (binary TV; or binary KL)
    masked if D > threshold AND the update pushes further away (A > 0 & ratio > 1, or A < 0 & ratio < 1)
    loss  = -A * min(ratio, cap).detach() * log pi        (0 on masked tokens)
so the gradient is -A * min(ratio, cap) * grad log pi on kept tokens and 0 on masked ones.

Run config: gateway.custom_losses {"dppo": "harnessforge.train.losses.dppo:dppo_loss"}, train.loss_fn "dppo",
train.loss_fn_config {threshold, ratio_cap, divergence}. Tinker's SDK sends loss_fn_config as floats, so the
divergence is a number there: 0 = tv (default), 1 = kl; the strings "tv"/"kl" also work.

Metrics per datum (summed per request by miles-tinker; platoon-dppo.patch turns them into optim/dppo_*), over the
tokens the loss trains (A != 0, inside the loss mask): dppo_tokens, dppo_masked_tokens, dppo_capped_tokens,
dppo_divergence (sum of D).

Imports only torch.
"""
from __future__ import annotations

from typing import Any

import torch

DEFAULTS: dict[str, Any] = {"divergence": "tv", "threshold": 0.1, "ratio_cap": 10.0}
DIVERGENCE_CODES = {0.0: "tv", 1.0: "kl"}
LOG_RATIO_LIMIT = 20.0
METRICS = ("dppo_tokens", "dppo_masked_tokens", "dppo_capped_tokens", "dppo_divergence")


def parse_config(config: dict[str, Any] | None) -> tuple[str, float, float]:
    """(divergence, threshold, ratio_cap) of a loss_fn_config; ratio_cap <= 0 means no cap. Unknown keys are
    refused (a typo would otherwise train with a default)."""
    config = dict(config or {})
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"dppo loss_fn_config: unknown keys {sorted(unknown)} (takes {sorted(DEFAULTS)})")
    div = config.get("divergence", DEFAULTS["divergence"])
    kind = div.strip().lower() if isinstance(div, str) else DIVERGENCE_CODES.get(float(div))
    if kind not in ("tv", "kl"):
        raise ValueError(f"dppo loss_fn_config: divergence must be tv (0) or kl (1), got {div!r}")
    return kind, float(config.get("threshold", DEFAULTS["threshold"])), float(config.get("ratio_cap", DEFAULTS["ratio_cap"]))


def binary_divergence(mu_lp: torch.Tensor, pi_lp: torch.Tensor, kind: str) -> torch.Tensor:
    """Divergence between the sampler's and the trainer's Bernoulli on the sampled token."""
    mu = torch.exp(mu_lp.clamp(min=-30.0, max=0.0))
    pi = torch.exp(pi_lp.clamp(min=-30.0, max=0.0))
    if kind == "tv":
        return (mu - pi).abs()
    if kind == "kl":
        eps = 1e-9
        m, p = mu.clamp(eps, 1 - eps), pi.clamp(eps, 1 - eps)
        return m * (m.log() - p.log()) + (1 - m) * ((1 - m).log() - (1 - p).log())
    raise ValueError(f"dppo divergence must be tv or kl, got {kind!r}")


def dppo_weights(pi_lp: torch.Tensor, mu_lp: torch.Tensor, advantages: torch.Tensor, *, threshold: float,
                 ratio_cap: float, kind: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Detached per-token weight mask * min(pi / mu, cap) (fp32), and per-token masked / capped / divergence."""
    with torch.no_grad():
        pi_lp, mu_lp, adv = pi_lp.detach().float(), mu_lp.detach().float(), advantages.detach().float()
        ratio = torch.exp((pi_lp - mu_lp).clamp(-LOG_RATIO_LIMIT, LOG_RATIO_LIMIT))
        div = binary_divergence(mu_lp, pi_lp, kind)
        outside = div > threshold
        masked = ((adv > 0) & (ratio > 1.0) & outside) | ((adv < 0) & (ratio < 1.0) & outside)
        if ratio_cap > 0:
            weight, capped = ratio.clamp(max=ratio_cap), ratio > ratio_cap
        else:
            weight, capped = ratio, torch.zeros_like(masked)
        weight = weight * (~masked).float()
    return weight, {"masked": masked, "capped": capped, "divergence": div}


def dppo_loss(*, logprobs: torch.Tensor, sampling_logprobs: torch.Tensor, advantages: torch.Tensor,
              loss_mask: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """miles-tinker custom loss: per-token -A * mask * min(pi/mu, cap).detach() * log pi, and per-datum metrics."""
    kind, threshold, ratio_cap = parse_config(config)
    weight, tok = dppo_weights(logprobs, sampling_logprobs, advantages,
                               threshold=threshold, ratio_cap=ratio_cap, kind=kind)
    per_token = -advantages * weight.to(logprobs.dtype) * logprobs
    with torch.no_grad():
        active = (advantages != 0) & (loss_mask > 0)
        metrics = {
            "dppo_tokens": active.sum().float(),
            "dppo_masked_tokens": (tok["masked"] & active).sum().float(),
            "dppo_capped_tokens": (tok["capped"] & active).sum().float(),
            "dppo_divergence": (tok["divergence"] * active).sum(),
        }
    return per_token, metrics
