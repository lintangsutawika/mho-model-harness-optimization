"""mho.train.dppo vs TMax's reference DPPO (hamishivi/tmax grpo_utils.py), via Miles' pg_loss path."""
import pytest

torch = pytest.importorskip("torch")

from mho.train import dppo, objectives


def _ref_loss(new_lp, mu_lp, adv, delta, cap):
    # verbatim semantics of compute_dppo_mask (tv) + compute_dppo_policy_loss
    ratio = torch.exp((new_lp - mu_lp).clamp(-20, 20))
    with torch.no_grad():
        div = (torch.exp(mu_lp) - torch.exp(new_lp)).abs()
        out = div > delta
        bad = ((adv > 0) & (ratio > 1) & out) | ((adv < 0) & (ratio < 1) & out)
        mask = (~bad).float()
    loss = -adv * ratio.clamp(max=cap).detach() * new_lp
    return (loss * mask).sum()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_tmax_reference_gradient(seed):
    g = torch.Generator().manual_seed(seed)
    n = 4000
    mu_lp = -torch.rand(n, generator=g) * 4            # rollout log-probs
    new0 = mu_lp + torch.randn(n, generator=g) * 1.5    # big drift: exercises mask and cap
    new0 = new0.clamp(max=0.0)
    adv = torch.randn(n, generator=g)
    adv[::7] = 0.0

    # Miles with --skip-actor-forward-only: old = new.detach(), so ratio = 1 carrying grad log pi
    new_m = new0.clone().requires_grad_(True)
    ratio = torch.exp(-(new_m.detach() - new_m))
    pg = torch.maximum(-ratio * adv, -ratio.clamp(1 - 1e9, 1 + 1e9) * adv)
    out, masks, metrics = dppo.dppo_tis_function(
        None, pg_loss=pg, train_log_probs=[new_m.detach()[:1000], new_m.detach()[1000:]],
        rollout_log_probs=[mu_lp[:1000], mu_lp[1000:]], loss_masks=["m"])
    out.sum().backward()

    new_r = new0.clone().requires_grad_(True)
    _ref_loss(new_r, mu_lp, adv, 0.1, 10.0).backward()

    assert masks == ["m"]
    assert 0.01 < metrics["dppo_maskfrac"].mean() < 0.9
    assert metrics["dppo_capfrac"].sum() > 0
    torch.testing.assert_close(new_m.grad, new_r.grad, rtol=1e-4, atol=1e-6)


def test_objective_flags(monkeypatch):
    monkeypatch.setenv("MHO_PG_LOSS", "dppo")
    a = objectives.objective_train_args("grpo")
    assert "--use-kl-loss" not in a and "--use-rollout-logprobs" not in a
    assert "--skip-actor-forward-only" in a
    assert f"--custom-tis-function-path {objectives.DPPO_TIS}" in a
    monkeypatch.setenv("MHO_PG_LOSS", "ppo")
    assert "--use-kl-loss" in objectives.objective_train_args("grpo")
    monkeypatch.setenv("MHO_PG_LOSS", "bogus")
    with pytest.raises(ValueError):
        objectives.objective_train_args("grpo")
