"""harnessforge.train.losses.dppo: TMax's DPPO as a miles-tinker custom loss (mask, cap, gradient, metrics) and
parity with the Miles-path port validated against TMax (src/mho/train/dppo.py at the repo root)."""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from harnessforge.train.losses import dppo  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MHO_DPPO = REPO / "src" / "mho" / "train" / "dppo.py"
L = math.log
# mu = 0.5 for every token; pi chosen for ratio above/below 1 and |mu - pi| above/below 0.1
PI = {"far_up": 0.7, "near_up": 0.55, "far_down": 0.3, "near_down": 0.45}


def run(pi_lp, mu_lp, adv, config=None, mask=None):
    lp = torch.tensor(pi_lp, dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(mu_lp, dtype=torch.float64)
    a = torch.tensor(adv, dtype=torch.float64)
    m = torch.ones_like(a) if mask is None else torch.tensor(mask, dtype=torch.float64)
    per_token, metrics = dppo.dppo_loss(logprobs=lp, sampling_logprobs=mu, advantages=a, loss_mask=m,
                                        config=config or {})
    per_token.sum().backward()
    return per_token, {k: float(v) for k, v in metrics.items()}, lp.grad


@pytest.mark.parametrize(("adv", "pi", "masked"), [
    (1.0, "far_up", True),       # A > 0, ratio > 1, D = 0.2: pushes further up -> masked
    (1.0, "near_up", False),     # inside the trust region
    (1.0, "far_down", False),    # A > 0 pulls back toward mu
    (1.0, "near_down", False),
    (-1.0, "far_down", True),    # A < 0, ratio < 1, D = 0.2: pushes further down -> masked
    (-1.0, "near_down", False),
    (-1.0, "far_up", False),     # A < 0 pulls back toward mu
    (-1.0, "near_up", False),
])
def test_mask_and_gradient(adv, pi, masked):
    per_token, metrics, grad = run([L(PI[pi])], [L(0.5)], [adv], {"threshold": 0.1, "ratio_cap": 10.0})
    ratio = PI[pi] / 0.5
    assert grad[0].item() == pytest.approx(0.0 if masked else -adv * ratio)
    assert per_token[0].item() == pytest.approx(0.0 if masked else -adv * ratio * L(PI[pi]))
    assert metrics["dppo_tokens"] == 1 and metrics["dppo_masked_tokens"] == (1 if masked else 0)
    assert metrics["dppo_divergence"] == pytest.approx(abs(PI[pi] - 0.5))


def test_ratio_cap():
    """A < 0 with ratio 40 pulls back (kept); its weight is capped at 10. ratio_cap <= 0: no cap."""
    _, metrics, grad = run([L(0.4)], [L(0.01)], [-0.5], {"ratio_cap": 10.0})
    assert grad[0].item() == pytest.approx(5.0)
    assert metrics["dppo_capped_tokens"] == 1 and metrics["dppo_masked_tokens"] == 0
    _, _, grad = run([L(0.4)], [L(0.01)], [-0.5], {"ratio_cap": 0.0})
    assert grad[0].item() == pytest.approx(20.0)


def test_config():
    assert dppo.parse_config({}) == ("tv", 0.1, 10.0)                      # TMax 4B
    assert dppo.parse_config({"divergence": 1.0, "threshold": 0.05}) == ("kl", 0.05, 10.0)
    assert dppo.parse_config({"divergence": "TV"})[0] == "tv"
    with pytest.raises(ValueError, match="unknown keys"):
        dppo.parse_config({"clip_low_threshold": 0.8})
    with pytest.raises(ValueError, match="divergence"):
        dppo.parse_config({"divergence": 2.0})


def test_zero_advantage_and_masked_out_tokens_are_not_counted():
    _, metrics, grad = run([L(0.7), L(0.7), L(0.7)], [L(0.5)] * 3, [0.0, 1.0, -1.0], mask=[1, 1, 0])
    assert metrics["dppo_tokens"] == 1 and metrics["dppo_masked_tokens"] == 1
    assert grad[0].item() == 0.0


def test_gradient_on_random_tokens():
    g = torch.Generator().manual_seed(0)
    n = 512
    mu = torch.log(torch.rand(n, generator=g, dtype=torch.float64).clamp(1e-4, 1))
    pi = (mu + 0.6 * torch.randn(n, generator=g, dtype=torch.float64)).clamp(max=0.0)
    adv = torch.randn(n, generator=g, dtype=torch.float64)
    adv[::7] = 0.0
    _, metrics, grad = run(pi.tolist(), mu.tolist(), adv.tolist())
    ratio = torch.exp(pi - mu)
    div = (pi.exp() - mu.exp()).abs()
    masked = (div > 0.1) & (((adv > 0) & (ratio > 1)) | ((adv < 0) & (ratio < 1)))
    expected = torch.where(masked, torch.zeros_like(adv), -adv * ratio.clamp(max=10.0))
    torch.testing.assert_close(grad, expected, rtol=1e-5, atol=1e-6)   # weights in fp32
    active = adv != 0
    assert 0 < metrics["dppo_masked_tokens"] == float((masked & active).sum()) < metrics["dppo_tokens"]


@pytest.fixture(scope="module")
def mho_dppo():
    if not MHO_DPPO.is_file():
        pytest.skip(f"{MHO_DPPO} not in this checkout")
    spec = importlib.util.spec_from_file_location("mho_train_dppo_reference", MHO_DPPO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(("kind", "delta", "cap"), [("tv", 0.1, 10.0), ("kl", 0.05, 10.0), ("tv", 0.02, 0.0)])
def test_parity_with_the_validated_port(mho_dppo, kind, delta, cap):
    """mho's hook multiplies pg_loss = -A (ratio 1, carrying grad log pi) by its weight: the same per-token
    weight and mask as ours, and so the same gradient -A * weight."""
    g = torch.Generator().manual_seed(1)
    n = 4096
    mu = torch.log(torch.rand(n, generator=g).clamp(1e-6, 1))
    pi = (mu + torch.randn(n, generator=g)).clamp(max=0.0)
    adv = torch.randn(n, generator=g)
    ref_w, ref_m = mho_dppo.dppo_weights(-adv, pi, mu, delta=delta, cap=cap, kind=kind)
    w, tok = dppo.dppo_weights(pi, mu, adv, threshold=delta, ratio_cap=cap, kind=kind)
    torch.testing.assert_close(w, ref_w)
    assert torch.equal(tok["masked"].float(), ref_m["dppo_maskfrac"]) and tok["masked"].any()
    config = {"divergence": kind, "threshold": delta, "ratio_cap": cap}
    _, _, grad = run(pi.tolist(), mu.tolist(), adv.tolist(), config)
    torch.testing.assert_close(grad.float(), -adv * ref_w, rtol=1e-4, atol=1e-5)


def test_imports_only_torch():
    """The trainer imports the module from hf_src: nothing beyond torch and the stdlib may load with it."""
    code = ("import sys, torch; before = set(sys.modules); import harnessforge.train.losses.dppo; "
            "new = {m.split('.')[0] for m in set(sys.modules) - before}; "
            "bad = sorted(new - {'harnessforge'} - set(sys.stdlib_module_names)); "
            "assert not [m for m in bad if not m.startswith('_')], bad")
    src = Path(dppo.__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-c", code], check=True, env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"})
