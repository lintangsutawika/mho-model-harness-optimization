"""mho.train.rsft (the RSFT filters Miles calls) and mho.train.objectives (the launcher flags)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mho.train import objectives, rsft  # noqa: E402


class S:
    def __init__(self, r):
        self.r, self.remove_sample = r, False

    def get_reward_value(self, args):
        return self.r


def test_keep_successful_removes_failures():
    a = SimpleNamespace(global_batch_size=2)
    ok1, ok2, ok3 = S(1.0), S(1.0), S(1.0)
    data = [[ok1, S(0.0), S(None)], [S(0.0), S(0.0)], [ok2, [ok3, S(0.0)]]]
    rsft.keep_successful(a, data)
    # the all-failed group is gone; a multi-segment trajectory with a failed segment is dropped
    assert data == [[ok1], [ok2]]
    assert not any(s.remove_sample for g in data for s in g)


def test_too_few_successes_are_padded_with_masked_failures():
    a = SimpleNamespace(global_batch_size=4)
    ok = S(1.0)
    data = [[ok, S(0.0), S(0.0)], [S(0.0), S(0.0)]]
    rsft.keep_successful(a, data)
    flat = [s for g in data for s in g]
    assert len(flat) == 4 and flat[0] is ok and not ok.remove_sample
    assert all(s.remove_sample for s in flat[1:])        # fill only: no gradient


def test_threshold_is_strict_as_whale():
    assert not rsft.is_success(None, S(0.5)) and rsft.is_success(None, S(0.51))


def test_any_success_keeps_all_correct_groups(monkeypatch):
    assert rsft.any_success(None, [S(1.0)] * 8)             # GRPO would drop this (no variance)
    assert not rsft.any_success(None, [S(0.0)] * 8)
    monkeypatch.setenv("MHO_RSFT_THRESHOLD", "0.3")
    assert rsft.any_success(None, [S(0.0), S(0.4)])
    assert not rsft.any_success(None, [S(0.3)])                  # strictly above


def test_reward_dicts_and_plain_rewards():
    class D:
        reward, remove_sample = {"reward": 1.0}, False

        def get_reward_value(self, args):
            raise KeyError("no rm key")
    assert rsft.is_success(None, D())


AGENT = ("--custom-rm-path generate.reward_func --dynamic-sampling-filter-path "
         f"{objectives.GRPO_GROUP_FILTER} --tito-model qwen35 ")


def test_objective_flags():
    g = objectives.objective_train_args("grpo")
    assert "--advantage-estimator grpo" in g and "--use-kl-loss" in g
    r = objectives.objective_train_args("rsft")
    assert "--loss-type sft_loss" in r and "--disable-compute-advantages-and-returns" in r
    assert "--calculate-per-token-loss" in r and "kl" not in r
    assert f"--rollout-sample-filter-path {objectives.RSFT_SAMPLE_FILTER}" in r
    with pytest.raises(ValueError):
        objectives.objective_train_args("ppo")


def test_rollout_temperature(monkeypatch):
    monkeypatch.delenv("MHO_ROLLOUT_TEMPERATURE", raising=False)
    assert objectives.rollout_temperature("rsft") == 1.0
    assert objectives.rollout_temperature("grpo") == 1.0
    monkeypatch.setenv("MHO_ROLLOUT_TEMPERATURE", "0.7")
    assert objectives.rollout_temperature("rsft") == 0.7


def test_group_filter_swap(monkeypatch):
    monkeypatch.delenv("MHO_RSFT_GROUP_FILTER", raising=False)
    assert objectives.objective_group_filter(AGENT, "grpo") == AGENT
    out = objectives.objective_group_filter(AGENT, "rsft")             # WHALE: keep every group
    assert "--dynamic-sampling-filter-path" not in out and "--tito-model qwen35" in out
    monkeypatch.setenv("MHO_RSFT_GROUP_FILTER", "any_success")
    out = objectives.objective_group_filter(AGENT, "rsft")
    assert objectives.GRPO_GROUP_FILTER not in out and objectives.RSFT_GROUP_FILTER in out


ASYNC_AGENT = ("--custom-rm-path generate.reward_func --rollout-function-path generate.RolloutFn "
         "--dynamic-sampling-filter-path x.y ")


def test_async_args_default_on(monkeypatch):
    for v in ("MHO_FULLY_ASYNC", "MHO_ASYNC_MAX_STALENESS", "MHO_ASYNC_MAX_CONCURRENT"):
        monkeypatch.delenv(v, raising=False)
    g = objectives.async_train_args(ASYNC_AGENT, "grpo")
    assert "--rollout-function-path" not in g and "--fully-async " in g
    assert "--max-weight-staleness 3 " in g and "--use-tis " in g
    assert "--async-max-concurrent-samples 128 " in g and "--dynamic-sampling-filter-path x.y " in g
    assert "--use-tis" not in objectives.async_train_args(ASYNC_AGENT, "rsft")


def test_async_args_knobs_and_off(monkeypatch):
    monkeypatch.setenv("MHO_ASYNC_MAX_STALENESS", "1")
    monkeypatch.setenv("MHO_ASYNC_MAX_CONCURRENT", "256")
    g = objectives.async_train_args(ASYNC_AGENT, "grpo")
    assert "--max-weight-staleness 1 " in g and "--async-max-concurrent-samples 256 " in g
    monkeypatch.setenv("MHO_FULLY_ASYNC", "0")
    assert objectives.async_train_args(ASYNC_AGENT, "grpo") == ASYNC_AGENT


def test_dr_grpo_flags(monkeypatch):
    monkeypatch.delenv("MHO_DR_GRPO", raising=False)
    g = objectives.objective_train_args("grpo")
    assert "--disable-grpo-std-normalization " in g and objectives.DRGRPO_REDUCER in g
    assert "--disable-grpo-std-normalization" not in objectives.objective_train_args("rsft")
    monkeypatch.setenv("MHO_DR_GRPO", "0")
    g = objectives.objective_train_args("grpo")
    assert "--disable-grpo-std-normalization" not in g and objectives.DRGRPO_REDUCER not in g


def test_dr_grpo_reducer_divides_token_sums_by_constant(monkeypatch):
    torch = pytest.importorskip("torch")
    from mho.train import drgrpo
    monkeypatch.setenv("MHO_DRGRPO_DIVISOR", "10")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])          # two trajectories: 3 + 2 tokens
    masks = [torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0, 1.0])]
    red = drgrpo.get_pg_loss_reducer(None, [3, 2], masks)
    assert float(red(x)) == pytest.approx((1 + 2) / 10 + (4 + 5) / 10)
