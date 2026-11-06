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
    assert objectives.rollout_temperature("grpo") == 0.8
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
