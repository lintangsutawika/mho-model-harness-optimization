"""mho.train.rsft (the RSFT filters Miles calls) and mho.train.objectives (the launcher flags)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mho.train import objectives, rsft  # noqa: E402


class S:
    def __init__(self, r):
        self.r, self.remove_sample = r, False

    def get_reward_value(self, args):
        return self.r


def test_keep_successful_masks_failures_only():
    groups = [[S(1.0), S(0.0), S(None)], [S(0.0), [S(1.0), S(0.4)]]]
    rsft.keep_successful(None, groups)
    flat = [groups[0][0], groups[0][1], groups[0][2], groups[1][0], *groups[1][1]]
    assert [s.remove_sample for s in flat] == [False, True, True, True, False, True]


def test_any_success_keeps_all_correct_groups(monkeypatch):
    assert rsft.any_success(None, [S(1.0)] * 8)             # GRPO would drop this (no variance)
    assert not rsft.any_success(None, [S(0.0)] * 8)
    monkeypatch.setenv("MHO_RSFT_THRESHOLD", "0.3")
    assert rsft.any_success(None, [S(0.0), S(0.4)])


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
    assert f"--rollout-sample-filter-path {objectives.RSFT_SAMPLE_FILTER}" in r and "kl" not in r
    with pytest.raises(ValueError):
        objectives.objective_train_args("ppo")


def test_group_filter_swap(monkeypatch):
    assert objectives.objective_group_filter(AGENT, "grpo") == AGENT
    out = objectives.objective_group_filter(AGENT, "rsft")
    assert objectives.GRPO_GROUP_FILTER not in out and objectives.RSFT_GROUP_FILTER in out
    monkeypatch.setenv("MHO_RSFT_GROUP_FILTER", "none")
    out = objectives.objective_group_filter(AGENT, "rsft")
    assert "--dynamic-sampling-filter-path" not in out and "--tito-model qwen35" in out
