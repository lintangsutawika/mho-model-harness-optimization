"""Proposer condenser / LLM-limit env parsing (no OpenHands SDK needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mho.optimizer.agent import _condenser_config, _llm_limits


def test_unset_means_no_condenser_and_no_limits():
    assert _condenser_config({}) is None
    assert _llm_limits({}) == {}


def test_limits_parsed():
    env = {"OPTIMIZER_MAX_INPUT_TOKENS": "131072", "OPTIMIZER_MAX_OUTPUT_TOKENS": " 16384 "}
    assert _llm_limits(env) == {"max_input_tokens": 131072, "max_output_tokens": 16384}


def test_condenser_defaults():
    assert _condenser_config({"OPTIMIZER_CONDENSE_AT_TOKENS": "90000"}) == {
        "max_tokens": 90000, "max_size": 240, "keep_first": 2}


def test_condenser_overrides():
    env = {"OPTIMIZER_CONDENSE_AT_TOKENS": "80000", "OPTIMIZER_CONDENSE_MAX_EVENTS": "400",
           "OPTIMIZER_CONDENSE_KEEP_FIRST": "4"}
    assert _condenser_config(env) == {"max_tokens": 80000, "max_size": 400, "keep_first": 4}


@pytest.mark.parametrize("env", [
    {"OPTIMIZER_CONDENSE_AT_TOKENS": "0"},
    {"OPTIMIZER_CONDENSE_AT_TOKENS": "abc"},
    {"OPTIMIZER_CONDENSE_AT_TOKENS": "140000", "OPTIMIZER_MAX_INPUT_TOKENS": "131072"},
    {"OPTIMIZER_CONDENSE_AT_TOKENS": "90000", "OPTIMIZER_CONDENSE_MAX_EVENTS": "8",
     "OPTIMIZER_CONDENSE_KEEP_FIRST": "3"},
])
def test_bad_values_fail_loudly(env):
    with pytest.raises(ValueError):
        _condenser_config(env)
