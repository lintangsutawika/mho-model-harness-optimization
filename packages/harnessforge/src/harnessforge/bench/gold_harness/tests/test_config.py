"""Config resolution is self-contained: `-c mini` -> micro's own mini.yaml."""

import micro_swe.config as cfgmod
from micro_swe.config import get_config_from_spec
from micro_swe.__main__ import _merge_configs


def test_builtin_config_dir_is_micro_owned():
    # The resolver must point at micro's package, not mini's.
    assert cfgmod.builtin_config_dir.name == "config"
    assert "micro_swe" in str(cfgmod.builtin_config_dir)
    assert (cfgmod.builtin_config_dir / "mini.yaml").exists()


def test_dash_c_mini_resolves_with_prompts():
    cfg = get_config_from_spec("mini")
    assert "system_template" in cfg["agent"]
    assert "instance_template" in cfg["agent"]


def test_key_value_spec_becomes_nested_dict():
    assert get_config_from_spec("model.model_kwargs.temperature=0.6") == {
        "model": {"model_kwargs": {"temperature": 0.6}}
    }


def test_merge_configs_left_to_right():
    cfg = _merge_configs(["mini", "agent.step_limit=42"])
    assert cfg["agent"]["step_limit"] == 42
    assert "system_template" in cfg["agent"]  # base preserved


def test_empty_specs_default_to_mini():
    assert "agent" in _merge_configs([])
