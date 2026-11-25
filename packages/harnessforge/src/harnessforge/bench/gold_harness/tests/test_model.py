"""Model: no-Claude message prep + action parsing (no network)."""

from types import SimpleNamespace

from micro_swe.models import Model
from tests.conftest import make_tool_call


def _model():
    return Model(model_name="litellm_proxy/x", cost_tracking="ignore_errors")


def test_format_message_is_passthrough():
    # multimodal removed -> format_message just returns its kwargs
    m = _model()
    assert m.format_message(role="user", content="hi") == {"role": "user", "content": "hi"}


def test_prepare_messages_strips_extra_only():
    m = _model()
    prepared = m._prepare_messages_for_api([{"role": "user", "content": "hi", "extra": {"k": 1}}])
    assert prepared == [{"role": "user", "content": "hi"}]


def test_parse_actions_from_response():
    m = _model()
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[make_tool_call()]), finish_reason="tool_calls")]
    )
    assert m._parse_actions(resp) == [{"command": "ls", "tool_call_id": "c1"}]


def test_format_observation_messages_via_model():
    m = _model()
    message = {"extra": {"actions": [{"command": "ls", "tool_call_id": "c1"}]}}
    outputs = [{"output": "out", "returncode": 0, "exception_info": ""}]
    msgs = m.format_observation_messages(message, outputs)
    assert msgs[0]["role"] == "tool" and msgs[0]["tool_call_id"] == "c1"


def test_serialize_shape():
    info = _model().serialize()["info"]["config"]
    assert "model" in info and info["model_type"].endswith("Model")
