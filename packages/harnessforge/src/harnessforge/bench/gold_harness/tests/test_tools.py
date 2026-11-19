"""The bash tool: action parsing + observation formatting."""

import pytest

from micro_swe.exceptions import FormatError
from micro_swe.tools import BASH_TOOL, format_toolcall_observation_messages, parse_toolcall_actions
from tests.conftest import make_tool_call

_FMT = "{{ error }}"


def test_bash_tool_schema():
    assert BASH_TOOL["function"]["name"] == "bash"
    assert "command" in BASH_TOOL["function"]["parameters"]["properties"]


def test_parse_valid_bash_call():
    actions = parse_toolcall_actions([make_tool_call()], format_error_template=_FMT)
    assert actions == [{"command": "ls", "tool_call_id": "c1"}]


def test_parse_no_tool_calls_raises():
    with pytest.raises(FormatError) as e:
        parse_toolcall_actions([], format_error_template=_FMT)
    assert e.value.messages[0]["extra"]["interrupt_type"] == "FormatError"


def test_parse_unknown_tool_raises():
    with pytest.raises(FormatError):
        parse_toolcall_actions([make_tool_call(name="python")], format_error_template=_FMT)


def test_parse_missing_command_raises():
    with pytest.raises(FormatError):
        parse_toolcall_actions([make_tool_call(arguments="{}")], format_error_template=_FMT)


def test_parse_bad_json_raises():
    with pytest.raises(FormatError):
        parse_toolcall_actions([make_tool_call(arguments="{not json")], format_error_template=_FMT)


def test_format_observation_tool_role_and_id():
    msgs = format_toolcall_observation_messages(
        actions=[{"command": "ls", "tool_call_id": "c1"}],
        outputs=[{"output": "file.txt", "returncode": 0, "exception_info": ""}],
        observation_template="<rc>{{output.returncode}}</rc>{{output.output}}",
    )
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "c1"
    assert "file.txt" in msgs[0]["content"]


def test_format_observation_pads_missing_outputs():
    # more actions than outputs -> padded with a "not executed" record
    msgs = format_toolcall_observation_messages(
        actions=[{"command": "a", "tool_call_id": "1"}, {"command": "b", "tool_call_id": "2"}],
        outputs=[{"output": "ok", "returncode": 0, "exception_info": ""}],
        observation_template="{{output.exception_info}}",
    )
    assert len(msgs) == 2
    assert "not executed" in msgs[1]["content"]
