"""Shared fakes for micro-swe-agent tests (no network / no real LLM)."""

from types import SimpleNamespace

import pytest


class FakeModel:
    """Minimal Model implementing the interface Agent uses."""

    def __init__(self, actions_per_turn=None):
        # each call to query() pops the next action list; empty -> a no-op command
        self._script = list(actions_per_turn or [[{"command": "echo hi", "tool_call_id": "t1"}]])
        self.calls = 0

    def format_message(self, **kwargs):
        return dict(kwargs)

    def query(self, messages):
        self.calls += 1
        actions = self._script.pop(0) if self._script else [{"command": "echo hi", "tool_call_id": "t1"}]
        return {"role": "assistant", "content": "ok", "extra": {"actions": actions, "cost": 0.0}}

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [
            {"role": "tool", "content": str(o.get("output", "")), "tool_call_id": a.get("tool_call_id"), "extra": {}}
            for a, o in zip(message.get("extra", {}).get("actions", []), outputs)
        ]

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {"info": {"config": {"model": {}, "model_type": "FakeModel"}}}


class FakeEnv:
    """Minimal Environment: echoes the command back as output."""

    def __init__(self, raise_exc=None):
        self._raise = raise_exc

    def execute(self, action, cwd="", *, timeout=None):
        if self._raise is not None:
            raise self._raise
        return {"output": action.get("command", ""), "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return {"system": "Linux", "release": "6", "version": "#1", "machine": "x86_64"}

    def serialize(self):
        return {}


def make_tool_call(name="bash", arguments='{"command": "ls"}', call_id="c1"):
    """Build a fake litellm tool_call object (duck-typed)."""
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments), id=call_id)


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def fake_env():
    return FakeEnv()
