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


# --- LLM timeout, budget note, nudges (fake OpenHands SDK) ---------------------------------------
import types

from mho.optimizer import agent as agent_mod


def test_llm_timeout_parsed():
    assert _llm_limits({"OPTIMIZER_LLM_TIMEOUT_S": "900"}) == {"timeout": 900}


def test_budget_note():
    assert agent_mod._budget_note({}) == ""
    note = agent_mod._budget_note({"OPTIMIZER_STEP_BUDGET": "150"})
    assert "about 150 tool calls" in note and "about 50 on analysis" in note


class _FakeConversation:
    instances: list = []
    script: list = []          # per run(): None = return, Exception = raise

    def __init__(self, agent, workspace, **kw):
        self.kw, self.messages, self.runs = kw, [], 0
        _FakeConversation.instances.append(self)

    def send_message(self, m):
        self.messages.append(m)

    def run(self):
        step = _FakeConversation.script[self.runs]
        self.runs += 1
        if isinstance(step, Exception):
            raise step


@pytest.fixture
def fake_sdk(monkeypatch, tmp_path):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    class _Any:
        name = "x"

        def __init__(self, *a, **k):
            self.kw = k

        def model_copy(self, update=None):
            return self

        @classmethod
        def load(cls, *a, **k):
            return cls()

    mod("openhands")
    mod("openhands.sdk", LLM=_Any, Agent=_Any, AgentContext=_Any, Conversation=_FakeConversation,
        LocalWorkspace=_Any, Tool=_Any)
    mod("openhands.sdk.skills"); mod("openhands.sdk.skills.skill", Skill=_Any)
    mod("openhands.tools")
    for sub, cls in (("file_editor", "FileEditorTool"), ("task_tracker", "TaskTrackerTool"),
                     ("terminal", "TerminalTool")):
        mod(f"openhands.tools.{sub}", **{cls: _Any})
    for k in ("OPTIMIZER_NUDGES", "OPTIMIZER_MAX_ITERATIONS", "OPTIMIZER_CONDENSE_AT_TOKENS",
              "OPTIMIZER_STEP_BUDGET"):
        monkeypatch.delenv(k, raising=False)
    _FakeConversation.instances.clear()
    (tmp_path / "SKILL.md").write_text("x")
    return tmp_path


def test_nudge_resumes_after_error_until_done(fake_sdk, monkeypatch):
    monkeypatch.setenv("OPTIMIZER_NUDGES", "2")
    monkeypatch.setenv("OPTIMIZER_MAX_ITERATIONS", "200")
    _FakeConversation.script = [RuntimeError("LLM timeout"), None]
    state = {"done": False}

    def unfinished():
        if state["done"]:
            return None
        state["done"] = True
        return "candidate c is still identical to its parent", "/w/c/harness"

    agent_mod._run_openhands(fake_sdk, "BRIEF", fake_sdk, unfinished=unfinished)
    conv = _FakeConversation.instances[0]
    assert conv.kw == {"max_iteration_per_run": 200}
    assert conv.runs == 2 and conv.messages[0] == "BRIEF"
    assert "still identical" in conv.messages[1] and "/w/c/harness" in conv.messages[1]


def test_no_nudges_reraises_like_before(fake_sdk):
    _FakeConversation.script = [RuntimeError("boom")]
    with pytest.raises(RuntimeError):
        agent_mod._run_openhands(fake_sdk, "BRIEF", fake_sdk, unfinished=lambda: ("x", "y"))
    assert _FakeConversation.instances[0].runs == 1


def test_nudges_are_bounded(fake_sdk, monkeypatch):
    monkeypatch.setenv("OPTIMIZER_NUDGES", "1")
    _FakeConversation.script = [None, None, None]
    agent_mod._run_openhands(fake_sdk, "BRIEF", fake_sdk, unfinished=lambda: ("still unchanged", "/w"))
    assert _FakeConversation.instances[0].runs == 2   # first run + 1 nudge, then give up


def test_budget_note_reaches_the_brief(fake_sdk, monkeypatch):
    monkeypatch.setenv("OPTIMIZER_STEP_BUDGET", "90")
    _FakeConversation.script = [None]
    agent_mod._run_openhands(fake_sdk, "BRIEF", fake_sdk)
    assert "about 90 tool calls" in _FakeConversation.instances[0].messages[0]
