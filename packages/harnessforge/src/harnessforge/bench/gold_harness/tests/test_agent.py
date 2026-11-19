"""Agent loop with fakes: runs steps, honors limits, saves a trajectory."""

import json

from micro_swe.agents import Agent
from micro_swe.exceptions import Submitted
from tests.conftest import FakeEnv, FakeModel

# Minimal templates that only reference {{task}} (provided by run()).
_TPL = {"system_template": "You are an agent.", "instance_template": "Task: {{task}}"}


def test_run_stops_at_step_limit(fake_model, fake_env):
    agent = Agent(fake_model, fake_env, step_limit=2, **_TPL)
    result = agent.run("do it")
    assert result["exit_status"] == "LimitsExceeded"
    # 2 executed steps (system+user + 2 assistant + 2 tool + 1 exit)
    assert fake_model.calls == 2
    assert agent.messages[-1]["role"] == "exit"


def test_run_observes_command_output():
    agent = Agent(FakeModel(), FakeEnv(), step_limit=1, **_TPL)
    agent.run("x")
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_msgs and "echo hi" in tool_msgs[0]["content"]


def test_submitted_exits_cleanly():
    # env raises Submitted -> caught as InterruptAgentFlow -> exit
    submit = Submitted({"role": "exit", "content": "done", "extra": {"exit_status": "Submitted", "submission": "ok"}})
    agent = Agent(FakeModel(), FakeEnv(raise_exc=submit), step_limit=10, **_TPL)
    result = agent.run("x")
    assert result["exit_status"] == "Submitted"


def test_trajectory_saved(tmp_path):
    out = tmp_path / "traj.json"
    agent = Agent(FakeModel(), FakeEnv(), step_limit=1, output_path=out, **_TPL)
    agent.run("x")
    assert out.exists()
    data = json.loads(out.read_text())
    assert "messages" in data and "info" in data
    assert data["info"]["config"]["agent"]["step_limit"] == 1
