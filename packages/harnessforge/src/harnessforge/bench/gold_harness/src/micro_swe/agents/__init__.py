"""Agent scaffolds. `Agent` is the default plan/act/observe loop (agents/default.py) --
edit it or add sibling agents here to change the control flow."""

from micro_swe.agents.default import Agent, AgentConfig

__all__ = ["Agent", "AgentConfig"]
