"""Trajectory helpers for sigma checks. Stdlib only: this file ships into every staged task's
tests/ and runs inside the rollout sandbox (any python3 >= 3.8).

A sigma check is a function the model proposer writes in model_specs/<name>.py:

    from sigma_checklib import *

    def ran_code_before_answering(traj: dict) -> bool | float:
        ...

`traj` is the parsed agent trajectory (mini-swe-agent: {"messages": [...], "info": {...}}).
Return a bool, or a float in [0, 1] for partial credit. The helpers below give the usual views
of it so a check doesn't have to re-parse the message format.
"""
from __future__ import annotations

import json

__all__ = ["messages", "assistant_messages", "assistant_texts", "reasoning_texts", "commands",
           "tool_results", "n_steps", "task_prompt"]


def messages(traj) -> list:
    """All messages, in order (system, user, assistant, tool)."""
    m = traj.get("messages") if isinstance(traj, dict) else traj
    return [x for x in m if isinstance(x, dict)] if isinstance(m, list) else []


def assistant_messages(traj) -> list:
    return [m for m in messages(traj) if m.get("role") == "assistant"]


def assistant_texts(traj) -> list:
    """The visible text of each assistant turn (not the reasoning, not the tool call)."""
    return [str(m.get("content") or "") for m in assistant_messages(traj)]


def reasoning_texts(traj) -> list:
    """The reasoning ("thinking") text of each assistant turn that has one."""
    return [str(m["reasoning_content"]) for m in assistant_messages(traj) if m.get("reasoning_content")]


def _calls(m: dict) -> list:
    tc = m.get("tool_calls")
    if isinstance(tc, str):
        try:
            tc = json.loads(tc)
        except ValueError:
            return []
    return tc if isinstance(tc, list) else []


def commands(traj) -> list:
    """Every shell command the agent issued, in order."""
    out = []
    for m in assistant_messages(traj):
        for c in _calls(m):
            args = ((c or {}).get("function") or {}).get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"command": args}
            if isinstance(args, dict) and args.get("command") is not None:
                out.append(str(args["command"]))
    return out


def tool_results(traj) -> list:
    """Each command's observation, in order: {"returncode": int | None, "output": str}."""
    out = []
    for m in messages(traj):
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        rc, text = None, content if isinstance(content, str) else json.dumps(content)
        try:
            d = json.loads(content) if isinstance(content, str) else content
        except ValueError:
            d = None
        if isinstance(d, dict):
            rc, text = d.get("returncode"), str(d.get("output", ""))
        extra = m.get("extra")
        if rc is None and isinstance(extra, dict):
            rc = extra.get("returncode")
        out.append({"returncode": rc, "output": text})
    return out


def n_steps(traj) -> int:
    """Number of assistant turns."""
    return len(assistant_messages(traj))


def task_prompt(traj) -> str:
    """The first user message: the task as the agent saw it."""
    for m in messages(traj):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""
