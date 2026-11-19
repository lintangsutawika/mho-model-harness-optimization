"""Exceptions hierarchy, serialize merge, and the console entrypoint."""

import subprocess
import sys

from micro_swe import exceptions as ex
from micro_swe.utils.serialize import recursive_merge


def test_exception_hierarchy():
    assert issubclass(ex.FormatError, ex.InterruptAgentFlow)
    assert issubclass(ex.LimitsExceeded, ex.InterruptAgentFlow)
    assert issubclass(ex.TimeExceeded, ex.LimitsExceeded)
    assert issubclass(ex.Submitted, ex.InterruptAgentFlow)


def test_interrupt_carries_messages():
    e = ex.FormatError({"role": "user", "content": "x"})
    assert e.messages[0]["content"] == "x"


def test_recursive_merge_deep():
    assert recursive_merge({"a": {"x": 1}}, {"a": {"y": 2}}, {"b": 3}) == {"a": {"x": 1, "y": 2}, "b": 3}


def test_recursive_merge_later_wins():
    assert recursive_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_entrypoint_help_runs():
    """The `mini-swe-agent` console entrypoint imports & runs (--help)."""
    r = subprocess.run(
        [sys.executable, "-m", "micro_swe", "--help"], capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 0
    assert "--model" in r.stdout
