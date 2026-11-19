"""Run one probe scenario against one harness directory (executed in a fresh subprocess).

Standalone on purpose (no harnessforge import): it is started as
`python -I _probe_runner.py --harness DIR --scenario NAME --out FILE` from an empty temp dir.

The harness runs end to end through its own CLI (`micro_swe.__main__.main`, the same entrypoint
an eval runner shells out to) with its own config, agent, model wrapper and local environment.
Only the LLM is fake: `litellm` is replaced by a scripted stub that records every request
(messages + sampling kwargs) and answers from the scenario's script. Commands are real but
harmless (echo/printf/python3 -c/sleep/mkdir) and run in the temp dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import types
from pathlib import Path

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
BASE_MAX_TOKENS = 1000
MAX_CALLS = 200


def _big(marker: str, n: int) -> str:
    return f"python3 -c \"print('{marker}' + 'z' * {n})\""


# Each step: ("cmd", command) | ("unknown_tool",) | ("missing_cmd",) | ("length_cut", command)
# After the script runs out the model always submits (twice if the harness asks it to check).
SCENARIOS: dict[str, dict] = {
    "output_truncation": {"script": [
        ("cmd", "python3 -c \"print('HEADMARK'); [print('line %05d ' % i + 'y' * 50) for i in range(1000)]; "
                "print('TAILMARK')\""),
    ]},
    "timeout_guidance": {"script": [
        ("cmd", "echo hello"),
        ("cmd", "echo PARTIALMARK; sleep 20"),
    ], "config": ["environment.timeout=2"]},
    "persistent_cwd": {"script": [
        ("cmd", "mkdir -p probe_sub && cd probe_sub"),
        ("cmd", "pwd"),
    ]},
    "length_recovery": {"script": [
        ("cmd", "echo first"),
        ("length_cut", "echo recovered"),
        ("cmd", "echo after"),
    ]},
    "loop_warning": {"script": [("cmd", "echo SAME")] * 5 + [("cmd", "echo DIFFERENT")]},
    "format_streak_trim": {"script": [
        ("cmd", "echo a"), ("unknown_tool",), ("unknown_tool",), ("cmd", "echo b"),
    ]},
    "missing_command_repair": {"script": [
        ("cmd", "echo a"), ("missing_cmd",), ("cmd", "echo b"),
    ]},
    "submit_confirmation": {"script": [
        ("cmd", "echo work"), ("cmd", SUBMIT),
    ]},
    "context_trim": {"script": [("cmd", _big("OBS%02d" % k, 4000)) for k in range(1, 31)]},
    "raw_observation": {"script": [
        ("cmd", "python3 -c \"import sys; sys.stdout.write(chr(113) + chr(34) + 'uote' + chr(92) + 'back' "
                "+ chr(10) + 'line2' + chr(10))\""),
    ]},
}


# ----------------------------------------------------------------------------- fake litellm
class _ProbeAbort(KeyboardInterrupt):
    """Stops a runaway harness (listed in the harness's abort exceptions via KeyboardInterrupt)."""


class _Function:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, i, name, arguments):
        self.id, self.type, self.function = f"call_{i}", "function", _Function(name, arguments)

    def model_dump(self, **_):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


class _Message:
    def __init__(self, content, tool_calls):
        self.role, self.content, self.tool_calls = "assistant", content, tool_calls
        self.function_call = None
        self.reasoning_content = None
        self.provider_specific_fields = None

    def model_dump(self, **_):
        return {"role": "assistant", "content": self.content,
                "tool_calls": [t.model_dump() for t in self.tool_calls] if self.tool_calls else None}


class _Usage:
    def __init__(self, completion_tokens):
        self.prompt_tokens, self.completion_tokens = 100, completion_tokens
        self.total_tokens = 100 + completion_tokens

    def model_dump(self, **_):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}


class _Choice:
    def __init__(self, message, finish_reason):
        self.index, self.message, self.finish_reason = 0, message, finish_reason
        self.logprobs = None
        self.provider_specific_fields = None


class _Response:
    def __init__(self, i, content, tool_calls, finish_reason, completion_tokens=20):
        self.id, self.model, self.object = f"resp_{i}", "fake/probe", "chat.completion"
        self.choices = [_Choice(_Message(content, tool_calls), finish_reason)]
        self.usage = _Usage(completion_tokens)
        self.prompt_token_ids = None

    def model_dump(self, **_):
        c = self.choices[0]
        return {"id": self.id, "model": self.model, "object": self.object,
                "choices": [{"index": 0, "message": c.message.model_dump(), "finish_reason": c.finish_reason}],
                "usage": self.usage.model_dump()}

    def __getitem__(self, key):
        return self.model_dump()[key]

    def get(self, key, default=None):
        return self.model_dump().get(key, default)


class Scripted:
    def __init__(self, script):
        self.script = list(script)
        self.pos = 0
        self.calls: list[dict] = []

    def completion(self, model=None, messages=None, tools=None, **kwargs):
        i = len(self.calls)
        self.calls.append({"messages": json.loads(json.dumps(messages, default=str)),
                           "kwargs": json.loads(json.dumps(kwargs, default=str))})
        if i >= MAX_CALLS:
            raise _ProbeAbort("probe call budget exhausted")
        step = self.script[self.pos] if self.pos < len(self.script) else ("cmd", SUBMIT)
        self.pos += 1
        kind = step[0]
        if kind == "cmd":
            return _Response(i, f"step {i}", [_ToolCall(i, "bash", json.dumps({"command": step[1]}))], "tool_calls")
        if kind == "unknown_tool":
            return _Response(i, f"step {i}", [_ToolCall(i, "python", json.dumps({"code": "print(1)"}))], "tool_calls")
        if kind == "missing_cmd":
            return _Response(i, f"step {i}", [_ToolCall(i, "bash", "{}")], "tool_calls")
        if kind == "length_cut":
            budget = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 0
            if budget and int(budget) > BASE_MAX_TOKENS:
                return _Response(i, f"step {i}", [_ToolCall(i, "bash", json.dumps({"command": step[1]}))],
                                 "tool_calls")
            return _Response(i, "Let me think about this very carefully. " * 40, None, "length",
                             completion_tokens=BASE_MAX_TOKENS)
        raise ValueError(f"unknown step {step!r}")


def _fake_litellm(scripted: Scripted) -> types.ModuleType:
    mod = types.ModuleType("litellm")
    exc = types.ModuleType("litellm.exceptions")
    for name in ("UnsupportedParamsError", "NotFoundError", "PermissionDeniedError", "ContextWindowExceededError",
                 "AuthenticationError", "APIError", "APIConnectionError", "BadRequestError", "RateLimitError",
                 "Timeout", "InternalServerError", "ServiceUnavailableError", "ContentPolicyViolationError",
                 "BudgetExceededError", "OpenAIError"):
        setattr(exc, name, type(name, (Exception,), {}))
    mod.exceptions = exc
    for name in dir(exc):
        if not name.startswith("_"):
            setattr(mod, name, getattr(exc, name))
    mod.completion = scripted.completion
    mod.utils = types.SimpleNamespace(register_model=lambda *a, **k: None,
                                      token_counter=lambda *a, **k: 0,
                                      get_model_info=lambda *a, **k: {})
    mod.cost_calculator = types.SimpleNamespace(completion_cost=lambda *a, **k: 0.0)
    mod.completion_cost = mod.cost_calculator.completion_cost
    mod.token_counter = mod.utils.token_counter
    mod.register_model = mod.utils.register_model
    mod.drop_params = True
    mod.suppress_debug_info = True
    mod.set_verbose = False
    sys.modules["litellm"] = mod
    sys.modules["litellm.exceptions"] = exc
    sys.modules["litellm.utils"] = mod.utils  # type: ignore[assignment]
    sys.modules["litellm.cost_calculator"] = mod.cost_calculator  # type: ignore[assignment]
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenario = SCENARIOS[args.scenario]
    scripted = Scripted(scenario["script"])
    _fake_litellm(scripted)
    os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
    src = Path(args.harness).resolve() / "src"
    sys.path.insert(0, str(src))

    traj = Path.cwd() / "traj.json"
    argv = ["--yolo", "--model", "fake/probe", "--task", "Probe task: follow your instructions.",
            "--output", str(traj), "-c", "mini", "-c", "model.cost_tracking=ignore_errors",
            "-c", f"model.model_kwargs.max_tokens={BASE_MAX_TOKENS}", "-c", "agent.step_limit=100",
            "-c", "agent.cost_limit=0"]
    for spec in scenario.get("config", []):
        argv += ["-c", spec]
    result: dict = {"scenario": args.scenario, "error": None}
    try:
        import micro_swe.__main__ as cli  # noqa: PLC0415  (import after the stub is installed)
        cli.main(argv)
    except BaseException as e:  # noqa: BLE001  (a probe must survive any harness failure)
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
    result["calls"] = scripted.calls
    try:
        result["trajectory"] = json.loads(traj.read_text()) if traj.exists() else None
    except Exception as e:  # noqa: BLE001
        result["trajectory"] = None
        result["error"] = (result["error"] or "") + f"\nbad trajectory: {e}"
    Path(args.out).write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
