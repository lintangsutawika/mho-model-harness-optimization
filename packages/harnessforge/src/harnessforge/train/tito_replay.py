"""Check platoon's token-in/token-out gateway patch (`patches/platoon-tito.patch`) offline.

Runs in a platoon venv (the training copy, with the patch), never with the host python: it imports platoon,
tinker and transformers. Two uses:

* the helpers the tests use (`tests/test_platoon_tito.py`): a scripted sampler (`ScriptedSampler`), a harness that
  keeps history as micro-swe-agent does (`MicroSweHistory`), and `datums()` (platoon's own `trajectory_to_data`
  over an agent's recorded calls);
* `python -m harnessforge.train.tito_replay --tokenizer <snapshot> <rollouts/train/K> ...`: replays the recorded
  rollouts of a run through the real gateway with TITO off and on, and prints datums per trajectory and the
  training tokens of each. The recordings hold each call's returned assistant message, its usage and its
  generation record, not the observations or the sampled token ids, so the replay rebuilds them: each call samples
  the text of its recorded message (thinking, content, qwen3_coder tool calls; cut calls without the end token),
  and each tool result is filler sized from the recorded prompt growth. The harness side follows micro-swe-agent:
  a reply with valid bash tool calls is kept with one tool message per call; any other reply (no tool call, a cut
  generation, an unparsed or invalid call) raises FormatError, so the reply is dropped and a user message with the
  format error follows.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Sequence
from typing import Any

SYSTEM = "You are a helpful assistant that can interact with a computer.\n"
# micro-swe-agent's mini.yaml instance_template, rendered with a typical ABCI sandbox uname
INSTANCE = 'Please solve this issue: {task}\n\nYou can execute bash commands and edit files to implement the necessary changes.\n\n## Command Execution Rules\n\nYou are operating in an environment where\n\n1. You issue at least one command\n2. The system executes the command(s) in a subshell\n3. You see the result(s)\n4. You write your next command(s)\n\nEach response should include:\n\n1. **Reasoning text** where you explain your analysis and plan\n2. At least one tool call with your command\n\n**CRITICAL REQUIREMENTS:**\n\n- Your response SHOULD include reasoning text explaining what you\'re doing\n- Your response MUST include AT LEAST ONE bash tool call\n- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.\n- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files\n- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\n  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>\n\nExample of a CORRECT response:\n<example_response>\nI need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.\n\n[Makes bash tool call with {"command": "ls -la"} as arguments]\n</example_response>\n\n<system_information>\nLinux 5.14.0-503.40.1.el9_5.x86_64 #1 SMP PREEMPT_DYNAMIC x86_64\n</system_information>\n\n## Useful command examples\n\n### Create a new file:\n\n```bash\ncat <<\'EOF\' > newfile.py\nimport numpy as np\nhello = "world"\nprint(hello)\nEOF\n```\n\n### Edit files with sed:```bash\n# Replace all occurrences\nsed -i \'s/old_string/new_string/g\' filename.py\n\n# Replace only first occurrence\nsed -i \'s/old_string/new_string/\' filename.py\n\n# Replace first occurrence on line 1\nsed -i \'1s/old_string/new_string/\' filename.py\n\n# Replace all occurrences in lines 1-10\nsed -i \'1,10s/old_string/new_string/g\' filename.py\n```\n\n### View file content:\n\n```bash\n# View specific lines with numbers\nnl -ba filename.py | sed -n \'10,20p\'\n```\n\n### Any other command you want to run\n\n```bash\nanything\n```'
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}
FORMAT_ERROR_LENGTH = (
    "Your previous response reached the output token limit (finish_reason=length) before you produced a tool call, "
    "so it was cut off. Respond more concisely and finish with exactly one bash tool call. If you need to think "
    "more, do so briefly.\n"
)
FORMAT_ERROR_TOOL = (
    "Tool call error:\n\n<error>\n{error}\n</error>\n\nHere is general guidance on how to submit correct "
    "toolcalls:\n\nEvery response needs to use the 'bash' tool at least once to execute commands.\n\nCall the bash "
    'tool with your command as the argument:\n- Tool: bash\n- Arguments: {{"command": "your_command_here"}}\n\nIf '
    "you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
    "without any other command.\n"
)


# --------------------------------------------------------------------------- platoon pieces


def load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


KEEP_THINKING_TEMPLATE = os.path.join(os.path.dirname(__file__), "templates", "qwen3.5-keep-thinking-v1.jinja")


def make_gateway(
    tokenizer: Any,
    sampler: Any,
    *,
    tito: bool,
    context_window: int = 65536,
    max_tokens: int = 16384,
    chat_template: str | None = None,
    min_completion_tokens: int = 0,
):
    """The gateway under test. ``chat_template``: the renderer's template override (the platoon-tito2 patch);
    ``min_completion_tokens``: the context limit (0: clamp as platoon 0.6.0 does)."""
    from platoon.gateway.core import ModelGateway
    from platoon.train.tinker.renderers import ChatTemplateRenderer

    renderer = ChatTemplateRenderer(tokenizer, tool_call_parser="qwen3_coder")
    extra: dict[str, Any] = {}
    if chat_template:
        renderer.chat_template = chat_template
    if min_completion_tokens:
        extra["min_completion_tokens"] = min_completion_tokens
    elif "min_completion_tokens" in ModelGateway.__init__.__code__.co_varnames:
        extra["min_completion_tokens"] = 0
    return ModelGateway(
        sampler=sampler,
        renderer=renderer,
        tokenizer=tokenizer,
        model_name="Qwen/Qwen3.5-4B",
        max_tokens=max_tokens,
        context_window_length=context_window,
        tito=tito,
        **extra,
    )


def open_agent(gateway: Any, name: str = "r0") -> Any:
    return gateway.open_agent(rollout_id=name, trajectory_id=name, interactions={})


class ScriptedSampler:
    """Answers each call with the next scripted completion: (text, stopped) or a callable(prompt_tokens) giving it.
    The text is encoded as sampled; ``stopped`` appends the end token. Logprobs are seeded random negatives."""

    def __init__(self, tokenizer: Any, script: list[Any], seed: int = 0) -> None:
        self.tokenizer = tokenizer
        self.script = list(script)
        self.prompts: list[list[int]] = []
        self.end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.rng = random.Random(seed)

    def _result(self, prompt: Any, max_tokens: int) -> Any:
        import tinker.types as tt

        tokens_in = prompt.to_ints()
        self.prompts.append(tokens_in)
        item = self.script.pop(0)
        if callable(item):
            item = item(tokens_in)
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            text, stopped = item
            tokens = list(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            tokens, stopped = list(item[0]), item[1]
        if len(tokens) >= max_tokens:  # a cut generation: as many tokens as allowed, no end token
            tokens, stopped = tokens[:max_tokens], False
        if stopped:
            tokens.append(self.end)
        logprobs = [-self.rng.random() for _ in tokens]
        sequence = tt.SampledSequence(tokens=tokens, logprobs=logprobs, stop_reason="stop" if stopped else "length")
        return tt.SampleResponse(sequences=[sequence])

    async def sample_async(self, *, prompt: Any, sampling_params: Any, num_samples: int) -> Any:
        return self._result(prompt, sampling_params.max_tokens)

    def sample(self, *, prompt: Any, sampling_params: Any, num_samples: int) -> Any:
        return self._result(prompt, sampling_params.max_tokens)


def echo(message: dict[str, Any]) -> dict[str, Any]:
    """The gateway's reply as micro-swe-agent sends it back through litellm: a JSON round trip, empty content as
    None, the tool calls' arguments as the JSON strings returned."""
    out = json.loads(json.dumps(message))
    if out.get("content") == "":
        out["content"] = None
    out.setdefault("function_call", None)
    return out


def valid_bash_calls(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """micro-swe-agent's parse_toolcall_actions: the calls when every one is `bash` with a JSON `command`, else None
    (FormatError)."""
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    for call in calls:
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception:
            return None
        if call["function"]["name"] != "bash" or not isinstance(args, dict) or "command" not in args:
            return None
    return calls


class MicroSweHistory:
    """A conversation kept as micro-swe-agent keeps it (agents/default.py, models/litellm.py, tools/bash.py)."""

    def __init__(self, task: str) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTANCE.replace("{task}", task)},
        ]

    def add_reply(self, reply: dict[str, Any], observations: Sequence[str], *, cut: bool) -> bool:
        """Keep ``reply`` and one tool message per call (True), or drop it for a format error (False)."""
        calls = None if cut else valid_bash_calls(reply)
        if calls is None:
            if cut:
                error = FORMAT_ERROR_LENGTH
            else:
                error = FORMAT_ERROR_TOOL.format(
                    error="No tool calls found in the response. Every response MUST include at least one tool call."
                )
            self.messages.append({"role": "user", "content": error})
            return False
        self.messages.append(echo(reply))
        for call, output in zip(calls, list(observations) + [""] * len(calls)):
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        return True


def chat_request(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
    from platoon.gateway.core import ChatRequest

    return ChatRequest.from_params(json.loads(json.dumps(messages)), {"tools": tools or [BASH_TOOL]})


def datums(agent: Any) -> list[Any]:
    """platoon's own datums for an agent's recorded calls (trajectory_to_data, advantage 1)."""
    from platoon.utils.tinker_data_processing import trajectory_to_data

    steps = [{"misc": {"action_misc": {"completion_id": call.call_id}}} for call in agent.calls]
    result = trajectory_to_data(
        {"steps": steps, "misc": {}}, agent.interactions, "task", agent.trajectory_id, 1.0, 0
    )
    return result.datums


def datum_tokens(datum: Any) -> int:
    return int(datum.model_input.length) + 1


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# --------------------------------------------------------------------------- end to end: micro-swe-agent over HTTP


def _call_block(command: str) -> str:
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>\n</tool_call>"


def e2e_script(tokenizer: Any) -> list[Any]:
    """Six turns: whitespace the template renders differently, two calls in one reply after content, an unparsable
    call (micro-swe-agent: format error), a canonical call, a generation cut at max_tokens (format error), submit."""
    long_thinking = tokenizer.encode("Let me think about this for a long time. " * 40, add_special_tokens=False)
    return [
        ("I'll look first.\n\n</think>\n\n" + _call_block("ls"), True),
        ("Two at once.\n</think>\n\nLet me run two commands.\n" + _call_block("pwd") + "\n" + _call_block("echo hi"), True),
        ("Hmm.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la", True),
        ("Again.\n</think>\n\n" + _call_block("echo again"), True),
        (long_thinking, False),
        ("Done.\n</think>\n\n" + _call_block("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"), True),
    ]


async def run_micro_swe(
    tokenizer: Any,
    *,
    tito: bool,
    harness_python: str,
    micro_swe_src: str,
    workdir: str,
    timeout: float = 300,
    script: list[Any] | None = None,
    check: bool = True,
    **gateway_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Serve the gateway (scripted sampler) over HTTP and run micro-swe-agent against it (litellm_proxy, as the
    harness spec launches it). Returns (the agent session, micro-swe-agent's trajectory, with its exit code under
    ``_exit_code``). ``check``: raise when the harness exits non-zero."""
    from platoon.gateway.server import GatewayServer

    sampler = ScriptedSampler(tokenizer, script if script is not None else e2e_script(tokenizer))
    gateway = make_gateway(tokenizer, sampler, tito=tito, **gateway_kwargs)
    agent = open_agent(gateway, f"e2e-{int(tito)}")
    server = GatewayServer(gateway)
    base = await server.start("127.0.0.1")
    os.makedirs(workdir, exist_ok=True)
    output = os.path.join(workdir, f"trajectory-{int(tito)}.json")
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": workdir,
        "PYTHONPATH": micro_swe_src,
        "LITELLM_PROXY_API_BASE": f"{base}/v1",
        "LITELLM_PROXY_API_KEY": agent.api_key,
        "MSWEA_API_KEY": agent.api_key,
        "MSWEA_CONFIGURED": "true",
        "MSWEA_COST_TRACKING": "ignore_errors",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    argv = [harness_python, "-I", "-c", "import sys; sys.path.insert(0, sys.argv.pop(1)); "
            "from micro_swe.__main__ import main; main(sys.argv[1:])", micro_swe_src,
            "--yolo", "--model=litellm_proxy/Qwen/Qwen3.5-4B", f"--output={output}", "--cost-limit", "0",
            "-c", "mini", "-c", "model.model_kwargs.timeout=600", "--exit-immediately", "--task=Print the hostname."]
    try:
        proc = await asyncio.create_subprocess_exec(*argv, cwd=workdir, env=env, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        if check and proc.returncode != 0:
            raise RuntimeError(f"micro-swe-agent exited {proc.returncode}: {out.decode()[-3000:]}")
    finally:
        await server.stop()
    with open(output) as f:
        trajectory = json.load(f)
    trajectory["_exit_code"] = proc.returncode
    return agent, trajectory


# --------------------------------------------------------------------------- replay of recorded rollouts


def _load_events(path: str) -> list[dict[str, Any]]:
    trajectories: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            event = json.loads(line)
            kind = event.get("type")
            tid = event.get("trajectory_id") or (event.get("trajectory") or {}).get("id")
            tr = trajectories.setdefault(tid, {"id": tid, "steps": [], "gens": {}, "goal": "", "role": None})
            if kind == "trajectory_task_set":
                tr["goal"] = (event.get("task") or {}).get("goal") or ""
            elif kind == "trajectory_step_added":
                misc = event["step"]["misc"]
                action = misc.get("action_misc") or {}
                tr["steps"].append((action.get("completion_id"), action.get("usage") or {}, misc.get("message") or {}))
            elif kind == "trajectory_finished":
                tr["reward"] = event.get("reward")
                misc = event.get("misc") or {}
                tr["gens"] = misc.get("model_generations") or {}
                tr["role"] = (misc.get("agent") or {}).get("role")
    return [t for t in trajectories.values() if t["steps"]]


def sample_text(message: dict[str, Any], *, variant: bool = False) -> str:
    """The text a recorded reply was most likely sampled as, after the generation prompt's ``<think>\\n``.
    ``variant``: with a blank line before ``</think>``, which parses the same but the template renders without (a
    sample the from-scratch render does not reproduce)."""
    reasoning = (message.get("reasoning_content") or "").strip()
    content = (message.get("content") or "").strip()
    text = reasoning + ("\n\n</think>\n\n" if variant else "\n</think>\n\n") + content
    calls = message.get("tool_calls") or []
    blocks = []
    for call in calls:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        params = "".join(
            f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items()
        )
        blocks.append(f"<tool_call>\n<function={fn.get('name')}>\n{params}</function>\n</tool_call>")
    if blocks:
        text = (text + "\n\n" if content else text) + "\n".join(blocks)
    return text


def _filler(tokenizer: Any, n: int) -> str:
    return " ".join(["ok"] * max(0, n))


# What the replay compares: the gateway before (off), with TITO (merged), and with TITO plus (a) the keep-thinking
# template and/or (b) the context limit (min_completion_tokens 2048), all at the run's 65536-token window.
VARIANTS: dict[str, dict[str, Any]] = {
    "off": {"tito": False},
    "tito": {"tito": True},
    "tito+a": {"tito": True, "keep_thinking": True},
    "tito+b": {"tito": True, "min_completion_tokens": 2048},
    "tito+a+b": {"tito": True, "keep_thinking": True, "min_completion_tokens": 2048},
}


async def replay_trajectory(
    tokenizer: Any,
    tr: dict[str, Any],
    *,
    tito: bool,
    perturb: float = 0.0,
    keep_thinking: bool = False,
    min_completion_tokens: int = 0,
    context_window: int = 65536,
    template: str = KEEP_THINKING_TEMPLATE,
) -> dict[str, Any]:
    from platoon.gateway.core import ContextLimitError, ContextWindowExceededError

    steps = tr["steps"]
    script: list[Any] = []
    rng = random.Random(tr["id"])  # the same samples in every variant
    for cid, usage, message in steps:
        record = tr["gens"].get(cid, {})
        text = sample_text(message, variant=rng.random() < perturb)
        if record.get("stop_reason") == "length":
            ids = tokenizer.encode(text, add_special_tokens=False)[: max(1, int(usage.get("completion_tokens") or 1))]
            script.append((ids, False))
        else:
            script.append((text, True))
    sampler = ScriptedSampler(tokenizer, script)
    gateway = make_gateway(
        tokenizer,
        sampler,
        tito=tito,
        context_window=context_window,
        chat_template=template if keep_thinking else None,
        min_completion_tokens=min_completion_tokens,
    )
    agent = open_agent(gateway, tr["id"])
    history = MicroSweHistory(tr["goal"])
    prompt_diffs = []
    kept = dropped = 0
    ended = None
    base = previous = 0  # drift the variant causes by design (thinking kept after a dropped reply), not steered away
    after_drop = False
    for i, (cid, usage, message) in enumerate(steps):
        try:
            response = await gateway.chat(agent, chat_request(history.messages))
        except ContextLimitError:
            ended = "context_limit"
            break
        except ContextWindowExceededError:
            ended = "context_overflow"
            break
        drift = response.prompt_tokens - int(usage.get("prompt_tokens") or 0)
        if after_drop:
            base += drift - previous
        previous = drift
        prompt_diffs.append(drift - base)
        cut = response.stop_reason == "length"
        reply = response.message()
        calls = None if cut else valid_bash_calls(reply)
        observations: list[str] = []
        if calls and i + 1 < len(steps):
            grow = int(steps[i + 1][1].get("prompt_tokens") or 0) - int(usage.get("prompt_tokens") or 0)
            grow -= int(usage.get("completion_tokens") or 0) + (drift - base)  # steer back to the recorded sizes
            per_call = max(0, grow - 12 * len(calls)) // len(calls)
            observations = [
                '{\n  "returncode": 0,\n  "output": "' + _filler(tokenizer, per_call - 10) + '"\n}' for _ in calls
            ]
        after_drop = not history.add_reply(reply, observations, cut=cut)
        if after_drop:
            dropped += 1
        else:
            kept += 1
    data = datums(agent)
    return {
        "datums": len(data),
        "tokens": sum(datum_tokens(d) for d in data),
        "calls": len(agent.calls),
        "recorded_calls": len(steps),
        "kept": kept,
        "dropped": dropped,
        "ended": ended,
        "reward": tr.get("reward"),
        "prompt_diff_mean_abs": sum(abs(d) for d in prompt_diffs) / max(1, len(prompt_diffs)),
        "tito": dict(agent.misc.get("tito") or {}),
    }


def usage_estimate(tr: dict[str, Any]) -> tuple[int, int]:
    """(datums, training tokens) under TITO from the recorded usage alone: a datum ends where micro-swe-agent drops
    a reply (format error) and at the end; each holds its last call's prompt + completion."""
    steps = tr["steps"]
    datums = tokens = 0
    for i, (cid, usage, message) in enumerate(steps):
        record = tr["gens"].get(cid, {})
        dropped = record.get("stop_reason") == "length" or valid_bash_calls(message) is None
        if dropped or i == len(steps) - 1:
            datums += 1
            tokens += int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    return datums, tokens


_WORKER: dict[str, Any] = {}


def _worker_init(tokenizer_path: str, perturb: float, variants: list[str]) -> None:
    os.environ.setdefault("PLATOON_TITO_VERIFY", "1")
    _WORKER["tokenizer"] = load_tokenizer(tokenizer_path)
    _WORKER["perturb"] = perturb
    _WORKER["variants"] = variants


def _replay_row(tr: dict[str, Any]) -> dict[str, Any]:
    tokenizer, perturb = _WORKER["tokenizer"], _WORKER["perturb"]
    row: dict[str, Any] = {"id": tr["id"], "calls": len(tr["steps"]), "usage_estimate": usage_estimate(tr),
                           "reward": tr.get("reward")}
    for name in _WORKER["variants"]:
        row[name] = run(replay_trajectory(tokenizer, tr, perturb=perturb, **VARIANTS[name]))
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessforge.train.tito_replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True, help="local Qwen3.5 snapshot (tokenizer + chat template)")
    ap.add_argument("--limit", type=int, default=0, help="at most this many trajectories (0: all)")
    ap.add_argument("--workers", type=int, default=1, help="processes (each replays whole trajectories)")
    ap.add_argument("--out", default=None, help="write per-trajectory rows here (JSONL)")
    ap.add_argument("--variants", default="off,tito", help=f"comma-separated, of {', '.join(VARIANTS)}")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="fraction of calls sampled with whitespace the template does not render (the recordings "
                         "hold parsed replies, so rebuilt samples are otherwise canonical and never break TITO-off)")
    ap.add_argument("dirs", nargs="+", help="rollouts/train/<k> directories (holding events/*.jsonl)")
    args = ap.parse_args(argv)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants {unknown}")
    trajectories = []
    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, "events", "*.jsonl"))):
            trajectories.extend(_load_events(path))
    if args.limit:
        trajectories = trajectories[: args.limit]
    # longest first, so the pool does not end on one long trajectory
    order = sorted(range(len(trajectories)), key=lambda k: -len(trajectories[k]["steps"]))
    rows: list[dict[str, Any]] = []
    init = (args.tokenizer, args.perturb, variants)
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers, _worker_init, init) as pool:
            for k, row in enumerate(pool.imap_unordered(_replay_row, [trajectories[j] for j in order])):
                rows.append(row)
                if (k + 1) % 20 == 0:
                    print(f"... {k + 1}/{len(trajectories)}", file=sys.stderr, flush=True)
    else:
        _worker_init(*init)
        for k, j in enumerate(order):
            rows.append(_replay_row(trajectories[j]))
            if (k + 1) % 20 == 0:
                print(f"... {k + 1}/{len(trajectories)}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(summarize(rows, variants, perturb=args.perturb))
    return 0


def summarize(rows: list[dict[str, Any]], variants: list[str], *, perturb: float = 0.0) -> str:
    n = max(1, len(rows))
    out = [f"trajectories {len(rows)}, recorded calls {sum(r['calls'] for r in rows)}, perturb {perturb}"]
    d = sum(r["usage_estimate"][0] for r in rows)
    t = sum(r["usage_estimate"][1] for r in rows)
    out.append(f"TITO from recorded usage: datums {d} ({d / n:.3f}/trajectory), training tokens {t:,}")
    for key in variants:
        d = sum(r[key]["datums"] for r in rows)
        t = sum(r[key]["tokens"] for r in rows)
        calls = sum(r[key]["calls"] for r in rows)
        line = (f"{key:9s}: datums {d} ({d / n:.3f}/trajectory), training tokens {t:,} ({t / n:,.0f}/trajectory), "
                f"calls {calls}")
        for reason in ("context_limit", "context_overflow"):
            ended = [r for r in rows if r[key]["ended"] == reason]
            if ended:
                rewards = [float(r["reward"] or 0.0) for r in ended]
                line += (f"; ended by {reason}: {len(ended)} ({len(ended) / n:.1%}), their recorded final reward "
                         f"mean {sum(rewards) / len(rewards):.3f}, passed {sum(1 for x in rewards if x > 0)}")
        out.append(line)
        reasons: Counter = Counter()
        for r in rows:
            reasons.update(r[key]["tito"].get("reasons") or {})
            for name in ("text_equal", "text_differs"):
                reasons[name] += int(r[key]["tito"].get(name) or 0)
        if key != "off":
            out.append(f"  prompt outcomes: {dict(reasons)}")
        diffs = [r[key]["prompt_diff_mean_abs"] for r in rows]
        out.append(f"  rebuilt vs recorded prompt tokens, mean |diff| per call: {sum(diffs) / n:.1f}")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
