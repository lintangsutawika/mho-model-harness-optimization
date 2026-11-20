"""platoon-tito2.patch on top of platoon-tito.patch: (a) the keep-thinking chat template override
(`train.renderer_kwargs.chat_template`), (b) the context limit (`PLATOON_MIN_COMPLETION_TOKENS`, outcome reason
`context_limit`). Real Qwen3.5 tokenizer and template; runs only in a venv with both patches:

    QWEN35_TOKENIZER=<snapshot> <venv>/bin/python -m pytest tests/test_platoon_tito2.py
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("platoon.gateway.tito", reason="needs a platoon venv with the TITO patches")
core = pytest.importorskip("platoon.gateway.core")
if not hasattr(core, "ContextLimitError"):
    pytest.skip("needs platoon-tito2.patch", allow_module_level=True)

from harnessforge.train import tito_replay as R  # noqa: E402

SNAPSHOT = os.environ.get(
    "QWEN35_TOKENIZER",
    "/home/aci18914wh/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
)
if not os.path.exists(os.path.join(SNAPSHOT, "tokenizer.json")):
    pytest.skip(f"no Qwen3.5 tokenizer at {SNAPSHOT} (set QWEN35_TOKENIZER)", allow_module_level=True)

TEMPLATE = R.KEEP_THINKING_TEMPLATE


@pytest.fixture(scope="module")
def tok():
    return R.load_tokenizer(SNAPSHOT)


def call(command: str) -> str:
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>\n</tool_call>"


def turn(thinking: str, command: str) -> tuple[str, bool]:
    return thinking + "\n</think>\n\n" + call(command), True


def conversation():
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "reasoning_content": "EARLIER-THINKING",
         "tool_calls": [{"id": "a", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": "out"},
        {"role": "user", "content": "Tool call error: ..."},
    ]


# ------------------------------------------------------------------------------------------- (a) keep thinking


def test_template_override_is_the_stock_template_but_one_condition(tok):
    stock = tok.chat_template
    ours = open(TEMPLATE).read()
    assert stock.count("loop.index0 > ns.last_query_index") == 1
    assert ours.replace(
        "        {#- keep-thinking v1 (harnessforge): render every assistant turn's thinking, not only those after the "
        "last user query #}\n        {%- if true %}\n", "        {%- if loop.index0 > ns.last_query_index %}\n") == stock


def test_override_keeps_thinking_after_a_user_message(tok):
    from platoon.train.tinker.renderers import ChatTemplateRenderer

    stock = ChatTemplateRenderer(tok, tool_call_parser="qwen3_coder")
    ours = ChatTemplateRenderer(tok, tool_call_parser="qwen3_coder", chat_template=TEMPLATE)
    assert "EARLIER-THINKING" not in stock.render_text(conversation())
    assert "EARLIER-THINKING" in ours.render_text(conversation())
    # without a user query after the turn, both render the same tokens
    head = conversation()[:4]
    assert stock.render_text(head) == ours.render_text(head)
    assert ours.resolve_tool_call_parser()[1].verified


def test_override_set_as_renderer_kwargs_attribute(tok):
    """platoon sets a config's renderer_kwargs as attributes (proxy.build_renderer)."""
    from platoon.train.tinker.renderers import ChatTemplateRenderer

    r = ChatTemplateRenderer(tok, tool_call_parser="qwen3_coder")
    assert hasattr(r, "chat_template")
    setattr(r, "chat_template", TEMPLATE)
    assert "EARLIER-THINKING" in r.render_text(conversation())


def test_format_error_no_longer_shrinks_the_prompt(tok):
    """micro-swe-agent drops a reply without a tool call and adds a user message: the stock template then drops all
    earlier thinking; the override keeps it, so the next prompt only grows."""
    script = [turn("plan " * 400, "ls"), turn("more " * 400, "pwd"), ("Just text.\n</think>\n\nNo call.", True),
              turn("again", "echo x")]
    results = {}
    for name, template in (("stock", None), ("ours", TEMPLATE)):
        sampler = R.ScriptedSampler(tok, list(script))
        gw = R.make_gateway(tok, sampler, tito=True, chat_template=template)
        agent = R.open_agent(gw)
        history = R.MicroSweHistory("t")
        for _ in script:
            r = R.run(gw.chat(agent, R.chat_request(history.messages)))
            calls = R.valid_bash_calls(r.message()) or []
            history.add_reply(r.message(), ["out"] * len(calls), cut=r.stop_reason == "length")
        results[name] = (sampler.prompts, agent)
    stock_prompts, _ = results["stock"]
    our_prompts, agent = results["ours"]
    assert len(stock_prompts[3]) < len(stock_prompts[2])          # thinking dropped
    assert len(our_prompts[3]) > len(our_prompts[2])               # kept
    assert agent.misc["tito"]["reasons"] == {"first": 1, "stitched": 2, "reply_dropped_or_changed": 1}
    # TITO still stitches every kept turn with the override, token for token as the override renders
    assert our_prompts[1][: len(our_prompts[0])] == our_prompts[0]


# ------------------------------------------------------------------------------------------- (b) context limit


def gateway(tok, window, min_completion=2048, script=()):
    sampler = R.ScriptedSampler(tok, list(script) or [turn("x", "ls")] * 4)
    return R.make_gateway(tok, sampler, tito=True, context_window=window, min_completion_tokens=min_completion)


def test_fit_max_tokens_ends_at_the_limit(tok):
    from platoon.gateway.core import ContextLimitError, ContextWindowExceededError
    from tinker.types import ModelInput

    gw = gateway(tok, 10000)
    prompt = ModelInput.from_ints([1] * 8000)
    with pytest.raises(ContextLimitError) as e:  # headroom 2000 < 2048
        gw.fit_max_tokens(prompt, 16384)
    assert isinstance(e.value, ContextWindowExceededError) and e.value.min_completion_tokens == 2048
    assert gw.fit_max_tokens(ModelInput.from_ints([1] * 7000), 16384) == 3000   # clamped as before
    assert gw.fit_max_tokens(prompt, 1024) == 1024        # a call that asks for less than the limit still fits
    off = gateway(tok, 10000, min_completion=0)
    assert off.fit_max_tokens(prompt, 16384) == 2000      # 0: platoon 0.6.0's clamp


def test_env_sets_the_limit(tok, monkeypatch):
    from platoon.gateway.core import DEFAULT_MIN_COMPLETION_TOKENS, ModelGateway

    assert DEFAULT_MIN_COMPLETION_TOKENS == 2048
    monkeypatch.setenv("PLATOON_MIN_COMPLETION_TOKENS", "4096")
    gw = ModelGateway(sampler=None, renderer=R.make_gateway(tok, None, tito=True).renderer, tokenizer=tok,
                      model_name="m")
    assert gw.min_completion_tokens == 4096
    monkeypatch.setenv("PLATOON_MIN_COMPLETION_TOKENS", "0")
    assert ModelGateway(sampler=None, renderer=gw.renderer, tokenizer=tok, model_name="m").min_completion_tokens == 0


def test_chat_records_the_limit_on_the_agent(tok):
    from platoon.gateway.core import ContextLimitError

    history = R.MicroSweHistory("t")
    probe = gateway(tok, 1 << 20)
    first = R.run(probe.chat(R.open_agent(probe), R.chat_request(history.messages))).prompt_tokens
    gw = gateway(tok, first + 1000)
    agent = R.open_agent(gw)
    with pytest.raises(ContextLimitError):
        R.run(gw.chat(agent, R.chat_request(history.messages)))
    assert agent.context_limits == 1 and agent.context_overflows == 1 and not agent.calls
    assert agent.misc["context_limit"]["prompt_tokens"] == first


def test_failure_outcome_grades_with_reason_context_limit():
    from platoon.harness.agent import AgentResult
    from platoon.harness.inline import failure_outcome

    for status in ("error", "completed"):
        result = AgentResult(status=status, error="ContextWindowExceededError" if status == "error" else None)
        assert failure_outcome(result, context_overflows=1, on_context_overflow="grade", harness_ref="h",
                               context_limits=1) == (None, "context_limit")
    failed, _ = failure_outcome(AgentResult(status="error", error="x"), context_overflows=1,
                                on_context_overflow="exclude", harness_ref="h", context_limits=1)
    assert failed is not None and failed.status == "error"  # Outcome.infra: excluded from training
    # a plain overflow keeps its reason
    assert failure_outcome(AgentResult(status="error", error="x"), context_overflows=1, on_context_overflow="grade",
                           harness_ref="h") == (None, "context_overflow")


def test_trajectory_stat_counts_the_limit():
    from platoon.utils.tinker_data_processing import _ended_by_context_limit

    assert _ended_by_context_limit({"misc": {"outcome": {"status": "scored", "reason": "context_limit"}}}) == 1
    assert _ended_by_context_limit({"misc": {"outcome": {"status": "scored", "reason": "context_overflow"}}}) == 0
    assert _ended_by_context_limit({"misc": {}}) == 0


HARNESS_PYTHON = os.environ.get("TITO_HARNESS_PYTHON")
MICRO_SWE_SRC = os.environ.get("TITO_MICRO_SWE_SRC")


@pytest.mark.skipif(not (HARNESS_PYTHON and MICRO_SWE_SRC), reason="set TITO_HARNESS_PYTHON and TITO_MICRO_SWE_SRC")
def test_micro_swe_agent_stops_at_the_limit(tok, tmp_path):
    """The real harness gets the context-length error and exits with an error after the calls that fit; the agent
    records the limit, which rollout.py grades with reason context_limit."""
    history = R.MicroSweHistory("Print the hostname.")
    probe = gateway(tok, 1 << 20)
    import json as _json
    first = R.run(probe.chat(R.open_agent(probe), R.chat_request(history.messages))).prompt_tokens
    script = [turn("look " * 200, "ls"), turn("next " * 200, "pwd"), turn("more " * 200, "echo x")] * 3
    agent, traj = R.run(R.run_micro_swe(
        tok, tito=True, harness_python=HARNESS_PYTHON, micro_swe_src=MICRO_SWE_SRC, workdir=str(tmp_path),
        script=script, check=False, context_window=first + 400 + 2048 + 600, min_completion_tokens=2048))
    assert traj["_exit_code"] != 0
    assert agent.context_limits == 1 and 1 <= len(agent.calls) < len(script)
    assert "context_limit" in agent.misc and len(R.datums(agent)) == 1
    _json.dumps(agent.misc)  # recorded misc stays JSON
