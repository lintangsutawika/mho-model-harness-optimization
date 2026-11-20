"""platoon's token-in/token-out gateway patch (patches/platoon-tito.patch), with the real Qwen3.5 tokenizer and chat
template. Runs only in a patched platoon venv (the training copy) with the tokenizer snapshot present:

    QWEN35_TOKENIZER=<snapshot> <venv>/bin/python -m pytest tests/test_platoon_tito.py

Skipped elsewhere (the host python has no platoon).
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("platoon.gateway.tito", reason="needs a platoon venv with platoon-tito.patch")
pytest.importorskip("transformers")

from harnessforge.train import tito_replay as R  # noqa: E402

SNAPSHOT = os.environ.get(
    "QWEN35_TOKENIZER",
    "/home/aci18914wh/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
)
if not os.path.exists(os.path.join(SNAPSHOT, "tokenizer.json")):
    pytest.skip(f"no Qwen3.5 tokenizer at {SNAPSHOT} (set QWEN35_TOKENIZER)", allow_module_level=True)


@pytest.fixture(scope="module")
def tok():
    return R.load_tokenizer(SNAPSHOT)


def call(command: str) -> str:
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>\n</tool_call>"


def canonical(thinking: str, *commands: str) -> str:
    return thinking + "\n</think>\n\n" + "\n".join(call(c) for c in commands)


# Samples the template would render differently: extra blank lines around the thinking, a space before a tool call.
NONCANONICAL = [
    "Look around first.\n\n\n</think>\n\n" + call("ls -la"),
    "Read it.\n</think>\n\n\n" + call("cat a.txt"),
    "Fine.  \n</think>\n\n " + call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
]


def drive(tok, script, *, tito, observe=lambda i: f'{{"returncode": 0, "output": "out {i}"}}', task="fix it",
          rewrite=None):
    """Run a micro-swe-agent-like harness over ``script``; returns (gateway, agent, sampler, history)."""
    sampler = R.ScriptedSampler(tok, list(script))
    gateway = R.make_gateway(tok, sampler, tito=tito)
    agent = R.open_agent(gateway)
    history = R.MicroSweHistory(task)
    for i in range(len(script)):
        if rewrite is not None:
            rewrite(i, history)
        response = R.run(gateway.chat(agent, R.chat_request(history.messages)))
        calls = R.valid_bash_calls(response.message()) or []
        history.add_reply(response.message(), [observe(i)] * len(calls), cut=response.stop_reason == "length")
    return gateway, agent, sampler, history


def stream_of(agent):
    """The last datum's full token sequence, mask and logprobs (positions of the sequence, not shifted)."""
    data = R.datums(agent)
    d = data[-1]
    targets = d.loss_fn_inputs["target_tokens"].to_torch().tolist()
    first = d.model_input.to_ints()[0]
    return data, [first] + targets, [0.0] + d.loss_fn_inputs["mask"].to_torch().tolist(), \
        [0.0] + d.loss_fn_inputs["logprobs"].to_torch().tolist()


def test_noncanonical_samples_break_without_tito_and_merge_with_it(tok):
    _, off, _, _ = drive(tok, [(t, True) for t in NONCANONICAL], tito=False)
    _, on, _, _ = drive(tok, [(t, True) for t in NONCANONICAL], tito=True)
    assert len(R.datums(off)) == 3          # every turn re-rendered differently: a datum per call
    assert len(R.datums(on)) == 1
    assert on.misc["tito"]["stitched"] == 2 and on.misc["tito"]["fresh"] == 1


def test_canonical_samples_stitch_to_exactly_the_rendered_prompt(tok):
    script = [(canonical("Plan.", "ls"), True), (canonical("Next.", "cat x"), True), (canonical("Done.", "echo hi"), True)]
    gw_off, off, s_off, _ = drive(tok, script, tito=False)
    gw_on, on, s_on, _ = drive(tok, script, tito=True)
    assert s_on.prompts == s_off.prompts  # token for token what rendering from scratch gives
    assert len(R.datums(on)) == len(R.datums(off)) == 1


def test_stitched_prompt_extends_stream_and_keeps_sampled_tokens(tok):
    gw, agent, sampler, history = drive(tok, [(t, True) for t in NONCANONICAL], tito=True)
    p0, p1, p2 = sampler.prompts
    c0 = agent.calls[0].tokens
    assert p1[: len(p0) + len(c0)] == p0 + c0
    # the from-scratch render of the same conversation (what the model would otherwise see) differs only in tokens
    # of the model's own turns: as text, the only difference is the whitespace the model sampled
    rendered = gw.renderer.render_text(history.messages[:4], tools=[R.BASH_TOOL])
    decoded = tok.decode(p1)
    assert decoded.replace("\n", "").replace(" ", "") == rendered.replace("\n", "").replace(" ", "")
    assert decoded.endswith("<|im_start|>assistant\n<think>\n")


def test_loss_mask_and_logprobs_cover_exactly_the_sampled_tokens(tok):
    gw, agent, sampler, _ = drive(tok, [(t, True) for t in NONCANONICAL], tito=True)
    data, seq, mask, logprobs = stream_of(agent)
    assert len(data) == 1
    sampled = [(c.prompt_tokens, c.tokens, c.logprobs) for c in agent.calls]
    expected = [0.0] * len(seq)
    for start, tokens, lps in sampled:
        assert seq[start:start + len(tokens)] == tokens
        for k, lp in enumerate(lps):
            expected[start + k] = 1.0
            assert abs(logprobs[start + k] - lp) < 1e-6
    assert mask == expected
    assert sum(mask) == sum(len(t) for _, t, _ in sampled)


def test_multiple_tool_calls_stay_in_one_datum(tok):
    script = [("Two at once.\n</think>\n\n" + call("ls") + "\n\n" + call("pwd"), True),
              (canonical("Then.", "cat f"), True), (canonical("End.", "echo done"), True)]
    _, on, _, history = drive(tok, script, tito=True)
    assert sum(1 for m in history.messages if m["role"] == "tool") == 4
    assert len(R.datums(on)) == 1


def test_unparsable_call_kept_by_an_echoing_harness_keeps_raw_tokens(tok):
    """A harness that keeps a reply whose tool call did not parse (content = the raw block)."""
    sampler = R.ScriptedSampler(tok, [("Try.\n</think>\n\n<tool_call>\n<function+bash>\nls\n</tool_call>", True),
                                      (canonical("Fix it.", "ls"), True), (canonical("ok", "pwd"), True)])
    gw = R.make_gateway(tok, sampler, tito=True)
    agent = R.open_agent(gw)
    messages = R.MicroSweHistory("t").messages
    for i in range(3):
        r = R.run(gw.chat(agent, R.chat_request(messages)))
        messages.append(R.echo(r.message()))
        if r.tool_calls:
            messages.extend({"role": "tool", "tool_call_id": c["id"], "content": "out"} for c in r.tool_calls)
        else:
            assert r.unparsed_tool_calls == 1
            messages.append({"role": "tool", "tool_call_id": "x", "content": "unparsed: try again"})
    assert agent.misc["tito"]["stitched"] == 2
    assert len(R.datums(agent)) == 1
    p0, p1 = sampler.prompts[:2]
    assert p1[: len(p0) + len(agent.calls[0].tokens)] == p0 + agent.calls[0].tokens


def test_cut_generation_gets_a_masked_end_token(tok):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    long_thinking = "word " * 400
    sampler = R.ScriptedSampler(tok, [(long_thinking, True), (canonical("short", "ls"), True)])
    gw = R.make_gateway(tok, sampler, tito=True, max_tokens=64)
    agent = R.open_agent(gw)
    messages = R.MicroSweHistory("t").messages
    r = R.run(gw.chat(agent, R.chat_request(messages)))
    assert r.stop_reason == "length" and len(r.tokens) == 64 and r.tokens[-1] != end
    messages.append(R.echo(r.message()))  # a harness that keeps the cut turn, and answers in a tool message
    # (a user message would not stitch with Qwen3.5: a new user query drops the thinking of the turns before it)
    messages.append({"role": "tool", "tool_call_id": "x", "content": "You were cut off; be brief."})
    R.run(gw.chat(agent, R.chat_request(messages)))
    assert agent.misc["tito"]["stitched"] == 1
    p0, p1 = sampler.prompts
    assert p1[: len(p0) + 64] == p0 + r.tokens
    assert p1[len(p0) + 64] == end  # the repaired close of the cut turn
    data, seq, mask, _ = stream_of(agent)
    assert len(data) == 1
    assert seq[len(p0) + 64] == end and mask[len(p0) + 64] == 0.0


def test_micro_swe_format_error_drops_the_reply_and_starts_a_new_datum(tok):
    """micro-swe-agent drops a reply without a valid tool call and adds a user message; Qwen3.5 then also drops the
    thinking of every earlier turn. That history differs, so TITO renders it from scratch: a new datum."""
    script = [(canonical("a", "ls"), True), ("I will just answer.\n</think>\n\nThe answer is 4.", True),
              (canonical("b", "pwd"), True), (canonical("c", "echo x"), True)]
    gw_on, on, s_on, history = drive(tok, script, tito=True)
    _, off, s_off, _ = drive(tok, script, tito=False)
    assert [m["role"] for m in history.messages[4:6]] == ["user", "assistant"]
    assert on.misc["tito"]["reasons"].get("reply_dropped_or_changed") == 1
    assert s_on.prompts[2] == s_off.prompts[2]  # rendered from scratch, exactly as without TITO
    assert len(R.datums(on)) == 2


def test_history_truncation_starts_a_new_datum(tok):
    def truncate(i, history):
        if i == 3:  # the harness drops the oldest exchange (compaction)
            del history.messages[2:4]
    script = [(canonical(f"s{i}", f"echo {i}"), True) for i in range(5)]
    _, on, _, _ = drive(tok, script, tito=True, rewrite=truncate)
    assert on.misc["tito"]["reasons"]["history_changed"] == 1
    assert len(R.datums(on)) == 2


def test_changed_tools_render_from_scratch(tok):
    sampler = R.ScriptedSampler(tok, [(t, True) for t in NONCANONICAL[:2]])
    gw = R.make_gateway(tok, sampler, tito=True)
    agent = R.open_agent(gw)
    history = R.MicroSweHistory("t")
    r = R.run(gw.chat(agent, R.chat_request(history.messages)))
    history.add_reply(r.message(), ["out"], cut=False)
    other = json.loads(json.dumps(R.BASH_TOOL))
    other["function"]["description"] = "Run bash"
    R.run(gw.chat(agent, R.chat_request(history.messages, tools=[other])))
    assert agent.misc["tito"]["reasons"]["request_changed"] == 1


def test_tito_off_by_env(tok, monkeypatch):
    monkeypatch.setenv("PLATOON_TITO", "0")
    sampler = R.ScriptedSampler(tok, [(t, True) for t in NONCANONICAL])
    from platoon.gateway.core import ModelGateway
    from platoon.train.tinker.renderers import ChatTemplateRenderer

    gw = ModelGateway(sampler=sampler, renderer=ChatTemplateRenderer(tok, tool_call_parser="qwen3_coder"),
                      tokenizer=tok, model_name="m")
    assert gw.tito is False and gw.tito_stitcher() is None


def test_verify_counts_text_equality(tok, monkeypatch):
    monkeypatch.setenv("PLATOON_TITO_VERIFY", "1")
    script = [(canonical("x", "ls"), True), (canonical("y", "pwd"), True)] + [(t, True) for t in NONCANONICAL[1:]]
    _, on, _, _ = drive(tok, script, tito=True)
    counts = on.misc["tito"]
    # calls 2 and 3 follow the canonical turns, call 4 follows a turn sampled with other whitespace
    assert counts["text_equal"] == 2 and counts["text_differs"] == 1


def test_http_server_passes_the_agent(tok):
    """The OpenAI-compatible server path stitches too (GatewayServer -> prepare(agent=...))."""
    import asyncio

    import aiohttp
    from platoon.gateway.server import GatewayServer

    async def main():
        sampler = R.ScriptedSampler(tok, [(t, True) for t in NONCANONICAL])
        gw = R.make_gateway(tok, sampler, tito=True)
        agent = R.open_agent(gw)
        server = GatewayServer(gw)
        base = await server.start("127.0.0.1")
        history = R.MicroSweHistory("t")
        try:
            async with aiohttp.ClientSession() as session:
                for i in range(3):
                    body = {"model": "m", "messages": history.messages, "tools": [R.BASH_TOOL]}
                    async with session.post(f"{base}/v1/chat/completions", json=body,
                                            headers={"Authorization": f"Bearer {agent.api_key}"}) as resp:
                        assert resp.status == 200, await resp.text()
                        message = (await resp.json())["choices"][0]["message"]
                    calls = R.valid_bash_calls(message) or []
                    history.add_reply(message, ["out"] * len(calls), cut=False)
        finally:
            await server.stop()
        return agent

    agent = asyncio.run(main())
    assert agent.misc["tito"]["stitched"] == 2
    assert len(R.datums(agent)) == 1


HARNESS_PYTHON = os.environ.get("TITO_HARNESS_PYTHON")  # a python with litellm (the hf-micro-swe-1 toolchain's)
MICRO_SWE_SRC = os.environ.get("TITO_MICRO_SWE_SRC")  # micro-swe-agent's src/


@pytest.mark.skipif(not (HARNESS_PYTHON and MICRO_SWE_SRC), reason="set TITO_HARNESS_PYTHON and TITO_MICRO_SWE_SRC")
def test_micro_swe_agent_end_to_end(tok, tmp_path):
    """The real harness (litellm_proxy -> the gateway's HTTP server) echoes the gateway's replies faithfully enough to
    stitch: only its format errors (which drop the reply) start new datums."""
    results = {}
    for tito in (False, True):
        agent, traj = R.run(R.run_micro_swe(tok, tito=tito, harness_python=HARNESS_PYTHON,
                                            micro_swe_src=MICRO_SWE_SRC, workdir=str(tmp_path / f"w{int(tito)}")))
        results[tito] = (agent, traj)
    on, traj = results[True]
    off, _ = results[False]
    assert len(on.calls) == len(off.calls) == 6
    assert traj["info"]["exit_status"] == "Submitted"
    kept = [m for m in traj["messages"] if m.get("role") == "assistant"]
    assert kept and all(m.get("reasoning_content") for m in kept)  # litellm keeps the thinking it was given
    reasons = on.misc["tito"]["reasons"]
    assert reasons == {"first": 1, "stitched": 3, "reply_dropped_or_changed": 2}, reasons
    assert len(R.datums(on)) == 3          # breaks only where micro-swe-agent dropped the reply
    assert len(R.datums(off)) >= 4         # plus the re-rendered whitespace without TITO
