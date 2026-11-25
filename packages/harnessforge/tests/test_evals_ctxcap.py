"""harnessforge.evals.ctxcap: the context-window cap proxy, against a fake upstream and a fake token counter."""
from __future__ import annotations

import asyncio
import json

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from harnessforge.evals.ctxcap import CapProxy, Stats, apply_cap, context_error, template_inputs  # noqa: E402


def test_apply_cap_rule():
    # room = 100 - 50 = 50: a request for 16 fits unchanged, one for 80 is clamped, none gets the room
    b = {"max_tokens": 16}
    assert apply_cap(b, 50, window=100, min_completion=20) == ("ok", 16) and b["max_tokens"] == 16
    b = {"max_tokens": 80, "max_completion_tokens": 90}
    assert apply_cap(b, 50, window=100, min_completion=20) == ("clamped", 50)
    assert b == {"max_tokens": 50, "max_completion_tokens": 50}
    b = {}
    assert apply_cap(b, 50, window=100, min_completion=20) == ("clamped", 50) and b["max_tokens"] == 50
    # room 15 < min(20, 80): refused (end-at-limit); a request asking for only 10 still fits
    assert apply_cap({"max_tokens": 80}, 85, window=100, min_completion=20) == ("refuse", 20)
    assert apply_cap({"max_tokens": 10}, 85, window=100, min_completion=20) == ("ok", 10)
    # margin counts against the room; a full window is refused
    assert apply_cap({"max_tokens": 80}, 75, window=100, min_completion=20, margin=10) == ("refuse", 20)
    assert apply_cap({"max_tokens": 1}, 100, window=100, min_completion=20)[0] == "refuse"


def test_context_error_is_recognized_by_platoon_wording():
    msg = context_error(65536, 64000, 2048)["message"]
    assert msg.startswith("This model's maximum context length is 65536 tokens.")
    try:  # platoon's own classifier, when platoon is installed
        from platoon.gateway.hosted import context_length_error
    except ImportError:
        return
    err = context_length_error(400, msg)
    assert err is not None


def test_template_inputs_like_sglang():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                         {"role": "assistant", "content": "",
                          "tool_calls": [{"id": "1", "type": "function",
                                          "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]}],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
            "chat_template_kwargs": {"enable_thinking": True}}
    msgs, tools, kw = template_inputs(body)
    assert msgs[0]["content"] == "ab"
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert tools == [{"name": "bash", "parameters": {}, "strict": False}]
    assert kw == {"enable_thinking": True}
    assert isinstance(body["messages"][1]["tool_calls"][0]["function"]["arguments"], str)  # input untouched


def _fake_counter(body: dict) -> int:
    """1 token per character of message content (deterministic, no tokenizer)."""
    return sum(len(m.get("content") or "") for m in body["messages"])


async def _run_case():
    seen = []

    async def upstream(request):
        if request.method == "GET":
            return web.json_response({"data": [{"id": "policy"}]})
        body = await request.json()
        seen.append(body)
        prompt = _fake_counter(body) + 3          # the engine counts 3 more than we do
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                  "usage": {"prompt_tokens": prompt, "completion_tokens": 1}})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", upstream)
    up_runner = web.AppRunner(app)
    await up_runner.setup()
    site = web.TCPSite(up_runner, "127.0.0.1", 0)
    await site.start()
    up_port = site._server.sockets[0].getsockname()[1]

    stats = Stats(None, 60)
    proxy = CapProxy(f"http://127.0.0.1:{up_port}", window=100, min_completion=20, counter=_fake_counter,
                     stats=stats)
    runner, port = await proxy.start("127.0.0.1", 0)
    base = f"http://127.0.0.1:{port}/v1"
    out = {}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(base + "/models") as r:
                out["models"] = (r.status, await r.json())
            req = {"model": "policy", "messages": [{"role": "user", "content": "x" * 30}], "max_tokens": 16}
            async with s.post(base + "/chat/completions", json=req) as r:
                out["ok"] = (r.status, await r.json())
            req = {"model": "policy", "messages": [{"role": "user", "content": "x" * 60}], "max_tokens": 64}
            async with s.post(base + "/chat/completions", json=req) as r:
                out["clamped"] = (r.status, await r.json())
            req = {"model": "policy", "messages": [{"role": "user", "content": "x" * 90}], "max_tokens": 64}
            async with s.post(base + "/chat/completions", json=req) as r:
                out["refused"] = (r.status, await r.json())
    finally:
        await proxy.close()
        await runner.cleanup()
        await up_runner.cleanup()
    return out, seen, stats.snapshot()


def test_proxy_end_to_end():
    out, seen, snap = asyncio.run(_run_case())
    assert out["models"] == (200, {"data": [{"id": "policy"}]})
    assert out["ok"][0] == 200 and out["clamped"][0] == 200
    assert [b["max_tokens"] for b in seen] == [16, 40]            # clamped to 100 - 60; the refused one never went up
    status, err = out["refused"]
    assert status == 400 and "maximum context length is 100 tokens" in err["error"]["message"]
    assert snap["calls"] == 3 and snap["clamped"] == 1 and snap["refused"] == 1 and snap["other"] == 1
    assert snap["est_minus_engine"]["n"] == 2 and snap["est_minus_engine"]["mean"] == -3
    json.dumps(snap)
