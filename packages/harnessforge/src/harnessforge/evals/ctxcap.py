"""A context-window cap for hosted evaluations: an OpenAI-compatible proxy in front of a model endpoint.

    python -m harnessforge.evals.ctxcap --upstream http://<router>:30000 --tokenizer <HF snapshot dir> \
        --window 65536 --min-completion 2048 --port 0 --url-file ctxcap.url --stats ctxcap.json

Why: training enforces its context window at rollout in platoon's gateway (`train.context_window_length` 65536,
`ModelGateway.fit_max_tokens`, and with `platoon-tito2.patch` the end-at-limit rule: a prompt that leaves less than
`min(min_completion_tokens, max_tokens)` under the window ends the trajectory, which is then graded). A hosted
evaluation has no such cap: platoon's hosted gateway leaves the window to the provider, and the pool's SGLang engines
serve 262144 tokens. Restarting the engines with a smaller `--context-length` is a pool restart (and would cap every
other user of the pool). This proxy applies the training rule per request instead, so the evaluation's endpoint is
`http://127.0.0.1:<port>/v1` and nothing else changes:

- `POST .../chat/completions`: the prompt is rendered and counted with the model's own chat template, the way SGLang
  renders it (`apply_chat_template(messages, tools=[function...], add_generation_prompt=True,
  **chat_template_kwargs)`, tool-call arguments parsed from JSON). With `room = window - prompt - margin`:
  `room < min(min_completion, max_tokens)` is refused with HTTP 400 and the OpenAI/vLLM context-length wording
  ("This model's maximum context length is ..."), which platoon's hosted gateway counts as a context overflow
  (`AgentSession.context_overflows`; the rollout is graded with reason `context_overflow`, as training grades the
  limit) and litellm raises as `ContextWindowExceededError` (micro-swe-agent aborts on it); otherwise `max_tokens`
  (or `max_completion_tokens`) is clamped to `room`, as platoon's `fit_max_tokens` clamps. A request without either
  gets `room`.
- Everything else (`/v1/models`, ...) and the replies are passed through unchanged, streaming included.
- A request the template cannot render is forwarded unchanged and counted as `uncounted`.

`--stats` is rewritten every `--stats-every` seconds: calls, clamped, refused, uncounted, in flight, render time, and
the count's error against the engine's own `usage.prompt_tokens` (`est_minus_engine`: positive = we over-count),
which tells whether `--margin` is needed. Counting runs in `--workers` processes (a 60k-token conversation takes tens
of milliseconds to render and encode).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}

# --------------------------------------------------------------------------- counting (worker processes)

_TOKENIZER: Any = None


def _init_tokenizer(path: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer  # noqa: PLC0415 - only in the workers

    _TOKENIZER = AutoTokenizer.from_pretrained(path)


def template_inputs(body: dict[str, Any]) -> tuple[list[dict], list[dict] | None, dict[str, Any]]:
    """What SGLang's chat serving hands the template: messages with text content and parsed tool-call arguments,
    the tools' function definitions, and the request's chat_template_kwargs."""
    messages = []
    for m in body.get("messages") or []:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                call = dict(call)
                fn = dict(call.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except ValueError:
                        pass
                call["function"] = fn
                calls.append(call)
            m["tool_calls"] = calls
        messages.append(m)
    tools = None
    if body.get("tools"):
        tools = []
        for t in body["tools"]:
            fn = dict(t.get("function") or t)
            fn.setdefault("strict", False)  # pydantic's model_dump in SGLang adds it
            tools.append(fn)
    kwargs = dict(body.get("chat_template_kwargs") or {})
    return messages, tools, kwargs


def count_prompt(body: dict[str, Any]) -> int:
    """Prompt tokens of a chat request under the worker's tokenizer (`_init_tokenizer`)."""
    messages, tools, kwargs = template_inputs(body)
    ids = _TOKENIZER.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True, **kwargs)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return len(ids)


# --------------------------------------------------------------------------- the rule


def context_error(window: int, prompt: int, completion: int) -> dict[str, Any]:
    """The OpenAI/vLLM context-length refusal (what platoon's hosted gateway and litellm recognize)."""
    message = (f"This model's maximum context length is {window} tokens. However, you requested "
               f"{prompt + completion} tokens ({prompt} in the messages, {completion} in the completion). "
               "Please reduce the length of the messages or completion.")
    return {"object": "error", "message": message, "type": "BadRequestError", "param": None, "code": 400,
            "error": {"message": message, "type": "invalid_request_error", "param": "messages",
                      "code": "context_length_exceeded"}}


def apply_cap(body: dict[str, Any], prompt: int, *, window: int, min_completion: int,
              margin: int = 0) -> tuple[str, int]:
    """Decide one request: ("refuse", needed) or ("ok"/"clamped", max_tokens), editing `body` in place."""
    room = window - prompt - margin
    keys = [k for k in ("max_completion_tokens", "max_tokens") if body.get(k) is not None]
    asked = min(int(body[k]) for k in keys) if keys else None
    need = min_completion if asked is None else min(min_completion, asked)
    if room < max(need, 1):
        return "refuse", need
    if asked is None:
        body["max_tokens"] = room
        return "clamped", room
    if asked > room:
        for k in keys:
            body[k] = room
        return "clamped", room
    return "ok", asked


# --------------------------------------------------------------------------- the proxy


class Stats:
    def __init__(self, path: Path | None, every: float) -> None:
        self.path, self.every = path, every
        self.started = time.time()
        self.c = {"calls": 0, "clamped": 0, "refused": 0, "uncounted": 0, "upstream_errors": 0, "other": 0}
        self.inflight = 0
        self.render_ms: deque[float] = deque(maxlen=5000)
        self.err: deque[int] = deque(maxlen=5000)
        self.err_all = {"n": 0, "sum": 0, "min": None, "max": None}
        self.prompt_max = 0

    def note_err(self, d: int) -> None:
        self.err.append(d)
        a = self.err_all
        a["n"] += 1
        a["sum"] += d
        a["min"] = d if a["min"] is None else min(a["min"], d)
        a["max"] = d if a["max"] is None else max(a["max"], d)

    def snapshot(self) -> dict[str, Any]:
        def q(xs, p):
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
        a = self.err_all
        return {"time": time.time(), "uptime_s": round(time.time() - self.started), **self.c,
                "inflight": self.inflight, "prompt_max": self.prompt_max,
                "render_ms_p50": q(self.render_ms, 0.5), "render_ms_p90": q(self.render_ms, 0.9),
                "render_ms_max": max(self.render_ms) if self.render_ms else None,
                "est_minus_engine": {"n": a["n"], "mean": round(a["sum"] / a["n"], 2) if a["n"] else None,
                                     "min": a["min"], "max": a["max"],
                                     "recent_median": statistics.median(self.err) if self.err else None}}

    def write(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=1))
        tmp.replace(self.path)


class CapProxy:
    def __init__(self, upstream: str, *, window: int, min_completion: int, margin: int = 0,
                 counter: Callable[[dict], int] = count_prompt, executor: Executor | None = None,
                 stats: Stats | None = None, timeout_s: float = 3600.0) -> None:
        self.upstream = upstream.rstrip("/")
        self.window, self.min_completion, self.margin = window, min_completion, margin
        self.counter, self.executor = counter, executor
        self.stats = stats or Stats(None, 60)
        self.timeout_s = timeout_s
        self.session: Any = None

    async def count(self, body: dict) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.counter, body)

    async def handle(self, request: Any) -> Any:
        from aiohttp import web  # noqa: PLC0415

        raw = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        path = request.rel_url.path_qs
        body = None
        estimate = None
        if request.method == "POST" and request.rel_url.path.endswith("/chat/completions"):
            self.stats.c["calls"] += 1
            try:
                body = json.loads(raw)
            except ValueError:
                body = None
            if isinstance(body, dict):
                t0 = time.perf_counter()
                try:
                    estimate = await self.count(body)
                except Exception as exc:  # noqa: BLE001 - forward what we cannot count, as is
                    print(f"[ctxcap] uncounted request: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr,
                          flush=True)
                    self.stats.c["uncounted"] += 1
                    estimate = None
                self.stats.render_ms.append(round((time.perf_counter() - t0) * 1000, 1))
                if estimate is not None:
                    self.stats.prompt_max = max(self.stats.prompt_max, estimate)
                    verdict, value = apply_cap(body, estimate, window=self.window,
                                               min_completion=self.min_completion, margin=self.margin)
                    if verdict == "refuse":
                        self.stats.c["refused"] += 1
                        return web.json_response(context_error(self.window, estimate + self.margin, value),
                                                 status=400)
                    if verdict == "clamped":
                        self.stats.c["clamped"] += 1
                    raw = json.dumps(body).encode()
                    headers["Content-Type"] = "application/json"
        else:
            self.stats.c["other"] += 1
        self.stats.inflight += 1
        try:
            async with self.session.request(request.method, self.upstream + path, data=raw,
                                            headers=headers) as up:
                out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP_HEADERS
                               and k.lower() != "content-encoding"}
                if body is not None and body.get("stream"):
                    resp = web.StreamResponse(status=up.status, headers=out_headers)
                    await resp.prepare(request)
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
                data = await up.read()
                if estimate is not None and up.status == 200:
                    try:
                        usage = json.loads(data).get("usage") or {}
                        if usage.get("prompt_tokens") is not None:
                            self.stats.note_err(estimate - int(usage["prompt_tokens"]))
                    except ValueError:
                        pass
                return web.Response(status=up.status, body=data, headers=out_headers)
        except Exception as exc:  # noqa: BLE001 - the client gets a 502, platoon retries it
            self.stats.c["upstream_errors"] += 1
            return web.json_response({"error": {"message": f"ctxcap upstream error: {type(exc).__name__}: {exc}",
                                                "type": "upstream_error"}}, status=502)
        finally:
            self.stats.inflight -= 1

    async def start(self, host: str, port: int) -> tuple[Any, int]:
        import aiohttp  # noqa: PLC0415
        from aiohttp import web  # noqa: PLC0415

        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, force_close=False),
            timeout=aiohttp.ClientTimeout(total=None, sock_read=self.timeout_s, sock_connect=60),
            auto_decompress=False)
        app = web.Application(client_max_size=1 << 30)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        bound = site._server.sockets[0].getsockname()[1]  # noqa: SLF001 - the port when port=0
        return runner, bound

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()


async def _serve(args: argparse.Namespace) -> None:
    stats = Stats(Path(args.stats) if args.stats else None, args.stats_every)
    executor = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_tokenizer,
                                   initargs=(os.path.expanduser(args.tokenizer),))
    proxy = CapProxy(args.upstream, window=args.window, min_completion=args.min_completion, margin=args.margin,
                     executor=executor, stats=stats, timeout_s=args.timeout)
    # warm every worker (tokenizer load) before announcing the URL
    probe = {"messages": [{"role": "user", "content": "hello"}]}
    await asyncio.gather(*(proxy.count(probe) for _ in range(args.workers)))
    runner, port = await proxy.start(args.host, args.port)
    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{port}/v1"
    if args.url_file:
        Path(args.url_file).write_text(url + "\n")
    print(f"[ctxcap] {url} -> {args.upstream} window={args.window} min_completion={args.min_completion} "
          f"margin={args.margin} workers={args.workers}", flush=True)
    try:
        while True:
            await asyncio.sleep(args.stats_every)
            stats.write()
    finally:
        stats.write()
        await proxy.close()
        await runner.cleanup()
        executor.shutdown(cancel_futures=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessforge.evals.ctxcap", description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", required=True, help="the model endpoint's origin, e.g. http://10.0.0.1:30000")
    ap.add_argument("--tokenizer", required=True, help="HF model dir (or id) whose chat template the engine uses")
    ap.add_argument("--window", type=int, default=65536)
    ap.add_argument("--min-completion", type=int, default=2048,
                    help="refuse when less than min(this, max_tokens) is left (platoon-tito2's end-at-limit)")
    ap.add_argument("--margin", type=int, default=0, help="tokens kept free for render differences")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds an upstream reply may take")
    ap.add_argument("--url-file", help="write the proxy's /v1 URL here once it serves")
    ap.add_argument("--stats", help="JSON stats file, rewritten every --stats-every seconds")
    ap.add_argument("--stats-every", type=float, default=60.0)
    args = ap.parse_args(argv)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
