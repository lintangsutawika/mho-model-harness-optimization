"""Reverse-relay web server: the public rendezvous between cloud sandboxes and a private node.

Cloud agents call this server over HTTPS; the node's bridge (bridge.py) dials OUT to /bridge and
holds a WebSocket. Each HTTP request is forwarded as a frame over that socket, the bridge proxies
it to the node-local model server, and the response comes back the same way.

    agent ──HTTPS──► relay (this server) ──WS──► node bridge ──► local vLLM / session server

Runs inside a Modal Sandbox (sandbox_relay.py starts it and exposes its port through a tunnel),
so the same routes and auth as the Modal-function relay (modal_relay.py):
  /bridge                WebSocket for the node bridge (Authorization: Bearer <secret>)
  /health                {"ok", "bridge_connected", "pending", "bridge_connects"}
  /v1/{path}             OpenAI-compatible proxy (Authorization: Bearer <secret>)
  /k/{secret}/{path}     key-in-path proxy, for clients that cannot set the header (Miles'
                         session URLs); forwards /{path}

Surviving a dropped bridge socket: a request stays pending (with its frame) until its response
arrives or it times out, not just until its first send. When the bridge (re)connects, every
pending frame is sent again; the bridge de-duplicates by id (it answers from its response cache
or ignores a request it is still working on), so a socket drop no longer loses requests or
responses in flight. Requests that arrive while no bridge is connected wait for one
(RELAY_BRIDGE_WAIT) instead of failing at once.

Env: RELAY_SECRET (required), RELAY_REQUEST_TIMEOUT (s, default 600), RELAY_PORT (default 8000),
     RELAY_BRIDGE_WAIT (s, default 120).
"""
import asyncio
import hmac
import json
import os
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket


def build_app(secret: str, request_timeout: float, bridge_wait: float = 120.0) -> FastAPI:
    api = FastAPI()
    # One bridge per relay; a reconnecting bridge replaces the old socket.
    state: dict = {"ws": None, "connects": 0}
    connected = None  # asyncio.Event, created on the server's loop (see _connected)
    # id -> (future, serialized request frame); kept until answered or timed out.
    pending: dict[str, tuple[asyncio.Future, str]] = {}
    # Starlette's send_text is not concurrency-safe: interleaved writes corrupt frames and the
    # peer drops the socket (1002). Serialize every send.
    send_lock = asyncio.Lock()

    def _connected() -> asyncio.Event:
        nonlocal connected
        if connected is None:
            connected = asyncio.Event()
        return connected

    async def send(ws, payload: str) -> bool:
        try:
            async with send_lock:
                await ws.send_text(payload)
            return True
        except Exception:
            return False

    @api.websocket("/bridge")
    async def bridge(ws: WebSocket):
        if not hmac.compare_digest(ws.headers.get("authorization", ""), f"Bearer {secret}"):
            await ws.close(code=1008)
            return
        await ws.accept()
        state["ws"] = ws
        state["connects"] += 1
        _connected().set()
        # Replay everything still unanswered: frames sent on the old socket may never have
        # reached the bridge, and their responses may have been lost on the way back.
        for _fut, payload in list(pending.values()):
            if not await send(ws, payload):
                break
        try:
            while True:
                raw = await ws.receive_text()
                # Parse large response bodies off the loop so keepalives keep flowing.
                frame = await asyncio.to_thread(json.loads, raw)
                entry = pending.pop(frame["id"], None)
                if entry is not None and not entry[0].done():
                    entry[0].set_result(frame)
        except Exception:
            pass
        finally:
            if state["ws"] is ws:
                state["ws"] = None
                _connected().clear()

    @api.get("/health")
    async def health():
        return {"ok": True, "bridge_connected": state["ws"] is not None,
                "pending": len(pending), "bridge_connects": state["connects"]}

    @api.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {secret}"):
            raise HTTPException(status_code=401, detail="bad relay token")
        return await forward(f"/v1/{path}", request)

    @api.api_route("/k/{key}/{path:path}", methods=["GET", "POST"])
    async def proxy_keyed(key: str, path: str, request: Request):
        if not hmac.compare_digest(key, secret):
            raise HTTPException(status_code=401, detail="bad relay token")
        return await forward(f"/{path}", request)

    async def forward(path: str, request: Request):
        if state["ws"] is None:
            try:
                await asyncio.wait_for(_connected().wait(), timeout=bridge_wait)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="no bridge connected")
        rid = uuid.uuid4().hex
        frame = {"id": rid, "method": request.method, "path": path,
                 "query": request.url.query or "", "body": (await request.body()).decode()}
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        payload = await asyncio.to_thread(json.dumps, frame)
        pending[rid] = (fut, payload)
        ws = state["ws"]
        # A failed send is not fatal: the frame stays pending and is replayed on reconnect.
        if ws is not None:
            await send(ws, payload)
        try:
            resp = await asyncio.wait_for(fut, timeout=request_timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="relay timeout waiting for node")
        finally:
            pending.pop(rid, None)
        return Response(content=resp["body"], status_code=resp["status"],
                        media_type=resp.get("content_type", "application/json"))

    return api


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(os.environ["RELAY_SECRET"],
                          float(os.environ.get("RELAY_REQUEST_TIMEOUT", "600")),
                          float(os.environ.get("RELAY_BRIDGE_WAIT", "120"))),
                host="0.0.0.0", port=int(os.environ.get("RELAY_PORT", "8000")),
                # Long-lived bridge socket: ping rarely, tolerate a busy loop (see bridge.py).
                ws_ping_interval=30, ws_ping_timeout=90, log_level="warning")
