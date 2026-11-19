"""
Node-side bridge for the Modal reverse relay (harbor/relay/modal_relay.py).

Runs on the vLLM node (same Slurm job as the server). Dials OUT to the relay's
WebSocket and holds it open; for each request frame the relay forwards, it calls
the local vLLM and sends the response back. This is the leg that makes a private
node reachable from Modal without accepting any inbound connection.

Env:
    RELAY_WS_URL    wss://<workspace>--tts-vllm-relay-web.modal.run/bridge
    RELAY_SECRET    shared secret (matches the Modal 'tts-relay' secret)
    VLLM_LOCAL_URL  local vLLM base, e.g. http://localhost:8000  (no trailing /v1)
    BACKEND_API_KEY optional bearer key sent to the local server (SGLang --api-key)

Run (deps resolved by uv):
    uv run --with websockets --with httpx python harbor/relay/bridge.py
"""

import asyncio
import collections
import json
import os
import re
import socket
import subprocess
import sys
import time

import httpx
import websockets

RELAY_WS_URL = os.environ["RELAY_WS_URL"]
RELAY_SECRET = os.environ["RELAY_SECRET"]
VLLM_LOCAL_URL = os.environ.get("VLLM_LOCAL_URL", "http://localhost:8000").rstrip("/")
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "")

# A path of the form /_hp/<host>:<port>/rest targets that host:port instead of
# VLLM_LOCAL_URL (Miles runs one session server per cell on the node IP, ports
# base..base+N-1, and each session lives on one of them). Only this node's own
# addresses and BRIDGE_ALLOWED_PORTS are reachable, so the relay is not an open proxy.
_HP = re.compile(r"^/_hp/([0-9A-Za-z.\-]+):(\d+)(/.*)?$")


def _local_hosts() -> set:
    hs = {"127.0.0.1", "localhost", socket.gethostname()}
    try:
        hs.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        hs.update(subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split())
    except Exception:
        pass
    return hs


ALLOWED_HOSTS = _local_hosts()
_PORT_LO, _PORT_HI = (int(x) for x in os.environ.get("BRIDGE_ALLOWED_PORTS", "30000-31023").split("-"))


def target_url(path: str) -> str:
    m = _HP.match(path)
    if not m:
        return VLLM_LOCAL_URL + path
    host, port, rest = m.group(1), int(m.group(2)), m.group(3) or "/"
    if host not in ALLOWED_HOSTS or not _PORT_LO <= port <= _PORT_HI:
        raise ValueError(f"target {host}:{port} not allowed")
    return f"http://{host}:{port}{rest}"


def log(msg: str) -> None:
    print(f"[bridge] {msg}", flush=True, file=sys.stderr)


# Surviving a dropped socket (with relay_server.py, which replays every unanswered request
# frame when the bridge reconnects): responses go out on whichever socket is current, a response
# that could not be sent stays in RECENT and is re-sent when the relay asks for that id again,
# and a replayed id we are still working on is ignored, so a request is never run twice.
# (modal_relay.py does not replay; there the cache is simply unused.)
RESP_TTL = float(os.environ.get("BRIDGE_RESP_TTL", "300"))   # keep sent responses this long
ID_TTL = float(os.environ.get("BRIDGE_ID_TTL", "7200"))      # remember answered ids this long
CUR: dict = {"ws": None, "lock": None}
INFLIGHT: set = set()
RECENT: "collections.OrderedDict[str, tuple[float, str]]" = collections.OrderedDict()
DONE_IDS: "collections.OrderedDict[str, float]" = collections.OrderedDict()


def _prune(now: float) -> None:
    while RECENT and now - next(iter(RECENT.values()))[0] > RESP_TTL:
        RECENT.popitem(last=False)
    while DONE_IDS and now - next(iter(DONE_IDS.values())) > ID_TTL:
        DONE_IDS.popitem(last=False)


async def send_current(payload: str, rid: str) -> None:
    """Send on the current socket; on failure keep it cached for the relay's replay."""
    ws, lock = CUR["ws"], CUR["lock"]
    if ws is None:
        log(f"deferring response {rid}: no relay connection (resent on replay)")
        return
    # Serialize sends: concurrent writes on one socket interleave frames (1002 on the relay).
    try:
        async with lock:
            await ws.send(payload)
    except Exception as e:
        log(f"deferring response {rid}: ws send failed ({e}); resent on replay")


async def handle(frame: dict, client: httpx.AsyncClient) -> None:
    """Proxy one relayed request to the local server and send back the response frame."""
    try:
        url = target_url(frame["path"])
        if frame.get("query"):
            url += "?" + frame["query"]
        # A connect failure means the request never reached the server, so retrying is safe
        # (a burst of replayed requests after a reconnect can briefly overflow its accept queue).
        for attempt in range(4):
            try:
                r = await client.request(
                    frame["method"],
                    url,
                    content=frame["body"].encode(),
                    headers={"content-type": "application/json",
                             **({"authorization": f"Bearer {BACKEND_API_KEY}"} if BACKEND_API_KEY else {})},
                )
                break
            except httpx.ConnectError:
                if attempt == 3:
                    raise
                await asyncio.sleep(0.5 * 2 ** attempt)
        resp = {
            "id": frame["id"],
            "status": r.status_code,
            "body": r.text,
            "content_type": r.headers.get("content-type", "application/json"),
        }
    except Exception as e:  # return a 502 rather than dropping the request
        resp = {
            "id": frame["id"],
            "status": 502,
            "body": json.dumps({"error": {"message": f"bridge->vllm failed: {type(e).__name__}: {e}"}}),
            "content_type": "application/json",
        }
    # Serialize the response body (a completion, up to ~100s of KB) off the event loop -- a
    # synchronous json.dumps of a large payload would starve the WebSocket keepalive.
    payload = await asyncio.to_thread(json.dumps, resp)
    now = time.monotonic()
    RECENT[frame["id"]] = (now, payload)
    DONE_IDS[frame["id"]] = now
    INFLIGHT.discard(frame["id"])
    _prune(now)
    await send_current(payload, frame["id"])


def on_frame(frame: dict, client: httpx.AsyncClient) -> None:
    rid = frame["id"]
    if rid in RECENT:                      # replayed: we answered, the answer was lost
        asyncio.create_task(send_current(RECENT[rid][1], rid))
    elif rid in INFLIGHT:                  # replayed while still running: answer comes later
        pass
    elif rid in DONE_IDS:                  # answered long ago, body no longer cached
        lost = {"id": rid, "status": 502, "content_type": "application/json",
                "body": json.dumps({"error": {"message": "bridge: response lost on reconnect"}})}
        asyncio.create_task(send_current(json.dumps(lost), rid))
    else:
        INFLIGHT.add(rid)
        asyncio.create_task(handle(frame, client))


async def main() -> None:
    headers = {"Authorization": f"Bearer {RELAY_SECRET}"}
    async with httpx.AsyncClient(trust_env=False, timeout=float(os.environ.get("BRIDGE_TIMEOUT", "600")),
                                 limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)) as client:
        while True:
            try:
                async with websockets.connect(
                    RELAY_WS_URL,
                    additional_headers=headers,
                    max_size=None,
                    # A momentarily-busy loop on either end must not be a death
                    # sentence: at 64-way concurrency the relay/bridge loops spend
                    # bursts shuttling large SWE-bench payloads, and a 20s pong
                    # deadline was tripping 1011 keepalive timeouts that dropped
                    # every in-flight request. Ping less often, wait far longer.
                    ping_interval=30,
                    ping_timeout=90,
                ) as ws:
                    CUR["ws"], CUR["lock"] = ws, asyncio.Lock()  # one lock per connection
                    log(f"connected to relay; forwarding to {VLLM_LOCAL_URL}")
                    async for msg in ws:
                        # Parse the (large) request frame off the loop so keepalives
                        # keep flowing while a hefty prompt body deserializes.
                        on_frame(await asyncio.to_thread(json.loads, msg), client)
            except Exception as e:
                log(f"disconnected ({e}); reconnecting in 2s")
            finally:
                CUR["ws"] = None
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
