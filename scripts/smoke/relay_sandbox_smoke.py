"""Smoke + load test for Sandbox relays (src/harness/relay/sandbox_relay.py).

Runs a stub model server on this host, starts a relay Sandbox, connects the real node bridge
(bridge.py) to it over the tunnel's WebSocket, and checks:
  1. auth: a bad key gets 401 on both routes
  2. load: N concurrent ~60 KB requests (half Bearer /v1, half key-in-path /k/<secret>/v1) all
     return 200 with the right body
  3. rotation: a second relay + bridge serves traffic while the first one is still up; then the
     first is terminated and the second keeps working
  4. reaping: a relay with no bridge and a short idle_timeout is gone on its own

    uv run --with fastapi --with uvicorn --with websockets --with httpx \
        python scripts/smoke/relay_sandbox_smoke.py [--n 256] [--skip-reap]
"""
import argparse
import asyncio
import os
import random
import secrets
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import modal

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "harness" / "relay"))
import sandbox_relay  # noqa: E402

BRIDGE = REPO / "src" / "harness" / "relay" / "bridge.py"


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def start_stub(port: int) -> None:
    """Stub OpenAI-ish server: sleeps 0.5-3 s, answers with ~20 KB that echoes the request size."""
    import uvicorn
    from fastapi import FastAPI, Request

    api = FastAPI()

    @api.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.body()
        await asyncio.sleep(random.uniform(0.5, 3.0))
        return {"choices": [{"message": {"content": "x" * 20000}}], "req_bytes": len(body)}

    cfg = uvicorn.Config(api, host="127.0.0.1", port=port, log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()


def start_bridge(url: str, secret: str, stub: int) -> subprocess.Popen:
    env = dict(os.environ, RELAY_WS_URL=url.replace("https://", "wss://") + "/bridge",
               RELAY_SECRET=secret, VLLM_LOCAL_URL=f"http://127.0.0.1:{stub}", BRIDGE_TIMEOUT="600")
    return subprocess.Popen([sys.executable, str(BRIDGE)], env=env)


def wait_bridge(url: str, s: float = 120) -> None:
    t = time.time() + s
    while time.time() < t:
        try:
            if httpx.get(url + "/health", timeout=10).json().get("bridge_connected"):
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"bridge never connected to {url}")


async def fire(url: str, secret: str, n: int) -> tuple[int, list[float], list[str]]:
    body = {"model": "stub", "messages": [{"role": "user", "content": "y" * 60000}]}
    lat, errs = [], []
    lim = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=300, limits=lim) as c:
        async def one(i: int):
            t = time.time()
            try:
                if i % 2:
                    r = await c.post(f"{url}/v1/chat/completions", json=body,
                                     headers={"authorization": f"Bearer {secret}"})
                else:
                    r = await c.post(f"{url}/k/{secret}/v1/chat/completions", json=body)
                ok = r.status_code == 200 and r.json().get("req_bytes", 0) > 60000
                if not ok:
                    errs.append(f"{r.status_code} {r.text[:100]}")
                lat.append(time.time() - t)
                return ok
            except Exception as e:
                errs.append(repr(e)[:120])
                return False
        res = await asyncio.gather(*(one(i) for i in range(n)))
    return sum(res), lat, errs


def report(tag: str, n: int, res) -> bool:
    ok, lat, errs = res
    q = sorted(lat)
    print(f"[{tag}] {ok}/{n} ok; latency p50 {statistics.median(q):.1f}s p95 {q[int(.95*len(q))-1]:.1f}s"
          f" max {q[-1]:.1f}s" + (f"; errors: {errs[:3]}" if errs else ""), flush=True)
    return ok == n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--skip-reap", action="store_true")
    a = ap.parse_args()
    os.environ["RELAY_SECRET"] = secret = secrets.token_hex(24)
    stub = free_port(); start_stub(stub)
    passed, procs, boxes = True, [], []
    try:
        t0 = time.time()
        A = sandbox_relay.up("smoke-A", 3600, 300, 4.0, 300); boxes.append(A["sandbox_id"])
        print(f"[up] relay A healthy in {time.time()-t0:.0f}s", flush=True)
        procs.append(start_bridge(A["url"], secret, stub)); wait_bridge(A["url"])

        bad = [httpx.post(A["url"] + "/v1/chat/completions", json={}, headers={"authorization": "Bearer nope"}).status_code,
               httpx.post(A["url"] + "/k/nope/v1/chat/completions", json={}).status_code]
        print(f"[auth] bad key -> {bad}", flush=True); passed &= bad == [401, 401]

        passed &= report("load A", a.n, asyncio.run(fire(A["url"], secret, a.n)))

        # Drop: kill the bridge mid-load and start a new one 5 s later; the relay replays the
        # unanswered requests to the new bridge, so every request still succeeds.
        async def drop():
            async def killer():
                await asyncio.sleep(3)
                procs[0].kill(); await asyncio.sleep(5)
                procs[0] = start_bridge(A["url"], secret, stub)
            r, _ = await asyncio.gather(fire(A["url"], secret, 128), killer())
            return r
        passed &= report("drop: bridge killed + restarted mid-load", 128, asyncio.run(drop()))

        B = sandbox_relay.up("smoke-B", 3600, 300, 4.0, 300); boxes.append(B["sandbox_id"])
        procs.append(start_bridge(B["url"], secret, stub)); wait_bridge(B["url"])

        async def both():
            return await asyncio.gather(fire(A["url"], secret, 64), fire(B["url"], secret, 64))
        ra, rb = asyncio.run(both())
        passed &= report("rotation: A while B up", 64, ra) & report("rotation: B while A up", 64, rb)
        sandbox_relay.down(A["sandbox_id"]); boxes.remove(A["sandbox_id"])
        time.sleep(5)
        try:
            gone = httpx.get(A["url"] + "/health", timeout=15).status_code != 200
        except Exception:
            gone = True
        print(f"[rotation] A terminated -> unreachable: {gone}", flush=True); passed &= gone
        passed &= report("rotation: B after A down", 64, asyncio.run(fire(B["url"], secret, 64)))

        if not a.skip_reap:
            C = sandbox_relay.up("smoke-C-noBridge", 3600, 60, 1.0, 300)
            print("[reap] relay C up with no bridge, idle_timeout 60 s; waiting up to 4 min", flush=True)
            sb, t = modal.Sandbox.from_id(C["sandbox_id"]), time.time()
            while sb.poll() is None and time.time() - t < 240:
                time.sleep(10)
            reaped = sb.poll() is not None
            print(f"[reap] relay C reaped on its own after {time.time()-t:.0f}s: {reaped}", flush=True)
            if not reaped:
                sb.terminate()
            passed &= reaped
    finally:
        for p in procs:
            p.terminate()
        for sid in boxes:
            try:
                sandbox_relay.down(sid)
            except Exception:
                pass
    print("PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
