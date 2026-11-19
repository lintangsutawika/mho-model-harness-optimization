"""Local drop test: stub model server <- bridge.py -> [cuttable TCP proxy] -> relay_server.py <- clients.
Cuts every proxied connection twice while 300 requests are in flight; expects all 300 to return the
right body and each request to have been executed exactly once by the stub."""
import asyncio, collections, os, random, socket, subprocess, sys, threading, time
from pathlib import Path
import httpx, uvicorn
from fastapi import FastAPI, Request

HERE = Path(__file__).resolve().parents[2] / "src" / "harness" / "relay"
sys.path.insert(0, str(HERE))
import relay_server

def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

SECRET = "s" * 48
EXEC = collections.Counter()

def serve(app, p):
    threading.Thread(target=uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=p, log_level="warning",
                     ws_ping_interval=30, ws_ping_timeout=90)).run, daemon=True).start()

stub = FastAPI()
@stub.post("/v1/chat/completions")
async def chat(req: Request):
    j = await req.json(); EXEC[j["n"]] += 1
    await asyncio.sleep(random.uniform(0.5, 4.0))
    return {"n": j["n"], "pad": "x" * 30000}

CONNS = set()
async def proxy_main(listen, target):
    async def pipe(r, w):
        try:
            while d := await r.read(65536): w.write(d); await w.drain()
        except Exception: pass
        finally: w.close()
    async def on(cr, cw):
        ur, uw = await asyncio.open_connection("127.0.0.1", target)
        CONNS.add(cw); CONNS.add(uw)
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
    srv = await asyncio.start_server(on, "127.0.0.1", listen)
    return srv

async def main():
    sp, rp, pp = port(), port(), port()
    serve(stub, sp); serve(relay_server.build_app(SECRET, 120, 60), rp)
    await proxy_main(pp, rp)
    env = dict(os.environ, RELAY_WS_URL=f"ws://127.0.0.1:{pp}/bridge", RELAY_SECRET=SECRET,
               VLLM_LOCAL_URL=f"http://127.0.0.1:{sp}")
    br = subprocess.Popen([sys.executable, str(HERE / "bridge.py")], env=env, stderr=subprocess.PIPE, text=True)
    async with httpx.AsyncClient(timeout=200, limits=httpx.Limits(max_connections=None)) as c:
        for _ in range(50):
            try:
                if (await c.get(f"http://127.0.0.1:{rp}/health")).json()["bridge_connected"]: break
            except Exception: pass
            await asyncio.sleep(0.2)
        async def one(n):
            await asyncio.sleep(random.uniform(0, 6))
            r = await c.post(f"http://127.0.0.1:{rp}/k/{SECRET}/v1/chat/completions", json={"n": n})
            return r.status_code == 200 and r.json()["n"] == n
        async def cutter():
            for t in (2.0, 5.0):
                await asyncio.sleep(t)
                k = len(CONNS); [w.transport.abort() for w in list(CONNS)]; CONNS.clear()
                print(f"[cut] aborted {k} proxied sockets", flush=True)
        t0 = time.time()
        res, _ = await asyncio.gather(asyncio.gather(*(one(n) for n in range(300))), cutter())
        h = (await c.get(f"http://127.0.0.1:{rp}/health")).json()
    br.terminate(); err = br.stderr.read()
    ok = sum(res); dup = sum(1 for v in EXEC.values() if v > 1); missing = 300 - len(EXEC)
    print(f"ok {ok}/300 in {time.time()-t0:.1f}s; executed twice: {dup}; never executed: {missing}; "
          f"bridge connects: {h['bridge_connects']}; pending after: {h['pending']}")
    print("bridge log:", sorted(collections.Counter(l.split(':')[0][:60] for l in err.splitlines()).items())[:6])
    print("PASS" if ok == 300 and dup == 0 and h["bridge_connects"] >= 3 else "FAIL")

asyncio.run(main())
