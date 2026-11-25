"""Node agent: one per pool node, the process a node's batch job runs (inside the SGLang
container). It

  * starts dp = gpus // tp SGLang engines on this node's GPUs for the pool's active policy,
  * registers them in <pool>/nodes/<i>.json once healthy (host, advertised IP, URLs),
  * restarts an engine whose process died (with the pool's *current* active policy; on a node
    lent to a trainer, with the trainer's latest published weights or not at all, see below),
  * on node 0 also runs the sglang-router, publishes <pool>/router.json, and keeps the router's
    worker list equal to the ready engines of every node (add new / remove gone),
  * exits cleanly when <pool>/STOP appears or the job is cancelled (SIGTERM).

    python -m harnessforge.pool.node --pool-dir <root>/<pool_id> --node <i>

Lent nodes (pool.json "lent", set by SGLangPool.lend): the trainer owns the engines' weights,
so an engine that dies is never restarted on the pool's policy (that would silently put the
base model back into training). With on_restart="latest" it restarts on the weights named by
the lend's policy_file ({"id", "path"}: the trainer's latest full HF export), else it is
*held* down and reported ("held": <reason> in the node file) until the node is returned.

Stdlib only; reads everything else from pool.json.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from harnessforge.pool import http
from harnessforge.pool.state import PoolDir, write_json

POLL = float(os.environ.get("HF_POOL_POLL", 5))
#: what this agent supports; the controller checks it before lending a node (an older agent
#: would route lent engines or restart them on the base policy)
FEATURES = ("lend",)


_LOG_FILE: Path | None = None      # live agent log (the batch log only lands at job end on PBS)


def log(msg: str) -> None:
    line = f"[node {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        try:
            with open(_LOG_FILE, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass          # the state dir may be moved aside by stop(); never fail on logging


# ----------------------------------------------------------------------------- helpers
def render_args(args: dict[str, Any]) -> list[str]:
    """{"tool-call-parser": "qwen3_coder", "enable-metrics": True, "x": None} -> CLI flags.
    True -> bare flag; False/None -> omitted; list -> flag followed by every item."""
    out: list[str] = []
    for k, v in args.items():
        flag = k if k.startswith("--") else "--" + k.replace("_", "-")
        if v is None or v is False:
            continue
        if v is True:
            out.append(flag)
        elif isinstance(v, (list, tuple)):
            out += [flag, *map(str, v)]
        elif isinstance(v, dict):
            import json
            out += [flag, json.dumps(v)]
        else:
            out += [flag, str(v)]
    return out


def visible_gpus(want: int) -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        ids = [g for g in cvd.split(",") if g.strip()]
    else:
        try:
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
            ids = [str(i) for i, l in enumerate(r.stdout.splitlines()) if l.startswith("GPU")]
        except (OSError, subprocess.TimeoutExpired):
            ids = []
        if not ids:                                   # no GPUs (tests / fake engines)
            ids = [str(i) for i in range(want)]
    if len(ids) < want:
        log(f"WARNING: spec wants {want} GPUs per node, {len(ids)} visible: {ids}")
    return ids[:want] if want else ids


def advertise_ip(mode: str | None) -> str:
    """The address other nodes use to reach this one. 'auto' = the hostname's address (ABCI:
    the 10.0.x compute network); a prefix like '10.0.' picks the first local IP matching it;
    anything else is used verbatim."""
    host = socket.gethostname()
    if not mode or mode == "auto":
        try:
            return socket.gethostbyname(host)
        except OSError:
            return "127.0.0.1"
    if mode == "hostname":
        return host
    if mode.endswith("."):
        try:
            r = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=10)
            for ip in r.stdout.split():
                if ip.startswith(mode):
                    return ip
        except (OSError, subprocess.TimeoutExpired):
            pass
        return socket.gethostbyname(host)
    return mode


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def pick_port(start: int, taken: set[int], span: int = 1, stride: int = 37) -> int:
    """A free port >= start not in `taken` (ports start..start+span-1 all free)."""
    p = start
    for _ in range(2000):
        if all(q not in taken and port_free(q) for q in range(p, p + span)):
            taken.update(range(p, p + span))
            return p
        p += stride
    raise RuntimeError(f"no free port from {start}")


def killpg(p: subprocess.Popen, timeout: float = 30.0) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


# ----------------------------------------------------------------------------- router API
class RouterAPI:
    """sglang-router (0.3.x) worker management: GET/POST /workers, DELETE /workers/{id}.
    Older routers: /list_workers, /add_worker?url=, /remove_worker?url=."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def workers(self) -> list[str]:
        try:
            d = http.get_json(self.url + "/workers", timeout=10)
            items = d.get("workers", d) if isinstance(d, dict) else d
            return [w["url"] if isinstance(w, dict) else str(w) for w in items]
        except (http.HTTPError, ValueError, KeyError, TypeError):
            d = http.get_json(self.url + "/list_workers", timeout=10)
            return list(d.get("urls", []))

    def _ids(self) -> dict[str, str]:
        try:
            d = http.get_json(self.url + "/workers", timeout=10)
        except (http.HTTPError, ValueError):
            return {}
        items = d.get("workers", d) if isinstance(d, dict) else d
        return {w["url"]: str(w.get("id") or w.get("worker_id") or "")
                for w in items if isinstance(w, dict) and "url" in w}

    def add(self, url: str) -> None:
        status, text = http.request("POST", self.url + "/workers", {"url": url}, timeout=60)
        if status in (404, 405):
            status, text = http.request(
                "POST", self.url + "/add_worker?url=" + urllib.parse.quote(url, safe=""),
                timeout=60)
        if status >= 300:
            raise http.HTTPError(self.url + "/workers", status, text)

    def remove(self, url: str) -> None:
        wid = self._ids().get(url)
        if wid:
            status, text = http.request("DELETE", f"{self.url}/workers/{wid}", timeout=60)
        else:
            status, text = http.request(
                "DELETE", f"{self.url}/workers/{urllib.parse.quote(url, safe='')}", timeout=60)
            if status in (404, 405):
                status, text = http.request(
                    "POST", self.url + "/remove_worker?url=" + urllib.parse.quote(url, safe=""),
                    timeout=60)
        if status >= 300:
            raise http.HTTPError(self.url + "/workers", status, text)


# ----------------------------------------------------------------------------- the agent
class Engine:
    def __init__(self, k: int, gpus: list[str], port: int):
        self.k, self.gpus, self.port = k, gpus, port
        self.proc: subprocess.Popen | None = None
        self.ready = False
        self.restarts = 0
        self.started_at = 0.0
        self.policy: str | None = None
        self.held: str | None = None        # why a lent engine is not (re)started


class NodeAgent:
    def __init__(self, pool_dir: PoolDir, index: int):
        self.d = pool_dir
        self.i = index
        cfg = self.d.pool()
        self.cfg = cfg
        spec = cfg["spec"]
        self.tp = int(spec.get("tp", 1))
        self.gpus = visible_gpus(int(spec.get("gpus_per_node", 1)))
        self.dp = max(1, len(self.gpus) // self.tp)
        self.bind = cfg.get("bind_host", "0.0.0.0")
        self.ip = advertise_ip(cfg.get("advertise", "auto"))
        self.host = socket.gethostname()
        self.taken: set[int] = set()
        base = int(cfg.get("engine_port", 31000))
        # distinct per node so that several agents on one host (local tests) don't race
        self.engines = [Engine(k, self.gpus[k * self.tp:(k + 1) * self.tp],
                               pick_port(base + 200 * index + 20 * k, self.taken))
                        for k in range(self.dp)]
        self.router_proc: subprocess.Popen | None = None
        self.router_port: int | None = None
        self.stop = False
        self.max_restarts = int(cfg.get("max_engine_restarts", 3))
        self.startup_timeout = float(cfg.get("startup_timeout", 1800))
        self.d.logs.mkdir(parents=True, exist_ok=True)

    # -- engines
    def engine_argv(self, e: Engine, policy: dict[str, Any]) -> list[str]:
        spec = self.cfg["spec"]
        argv = list(self.cfg.get("engine_cmd") or [sys.executable, "-m", "sglang.launch_server"])
        argv += ["--model-path", policy["path"], "--host", self.bind, "--port", str(e.port),
                 "--tp-size", str(self.tp), "--served-model-name", spec.get("served_name", "policy"),
                 "--weight-version", policy["id"]]
        if policy.get("base_model"):
            argv += ["--tokenizer-path", policy["base_model"]]
        if spec.get("max_context"):
            argv += ["--context-length", str(spec["max_context"])]
        return argv + render_args(spec.get("engine_args") or {})

    def lent_policy(self, lend: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """The policy a lent engine may (re)start on: the trainer's latest published weights
        (lend["policy_file"] -> {"id", "path"}) when on_restart == "latest", else none."""
        if lend.get("on_restart", "down") != "latest":
            return None, f"lent to {lend.get('to')}; on_restart=down"
        pf = lend.get("policy_file")
        pol = None
        if pf:
            from harnessforge.pool.state import read_json
            pol = read_json(Path(pf))
        if not pol or not pol.get("path") or not Path(pol["path"]).exists():
            return None, f"lent to {lend.get('to')}; no published weights at {pf}"
        return {"id": pol.get("id") or Path(pol["path"]).name, "path": pol["path"],
                "base_model": pol.get("base_model")}, ""

    def start_engine(self, e: Engine) -> None:
        cfg = self.d.pool()
        policy = cfg["policy"]                    # always the CURRENT active policy
        lend = (cfg.get("lent") or {}).get(str(self.i))
        if lend is not None:
            lp, why = self.lent_policy(lend)
            if lp is None:
                if e.held != why:
                    log(f"engine {e.k} held down: {why}")
                e.proc, e.ready, e.held = None, False, why
                return
            policy = lp
        e.held = None
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(e.gpus)
        env.update({k: str(v) for k, v in (self.cfg.get("engine_env") or {}).items()})
        logf = self.d.logs / f"node{self.i}-engine{e.k}.log"
        argv = self.engine_argv(e, policy)
        log(f"engine {e.k} on GPUs {e.gpus} port {e.port}: {' '.join(argv)}")
        with open(logf, "ab") as out:
            e.proc = subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
        e.ready, e.started_at, e.policy = False, time.time(), policy["id"]

    def engine_url(self, e: Engine, local: bool = False) -> str:
        return f"http://{'127.0.0.1' if local else self.ip}:{e.port}"

    def check_engine(self, e: Engine) -> None:
        if e.proc is None:
            if e.held is not None and not self.stop:
                self.start_engine(e)              # re-evaluate: returned, or weights published
            return
        if e.proc.poll() is not None:
            log(f"engine {e.k} exited rc={e.proc.returncode}")
            e.ready = False
            if e.restarts < self.max_restarts and not self.stop:
                e.restarts += 1
                self.start_engine(e)
            return
        if not e.ready:
            if http.ok(self.engine_url(e, local=True) + "/health", timeout=5):
                e.ready = True
                log(f"engine {e.k} ready after {time.time() - e.started_at:.0f}s")
            elif time.time() - e.started_at > self.startup_timeout:
                log(f"engine {e.k} not healthy after {self.startup_timeout:.0f}s; restarting")
                killpg(e.proc)

    def publish(self, status: str) -> None:
        if not self.d.pool_file.exists():       # pool stopped and moved aside: don't recreate
            return
        write_json(self.d.node_file(self.i), {
            "index": self.i, "host": self.host, "ip": self.ip, "status": status,
            "job_id": os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or "",
            "pid": os.getpid(), "tp": self.tp, "updated_at": time.time(),
            "features": list(FEATURES),
            "engines": [{"k": e.k, "url": self.engine_url(e), "gpus": e.gpus, "port": e.port,
                         "ready": e.ready, "restarts": e.restarts, "policy": e.policy,
                         "held": e.held, "pid": e.proc.pid if e.proc is not None else None,
                         "alive": e.proc is not None and e.proc.poll() is None}
                        for e in self.engines]})

    # -- router (node 0)
    def router_url(self, local: bool = False) -> str:
        return f"http://{'127.0.0.1' if local else self.ip}:{self.router_port}"

    def start_router(self, workers: list[str]) -> None:
        if self.router_port is None:
            self.router_port = pick_port(int(self.cfg.get("router_port", 30000)), self.taken)
        argv = list(self.cfg.get("router_cmd")
                    or [sys.executable, "-m", "sglang_router.launch_router"])
        argv += ["--host", self.bind, "--port", str(self.router_port)]
        if workers:
            argv += ["--worker-urls", *workers]
        argv += render_args(self.cfg.get("router_args") or {})
        log(f"router :{self.router_port}: {' '.join(argv)}")
        with open(self.d.logs / "router.log", "ab") as out:
            self.router_proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 300 and self.router_proc.poll() is None:
            if http.ok(self.router_url(local=True) + "/health", timeout=5) or \
                    http.ok(self.router_url(local=True) + "/v1/models", timeout=5):
                break
            time.sleep(1)
        write_json(self.d.router_file, {"url": self.router_url(), "host": self.host,
                                        "ip": self.ip, "port": self.router_port,
                                        "node": self.i, "pid": self.router_proc.pid,
                                        "started_at": time.time()})

    def reconcile_router(self) -> None:
        if self.router_proc is None:
            return
        if self.router_proc.poll() is not None:
            log(f"router exited rc={self.router_proc.returncode}; restarting")
            self.start_router(self.d.engine_urls())
            return
        api = RouterAPI(self.router_url(local=True))
        try:
            have = set(api.workers())
        except (http.HTTPError, ValueError) as e:
            log(f"router worker list failed: {e}")
            return
        want = set(self.d.engine_urls())
        for u in sorted(want - have):
            try:
                api.add(u)
                log(f"router + {u}")
            except http.HTTPError as e:
                log(f"router add {u} failed: {e}")
        for u in sorted(have - want):
            try:
                api.remove(u)
                log(f"router - {u}")
            except http.HTTPError as e:
                log(f"router remove {u} failed: {e}")

    # -- main loop
    def shutdown(self, *_: Any) -> None:
        self.stop = True

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        log(f"node {self.i} on {self.host} ({self.ip}): {len(self.gpus)} GPUs, "
            f"tp={self.tp} dp={self.dp}")
        self.publish("starting")
        for e in self.engines:
            self.start_engine(e)
        status = "starting"
        try:
            while not self.stop and not self.d.stopping():
                for e in self.engines:
                    self.check_engine(e)
                if all(e.held is None and e.proc is not None and e.proc.poll() is not None
                       and e.restarts >= self.max_restarts for e in self.engines):
                    self.publish("failed")
                    log("all engines failed")
                    return 1
                ready = any(e.ready for e in self.engines)
                status = "ready" if ready else "starting"
                self.publish(status)
                if self.i == 0 and ready and self.router_proc is None:
                    self.start_router([self.engine_url(e) for e in self.engines if e.ready])
                if self.i == 0:
                    self.reconcile_router()
                time.sleep(POLL)
        finally:
            # kill children first: nothing below may prevent it
            log("shutting down")
            if self.router_proc is not None:
                killpg(self.router_proc, 10)
            for e in self.engines:
                if e.proc is not None:
                    killpg(e.proc)
            try:
                self.publish("stopped")
            except OSError:
                pass
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--node", type=int, required=True)
    a = ap.parse_args(argv)
    global _LOG_FILE
    d = PoolDir.at(a.pool_dir)
    d.logs.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = d.logs / f"node{a.node}.agent.log"
    return NodeAgent(d, a.node).run()


if __name__ == "__main__":
    sys.exit(main())
