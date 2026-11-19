"""SGLangPool: contracts.ModelPool on N batch-job nodes of SGLang engines behind one sglang-router.

    pool = SGLangPool(PoolSpec(nodes=2, gpus_per_node=1, engine_args={...},
                               launcher={"kind": "pbs", "queue": "R9920261300",
                                         "account": "gai51740", "resource_type": "rt_HG",
                                         "walltime": "2:00:00"}),
                      policy=PolicyRef("qwen3.5-4b", "Qwen/Qwen3.5-4B"),
                      pool_id="dev", sif="/path/miles.sif")
    pool.start()                 # one job per node; returns when the router serves every engine
    pool.endpoint()              # http://<node0>:<port>/v1  (OpenAI-compatible)
    pool.load(PolicyRef("step200", "/groups/.../hf/step200", base_model="Qwen/Qwen3.5-4B"))
    pool.health(); pool.stop()

Any other process (an eval job on another node) attaches with `SGLangPool.attach(pool_id)` or
just reads the endpoint with `harnessforge.pool.endpoint_of(pool_id)`.

Lending to a trainer: `pool.lend([3], to="grpo-run", policy_file=...)` hands node 3's engines to
a trainer (miles-tinker attaches them as external rollout engines and owns their weights). While
lent they are out of the router (no evals), `load()` skips them, and an engine that dies does not
come back on the pool's policy (node agent: held down, or on the trainer's latest published
weights). `pool.unlend([3])` reloads the pool's active policy into them, verifies it, and puts
them back into the router.

Design: the controller (this object) only writes pool.json, submits/cancels jobs and makes
control HTTP calls (load/health). Everything long-lived runs in the node jobs (pool/node.py):
engines, the router (node 0), router membership, engine restarts. So the controller can exit
after start() and a later process can load()/stop() the same pool.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from harnessforge.contracts import JobSpec, JobState, Launcher, PolicyRef, PoolSpec
from harnessforge.pool import http
from harnessforge.pool.state import DEFAULT_ROOT, PoolDir

SRC_DIR = Path(__file__).resolve().parents[2]     # .../src (contains harnessforge/)

LAUNCHER_KEYS = ("queue", "account", "resource_type", "rtype", "select", "host", "mem",
                 "extra_args")


class PoolError(RuntimeError):
    pass


def policy_to_dict(p: PolicyRef) -> dict[str, Any]:
    return asdict(p)


def policy_from_dict(d: dict[str, Any]) -> PolicyRef:
    return PolicyRef(id=d["id"], path=d["path"], base_model=d.get("base_model"))


def spec_from_dict(d: dict[str, Any]) -> PoolSpec:
    fields = PoolSpec.__dataclass_fields__
    return PoolSpec(**{k: v for k, v in d.items() if k in fields})


class SGLangPool:
    """See module docstring. Pool-level options beyond PoolSpec:

    sif           container image with SGLang + sglang-router (None = run on the host python,
                  e.g. local tests with fake engines)
    binds         extra singularity --bind paths (e.g. ["/groups"]) for checkpoints outside $HOME
    engine_env    extra env for engine processes
    router_args   sglang-router flags, e.g. {"policy": "cache_aware"}
    advertise     how nodes advertise their address: "auto" (hostname's IP), "hostname",
                  an IP prefix like "10.0.", or a literal address
    engine_cmd / router_cmd   override the engine/router argv prefix (tests use fakes)
    """

    def __init__(self, spec: PoolSpec, policy: PolicyRef | None = None, *, pool_id: str,
                 state_root: Path | str | None = None, sif: str | None = None,
                 launcher: Launcher | None = None, binds: list[str] | None = None,
                 engine_env: dict[str, str] | None = None,
                 router_args: dict[str, Any] | None = None, advertise: str = "auto",
                 engine_cmd: list[str] | None = None, router_cmd: list[str] | None = None,
                 python: str | None = None, startup_timeout: float = 1800,
                 job_env: dict[str, str] | None = None, container_bin: str | None = None):
        self.spec = spec
        self._policy = policy
        self.pool_id = pool_id
        self.dir = PoolDir(state_root or DEFAULT_ROOT, pool_id)
        self.sif = sif
        self._launcher = launcher
        self.binds = list(binds or [])
        self.engine_env = dict(engine_env or {})
        self.router_args = {"policy": "cache_aware"} if router_args is None else dict(router_args)
        self.advertise = advertise
        self.engine_cmd = engine_cmd
        self.router_cmd = router_cmd
        self.python = python or ("python" if sif else sys.executable)
        self.startup_timeout = startup_timeout
        self.job_env = dict(job_env or {})
        self.container_bin = container_bin

    # ------------------------------------------------------------------ construction
    @classmethod
    def attach(cls, pool_id: str, state_root: Path | str | None = None,
               launcher: Launcher | None = None) -> "SGLangPool":
        """The pool another process started (reads pool.json)."""
        d = PoolDir(state_root or DEFAULT_ROOT, pool_id)
        cfg = d.pool()
        o = cfg.get("options", {})
        return cls(spec_from_dict(cfg["spec"]), policy_from_dict(cfg["policy"]), pool_id=pool_id,
                   state_root=d.root, sif=cfg.get("sif"), launcher=launcher,
                   binds=o.get("binds"), engine_env=cfg.get("engine_env"),
                   router_args=cfg.get("router_args"), advertise=cfg.get("advertise", "auto"),
                   engine_cmd=cfg.get("engine_cmd"), router_cmd=cfg.get("router_cmd"),
                   python=o.get("python"), startup_timeout=cfg.get("startup_timeout", 1800),
                   job_env=o.get("job_env"), container_bin=o.get("container_bin"))

    @property
    def launcher(self) -> Launcher:
        if self._launcher is None:
            from harnessforge.launch import make_launcher
            kind = self.spec.launcher.get("kind")
            self._launcher = make_launcher(kind, defaults={k: v for k, v in
                                                           self.spec.launcher.items()
                                                           if k in LAUNCHER_KEYS})
        return self._launcher

    # ------------------------------------------------------------------ jobs
    def node_command(self, i: int) -> list[str]:
        agent = [self.python, "-m", "harnessforge.pool.node", "--pool-dir", str(self.dir.dir),
                 "--node", str(i)]
        if not self.sif:
            return agent
        # absolute path: PBS execs a `--` command without a login PATH
        container = (self.container_bin or shutil.which("singularity")
                     or shutil.which("apptainer") or "singularity")
        argv = [container, "exec", "--nv"]
        for b in self.binds:
            argv += ["--bind", b]
        return argv + [self.sif] + agent

    def node_job(self, i: int) -> JobSpec:
        L = self.spec.launcher
        env = {"PYTHONPATH": str(SRC_DIR), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONUNBUFFERED": "1", **self.job_env}
        return JobSpec(
            name=f"{L.get('name_prefix', 'hfpool')}-{self.pool_id}-{i}"[:63],
            command=self.node_command(i), nodes=1, gpus=self.spec.gpus_per_node,
            cpus=int(L.get("cpus", 0)), walltime=str(L.get("walltime", "4:00:00")), env=env,
            log=self.dir.logs / f"node{i}.job.log",
            options={k: v for k, v in L.items() if k in LAUNCHER_KEYS})

    def _config(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id, "spec": asdict(self.spec), "sif": self.sif,
            "policy": policy_to_dict(self._policy), "engine_cmd": self.engine_cmd,
            "router_cmd": self.router_cmd, "router_args": self.router_args,
            "engine_env": self.engine_env, "advertise": self.advertise,
            "startup_timeout": self.startup_timeout,
            "engine_port": int(self.spec.launcher.get("engine_port", 31000)),
            "router_port": int(self.spec.launcher.get("router_port", 30000)),
            "options": {"binds": self.binds, "python": self.python, "job_env": self.job_env,
                        "container_bin": self.container_bin},
            "status": "starting", "jobs": {}, "history": [],
            "created_at": time.time()}

    # ------------------------------------------------------------------ ModelPool
    def start(self, *, wait: bool = True, timeout: float | None = None) -> None:
        if self._policy is None:
            raise PoolError("start() needs an initial policy")
        if self.dir.exists() and self.dir.pool().get("status") not in ("stopped", None):
            raise PoolError(f"pool {self.pool_id!r} already exists at {self.dir.dir} "
                            "(attach to it, or stop it first)")
        if self.dir.dir.exists():          # leftovers of a stopped pool: keep them, aside
            self._move_aside()
        self.dir.dir.mkdir(parents=True, exist_ok=True)
        self.dir.logs.mkdir(parents=True, exist_ok=True)
        cfg = self._config()
        self.dir.write_pool(cfg)
        jobs = {}
        for i in range(self.spec.nodes):
            jobs[str(i)] = self.launcher.submit(self.node_job(i))
            self.dir.update_pool(jobs=dict(jobs))
        if wait:
            self.wait_ready(timeout=timeout)

    def wait_ready(self, *, timeout: float | None = None, poll: float = 5.0,
                   all_engines: bool = True) -> None:
        """Block until the router serves every engine of every node that is not lent (or,
        all_engines=False, until it serves at least one)."""
        timeout = self.startup_timeout + 600 if timeout is None else timeout
        t0 = time.time()
        serving = self.spec.nodes - len(self.dir.lent())
        want = serving * max(1, self.spec.gpus_per_node // self.spec.tp)
        last = ""
        while True:
            jobs = self.dir.pool().get("jobs", {})
            for i, jid in jobs.items():
                st = self.launcher.status(jid)
                if st.state == JobState.DONE:
                    raise PoolError(f"node {i} job {jid} ended (exit {st.exit_code}) during "
                                    f"startup; see {self.dir.logs}/node{i}.job.log")
            for i, n in self.dir.nodes().items():
                if n.get("status") == "failed":
                    raise PoolError(f"node {i} failed to start its engines; see {self.dir.logs}")
            routed = self.router_workers()
            ready = len(self.dir.engine_urls())
            msg = f"engines ready {ready}/{want}, routed {len(routed)}"
            if msg != last:
                print(f"[pool {self.pool_id}] {msg}", flush=True)
                last = msg
            if routed and (len(set(routed)) >= want or not all_engines) and \
                    set(self.dir.engine_urls()) <= set(routed):
                self.dir.update_pool(status="ready")
                return
            if time.time() - t0 > timeout:
                raise PoolError(f"pool not ready after {timeout:.0f}s ({msg})")
            time.sleep(poll)

    def router_url(self) -> str | None:
        r = self.dir.router()
        return r["url"].rstrip("/") if r and r.get("url") else None

    def router_workers(self) -> list[str]:
        url = self.router_url()
        if not url:
            return []
        from harnessforge.pool.node import RouterAPI
        try:
            return RouterAPI(url).workers()
        except (http.HTTPError, ValueError):
            return []

    def endpoint(self) -> str:
        url = self.router_url()
        if not url:
            raise PoolError(f"pool {self.pool_id!r} has no router yet")
        return url + "/v1"

    def policy(self) -> PolicyRef:
        return policy_from_dict(self.dir.pool()["policy"])

    def engines(self) -> list[str]:
        """Ready engines of the nodes that are not lent (what the router serves and load()
        swaps)."""
        return self.dir.engine_urls()

    # ------------------------------------------------------------------ lending to a trainer
    def lent(self) -> dict[int, dict[str, Any]]:
        return self.dir.lent()

    def lent_engines(self, nodes: list[int] | None = None, *, ready_only: bool = True) -> list[str]:
        """Engine URLs of lent nodes (all lent nodes, or `nodes`)."""
        want = set(self.dir.lent()) if nodes is None else set(nodes)
        return self.dir.engine_urls(include_lent=True, nodes=want, ready_only=ready_only)

    def lend(self, nodes: list[int], *, to: str, policy_file: str | None = None,
             on_restart: str = "down", drain_timeout: float = 600.0, poll: float = 2.0,
             wait_router: float = 120.0) -> list[str]:
        """Hand `nodes`' engines to a trainer `to`. Marks them lent in pool.json (node 0's agent
        then drops them from the router; load() and wait_ready() skip them), waits until the
        router no longer lists them and their in-flight requests finished (up to
        drain_timeout; then raises, leaving them lent), and returns their engine URLs.

        on_restart: "down" (default) = an engine that dies stays down until the node is returned;
        "latest" = it restarts on policy_file's {"id", "path"} (the trainer's latest full HF
        export) if that exists, else stays down. Never on the pool's policy."""
        if on_restart not in ("down", "latest"):
            raise PoolError(f"on_restart must be 'down' or 'latest', not {on_restart!r}")
        if on_restart == "latest" and not policy_file:
            raise PoolError("on_restart='latest' needs a policy_file")
        nodes = sorted(set(int(i) for i in nodes))
        cfg = self.dir.pool()
        known = {int(i) for i in cfg.get("jobs", {})}
        node_files = self.dir.nodes()
        lent = {str(k): v for k, v in (cfg.get("lent") or {}).items()}
        for i in nodes:
            if i not in known:
                raise PoolError(f"node {i} is not part of pool {self.pool_id!r}")
            if str(i) in lent and lent[str(i)].get("to") != to:
                raise PoolError(f"node {i} is already lent to {lent[str(i)].get('to')!r}")
        # the agents that must honour the lend: node 0 (router) and every lent node (restarts)
        for i in sorted({0, *nodes}):
            if "lend" not in (node_files.get(i) or {}).get("features", []):
                raise PoolError(
                    f"node {i}'s agent predates lending (no 'lend' feature): restart it with this "
                    f"code first (python -m harnessforge.pool restart-node --node {i})")
        if not self.dir.engine_urls(include_lent=True, nodes=set(nodes)):
            raise PoolError(f"nodes {nodes} have no ready engines to lend")
        now = time.time()
        for i in nodes:
            lent[str(i)] = {"to": to, "since": now, "policy_file": policy_file,
                            "on_restart": on_restart}
        self.dir.update_pool(lent=lent, lend_history=cfg.get("lend_history", []) + [
            {"op": "lend", "nodes": nodes, "to": to, "at": now}])
        urls = self.lent_engines(nodes)
        # 1. out of the router
        t0 = time.time()
        while self.router_url() and set(urls) & set(self.router_workers()):
            if time.time() - t0 > wait_router:
                raise PoolError(f"router still routes lent engines after {wait_router:.0f}s "
                                f"(is node 0's agent running this code?)")
            time.sleep(poll)
        # 2. drain what the router had already sent them
        t0 = time.time()
        while True:
            busy = {u: n for u in urls if (n := self._inflight(u))}
            if not busy:
                break
            if time.time() - t0 > drain_timeout:
                raise PoolError(f"lent engines still busy after {drain_timeout:.0f}s: {busy}")
            time.sleep(poll)
        return urls

    def _inflight(self, url: str) -> int:
        try:
            _, text = http.request("GET", url + "/metrics", timeout=5)
        except http.HTTPError:
            return 0
        m = http.prometheus_values(text, ("sglang:num_running_reqs", "sglang:num_queue_reqs"))
        return int(sum(v or 0 for v in m.values()))

    def unlend(self, nodes: list[int] | None = None, *, reload: bool = True,
               timeout: float = 3600) -> None:
        """Return lent nodes (all, or `nodes`) to the pool. reload=True first swaps their running
        engines to the pool's active policy (update_weights_from_disk, verified via
        weight_version); on any failure the nodes stay lent and this raises, so an engine with
        trainer weights never re-enters the router. Held (down) engines start on the active
        policy once the node is returned."""
        lent = {str(k): v for k, v in (self.dir.pool().get("lent") or {}).items()}
        nodes = sorted(int(k) for k in lent) if nodes is None else sorted(set(map(int, nodes)))
        nodes = [i for i in nodes if str(i) in lent]
        if not nodes:
            return
        urls = self._alive_engines(nodes, timeout=min(timeout, 1800))
        if reload and urls:
            errors = self._swap(urls, self.policy(), drain=False, timeout=timeout, verify=True)
            if errors:
                raise PoolError(f"unlend: reloading {self.policy().id} failed on "
                                f"{len(errors)}/{len(urls)} engines (nodes stay lent): {errors}")
        cfg = self.dir.pool()
        lent = {k: v for k, v in (cfg.get("lent") or {}).items() if int(k) not in nodes}
        self.dir.update_pool(lent=lent, lend_history=cfg.get("lend_history", []) + [
            {"op": "unlend", "nodes": nodes, "at": time.time(), "reloaded": bool(reload),
             "engines": len(urls)}])

    def _alive_engines(self, nodes: list[int], *, timeout: float, poll: float = 2.0) -> list[str]:
        """Every engine of `nodes` whose process is alive (not held down), waiting until each
        answers /health: an engine that restarted on trainer weights and is still loading must
        be reloaded too, not slip into the router afterwards."""
        t0 = time.time()
        while True:
            urls, waiting = [], []
            for i in nodes:
                for e in (self.dir.node(i) or {}).get("engines", []):
                    if e.get("held") or e.get("alive") is False:
                        continue
                    (urls if http.ok(e["url"] + "/health", timeout=5) else waiting).append(e["url"])
            if not waiting:
                return urls
            if time.time() - t0 > timeout:
                raise PoolError(f"unlend: engines {waiting} alive but not healthy after "
                                f"{timeout:.0f}s (nodes stay lent)")
            time.sleep(poll)

    def _swap(self, urls: list[str], policy: PolicyRef, *, drain: bool, timeout: float,
              verify: bool) -> dict[str, str]:
        payload = {"model_path": policy.path, "weight_version": policy.id,
                   "abort_all_requests": not drain, "flush_cache": True}

        def one(url: str) -> tuple[str, str | None]:
            try:
                r = http.post_json(url + "/update_weights_from_disk", payload, timeout=timeout)
                if isinstance(r, dict) and r.get("success") is False:
                    return url, f"update failed: {r.get('message')}"
                if verify:
                    info = self.engine_info(url)
                    if info.get("weight_version") != policy.id:
                        return url, f"weight_version is {info.get('weight_version')!r}"
                return url, None
            except http.HTTPError as e:
                return url, str(e)

        with cf.ThreadPoolExecutor(max_workers=min(64, max(1, len(urls)))) as ex:
            return {u: e for u, e in ex.map(one, urls) if e}

    def load(self, policy: PolicyRef, *, drain: bool = True, timeout: float = 3600,
             verify: bool = True) -> None:
        """Swap every engine to `policy` (SGLang /update_weights_from_disk, all engines in
        parallel). drain=True: each engine finishes its in-flight requests first (SGLang's model
        update writer lock; new requests wait for the swap); drain=False: in-flight requests are
        aborted. pool.json records the target policy before the swap (so an engine that
        restarts mid-swap comes up on it) and the outcome after; a partial failure leaves
        status="degraded" and raises. Lent nodes are skipped (they get the active policy when
        they are returned, see unlend)."""
        cfg = self.dir.pool()
        prev = cfg["policy"]
        urls = self.engines()
        if not urls:
            raise PoolError("no ready engines to load into")
        self.dir.update_pool(policy=policy_to_dict(policy), previous_policy=prev,
                             status="loading")
        t0 = time.time()
        errors = self._swap(urls, policy, drain=drain, timeout=timeout, verify=verify)
        entry = {"policy": policy_to_dict(policy), "at": time.time(), "drain": drain,
                 "seconds": round(time.time() - t0, 2), "engines": len(urls),
                 "errors": errors, "skipped_lent_nodes": sorted(self.dir.lent())}
        cfg = self.dir.pool()
        hist = cfg.get("history", []) + [entry]
        self.dir.update_pool(history=hist, status="degraded" if errors else "ready")
        if errors:
            raise PoolError(f"load({policy.id}) failed on {len(errors)}/{len(urls)} engines: "
                            f"{errors}")

    @staticmethod
    def engine_info(url: str) -> dict[str, Any]:
        for path in ("/model_info", "/get_model_info"):
            try:
                return http.get_json(url + path, timeout=10)
            except http.HTTPError as e:
                if e.status != 404:
                    raise
        return {}

    def health(self) -> dict[str, Any]:
        cfg = self.dir.pool()
        nodes_state = self.dir.nodes()
        lent = {int(k): v for k, v in (cfg.get("lent") or {}).items()}
        out: dict[str, Any] = {"pool_id": self.pool_id, "status": cfg.get("status"),
                               "policy": cfg.get("policy"), "router": None, "nodes": [],
                               "lent": {str(k): v for k, v in lent.items()}}
        url = self.router_url()
        if url:
            out["endpoint"] = url + "/v1"
            out["router"] = {"url": url, "ok": http.ok(url + "/health") or
                             http.ok(url + "/v1/models"), "workers": self.router_workers()}

        def engine(e: dict[str, Any]) -> dict[str, Any]:
            u = e["url"]
            r = {"url": u, "ok": http.ok(u + "/health", timeout=5), "gpus": e.get("gpus"),
                 "restarts": e.get("restarts", 0), "held": e.get("held")}
            if r["ok"]:
                try:
                    info = self.engine_info(u)
                    r["weight_version"] = info.get("weight_version")
                    r["model_path"] = info.get("model_path")
                except (http.HTTPError, ValueError):
                    pass
                try:
                    _, text = http.request("GET", u + "/metrics", timeout=5)
                    m = http.prometheus_values(text, ("sglang:num_running_reqs",
                                                      "sglang:num_queue_reqs"))
                    r["running"] = m.get("sglang:num_running_reqs")
                    r["queued"] = m.get("sglang:num_queue_reqs")
                except http.HTTPError:
                    pass
            return r

        with cf.ThreadPoolExecutor(max_workers=32) as ex:
            for i, jid in sorted(cfg.get("jobs", {}).items(), key=lambda kv: int(kv[0])):
                n = nodes_state.get(int(i), {})
                try:
                    st = self.launcher.status(jid)
                    job = {"state": st.state.value, "exit_code": st.exit_code,
                           "hosts": list(st.hosts)}
                except Exception as e:                     # scheduler hiccup: report, go on
                    job = {"state": "unknown", "error": str(e)}
                out["nodes"].append({
                    "index": int(i), "job_id": jid, "job": job, "host": n.get("host"),
                    "status": n.get("status", "pending"),
                    "updated_at": n.get("updated_at"), "lent_to": (lent.get(int(i)) or {}).get("to"),
                    "features": n.get("features", []),
                    "engines": list(ex.map(engine, n.get("engines", [])))})
        out["engines_ok"] = sum(e["ok"] for n in out["nodes"] for e in n["engines"])
        out["engines_total"] = sum(len(n["engines"]) for n in out["nodes"])
        return out

    def restart_node(self, i: int) -> str:
        """Cancel node i's job (if any) and submit a fresh one. Its engines come up on the
        active policy; node 0 also brings up a new router (the endpoint may move)."""
        cfg = self.dir.pool()
        jid = cfg.get("jobs", {}).get(str(i))
        if jid:
            try:
                self.launcher.cancel(jid)
            except RuntimeError:
                pass
        nf = self.dir.node_file(i)
        if nf.exists():
            nf.rename(nf.with_name(f"{i}.json.replaced-{int(time.time())}"))
        if i == 0 and self.dir.router_file.exists():
            self.dir.router_file.rename(self.dir.router_file.with_name(
                f"router.json.replaced-{int(time.time())}"))
        new = self.launcher.submit(self.node_job(i))
        cfg = self.dir.pool()
        jobs = dict(cfg.get("jobs", {}))
        jobs[str(i)] = new
        self.dir.update_pool(jobs=jobs, restarts=cfg.get("restarts", []) +
                             [{"node": i, "old": jid, "new": new, "at": time.time()}])
        return new

    def repair(self, *, stale_seconds: float = 300) -> list[int]:
        """Restart every node whose job ended, whose agent reported failure, or whose node file
        went stale (agent dead). Returns the restarted node indices."""
        restarted = []
        cfg = self.dir.pool()
        nodes_state = self.dir.nodes()
        now = time.time()
        for i, jid in cfg.get("jobs", {}).items():
            i = int(i)
            st = self.launcher.status(jid)
            n = nodes_state.get(i)
            dead = st.state == JobState.DONE
            dead |= bool(n and n.get("status") == "failed")
            dead |= bool(st.state == JobState.RUNNING and n and
                         now - n.get("updated_at", now) > stale_seconds)
            if dead:
                self.restart_node(i)
                restarted.append(i)
        return restarted

    def stop(self, *, wait: float = 120.0) -> None:
        """Signal agents (STOP file), cancel every node job, and move the state dir aside
        (<root>/.stopped/<pool_id>-<time>) so logs survive and the id can be reused."""
        if not self.dir.exists():
            return
        self.dir.stop_file.write_text(str(time.time()))
        cfg = self.dir.pool()
        for jid in cfg.get("jobs", {}).values():
            try:
                self.launcher.cancel(jid)
            except RuntimeError as e:
                print(f"[pool {self.pool_id}] cancel {jid}: {e}", file=sys.stderr)
        self.dir.update_pool(status="stopped", stopped_at=time.time())
        # let the agents finish writing their last node files before moving the dir
        deadline = time.time() + wait
        pending = list(cfg.get("jobs", {}).values())
        while pending and time.time() < deadline:
            pending = [j for j in pending if self.launcher.status(j).state != JobState.DONE]
            if pending:
                time.sleep(2)
        self._move_aside()

    def _move_aside(self) -> Path:
        aside = self.dir.root / ".stopped" / f"{self.pool_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        aside.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.dir.dir, aside)
        return aside
