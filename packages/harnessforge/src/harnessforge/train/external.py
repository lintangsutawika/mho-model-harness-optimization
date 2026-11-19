"""External rollout mode: the trainer attaches SGLang engines it borrows from a harnessforge pool.

    "rollout": {"mode": "external",
                "pool_spec": "~/hf-dev/pool/scripts/pool-n4hf72.json",   # or pool_id + state_root
                "lend_nodes": [1, 2, 3],
                "engine_local_dir": "/dev/shm/{user}-hftrain-{run_id}/engine-local"}

`launch` lends the nodes (`SGLangPool.lend`, tag = run id), checks the engines, writes
<run>/rollout.json, and submits a 1-node gateway job whose serve_tinker gets
`--rollout-external --rollout-external-engine-addrs host:port ...` (miles-tinker's attach path,
`external-engines-fixes`). The gateway job's teardown, `train stop` and `train unlend` return the
nodes (`SGLangPool.unlend`: reload the pool policy, verify it, else the nodes stay lent).

Preconditions (miles refuses or never readies an engine otherwise):

* the engines serve the trainer's `--hf-checkpoint` as a local path (disk-delta seeds each
  host's local checkpoint from the engine's `--model-path`; a hub id is refused): the pool's
  policy path must equal `model.hf_checkpoint`;
* each engine still serves the pool policy (`weight_version` = its id), not a previous
  trainer's weights (with disk-delta the first sync only records the baseline);
* tp 1 per engine (`--rollout-num-gpus-per-engine 1`);
* `--update-weight-local-checkpoint-dir` is run-unique (an earlier run's applied-version marker
  makes the first delta fail as out of order) and host-local with room for the checkpoint
  (~9.3 GB for Qwen3.5-4B; /tmp on ABCI compute nodes is ~8 GB, hence /dev/shm).

A lent engine that dies is restarted by its pool agent on the **base** checkpoint
(`on_restart: latest` with a policy file naming `model.hf_checkpoint`): miles treats the comeback
as a new generation and replays every published delta from the base. Restarting it on a later
HF export would apply the XOR deltas twice.
"""
from __future__ import annotations

import getpass
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from harnessforge.train.config import TrainRun

DEFAULT_LOCAL_DIR = "/dev/shm/{user}-hftrain-{run_id}/engine-local"


class ExternalRolloutError(RuntimeError):
    pass


def rollout_cfg(run: TrainRun) -> dict[str, Any]:
    return run.get("rollout") or {}


def is_external(run: TrainRun) -> bool:
    return rollout_cfg(run).get("mode", "managed") == "external"


def engine_local_dir(run: TrainRun) -> str:
    tmpl = rollout_cfg(run).get("engine_local_dir", DEFAULT_LOCAL_DIR)
    return tmpl.format(user=os.environ.get("USER") or getpass.getuser(), run_id=run.run_id)


def engine_addr(url: str) -> str:
    """http://10.0.82.8:31600[/] -> 10.0.82.8:31600 (miles wants host:port, no scheme)."""
    u = url.split("://", 1)[-1].rstrip("/")
    if "/" in u or ":" not in u:
        raise ValueError(f"not an engine URL: {url!r}")
    return u


def rollout_file(run: TrainRun) -> Path:
    return run.dir / "rollout.json"


def restart_policy_file(run: TrainRun) -> Path:
    return run.dir / "engine_restart_policy.json"


def read_rollout(run: TrainRun) -> dict[str, Any]:
    f = rollout_file(run)
    return json.loads(f.read_text()) if f.exists() else {}


def _write(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def engine_addrs(run: TrainRun) -> list[str]:
    """The attached engines' host:port, from <run>/rollout.json (written by `launch`)."""
    addrs = read_rollout(run).get("engines") or []
    if not addrs:
        raise ExternalRolloutError(f"{rollout_file(run)} lists no engines (was the run launched?)")
    return list(addrs)


def open_pool(run: TrainRun):
    """The pool named by rollout.pool_spec (a pool CLI spec file) or rollout.pool_id/state_root."""
    from harnessforge.pool import SGLangPool, from_config, load_config
    r = rollout_cfg(run)
    if r.get("pool_spec"):
        fresh = from_config(load_config(os.path.expanduser(r["pool_spec"])))
        return SGLangPool.attach(fresh.pool_id, fresh.dir.root)
    if r.get("pool_id"):
        return SGLangPool.attach(r["pool_id"], r.get("state_root"))
    raise ExternalRolloutError("rollout.mode external needs rollout.pool_spec or rollout.pool_id")


def check_engine(url: str, *, hf_checkpoint: str, policy_id: str | None,
                 get_json: Callable[[str], dict] | None = None) -> list[str]:
    """What would keep miles from attaching this engine (empty = fine). Reads /server_info and
    /model_info (a `get_json(url)` is injectable for tests)."""
    if get_json is None:
        from harnessforge.pool import http

        def get_json(u: str) -> dict:
            return http.get_json(u, timeout=15)
    problems = []
    try:
        info = get_json(url + "/server_info")
    except Exception as e:                     # unreachable / not up: miles would wait forever
        return [f"{url}: /server_info failed: {e}"]
    mp = info.get("model_path")
    if not (isinstance(mp, str) and mp.startswith("/")):
        problems.append(f"{url}: model_path {mp!r} is not a local path (disk-delta refuses hub ids)")
    elif os.path.realpath(mp) != os.path.realpath(hf_checkpoint):
        problems.append(f"{url}: model_path {mp!r} != model.hf_checkpoint {hf_checkpoint!r}")
    tp = info.get("tp_size", 1)
    if int(tp or 1) != 1:
        problems.append(f"{url}: tp_size {tp} != 1 (--rollout-num-gpus-per-engine 1)")
    if policy_id is not None:
        try:
            wv = get_json(url + "/model_info").get("weight_version")
        except Exception:
            wv = info.get("weight_version")
        if wv not in (policy_id, "default", None):
            problems.append(f"{url}: weight_version {wv!r}, not the pool policy {policy_id!r} "
                            f"(a previous trainer's weights?)")
    return problems


def lend_for_run(run: TrainRun, *, pool=None, check: Callable[..., list[str]] = check_engine,
                 log: Callable[[str], None] = print) -> list[str]:
    """Lend rollout.lend_nodes to this run and return their engines as host:port. On any failed
    check after lending, the nodes are returned (unlend) and this raises."""
    r = rollout_cfg(run)
    nodes = sorted(int(i) for i in r.get("lend_nodes", []))
    if not nodes:
        raise ExternalRolloutError("rollout.lend_nodes is empty")
    pool = pool or open_pool(run)
    hf = run["model"]["hf_checkpoint"]
    pol = pool.policy()
    if os.path.realpath(pol.path) != os.path.realpath(hf):
        raise ExternalRolloutError(
            f"pool {pool.pool_id} serves {pol.path!r}, the trainer starts from {hf!r}: the engines must "
            f"hold the --hf-checkpoint bytes at that path (load it, or start the pool on it)")
    lent = pool.lent()
    for i in nodes:
        if i in lent and lent[i].get("to") != run.run_id:
            raise ExternalRolloutError(f"pool node {i} is lent to {lent[i].get('to')!r}")
    # engines must answer now (pool.json's status may still say "starting" after `start --no-wait`)
    from harnessforge.pool import http
    want = int(r.get("engines_per_node", pool.spec.gpus_per_node // max(1, pool.spec.tp)))
    urls = pool.dir.engine_urls(include_lent=True, nodes=set(nodes))
    healthy = [u for u in urls if http.ok(u + "/health", timeout=10)]
    if len(healthy) < want * len(nodes):
        raise ExternalRolloutError(f"nodes {nodes}: {len(healthy)}/{want * len(nodes)} engines healthy")
    run.dir.mkdir(parents=True, exist_ok=True)
    _write(restart_policy_file(run), {"id": f"{run.run_id}-base", "path": hf,
                                      "base_model": run["model"].get("name")})
    log(f"[external] lending pool {pool.pool_id} nodes {nodes} to {run.run_id}")
    urls = pool.lend(nodes, to=run.run_id, policy_file=str(restart_policy_file(run)),
                     on_restart="latest", drain_timeout=float(r.get("drain_timeout", 1800)))
    problems = [p for u in urls for p in check(u, hf_checkpoint=hf, policy_id=pol.id)]
    if len(urls) < want * len(nodes):
        problems.append(f"only {len(urls)}/{want * len(nodes)} engines after lending")
    if problems:
        log("[external] engine checks failed; returning the nodes:\n  " + "\n  ".join(problems))
        try:
            pool.unlend(nodes)
        finally:
            _write(rollout_file(run), {"status": "refused", "pool_id": pool.pool_id, "nodes": nodes,
                                       "problems": problems, "at": time.time()})
        raise ExternalRolloutError("; ".join(problems))
    addrs = [engine_addr(u) for u in urls]
    _write(rollout_file(run), {"status": "lent", "pool_id": pool.pool_id,
                               "state_root": str(pool.dir.root), "nodes": nodes, "engines": addrs,
                               "engine_local_dir": engine_local_dir(run), "policy": pol.id,
                               "lent_at": time.time()})
    log(f"[external] {len(addrs)} engines: {' '.join(addrs)}")
    return addrs


def unlend_for_run(run: TrainRun, *, pool=None, log: Callable[[str], None] = print) -> bool:
    """Return this run's nodes to the pool (reload + verify the pool policy). True when returned;
    False when unlend's checks failed and the nodes stay lent (recorded in rollout.json)."""
    info = read_rollout(run)
    if info.get("status") != "lent":
        return True
    nodes = info.get("nodes", [])
    if pool is None:
        from harnessforge.pool import SGLangPool
        pool = SGLangPool.attach(info["pool_id"], info.get("state_root"))
    mine = [i for i, v in pool.lent().items() if i in nodes and v.get("to") == run.run_id]
    try:
        if mine:
            pool.unlend(mine)
    except Exception as e:
        log(f"[external] unlend of nodes {mine} failed, they stay lent: {e}")
        info.update(status="lent", unlend_error=str(e)[:2000], unlend_failed_at=time.time())
        _write(rollout_file(run), info)
        return False
    info.update(status="returned", returned_at=time.time())
    _write(rollout_file(run), info)
    log(f"[external] returned pool {info['pool_id']} nodes {mine or nodes}")
    return True
