"""The pool's shared state directory: the only channel between the controller (SGLangPool, in any
process), the node agents (one per node job) and clients (eval jobs on other nodes).

    <root>/<pool_id>/
        pool.json            controller: spec, sif, job ids, active policy, status, history,
                             lent nodes ({"<i>": {to, since, policy_file, on_restart}})
        router.json          node 0's agent: the router's URL (the pool endpoint)
        nodes/<i>.json       node i's agent: host, ip, job id, engines [{url, gpus, pid}], status
        logs/                node agents' job logs and per-engine / router logs
        STOP                 present => agents shut their processes down and exit

All writes are atomic (temp file + rename) so readers on other nodes never see half a file.
The directory must be on a filesystem every node sees (ABCI: /home or /groups).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(os.environ.get("HF_POOL_ROOT", Path.home() / "hf-dev" / "pool" / "state"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default        # never half-written (atomic rename), but tolerate hand edits


class PoolDir:
    """Paths and typed accessors for one pool's state dir."""

    def __init__(self, root: Path | str, pool_id: str):
        self.root = Path(root).expanduser()
        self.pool_id = pool_id
        self.dir = self.root / pool_id

    @classmethod
    def at(cls, path: Path | str) -> "PoolDir":
        p = Path(path).expanduser()
        return cls(p.parent, p.name)

    # paths
    @property
    def pool_file(self) -> Path:
        return self.dir / "pool.json"

    @property
    def router_file(self) -> Path:
        return self.dir / "router.json"

    @property
    def nodes_dir(self) -> Path:
        return self.dir / "nodes"

    @property
    def logs(self) -> Path:
        return self.dir / "logs"

    @property
    def stop_file(self) -> Path:
        return self.dir / "STOP"

    def node_file(self, i: int) -> Path:
        return self.nodes_dir / f"{i}.json"

    # accessors
    def exists(self) -> bool:
        return self.pool_file.exists()

    def pool(self) -> dict[str, Any]:
        d = read_json(self.pool_file)
        if d is None:
            raise FileNotFoundError(f"no pool state at {self.pool_file}")
        return d

    def write_pool(self, data: dict[str, Any]) -> None:
        data = dict(data)
        data["updated_at"] = time.time()
        write_json(self.pool_file, data)

    def update_pool(self, **changes: Any) -> dict[str, Any]:
        d = self.pool()
        d.update(changes)
        self.write_pool(d)
        return d

    def router(self) -> dict[str, Any] | None:
        return read_json(self.router_file)

    def node(self, i: int) -> dict[str, Any] | None:
        return read_json(self.node_file(i))

    def nodes(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        if not self.nodes_dir.is_dir():
            return out
        for p in self.nodes_dir.glob("*.json"):
            try:
                i = int(p.stem)
            except ValueError:
                continue
            d = read_json(p)
            if d is not None:
                out[i] = d
        return dict(sorted(out.items()))

    def lent(self) -> dict[int, dict[str, Any]]:
        """Nodes lent to a trainer (pool.json "lent"): index -> {to, since, policy_file,
        on_restart}. Lent engines serve the trainer only: no routing, no load()."""
        d = read_json(self.pool_file, {}) or {}
        return {int(k): v for k, v in (d.get("lent") or {}).items()}

    def engine_urls(self, *, ready_only: bool = True, max_age: float | None = 90.0,
                    include_lent: bool = False, nodes: "set[int] | None" = None) -> list[str]:
        """Engine URLs of every node. ready_only: only ready engines of ready nodes whose file
        was refreshed in the last `max_age` seconds (a SIGKILLed node stops refreshing).
        Engines of lent nodes are left out unless include_lent; `nodes` restricts to those."""
        urls = []
        now = time.time()
        lent = set() if include_lent else set(self.lent())
        for i, n in self.nodes().items():
            if i in lent or (nodes is not None and i not in nodes):
                continue
            if ready_only and n.get("status") != "ready":
                continue
            if ready_only and max_age is not None and now - n.get("updated_at", now) > max_age:
                continue
            urls += [e["url"] for e in n.get("engines", []) if not ready_only or e.get("ready")]
        return urls

    def stopping(self) -> bool:
        return self.stop_file.exists()


def endpoint_of(pool_id: str, root: Path | str | None = None) -> str:
    """The pool's OpenAI base URL (http://host:port/v1), for clients in other jobs."""
    d = PoolDir(root or DEFAULT_ROOT, pool_id)
    r = d.router()
    if not r or not r.get("url"):
        raise RuntimeError(f"pool {pool_id!r} has no router yet ({d.router_file})")
    return r["url"].rstrip("/") + "/v1"
