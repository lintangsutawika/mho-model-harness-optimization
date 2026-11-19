"""Pool CLI.

    python -m harnessforge.pool start   --spec pool.yaml [--no-wait]
    python -m harnessforge.pool status  --spec pool.yaml [--json]
    python -m harnessforge.pool endpoint --spec pool.yaml
    python -m harnessforge.pool load    --spec pool.yaml --id step200 --path /ckpt/step200 \\
                                        [--base-model Qwen/Qwen3.5-4B] [--no-drain]
    python -m harnessforge.pool repair  --spec pool.yaml
    python -m harnessforge.pool restart-node --spec pool.yaml --node 1
    python -m harnessforge.pool stop    --spec pool.yaml

Instead of --spec, an existing pool can be named with --pool-id [--state-root].
"""
from __future__ import annotations

import argparse
import json
import sys

from harnessforge.contracts import PolicyRef
from harnessforge.pool import SGLangPool, from_config, load_config
from harnessforge.pool.sglang import PoolError


def _pool(a, *, new: bool) -> SGLangPool:
    if a.spec:
        fresh = from_config(load_config(a.spec))
        if new:
            return fresh
        return SGLangPool.attach(fresh.pool_id, fresh.dir.root, launcher=fresh.launcher)
    if not a.pool_id:
        sys.exit("need --spec or --pool-id")
    if new:
        sys.exit("start needs --spec")
    return SGLangPool.attach(a.pool_id, a.state_root)


def _summary(h: dict) -> str:
    lines = [f"pool {h['pool_id']}  status={h['status']}  policy={h['policy']['id']} "
             f"({h['policy']['path']})",
             f"endpoint {h.get('endpoint', '-')}  engines ok {h['engines_ok']}/"
             f"{h['engines_total']}"]
    if h.get("router"):
        r = h["router"]
        lines.append(f"router {r['url']} ok={r['ok']} workers={len(r['workers'])}")
    for n in h["nodes"]:
        lines.append(f"  node {n['index']} job {n['job_id']} [{n['job']['state']}] "
                     f"host={n['host']} status={n['status']}")
        for e in n["engines"]:
            lines.append(f"    {e['url']} ok={e['ok']} version={e.get('weight_version')} "
                         f"running={e.get('running')} queued={e.get('queued')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessforge.pool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start", "status", "endpoint", "load", "repair",
                                    "restart-node", "stop"])
    ap.add_argument("--spec")
    ap.add_argument("--pool-id")
    ap.add_argument("--state-root")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--id")
    ap.add_argument("--path")
    ap.add_argument("--base-model")
    ap.add_argument("--no-drain", action="store_true")
    ap.add_argument("--node", type=int)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "start":
            p = _pool(a, new=True)
            p.start(wait=not a.no_wait, timeout=a.timeout)
            print(p.endpoint() if not a.no_wait else f"submitted {p.dir.pool()['jobs']}")
            return 0
        p = _pool(a, new=False)
        if a.cmd == "status":
            h = p.health()
            print(json.dumps(h, indent=1) if a.json else _summary(h))
        elif a.cmd == "endpoint":
            print(p.endpoint())
        elif a.cmd == "load":
            if not (a.id and a.path):
                sys.exit("load needs --id and --path")
            p.load(PolicyRef(a.id, a.path, a.base_model), drain=not a.no_drain)
            print(f"active policy {p.policy().id}")
        elif a.cmd == "repair":
            print(f"restarted nodes {p.repair()}")
        elif a.cmd == "restart-node":
            print(f"node {a.node} -> job {p.restart_node(a.node)}")
        elif a.cmd == "stop":
            p.stop()
            print("stopped")
    except PoolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
