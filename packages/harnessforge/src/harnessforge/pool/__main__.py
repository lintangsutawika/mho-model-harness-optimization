"""Pool CLI.

    python -m harnessforge.pool start   --spec pool.yaml [--no-wait]
    python -m harnessforge.pool status  --spec pool.yaml [--json]
    python -m harnessforge.pool endpoint --spec pool.yaml
    python -m harnessforge.pool wait    --spec pool.yaml [--timeout S]   # until all engines routed
    python -m harnessforge.pool load    --spec pool.yaml --id step200 --path /ckpt/step200 \\
                                        [--base-model Qwen/Qwen3.5-4B] [--no-drain]
    python -m harnessforge.pool repair  --spec pool.yaml
    python -m harnessforge.pool restart-node --spec pool.yaml --node 1
    python -m harnessforge.pool stop    --spec pool.yaml
    python -m harnessforge.pool lend    --spec pool.yaml --node 3 [--node 2] --to RUN \
                                        [--policy-file F --on-restart latest]   # prints engine URLs
    python -m harnessforge.pool unlend  --spec pool.yaml [--node 3] [--no-reload]

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
             f"({h['policy']['path']})"
             + (f"  lent={ {k: v.get('to') for k, v in h['lent'].items()} }" if h.get("lent") else ""),
             f"endpoint {h.get('endpoint', '-')}  engines ok {h['engines_ok']}/"
             f"{h['engines_total']}"]
    if h.get("router"):
        r = h["router"]
        lines.append(f"router {r['url']} ok={r['ok']} workers={len(r['workers'])}")
    for n in h["nodes"]:
        lines.append(f"  node {n['index']} job {n['job_id']} [{n['job']['state']}] "
                     f"host={n['host']} status={n['status']}"
                     + (f" LENT to {n['lent_to']}" if n.get("lent_to") else ""))
        for e in n["engines"]:
            lines.append(f"    {e['url']} ok={e['ok']} version={e.get('weight_version')} "
                         f"running={e.get('running')} queued={e.get('queued')}"
                         + (f" HELD ({e['held']})" if e.get("held") else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessforge.pool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start", "status", "endpoint", "wait", "load", "repair",
                                    "restart-node", "stop", "lend", "unlend"])
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
    ap.add_argument("--node", type=int, action="append",
                    help="node index (lend/unlend: repeatable)")
    ap.add_argument("--to", help="lend: who borrows the nodes (a training run id)")
    ap.add_argument("--policy-file", help="lend: the trainer's latest-weights pointer file")
    ap.add_argument("--on-restart", default="down", choices=["down", "latest"])
    ap.add_argument("--drain-timeout", type=float, default=600.0)
    ap.add_argument("--no-reload", action="store_true")
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
        elif a.cmd == "wait":
            p.wait_ready(timeout=a.timeout)
            print(p.endpoint())
        elif a.cmd == "load":
            if not (a.id and a.path):
                sys.exit("load needs --id and --path")
            p.load(PolicyRef(a.id, a.path, a.base_model), drain=not a.no_drain)
            print(f"active policy {p.policy().id}")
        elif a.cmd == "repair":
            print(f"restarted nodes {p.repair()}")
        elif a.cmd == "restart-node":
            for i in a.node or []:
                print(f"node {i} -> job {p.restart_node(i)}")
        elif a.cmd == "lend":
            if not (a.node and a.to):
                sys.exit("lend needs --node and --to")
            urls = p.lend(a.node, to=a.to, policy_file=a.policy_file, on_restart=a.on_restart,
                          drain_timeout=a.drain_timeout)
            print(json.dumps({"lent": a.node, "to": a.to, "engines": urls}) if a.json
                  else "\n".join(urls))
        elif a.cmd == "unlend":
            p.unlend(a.node, reload=not a.no_reload)
            print(f"returned; lent now {sorted(p.lent())}")
        elif a.cmd == "stop":
            p.stop()
            print("stopped")
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except PoolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
