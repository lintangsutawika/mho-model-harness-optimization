"""Sandbox host fleet CLI (see harnessforge.sandboxhost).

    python -m harnessforge.sandboxhost start  --spec fleet.json [--hosts N] [--no-wait] [--timeout S]
    python -m harnessforge.sandboxhost status --spec fleet.json [--json] [--no-ping]
    python -m harnessforge.sandboxhost stop   --spec fleet.json [--host I] [--grace S]
    python -m harnessforge.sandboxhost drain  --spec fleet.json --host I [--replace] [--timeout S]
        (no new sandboxes on host I; it exits once its own have stopped; --replace then starts a fresh one)
    python -m harnessforge.sandboxhost serve  --state DIR --index I          (inside a host job)

--spec also takes the fleet's state dir.
"""
from __future__ import annotations

import argparse
import json
import sys

from harnessforge.sandboxhost import fleet


def _summary(st: dict) -> str:
    lines = [f"fleet {st['fleet_id']}  ({st['dir']})"]
    tot_slots = tot_active = 0
    for h in st["hosts"]:
        tot_slots += h.get("slots") or 0
        tot_active += h.get("active") or 0
        load = h.get("load")
        lines.append(f"  host {h['index']} job {h['job']} [{h['job_state']}] {h.get('status')} {h.get('addr')} "
                     f"active {h.get('active')}/{h.get('slots')} starting {h.get('starting')} "
                     f"load {load[0] if load else '-'} age {h.get('age_s')}s ping {h.get('ping', '-')}")
    lines.append(f"  total: {tot_active}/{tot_slots} sandboxes")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessforge.sandboxhost", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start", "status", "stop", "drain", "serve"])
    ap.add_argument("--spec", help="fleet spec JSON (or its state dir)")
    ap.add_argument("--hosts", type=int)
    ap.add_argument("--host", type=int, help="stop: only this host index; drain: the host to drain")
    ap.add_argument("--replace", action="store_true", help="drain: start a fresh host for it once drained")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--grace", type=float, default=330.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-ping", action="store_true")
    ap.add_argument("--state")
    ap.add_argument("--index", type=int)
    ap.add_argument("--port", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from harnessforge.sandboxhost.daemon import serve
        if a.state is None or a.index is None:
            sys.exit("serve needs --state and --index")
        return serve(a.state, a.index, port=a.port)
    if not a.spec:
        sys.exit(f"{a.cmd} needs --spec")
    if a.cmd == "start":
        spec = fleet.load_spec(a.spec)
        fleet.start(spec, hosts=a.hosts, wait=not a.no_wait, timeout=a.timeout)
        return 0
    fd = fleet.FleetDir(fleet.state_dir_of(a.spec))
    if a.cmd == "status":
        st = fleet.status(fd, ping=not a.no_ping)
        print(json.dumps(st, indent=1) if a.json else _summary(st))
    elif a.cmd == "stop":
        fleet.stop(fd, index=a.host, grace_s=a.grace)
    elif a.cmd == "drain":
        if a.host is None:
            sys.exit("drain needs --host")
        spec = fleet.load_spec(a.spec) if a.spec.endswith(".json") else None
        timeout = a.timeout if a.timeout != 1800.0 else 4 * 3600.0
        return 0 if fleet.drain(fd, a.host, timeout=timeout, replace=a.replace, spec=spec) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
