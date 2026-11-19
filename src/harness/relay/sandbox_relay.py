"""Per-job reverse relays as Modal Sandboxes under ONE long-lived app (default "mho-relay").

Replaces deploying a Modal app per job (modal_relay.py): nothing is deployed, the app is created
once on first use, and each job starts its own Sandbox running relay_server.py with port 8000
exposed through an encrypted tunnel (its public HTTPS URL is the relay URL). A relay that loses
its job is reaped by Modal: `idle_timeout` fires once no TCP connection is open over the tunnel
(the bridge's WebSocket counts as one), and `timeout` caps the lifetime (Modal max: 24 h). Jobs
that outlive one Sandbox start a new one and move new trials to it (see train.sh rotation).

    python sandbox_relay.py up --label <job>        # prints {"sandbox_id": ..., "url": ...}
    python sandbox_relay.py down <sandbox_id>
    python sandbox_relay.py list                    # running relays of the app, with labels

Env: RELAY_SECRET (required for `up`), RELAY_REQUEST_TIMEOUT (s, default 600),
     RELAY_APP (default mho-relay).
"""
import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import modal

APP = os.environ.get("RELAY_APP", "mho-relay")
PORT = 8000
SERVER = Path(__file__).with_name("relay_server.py")
MAX_TIMEOUT_S = 24 * 3600


def _image() -> modal.Image:
    return (modal.Image.debian_slim(python_version="3.12")
            .pip_install("fastapi==0.115.*", "uvicorn[standard]==0.30.*", "websockets>=12")
            .add_local_file(str(SERVER), "/root/relay_server.py", copy=True))


def _ssl_context() -> ssl.SSLContext:
    # Verify with certifi's CA bundle (a dependency of modal): some Pythons (e.g. python.org
    # macOS builds) ship without system CA certs, and the health check would never pass.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def up(label: str, timeout_s: int, idle_timeout_s: int, cpu: float, wait_s: int) -> dict:
    secret = os.environ["RELAY_SECRET"]
    app = modal.App.lookup(APP, create_if_missing=True)
    sb = modal.Sandbox.create(
        "python", "/root/relay_server.py",
        app=app, image=_image(), encrypted_ports=[PORT],
        timeout=min(timeout_s, MAX_TIMEOUT_S), idle_timeout=idle_timeout_s, cpu=cpu,
        secrets=[modal.Secret.from_dict({
            "RELAY_SECRET": secret,
            "RELAY_REQUEST_TIMEOUT": os.environ.get("RELAY_REQUEST_TIMEOUT", "600"),
            "RELAY_PORT": str(PORT)})])
    sb.set_tags({"label": label, "kind": "mho-relay"})
    url = sb.tunnels(timeout=wait_s)[PORT].url.rstrip("/")
    deadline = time.time() + wait_s
    while True:
        try:
            with urllib.request.urlopen(url + "/health", timeout=10, context=_ssl_context()) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        if time.time() > deadline:
            sb.terminate()
            raise RuntimeError(f"relay sandbox {sb.object_id} never became healthy at {url}")
        time.sleep(2)
    return {"sandbox_id": sb.object_id, "url": url}


def down(sandbox_id: str) -> None:
    modal.Sandbox.from_id(sandbox_id).terminate()


def list_relays() -> list[dict]:
    app = modal.App.lookup(APP, create_if_missing=True)
    return [{"sandbox_id": sb.object_id, **sb.get_tags()} for sb in modal.Sandbox.list(app_id=app.app_id)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    u.add_argument("--label", required=True, help="job id or name (Sandbox tag)")
    u.add_argument("--timeout", type=int, default=MAX_TIMEOUT_S, help="lifetime s (max 24 h)")
    u.add_argument("--idle-timeout", type=int, default=900,
                   help="reap after this long with no open tunnel connection")
    u.add_argument("--cpu", type=float, default=4.0)
    u.add_argument("--wait", type=int, default=300, help="s to wait for the relay to be healthy")
    d = sub.add_parser("down")
    d.add_argument("sandbox_id")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "up":
        print(json.dumps(up(a.label, a.timeout, a.idle_timeout, a.cpu, a.wait)), flush=True)
    elif a.cmd == "down":
        down(a.sandbox_id)
    else:
        for r in list_relays():
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
