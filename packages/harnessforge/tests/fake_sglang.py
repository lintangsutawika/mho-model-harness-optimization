"""A fake SGLang engine and sglang-router (stdlib HTTP) for pool tests.

    python fake_sglang.py engine --port P --model-path M --weight-version V [any sglang flags]
    python fake_sglang.py router --port P [--worker-urls U ...] [any router flags]

Engine: /health, /model_info, /v1/models, /metrics, POST /update_weights_from_disk (a path
containing "BAD" fails), POST /v1/chat/completions -> content "<weight_version>: <last msg>".
Router: /health, GET/POST /workers, DELETE /workers/{id}, /v1/models and /v1/chat/completions
proxied round-robin to the workers.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Base(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, (bytes, str)) else json.dumps(obj)
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")


def engine(a, rest):
    st = {"model_path": a.model_path, "weight_version": a.weight_version, "running": 0}
    lock = threading.Lock()

    class H(Base):
        def do_GET(self):
            if self.path == "/health":
                self.send(200, {})
            elif self.path in ("/model_info", "/get_model_info"):
                self.send(200, {"model_path": st["model_path"], "served_model_name":
                                a.served_model_name, "weight_version": st["weight_version"]})
            elif self.path == "/v1/models":
                self.send(200, {"data": [{"id": a.served_model_name}]})
            elif self.path == "/metrics":
                self.send(200, f'sglang:num_running_reqs{{x="1"}} {st["running"]}\n'
                               f'sglang:num_queue_reqs{{x="1"}} 0\n', "text/plain")
            else:
                self.send(404, {"error": self.path})

        def do_POST(self):
            req = self.body()
            if self.path == "/update_weights_from_disk":
                time.sleep(0.2)
                if "BAD" in req["model_path"]:
                    self.send(400, {"success": False, "message": "cannot load"})
                    return
                with lock:
                    st["model_path"] = req["model_path"]
                    if req.get("weight_version"):
                        st["weight_version"] = req["weight_version"]
                self.send(200, {"success": True, "message": "ok"})
            elif self.path == "/v1/chat/completions":
                with lock:
                    st["running"] += 1
                msg = req["messages"][-1]["content"]
                with lock:
                    st["running"] -= 1
                self.send(200, {"model": req.get("model"), "choices": [
                    {"index": 0, "message": {"role": "assistant",
                                             "content": f"{st['weight_version']}: {msg}"}}],
                    "served_by": a.port})
            else:
                self.send(404, {"error": self.path})

    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


def router(a, rest):
    workers: dict[str, str] = {}
    ids = itertools.count()
    for u in a.worker_urls or []:
        workers[str(next(ids))] = u
    rr = itertools.count()

    def pick():
        urls = list(workers.values())
        return urls[next(rr) % len(urls)] if urls else None

    def proxy(h, method, data=None):
        u = pick()
        if u is None:
            h.send(503, {"error": "no workers"})
            return
        req = urllib.request.Request(u + h.path, method=method, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            h.send(r.status, r.read())

    class H(Base):
        def do_GET(self):
            if self.path == "/health":
                self.send(200, {})
            elif self.path == "/workers":
                self.send(200, {"workers": [{"id": i, "url": u} for i, u in workers.items()]})
            else:
                proxy(self, "GET")

        def do_POST(self):
            if self.path == "/workers":
                u = self.body()["url"]
                if u not in workers.values():
                    workers[str(next(ids))] = u
                self.send(202, {"status": "accepted"})
            else:
                n = int(self.headers.get("Content-Length") or 0)
                proxy(self, "POST", self.rfile.read(n))

        def do_DELETE(self):
            wid = self.path.rsplit("/", 1)[1]
            if workers.pop(wid, None) is None:
                self.send(404, {})
            else:
                self.send(200, {})

    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["engine", "router"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--weight-version", default="default")
    ap.add_argument("--served-model-name", default="policy")
    ap.add_argument("--worker-urls", nargs="*")
    a, rest = ap.parse_known_args()
    if a.host == "0.0.0.0":
        a.host = "127.0.0.1"
    (engine if a.role == "engine" else router)(a, rest)


if __name__ == "__main__":
    sys.exit(main())
