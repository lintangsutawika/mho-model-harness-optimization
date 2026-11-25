"""Tiny stdlib HTTP helpers (the core stays dependency-free; no httpx needed for control calls)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class HTTPError(RuntimeError):
    def __init__(self, url: str, status: int | None, body: str):
        super().__init__(f"{url} -> {status}: {body[:500]}")
        self.url, self.status, self.body = url, status, body


def request(method: str, url: str, data: Any = None, *, timeout: float = 10.0,
            headers: dict[str, str] | None = None) -> tuple[int, str]:
    """(status, body text). Network errors raise HTTPError with status None."""
    body = None
    hdrs = dict(headers or {})
    if data is not None:
        body = json.dumps(data).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise HTTPError(url, None, repr(e)) from e


def get_json(url: str, *, timeout: float = 10.0) -> Any:
    status, text = request("GET", url, timeout=timeout)
    if status != 200:
        raise HTTPError(url, status, text)
    return json.loads(text)


def post_json(url: str, data: Any = None, *, timeout: float = 10.0) -> Any:
    status, text = request("POST", url, data if data is not None else {}, timeout=timeout)
    if status >= 300:
        raise HTTPError(url, status, text)
    try:
        return json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return {"text": text}


def ok(url: str, *, timeout: float = 5.0) -> bool:
    try:
        return request("GET", url, timeout=timeout)[0] == 200
    except HTTPError:
        return False


def prometheus_values(text: str, names: tuple[str, ...]) -> dict[str, float]:
    """Sum of every sample of each metric name in a Prometheus text exposition."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in names:
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return out
