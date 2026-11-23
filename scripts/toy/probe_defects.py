#!/usr/bin/env python3
"""Which toy defects (micro-swe-agent branch defects/toy: q50, s30, t1024) a harness still has. Behavioural where it is
cheap, a source check otherwise; prints one JSON object.

    python scripts/toy/probe_defects.py <harness dir>

  q50    runs 32 single-quoted echo commands through the harness's own `environments.local._run`; present if any output
         differs from the plain shell's (the defect wraps ~half of commands in bash -c '...')
  s30    `step_limit` in config/mini.yaml: present when 0 < limit <= 40 (upstream: 0 = unlimited)
  t1024  calls the litellm model's `_query` with a stubbed `litellm` (max_tokens 16384 requested); present when the
         harness sends max_tokens <= 1024

PROBE_PYTHON: the interpreter for the behavioural checks (>= 3.10 with the harness deps: pydantic, pyyaml, jinja2,
tenacity); default this one. They fall back to a source check when the harness cannot be imported.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path


PY = os.environ.get("PROBE_PYTHON", sys.executable)


def _src(h: Path) -> Path:
    return h / "src" / "micro_swe"


def probe_q50(h: Path) -> dict:
    code = r"""
import json, os, subprocess, sys
sys.path.insert(0, sys.argv[1])
from micro_swe.environments.local import _run
bad = []
for i in range(32):
    cmd = f"echo 'q{i} two words'"
    p = _run(cmd, "/tmp", dict(os.environ), 20)
    if (p.stdout or "").strip() != f"q{i} two words":
        bad.append([cmd, (p.stdout or "").strip()])
print(json.dumps({"bad": bad}))
"""
    p = subprocess.run([PY, "-c", code, str(h / "src")], capture_output=True, text=True, timeout=120)
    if p.returncode == 0:
        bad = json.loads(p.stdout)["bad"]
        return {"present": bool(bad), "method": "behavioural", "broken": len(bad), "of": 32, "example": bad[:1]}
    text = (_src(h) / "environments" / "local.py").read_text() if (_src(h) / "environments" / "local.py").exists() else ""
    return {"present": "bash -c '" in text, "method": "source", "error": p.stderr.strip()[-300:]}


def probe_s30(h: Path) -> dict:
    f = _src(h) / "config" / "mini.yaml"
    m = re.search(r"^\s*step_limit:\s*(\d+)", f.read_text(), re.M) if f.exists() else None
    if not m:
        return {"present": None, "method": "source", "error": "no step_limit in config/mini.yaml"}
    v = int(m.group(1))
    return {"present": 0 < v <= 40, "method": "source", "step_limit": v}


def probe_t1024(h: Path) -> dict:
    code = r"""
import json, sys, types
sys.path.insert(0, sys.argv[1])
seen = {}
class _E(Exception): pass
exc = types.SimpleNamespace(**{n: type(n, (_E,), {}) for n in (
    "UnsupportedParamsError", "NotFoundError", "PermissionDeniedError", "ContextWindowExceededError",
    "AuthenticationError", "APIError", "RateLimitError", "Timeout", "APIConnectionError", "BadRequestError",
    "InternalServerError", "ServiceUnavailableError")})
def completion(**kw):
    seen.update(kw)
    raise _E("stub")
lit = types.ModuleType("litellm"); lit.completion = completion; lit.exceptions = exc
lit.utils = types.SimpleNamespace(register_model=lambda *a, **k: None)
lit.cost_calculator = types.SimpleNamespace(completion_cost=lambda *a, **k: 0.0)
lit.completion_cost = lambda *a, **k: 0.0
sys.modules["litellm"] = lit
import micro_swe.models.litellm as M
cls = next(v for v in vars(M).values() if isinstance(v, type) and hasattr(v, "_query") and v.__module__ == M.__name__)
m = cls(model_name="probe", model_kwargs={"max_tokens": 16384})
try:
    m._query([{"role": "user", "content": "hi"}])
except Exception:
    pass
print(json.dumps({"max_tokens": seen.get("max_tokens")}))
"""
    p = subprocess.run([PY, "-c", code, str(h / "src")], capture_output=True, text=True, timeout=120)
    if p.returncode == 0:
        mt = json.loads(p.stdout.strip().splitlines()[-1])["max_tokens"]
        if mt is not None:
            return {"present": int(mt) <= 1024, "method": "behavioural", "max_tokens_sent": mt}
    f = _src(h) / "models" / "litellm.py"
    text = f.read_text() if f.exists() else ""
    return {"present": bool(re.search(r"min\([^\n]*1024", text)), "method": "source",
            "error": p.stderr.strip()[-300:]}


def main() -> int:
    h = Path(sys.argv[1])
    out = {}
    for name, fn in (("q50", probe_q50), ("s30", probe_s30), ("t1024", probe_t1024)):
        try:
            out[name] = fn(h)
        except Exception as e:  # noqa: BLE001 - a probe failure is reported, not fatal
            out[name] = {"present": None, "error": f"{type(e).__name__}: {e}"}
    out["fixed"] = sorted(k for k, v in out.items() if isinstance(v, dict) and v.get("present") is False)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
