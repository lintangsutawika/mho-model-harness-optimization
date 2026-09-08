"""Host-side external executor for the local_singularity backend.

Runs as a SEPARATE process on the HPC node (outside the training SIF). Receives a harbor
trial config over HTTP, runs the full harbor Trial (agent + verifier + sandbox) on the host
via WritableSingularityEnvironment, and returns a TrialResult JSON. SkyRL's HarborGenerator
posts trials here (see .client) instead of running them in-process in the training container
-- the "Modal-like but local" external backend. The agent reaches the training container's
vLLM endpoint via the localhost api_base carried in the config (they share the node).

Run (on the node):
    EXECUTOR_PORT=8900 PYTHONPATH=$REPO/src \
        python -m mho.backends.local_singularity.service
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# harbor's Trial.create() imports agent/verifier classes (harness.agent_harness:*, etc.) in
# THIS process at trial time. Force src/ to the FRONT of sys.path: `python -m` puts the repo
# ROOT on sys.path[0] (cwd), and the repo root has its OWN `harness/` package that would
# shadow src/harness/. So src must precede the root even if it's already present.
_REPO_ROOT = Path(__file__).resolve().parents[4]  # src/mho/backends/local_singularity/ -> repo
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir():
    _src_s = str(_SRC)
    while _src_s in sys.path:
        sys.path.remove(_src_s)
    sys.path.insert(0, _src_s)

# Startup self-check (to the service log): proves the agent class imports in THIS process,
# so a failing trial's cause is obvious at boot, not per-request.
try:
    import importlib as _il
    _il.import_module("harness.agent_harness")
    print(f"[executor startup] OK: harness.agent_harness importable | repo={_REPO_ROOT}", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[executor startup] FAIL: harness.agent_harness NOT importable: {_e!r}\n"
          f"  __file__={__file__}\n  repo={_REPO_ROOT}\n  sys.path={sys.path}", flush=True)

from harbor.models.trial.config import TrialConfig  # noqa: E402
from harbor.models.trial.result import TrialResult  # noqa: E402
from harbor.trial.trial import Trial  # noqa: E402

# Import the backend so it loads in-process (fail fast) and captures the ORIGINAL
# asyncio.create_subprocess_exec before any wrapping.
from . import environment as _env  # noqa: E402,F401

log = logging.getLogger("mho.backends.local_singularity.service")
logging.basicConfig(level=logging.INFO)

# harbor's mini-swe-agent wrapper resolves the model API key from MSWEA_API_KEY in THIS
# process; local vLLM ignores the value, so default a placeholder rather than depend on the
# launch command.
os.environ.setdefault("MSWEA_API_KEY", "dummy")

app = FastAPI(title="mho-local-singularity-executor")

_BACKEND_IMPORT_PATH = "mho.backends.local_singularity.environment:WritableSingularityEnvironment"


async def _run_trial(config: dict) -> TrialResult:
    # The executor IS the sandbox host: pin the backend to our writable singularity
    # environment regardless of what the training-side config said.
    config = json.loads(json.dumps(config))  # defensively copy
    env = config.setdefault("environment", {})
    env["type"] = "singularity"
    env["import_path"] = _BACKEND_IMPORT_PATH
    env.pop("mounts", None)
    config.setdefault("timeout_multiplier", 1.0)
    trial_config = TrialConfig.model_validate(config)
    trial = await Trial.create(trial_config)
    return await trial.run()


@app.post("/trial")
async def post_trial(request: Request) -> JSONResponse:
    try:
        config = await request.json()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"invalid json: {e}"}, status_code=400)
    try:
        result = await _run_trial(config)
        return JSONResponse(json.loads(result.model_dump_json()))
    except Exception as e:  # noqa: BLE001
        log.exception("trial failed")
        return JSONResponse({"error": str(e), "type": type(e).__name__}, status_code=500)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    port = int(os.environ.get("EXECUTOR_PORT", "8900"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
