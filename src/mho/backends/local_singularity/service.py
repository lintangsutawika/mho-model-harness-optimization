"""Host-side sandbox service for the local_singularity backend.

Runs as a SEPARATE process on the HPC node (outside the training SIF). Serves two
calling conventions (see .client):

  * POST /trial            -- whole-trial (SkyRL): runs the entire harbor Trial
                              (agent + verifier + sandbox) on the host and returns a
                              TrialResult. The agent reaches the trainer's vLLM endpoint
                              via the localhost api_base in the config.

  * POST   /sandbox        -- sandbox lifecycle (Miles / lossless): launches ONLY the
    DELETE /sandbox/{id}      sandbox container on the host (reusing
                              WritableSingularityEnvironment.start()/stop()) and returns
                              its server_port + staging_dir. The agent + verifier run
                              in the CALLER's process (inside the training SIF), reachable
                              at localhost:{port} because singularity shares the host net
                              namespace -- so the trainer records token-in/out natively.

Run (on the node):
    EXECUTOR_PORT=8900 HB_STAGING_ROOT=$BASE_DIR/hbstaging PYTHONPATH=$REPO/src \
        python -m mho.backends.local_singularity.service
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# harbor's Trial.create() / verifier import agent classes (harness.agent_harness:*, etc.)
# in THIS process at trial time. Force src/ to the FRONT of sys.path: `python -m` puts the
# repo ROOT on sys.path[0] (cwd), and the repo root has its OWN `harness/` package that
# would shadow src/harness/. So src must precede the root even if already present.
_REPO_ROOT = Path(__file__).resolve().parents[4]  # src/mho/backends/local_singularity/ -> repo
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir():
    _src_s = str(_SRC)
    while _src_s in sys.path:
        sys.path.remove(_src_s)
    sys.path.insert(0, _src_s)

# Startup self-check (to the service log): proves the agent class imports in THIS process.
try:
    import importlib as _il
    _il.import_module("harness.agent_harness")
    print(f"[executor startup] OK: harness.agent_harness importable | repo={_REPO_ROOT}", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[executor startup] FAIL: harness.agent_harness NOT importable: {_e!r}\n"
          f"  __file__={__file__}\n  repo={_REPO_ROOT}\n  sys.path={sys.path}", flush=True)

from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.config import TrialConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402
from harbor.models.trial.result import TrialResult  # noqa: E402
from harbor.trial.trial import Trial  # noqa: E402

# Import the backend so it loads in-process (fail fast) and captures the ORIGINAL
# asyncio.create_subprocess_exec before any wrapping.
from . import environment as _env  # noqa: E402

log = logging.getLogger("mho.backends.local_singularity.service")
logging.basicConfig(level=logging.INFO)

# harbor's mini-swe-agent wrapper resolves the model API key from MSWEA_API_KEY in THIS
# process; local vLLM ignores the value, so default a placeholder.
os.environ.setdefault("MSWEA_API_KEY", "dummy")

# Staging dirs created by WritableSingularityEnvironment._start_server() must live on a
# host path bind-mounted into BOTH the training SIF and the sandbox container at the same
# absolute path (so the in-SIF caller's upload_file writes where the container reads).
# Put it under BASE_DIR via HB_STAGING_ROOT; falls back to the system tmp dir.
_STAGING_ROOT = os.environ.get("HB_STAGING_ROOT")
if _STAGING_ROOT:
    Path(_STAGING_ROOT).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = _STAGING_ROOT

app = FastAPI(title="mho-local-singularity-service")

_BACKEND_IMPORT_PATH = "mho.backends.local_singularity.environment:WritableSingularityEnvironment"

# Live sandboxes provisioned via /sandbox, keyed by sandbox_id.
_SANDBOXES: dict[str, _env.WritableSingularityEnvironment] = {}


# --------------------------------------------------------------------------- #
# Whole-trial path (SkyRL)
# --------------------------------------------------------------------------- #
async def _run_trial(config: dict) -> TrialResult:
    # The service IS the sandbox host: pin the backend to our writable singularity
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


# --------------------------------------------------------------------------- #
# Sandbox-lifecycle path (Miles / lossless)
# --------------------------------------------------------------------------- #
def _build_env(payload: dict) -> _env.WritableSingularityEnvironment:
    """Reconstruct the host-side writable singularity environment from a /sandbox payload.

    The manager only ever calls start()/stop() on this env (launch + tear down the
    container). exec/upload/download are issued by the in-SIF caller directly over HTTP,
    so a minimal TrialPaths (never dereferenced by the env) suffices here."""
    task_env_config = EnvironmentConfig.model_validate(payload["task_env_config"])
    trial_dir = Path(tempfile.mkdtemp(prefix="hbmgr_trial_"))
    trial_paths = TrialPaths(trial_dir=trial_dir)
    env_kwargs = {k: v for k, v in (payload.get("env_kwargs") or {}).items() if v is not None}
    # The .sif cache is a HOST concept: prefer the manager's own cache dir over whatever
    # in-SIF path the caller sent (paths may not coincide across the SIF boundary).
    _host_cache = os.environ.get("SIF_IMAGE_CACHE_DIR")
    if _host_cache:
        env_kwargs["singularity_image_cache_dir"] = _host_cache
    return _env.WritableSingularityEnvironment(
        environment_dir=Path(payload["environment_dir"]),
        environment_name=payload["environment_name"],
        session_id=payload["session_id"],
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        mounts=payload.get("mounts") or None,
        persistent_env=payload.get("persistent_env") or None,
        **env_kwargs,
    )


@app.post("/sandbox")
async def post_sandbox(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"invalid json: {e}"}, status_code=400)
    try:
        env = _build_env(payload)
        await env.start(force_build=bool(payload.get("force_build", False)))
        sandbox_id = uuid.uuid4().hex
        _SANDBOXES[sandbox_id] = env
        staging = env._staging_dir  # noqa: SLF001
        sif = env._sif_path  # noqa: SLF001
        log.info("provisioned sandbox %s port=%s staging=%s", sandbox_id, env._server_port, staging)  # noqa: SLF001
        return JSONResponse({
            "sandbox_id": sandbox_id,
            "server_port": env._server_port,  # noqa: SLF001
            "staging_dir": str(staging) if staging else None,
            "sif_path": str(sif) if sif else None,
        })
    except Exception as e:  # noqa: BLE001
        log.exception("sandbox provision failed")
        return JSONResponse({"error": str(e), "type": type(e).__name__}, status_code=500)


@app.delete("/sandbox/{sandbox_id}")
async def delete_sandbox(sandbox_id: str, delete: str = "true") -> JSONResponse:
    env = _SANDBOXES.pop(sandbox_id, None)
    if env is None:
        return JSONResponse({"error": "unknown sandbox_id", "type": "KeyError"}, status_code=404)
    try:
        await env.stop(delete=(str(delete).lower() == "true"))
        return JSONResponse({"status": "stopped", "sandbox_id": sandbox_id})
    except Exception as e:  # noqa: BLE001
        log.exception("sandbox stop failed")
        return JSONResponse({"error": str(e), "type": type(e).__name__}, status_code=500)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "sandboxes": len(_SANDBOXES)}


def main() -> None:
    import uvicorn

    port = int(os.environ.get("EXECUTOR_PORT", "8900"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
