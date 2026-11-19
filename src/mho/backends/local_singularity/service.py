"""Host-side sandbox service for the local_singularity backend.

Runs as a SEPARATE process on the HPC node (outside the training SIF). Serves the
sandbox-lifecycle convention (Miles): POST /sandbox launches ONLY the sandbox container
on the host (using harbor-singularity-hpc's SingularityWritableEnvironment) and returns
its server_port + staging_dir. The agent + verifier run in the CALLER's process (inside
the training SIF), reachable at localhost:{port} because singularity shares the host net
namespace -- so the trainer records token-in/out natively.

Run (on the node):
    EXECUTOR_PORT=8900 HB_STAGING_ROOT=$BASE_DIR/hbstaging PYTHONPATH=$REPO/src \
        python -m mho.backends.local_singularity.service
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# harbor's Trial.create() / verifier import agent classes (harness.micro_swe_agent:*, etc.)
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
    _il.import_module("harness.micro_swe_agent")
    print(f"[executor startup] OK: harness.micro_swe_agent importable | repo={_REPO_ROOT}", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[executor startup] FAIL: harness.micro_swe_agent NOT importable: {_e!r}\n"
          f"  __file__={__file__}\n  repo={_REPO_ROOT}\n  sys.path={sys.path}", flush=True)

from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

# Host-side sandbox engine: harbor-singularity-hpc's writable-rootfs Singularity
# environment (drift-guarded against harbor c178c20, proven on this cluster by the
# tts-tokens-that-suffice project). Replaces the hand-rolled WritableSingularityEnvironment
# that lived in ./environment.py. The in-SIF RemoteSingularityEnvironment still forwards
# /sandbox provisioning here; sandboxes run on THIS host, not nested in the training SIF.
from harbor_singularity_hpc.environment import SingularityWritableEnvironment as _WritableEnv  # noqa: E402

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


# Live sandboxes provisioned via /sandbox, keyed by sandbox_id.
_SANDBOXES: dict[str, _WritableEnv] = {}


def _cleanup_dirs(env: _WritableEnv | None, trial_dir: Path | None) -> None:
    """Remove a sandbox's host-side dirs (manager trial dir + staging dir). Without this every
    provision leaves two dirs behind under HB_STAGING_ROOT: ~10k per 8 GRPO steps."""
    dirs = [trial_dir]
    if env is not None:
        dirs += [getattr(env, "_staging_dir", None), getattr(getattr(env, "trial_paths", None), "trial_dir", None)]
    for d in {Path(x) for x in dirs if x}:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Sandbox-lifecycle path (Miles / lossless)
# --------------------------------------------------------------------------- #
def _build_env(payload: dict) -> _WritableEnv:
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
    return _WritableEnv(
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
    env = None
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
        if env is not None:
            try:
                await env.stop(delete=True)
            except Exception:  # noqa: BLE001
                pass
        _cleanup_dirs(env, None)
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
    finally:
        if str(delete).lower() == "true":
            _cleanup_dirs(env, None)


# --------------------------------------------------------------------------- #
# hpc-sandbox path (MHO_SANDBOX=hpcsb): the caller's RemoteHpcSandboxEnvironment
# (mho.backends.hpcsb_remote) forwards EVERY operation here; this process runs
# mho.backends.hpc_sandbox_env.HpcSandboxEnvironment (one pod per trial, own network
# namespace + egress) on the host. Transfers go through a per-sandbox staging dir under
# HB_STAGING_ROOT, which the training container sees at the same absolute path.
# Settings come from this process's environment: HPCSB_IMAGE_CACHE, HPCSB_NETWORK,
# HPCSB_RUNTIME, HPCSB_DOCKERFILE_STEPS, HPCSB_POD_USERS.
# --------------------------------------------------------------------------- #
_HPCSB: dict[str, tuple] = {}  # sandbox_id -> (env, staging_dir)


def _hpcsb_kwargs(payload: dict) -> dict:
    kw = {"hpc_image_cache": os.environ.get("HPCSB_IMAGE_CACHE") or None,
          "hpc_network": os.environ.get("HPCSB_NETWORK") or None,
          "hpc_runtime": os.environ.get("HPCSB_RUNTIME") or None,
          "hpc_dockerfile_steps": os.environ.get("HPCSB_DOCKERFILE_STEPS") or None,
          "hpc_pod_users": os.environ.get("HPCSB_POD_USERS") or None}
    kw.update({k: v for k, v in (payload.get("env_kwargs") or {}).items() if k.startswith("hpc_")})
    return {k: v for k, v in kw.items() if v is not None}


def _build_hpcsb_env(payload: dict, trial_dir: Path):
    from harbor.models.task.config import NetworkMode, NetworkPolicy
    from mho.backends.hpc_sandbox_env import HpcSandboxEnvironment

    policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK) if payload.get("network_disabled") else None
    return HpcSandboxEnvironment(
        environment_dir=Path(payload["environment_dir"]),
        environment_name=payload["environment_name"],
        session_id=payload["session_id"],
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig.model_validate(payload["task_env_config"]),
        mounts=payload.get("mounts") or None,
        persistent_env=payload.get("persistent_env") or None,
        network_policy=policy,
        **_hpcsb_kwargs(payload),
    )


def _err(e: Exception, status: int = 500) -> JSONResponse:
    return JSONResponse({"error": str(e), "type": type(e).__name__}, status_code=status)


def _hpcsb_get(sandbox_id: str):
    item = _HPCSB.get(sandbox_id)
    if item is None:
        raise KeyError(f"unknown hpcsb sandbox {sandbox_id}")
    return item


@app.post("/hpcsb/sandbox")
async def hpcsb_create(request: Request) -> JSONResponse:
    env = None
    trial_dir = Path(tempfile.mkdtemp(prefix="hpcsb_trial_"))
    # HPCSB_XFER_ROOT: transfer dirs on storage the caller can see. Multi-node runs put it on a
    # shared filesystem (the trainer may sit on another node); node-local scratch otherwise.
    staging = Path(tempfile.mkdtemp(prefix="hpcsb_stage_", dir=os.environ.get("HPCSB_XFER_ROOT") or None))
    staging.chmod(0o700)
    try:
        payload = await request.json()
        env = _build_hpcsb_env(payload, trial_dir)
        await env.start(force_build=bool(payload.get("force_build", False)))
        sandbox_id = uuid.uuid4().hex
        _HPCSB[sandbox_id] = (env, staging, trial_dir)
        log.info("hpcsb sandbox %s up for %s", sandbox_id, payload.get("session_id"))
        return JSONResponse({"sandbox_id": sandbox_id, "staging_dir": str(staging)})
    except Exception as e:  # noqa: BLE001
        log.exception("hpcsb provision failed")
        if env is not None:
            try:
                await env.stop(delete=True)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(trial_dir, ignore_errors=True)
        return _err(e)


@app.post("/hpcsb/{sandbox_id}/exec")
async def hpcsb_exec(sandbox_id: str, request: Request) -> JSONResponse:
    try:
        env = _hpcsb_get(sandbox_id)[0]
        p = await request.json()
        r = await env.exec(p["command"], cwd=p.get("cwd"), env=p.get("env"),
                           timeout_sec=p.get("timeout_sec"), user=p.get("user"))
        return JSONResponse({"stdout": r.stdout, "stderr": r.stderr, "return_code": r.return_code})
    except KeyError as e:
        return _err(e, 404)
    except Exception as e:  # noqa: BLE001
        log.exception("hpcsb exec failed")
        return _err(e)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@app.post("/hpcsb/{sandbox_id}/upload")
async def hpcsb_upload(sandbox_id: str, request: Request) -> JSONResponse:
    try:
        env, staging, _ = _hpcsb_get(sandbox_id)
        p = await request.json()
        src = Path(p["src"])
        if not _inside(src, staging):  # only what the caller staged for THIS sandbox
            raise PermissionError(f"upload source outside the sandbox's staging dir: {src}")
        if p.get("dir"):
            await env.upload_dir(src, p["dst"])
        else:
            await env.upload_file(src, p["dst"])
        return JSONResponse({"status": "ok"})
    except KeyError as e:
        return _err(e, 404)
    except Exception as e:  # noqa: BLE001
        log.exception("hpcsb upload failed")
        return _err(e)


@app.post("/hpcsb/{sandbox_id}/download")
async def hpcsb_download(sandbox_id: str, request: Request) -> JSONResponse:
    try:
        env, staging, _ = _hpcsb_get(sandbox_id)
        p = await request.json()
        dst = Path(p["dst"])
        if not _inside(dst.parent if not dst.exists() else dst, staging):
            raise PermissionError(f"download target outside the sandbox's staging dir: {dst}")
        if p.get("dir"):
            await env.download_dir(p["src"], dst)
        else:
            await env.download_file(p["src"], dst)
        return JSONResponse({"status": "ok"})
    except KeyError as e:
        return _err(e, 404)
    except FileNotFoundError as e:
        return _err(e, 404)
    except Exception as e:  # noqa: BLE001
        log.exception("hpcsb download failed")
        return _err(e)


@app.delete("/hpcsb/{sandbox_id}")
async def hpcsb_delete(sandbox_id: str, delete: str = "true") -> JSONResponse:
    item = _HPCSB.pop(sandbox_id, None)
    if item is None:
        return _err(KeyError(f"unknown hpcsb sandbox {sandbox_id}"), 404)
    env, staging, trial_dir = item
    try:
        await env.stop(delete=(str(delete).lower() == "true"))
        return JSONResponse({"status": "stopped", "sandbox_id": sandbox_id})
    except Exception as e:  # noqa: BLE001
        log.exception("hpcsb stop failed")
        return _err(e)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(trial_dir, ignore_errors=True)


# EXECUTOR_TOKEN_FILE: when set, every request must carry the token in X-Executor-Token. Multi-node
# runs bind the executor to the node's cluster address (other nodes' trainers call it), where
# anyone on the cluster network could otherwise start containers as this user.
def _load_token() -> str:
    path = os.environ.get("EXECUTOR_TOKEN_FILE", "").strip()
    return Path(path).read_text().strip() if path else ""


_TOKEN = _load_token()


@app.middleware("http")
async def _require_token(request: Request, call_next):
    if _TOKEN:
        import hmac
        if not hmac.compare_digest(request.headers.get("x-executor-token", ""), _TOKEN):
            return JSONResponse({"error": "unauthorized", "type": "PermissionError"}, status_code=401)
    return await call_next(request)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "sandboxes": len(_SANDBOXES), "hpcsb_sandboxes": len(_HPCSB)}


def main() -> None:
    import uvicorn

    port = int(os.environ.get("EXECUTOR_PORT", "8900"))
    # Hundreds of trials connect at once and agent-loop execs hold a request for minutes: a deep
    # accept backlog, and idle keep-alives long enough for clients that do reuse connections.
    host = os.environ.get("EXECUTOR_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host not in ("127.0.0.1", "localhost") and not _TOKEN:
        raise SystemExit("EXECUTOR_HOST is not loopback: set EXECUTOR_TOKEN_FILE")
    uvicorn.run(app, host=host, port=port, backlog=4096, timeout_keep_alive=120,
                limit_concurrency=None)


if __name__ == "__main__":
    main()
