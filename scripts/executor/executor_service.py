"""Local external sandbox executor for SkyRL HarborGenerator.

Runs as a SEPARATE process on the HPC node (outside the training container).
Receives a harbor trial config dict over HTTP and runs the full harbor
``Trial`` (agent + verifier + sandbox) on the host, returning a
``TrialResult`` JSON. The agent reaches the training container's vLLM endpoint
via the localhost ``api_base`` carried in the config (training and executor
share the node).

This is the "Modal-like but local" external backend: SkyRL posts trials here
instead of running them in-process inside the training container.

Writable-sandbox note: HPC nodes often lack ``user_allow_other`` in
/etc/fuse.conf, so ``singularity exec --writable-tmpfs <sif>`` silently
degrades to a read-only rootfs (underlay has no overlayfs) and harbor's
in-container server cannot create its venv. To sidestep that, the sandbox
image is extracted to a per-session DIRECTORY with ``singularity build
--sandbox`` and launched with ``--writable <dir>`` -- a plain directory needs
neither a squashfuse mount nor overlayfs.

Run (on the node; host uv env with `harbor` + the repo's `src/` on PYTHONPATH,
and this module importable):
    EXECUTOR_PORT=8900 PYTHONPATH=$REPO/src:$REPO/scripts/executor \
        uv run --python 3.12 python -m executor_service
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult

# harbor's Trial.create() imports the agent/verifier classes (e.g. harness.agent_harness:*)
# in THIS process at trial time, so the repo's src/ (and this dir) must be importable no
# matter the launch cwd/PYTHONPATH. Bootstrap sys.path from this file's location.
import sys as _sys
# Force src/ to the FRONT: `python -m` puts the repo ROOT on sys.path[0] (cwd), and the repo
# root has its OWN `harness/` package (harbor_run/scaffold) that would shadow src/harness/
# (agent_harness). Must precede the root even when already present, so remove-then-insert.
_REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/executor/ -> repo root
for _extra in (_REPO_ROOT / "scripts" / "executor", _REPO_ROOT / "src"):
    _extra_s = str(_extra)
    if _extra.is_dir():
        while _extra_s in _sys.path:
            _sys.path.remove(_extra_s)
        _sys.path.insert(0, _extra_s)

# Startup self-check (prints to the service log): proves whether the agent class is
# importable in THIS process, so a failing trial's cause is obvious at boot, not per-request.
try:
    import importlib as _il
    _il.import_module("harness.agent_harness")
    print(f"[executor startup] OK: harness.agent_harness importable | repo={_REPO_ROOT}", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[executor startup] FAIL: harness.agent_harness NOT importable: {_e!r}\n"
          f"  __file__={__file__}\n  repo={_REPO_ROOT}\n  sys.path={_sys.path}", flush=True)
from harbor.trial.trial import Trial
from harbor.environments.singularity.singularity import SingularityEnvironment

log = logging.getLogger("harbor.executor_service")
logging.basicConfig(level=logging.INFO)

# harbor's mini-swe-agent wrapper resolves the model API key from MSWEA_API_KEY in THIS
# (executor) process (agents/installed/mini_swe_agent.py: api_key_envs=("MSWEA_API_KEY",))
# and errors "No API key found ..." if unset. The agent talks to local vLLM, which ignores
# the value, so default a placeholder here rather than depending on the launch command.
os.environ.setdefault("MSWEA_API_KEY", "dummy")

app = FastAPI(title="harbor-local-external-executor")

_WRITABLE_REGISTRY: dict[str, Path] = {}


def _rewrite_singularity_argv(argv: list) -> list:
    """Rewrite a `singularity exec --writable-tmpfs <registered-sif>` argv to use
    `--writable <extracted-dir>`, pre-creating bind destinations and --pwd."""
    if (
        len(argv) < 3
        or os.path.basename(str(argv[0])) != "singularity"
        or str(argv[1]) != "exec"
        or "--writable-tmpfs" not in argv
    ):
        return argv
    sandbox: Path | None = None
    img_idx: int | None = None
    for i, tok in enumerate(argv):
        if str(tok) in _WRITABLE_REGISTRY:
            sandbox = _WRITABLE_REGISTRY[str(tok)]
            img_idx = i
            break
    if sandbox is None or img_idx is None:
        return argv
    dests: list[str] = []
    for i, tok in enumerate(argv):
        if tok == "-B" and i + 1 < len(argv):
            spec = str(argv[i + 1])
            dst = spec.split(":", 1)[1] if ":" in spec else spec
            dests.append(dst.split(":", 1)[0])
        elif tok == "--pwd" and i + 1 < len(argv):
            dests.append(str(argv[i + 1]))
    for dst in dests:
        if dst.startswith("/"):
            (sandbox / dst.lstrip("/")).mkdir(parents=True, exist_ok=True)
    out = list(argv)
    out[out.index("--writable-tmpfs")] = "--writable"
    out[img_idx] = str(sandbox)
    # harbor runs with --containall, which pairs with --writable-tmpfs to give a minimal
    # tmpfs /dev + host-derived resolver. Once we swap to --writable (a plain sandbox dir),
    # /dev is the image's own (empty) /dev -- no /dev/null (breaks bootstrap redirects/apt/
    # server) -- and /etc/resolv.conf is the image's (empty), so DNS fails ("Temporary
    # failure in name resolution") and in-sandbox `uv`/`pip` can't reach PyPI. Bind the host
    # copies back in (idempotently). Network namespace is shared (no --net), so with a
    # resolver the sandbox reaches PyPI just like the host.
    for _hostpath in ("/dev", "/etc/resolv.conf", "/etc/hosts"):
        _already = any(
            str(out[i]) in ("-B", "--bind") and i + 1 < len(out)
            and str(out[i + 1]).split(":", 1)[0] == _hostpath
            for i in range(len(out))
        )
        if not _already and os.path.exists(_hostpath):
            out[2:2] = ["--bind", _hostpath]
    return out


_ORIG_CREATE_SUBPROCESS = asyncio.create_subprocess_exec


async def _wrapped_create_subprocess_exec(*args: Any, **kwargs: Any):
    return await _ORIG_CREATE_SUBPROCESS(*_rewrite_singularity_argv(list(args)), **kwargs)


def _install_writable_argv_rewrite() -> None:
    """Idempotently route SingularityEnvironment's exec to a writable dir."""
    if getattr(asyncio, "create_subprocess_exec", None) is _wrapped_create_subprocess_exec:
        return
    asyncio.create_subprocess_exec = _wrapped_create_subprocess_exec  # type: ignore[assignment]


# One extraction per sif, guarded by a per-sif lock: with many concurrent trials, every
# trial's WritableSingularityEnvironment calls this for the SAME sif, and racing
# `rm -rf`+`singularity build --sandbox` on one path fails ("sandbox assemble failed: ...
# file exists"). Extract once under the lock, then reuse the registered sandbox dir.
_EXTRACT_LOCKS: dict[str, "asyncio.Lock"] = {}


async def _extract_writable_sandbox(sif_path: Path) -> Path:
    key = str(sif_path)
    lock = _EXTRACT_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        cached = _WRITABLE_REGISTRY.get(key)
        if cached is not None and cached.exists():
            return cached
        root = Path(os.environ.get("SBX_DIR", tempfile.gettempdir()))
        root.mkdir(parents=True, exist_ok=True)
        sandbox = root / f"hbsbx_{os.getpid()}_{sif_path.name.replace('.', '_')}"
        proc = await _ORIG_CREATE_SUBPROCESS(
            "rm", "-rf", str(sandbox),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        cmd = ["singularity", "build", "--fix-perms", "--sandbox", str(sandbox), str(sif_path)]
        log.info("Building writable sandbox: %s", " ".join(cmd))
        proc = await _ORIG_CREATE_SUBPROCESS(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to build writable sandbox from {sif_path}: "
                f"{stderr.decode(errors='replace')}"
            )
        _WRITABLE_REGISTRY[key] = sandbox
        return sandbox


class WritableSingularityEnvironment(SingularityEnvironment):
    """SingularityEnvironment whose exec is redirected to an extracted writable dir."""

    def __init__(self, *args: Any, writable_sandbox: bool = True, **kwargs: Any):
        self._writable_sandbox = writable_sandbox
        super().__init__(*args, **kwargs)
        _install_writable_argv_rewrite()

    async def _convert_docker_to_sif(self, docker_image: str, *, force_pull: bool = False) -> Path:
        sif = await super()._convert_docker_to_sif(docker_image, force_pull=force_pull)
        if self._writable_sandbox:
            sbx = await _extract_writable_sandbox(sif)
            _WRITABLE_REGISTRY[str(sif)] = sbx
        return sif


async def _run_trial(config: dict) -> TrialResult:
    # The executor IS the sandbox host: pin the backend to our writable
    # singularity environment regardless of what the training-side config said.
    config = json.loads(json.dumps(config))  # defensively copy
    env = config.setdefault("environment", {})
    env["type"] = "singularity"
    env["import_path"] = "executor_service:WritableSingularityEnvironment"
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