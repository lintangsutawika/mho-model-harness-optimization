"""HTTP client for the local host-side sandbox service.

Two calling conventions share one service (see .service):

  * Whole-trial (SkyRL path): ``run_trial_async`` POSTs a full trial config to ``/trial``;
    the host runs the entire harbor Trial (agent + verifier + sandbox) and returns a
    TrialResult. Used by SkyRL's HarborGenerator.

  * Sandbox lifecycle (Miles / lossless path): ``provision_sandbox`` / ``stop_sandbox``
    manage just the container via ``/sandbox``; the agent + verifier run in the caller's
    process (inside the training SIF) so token-in/out is recorded natively. Used by
    RemoteSingularityEnvironment.

Both talk to EXECUTOR_URL (default http://127.0.0.1:8900) on the same node.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from harbor.models.trial.result import TrialResult


def executor_url() -> str:
    return os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8900").rstrip("/")


def executor_enabled() -> bool:
    return os.environ.get("EXECUTOR_URL") is not None


def health() -> dict:
    r = httpx.get(f"{executor_url()}/health", timeout=5.0)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Whole-trial path (SkyRL)
# --------------------------------------------------------------------------- #
def run_trial(config: dict) -> TrialResult:
    """POST a trial config to the external executor; return the TrialResult."""
    base = executor_url()
    r = httpx.post(f"{base}/trial", json=config, timeout=None)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"executor trial failed: {data.get('type')}: {data.get('error')}")
    return TrialResult.model_validate(data)


async def run_trial_async(config: dict) -> TrialResult:
    """Async variant for use inside the generator's event loop."""
    base = executor_url()
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{base}/trial", json=config, timeout=None)
        r.raise_for_status()
        data = r.json()
    if "error" in data:
        raise RuntimeError(data.get("error", "executor trial failed"))
    return TrialResult.model_validate(data)


# --------------------------------------------------------------------------- #
# Sandbox-lifecycle path (Miles / lossless)
# --------------------------------------------------------------------------- #
async def provision_sandbox(payload: dict[str, Any]) -> dict[str, Any]:
    """Ask the host manager to launch a sandbox container.

    Returns ``{"sandbox_id", "server_port", "staging_dir", "sif_path"}``. Raises on error.
    """
    base = executor_url()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        r = await client.post(f"{base}/sandbox", json=payload)
        r.raise_for_status()
        data = r.json()
    if "error" in data:
        raise RuntimeError(f"sandbox provision failed: {data.get('type')}: {data.get('error')}")
    return data


async def stop_sandbox(sandbox_id: str, *, delete: bool = True) -> None:
    """Ask the host manager to tear down a sandbox container."""
    base = executor_url()
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.request(
            "DELETE", f"{base}/sandbox/{sandbox_id}", params={"delete": str(delete).lower()}
        )
        r.raise_for_status()
