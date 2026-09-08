"""HTTP client for the local external sandbox executor.

Used by SkyRL's HarborGenerator (running inside the training container) to
submit trials to the external executor service instead of running them
in-process. The executor runs on the same node and returns a TrialResult JSON.
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