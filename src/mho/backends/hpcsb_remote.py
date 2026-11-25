"""Training-side client for hpc-sandbox trials (MHO_SANDBOX=hpcsb).

Inside the training container there is no container runtime, so every sandbox operation is
forwarded to the host executor (mho.backends.local_singularity.service, /hpcsb/* routes), which
runs mho.backends.hpc_sandbox_env.HpcSandboxEnvironment on the host: one pod per trial, with its
own network namespace and egress. The agent inside the pod reaches the trainer's session servers
at the node's own address (pods with egress reach the node's network, not its loopback), so no
relay is needed.

File transfers go through a staging directory that the service creates per sandbox under
HB_STAGING_ROOT, a host path bound into the training container at the same absolute path: the
client copies an upload there, the service moves it into the pod; a download goes the other way.

Selected by backend.py's registry swap when MHO_SANDBOX=hpcsb (EnvironmentType.SINGULARITY ->
RemoteHpcSandboxEnvironment).
"""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import httpx
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities

log = logging.getLogger("mho.backends.hpcsb_remote")

# Exec of a whole agent loop can run for hours (harbor passes no timeout); bound it anyway.
_LONG = float(os.environ.get("HPCSB_CLIENT_MAX_EXEC_S", "90000"))


# Tests inject an in-process transport (httpx.ASGITransport over the service app).
_TRANSPORT: httpx.AsyncBaseTransport | None = None


def _http(timeout: float) -> httpx.AsyncClient:
    # No keep-alive: a sandbox start takes minutes, longer than the server's idle timeout, so a
    # pooled connection is often already closed when the next call reuses it (httpx.ReadError on
    # the first exec after start). One connection per request costs ~nothing on loopback.
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0), transport=_TRANSPORT,
                             limits=httpx.Limits(max_keepalive_connections=0))


def _executor_urls() -> list[str]:
    """EXECUTOR_URLS (comma-separated, one executor per node of a multi-node run), else EXECUTOR_URL."""
    raw = os.environ.get("EXECUTOR_URLS", "").strip() or os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8900")
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


def _executor_url() -> str:
    return _executor_urls()[0]


_TOKEN_CACHE: str | None = None


def _headers() -> dict[str, str]:
    global _TOKEN_CACHE
    if _TOKEN_CACHE is None:
        path = os.environ.get("EXECUTOR_TOKEN_FILE", "").strip()
        _TOKEN_CACHE = Path(path).read_text().strip() if path else ""
    return {"X-Executor-Token": _TOKEN_CACHE} if _TOKEN_CACHE else {}


# Live sandboxes per executor in this process: a new sandbox goes to the least-loaded node.
_LOAD: dict[str, int] = {}


def _pick_executor(exclude: set[str]) -> str:
    urls = [u for u in _executor_urls() if u not in exclude] or _executor_urls()
    return min(urls, key=lambda u: (_LOAD.get(u, 0), urls.index(u)))


class RemoteHpcSandboxEnvironment(BaseEnvironment):
    def __init__(self, *args: Any, **kwargs: Any):
        # HARBOR_ENV_KWARGS may carry singularity_* knobs meant for the old backend: the host
        # service owns the sandbox settings (HPCSB_* in its environment), so drop them here.
        self._hpc_kwargs = {k: v for k, v in kwargs.items() if k.startswith("hpc_")}
        for k in [k for k in kwargs if k.startswith(("singularity_", "hpc_"))]:
            kwargs.pop(k)
        super().__init__(*args, **kwargs)
        self._sandbox_id: str | None = None
        self._url: str = _executor_url()
        self._staging: Path | None = None
        self._client: httpx.AsyncClient | None = None

    # --- harbor contract -----------------------------------------------------------------------

    @staticmethod
    def type() -> str:
        return "hpc-sandbox-remote"

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True, disable_internet=True)

    def _validate_definition(self) -> None:
        return  # the host service validates (it can see the image cache and runtime)

    async def _post(self, path: str, payload: dict[str, Any], timeout: float = 600.0,
                    url: str | None = None) -> dict[str, Any]:
        async with _http(timeout) as c:
            r = await c.post(f"{url or self._url}{path}", json=payload, timeout=timeout, headers=_headers())
        data = r.json()
        if r.status_code >= 400 or (isinstance(data, dict) and "error" in data):
            raise RuntimeError(f"hpcsb {path} failed: {data.get('type')}: {data.get('error')}")
        return data

    async def start(self, force_build: bool) -> None:
        payload = {
            "environment_dir": str(self.environment_dir),
            "environment_name": self.environment_name,
            "session_id": self.session_id,
            "force_build": force_build,
            "mounts": [dict(m) for m in self._mounts],
            "persistent_env": dict(self._persistent_env),
            "task_env_config": self.task_env_config.model_dump(mode="json"),
            "network_disabled": bool(self._network_disabled),
            "env_kwargs": self._hpc_kwargs,
        }
        tried: set[str] = set()
        while True:
            url = _pick_executor(tried)
            _LOAD[url] = _LOAD.get(url, 0) + 1
            try:
                info = await self._post("/hpcsb/sandbox", payload, timeout=3600.0, url=url)
                break
            except httpx.ConnectError:
                # an unreachable executor (its node is gone): try the others before giving up.
                # Only a failed connect: after a read timeout the sandbox may exist there.
                _LOAD[url] -= 1
                tried.add(url)
                if len(tried) >= len(_executor_urls()):
                    raise
            except BaseException:
                _LOAD[url] -= 1
                raise
        self._url = url
        self._sandbox_id = info["sandbox_id"]
        self._staging = Path(info["staging_dir"])
        log.info("hpcsb sandbox %s (staging %s)", self._sandbox_id, self._staging)

    async def stop(self, delete: bool) -> None:
        sid, self._sandbox_id = self._sandbox_id, None
        try:
            if sid:
                _LOAD[self._url] = max(0, _LOAD.get(self._url, 0) - 1)
                async with _http(120.0) as c:
                    await c.request("DELETE", f"{self._url}/hpcsb/{sid}",
                                    params={"delete": str(delete).lower()}, headers=_headers())
        except Exception:  # noqa: BLE001
            log.exception("failed to stop hpcsb sandbox %s", sid)
        finally:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    def _sid(self) -> str:
        if not self._sandbox_id:
            raise RuntimeError("hpcsb environment not started")
        return self._sandbox_id

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        payload = {
            "command": command,
            "cwd": cwd,
            # persistent + per-call + scoped env merged HERE (the host-side env has none of it)
            "env": self._merge_env(env),
            "timeout_sec": timeout_sec,
            "user": self._resolve_user(user),
        }
        http_timeout = (float(timeout_sec) + 120.0) if timeout_sec else _LONG
        data = await self._post(f"/hpcsb/{self._sid()}/exec", payload, timeout=http_timeout)
        return ExecResult(stdout=data.get("stdout"), stderr=data.get("stderr"),
                          return_code=int(data.get("return_code", 1)))

    # --- transfers through the shared staging dir ----------------------------------------------

    def _stage_path(self, name: str) -> Path:
        assert self._staging is not None
        return self._staging / f"{uuid.uuid4().hex[:12]}-{Path(name).name or 'x'}"

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {src}")
        staged = self._stage_path(src.name)
        shutil.copy2(src, staged)
        try:
            await self._post(f"/hpcsb/{self._sid()}/upload",
                             {"src": str(staged), "dst": target_path, "dir": False})
        finally:
            staged.unlink(missing_ok=True)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        src = Path(source_dir)
        if not src.is_dir():
            raise FileNotFoundError(f"Source directory not found: {src}")
        staged = self._stage_path(src.name)
        shutil.copytree(src, staged, symlinks=True)
        try:
            await self._post(f"/hpcsb/{self._sid()}/upload",
                             {"src": str(staged), "dst": target_dir, "dir": True})
        finally:
            shutil.rmtree(staged, ignore_errors=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        staged = self._stage_path(source_path)
        try:
            await self._post(f"/hpcsb/{self._sid()}/download",
                             {"src": source_path, "dst": str(staged), "dir": False})
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, target)
        finally:
            staged.unlink(missing_ok=True)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        staged = self._stage_path(source_dir)
        try:
            await self._post(f"/hpcsb/{self._sid()}/download",
                             {"src": source_dir, "dst": str(staged), "dir": True})
            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staged, target, symlinks=True, dirs_exist_ok=True)
        finally:
            shutil.rmtree(staged, ignore_errors=True)
