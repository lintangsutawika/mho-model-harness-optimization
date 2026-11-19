"""Modal-style remote Singularity backend for harbor trials under Miles.

Keeps the harbor Trial -- agent (mini-swe-agent) and verifier -- running IN-PROCESS inside
the training container, so the agent's model calls flow through the trainer's native
generate path and token-in/out is recorded losslessly. Only the *sandbox container*
is provisioned out-of-process, on a HOST manager, exactly mirroring how the Modal
backend provisions its container in Modal's cloud:

    [training proc: agent loop + verifier]  --HTTP /exec-->  [host singularity container]
              |                                                        ^
              +----------- POST /sandbox ------> [host manager: WritableSingularityEnvironment.start()]

harbor's stock ``SingularityEnvironment`` is already a client/server split: ``start()``
launches a FastAPI server inside the container and reserves a host port; ``exec`` /
``upload_file`` / ``download_file`` just talk to ``http://localhost:{port}`` and a
bind-mounted staging dir. Singularity shares the host network namespace, so a container
launched by the host manager is reachable at ``localhost:{port}`` from inside the
training SIF. This subclass therefore overrides ONLY the provisioning half:

  * ``start()``   -> POST /sandbox to the host manager (which runs the real
                     WritableSingularityEnvironment on the host) and record the returned
                     port + staging dir.
  * ``stop()``    -> DELETE /sandbox/{id}.
  * ``exec`` / ``upload_file`` / ``download_file`` / ``upload_dir`` / ``download_dir``
    are inherited UNCHANGED.

Select per trial via
``environment.import_path = mho.backends.local_singularity.backend:RemoteSingularityEnvironment``
with ``environment.type = singularity``.

Staging: the manager creates its staging dir under a host path that is bind-mounted into
BOTH the training SIF and the sandbox container at the same absolute path (put it under
BASE_DIR). The in-SIF client's inherited ``upload_file`` writes there; the container
reads it at ``/staging``.
"""
from __future__ import annotations

import logging
import sys as _sys
from pathlib import Path
from typing import Any

import httpx

from harbor.environments.singularity.singularity import SingularityEnvironment

# Sandbox-lifecycle HTTP helpers to the host service (folded from the former client.py).
_EXECUTOR_URL = None


def _executor_url() -> str:
    global _EXECUTOR_URL
    if _EXECUTOR_URL is None:
        import os as _os
        _EXECUTOR_URL = _os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8900").rstrip("/")
    return _EXECUTOR_URL


async def _provision_sandbox(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /sandbox; return {sandbox_id, server_port, staging_dir, sif_path}."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as c:
        r = await c.post(f"{_executor_url()}/sandbox", json=payload)
        r.raise_for_status()
        data = r.json()
    if "error" in data:
        raise RuntimeError(f"sandbox provision failed: {data.get('type')}: {data.get('error')}")
    return data


async def _stop_sandbox(sandbox_id: str, *, delete: bool = True) -> None:
    """DELETE /sandbox/{id} to tear down a host-side sandbox."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
        r = await c.request("DELETE", f"{_executor_url()}/sandbox/{sandbox_id}",
                            params={"delete": str(delete).lower()})
        r.raise_for_status()

log = logging.getLogger("mho.backends.local_singularity.backend")


# The repo ROOT has its own `harness/` package that can shadow src/harness/ whenever the repo
# root lands on the worker's sys.path (cwd, etc.). Force src/ to the FRONT and drop any
# already-imported `harness` that resolved outside src, so src/harness (agent harness) wins.
_SRC = str(Path(__file__).resolve().parents[3])  # .../src
while _SRC in _sys.path:
    _sys.path.remove(_SRC)
_sys.path.insert(0, _SRC)
for _m in [m for m in list(_sys.modules) if m == "harness" or m.startswith("harness.")]:
    _f = getattr(_sys.modules.get(_m), "__file__", "") or ""
    if not _f.startswith(_SRC):
        del _sys.modules[_m]


class RemoteSingularityEnvironment(SingularityEnvironment):
    """SingularityEnvironment whose container is provisioned on a host manager."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._sandbox_id: str | None = None

    def _validate_definition(self) -> None:
        """No-op: the .sif is built and validated on the HOST manager, and need not be
        visible from inside the training container. (Stock SingularityEnvironment here
        asserts the .sif file exists locally.)"""
        return

    def _provision_payload(self, force_build: bool) -> dict[str, Any]:
        """Everything the host manager needs to reconstruct WritableSingularityEnvironment
        and call .start()."""
        return {
            "environment_dir": str(self.environment_dir),
            "environment_name": self.environment_name,
            "session_id": self.session_id,
            "force_build": force_build,
            "mounts": [dict(m) for m in self._mounts],
            "persistent_env": dict(self._persistent_env),
            "task_env_config": self.task_env_config.model_dump(mode="json"),
            # Sandbox operating knobs only (resource overrides are null/no-op and omitted).
            "env_kwargs": {
                "singularity_image_cache_dir": str(self._image_cache_dir),
                "singularity_no_mount": self._singularity_no_mount,
                "singularity_force_pull": self._force_pull,
                "singularity_writable_sandbox": True,
            },
        }

    async def start(self, force_build: bool) -> None:
        """Provision the sandbox on the host manager instead of spawning singularity."""
        info = await _provision_sandbox(self._provision_payload(force_build))
        self._sandbox_id = info["sandbox_id"]
        self._server_port = int(info["server_port"])
        self._staging_dir = Path(info["staging_dir"])
        sif_path = info.get("sif_path")
        self._sif_path = Path(sif_path) if sif_path else None
        self._http_client = httpx.AsyncClient(timeout=30.0)
        log.info(
            "provisioned sandbox %s on host manager: port=%s staging=%s",
            self._sandbox_id, self._server_port, self._staging_dir,
        )
        # NOTE: the manager's .start() already ran _upload_environment_dir_after_start();
        # do not re-upload the environment dir from here.

    async def stop(self, delete: bool) -> None:
        """Tear down the host-side sandbox."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        sid = self._sandbox_id
        if sid:
            try:
                await _stop_sandbox(sid, delete=delete)
            except Exception:  # noqa: BLE001
                log.exception("failed to stop sandbox %s", sid)
            finally:
                self._sandbox_id = None


# --------------------------------------------------------------------------- #
# Registry swap: make the built-in EnvironmentType.SINGULARITY resolve to OUR
# RemoteSingularityEnvironment, not harbor's stock SingularityEnvironment.
#
# Mile's in-process Harbor connector builds EnvironmentConfig(type=<HARBOR_ENV_TYPE>)
# (no import_path) and harbor's Trial instantiates the env via
# EnvironmentFactory.create_environment_from_config -> _ENVIRONMENT_REGISTRY lookup
# (NOT a direct import). Repoint the registry entry for the built-in SINGULARITY
# type at our class so HARBOR_ENV_TYPE=singularity yields our forwarding env
# (host-provisioned sandbox, no nested apptainer) while keeping the connector's
# agent<->generate wiring intact. Runs idempotently on import.
# --------------------------------------------------------------------------- #
def _install_registry_swap() -> None:
    from harbor.environments import factory
    from harbor.models.environment_type import EnvironmentType
    try:
        if factory._ENVIRONMENT_REGISTRY[EnvironmentType.SINGULARITY].class_name == "RemoteSingularityEnvironment":
            return  # already swapped
    except (KeyError, AttributeError):
        pass
    factory._ENVIRONMENT_REGISTRY[EnvironmentType.SINGULARITY] = factory._EnvEntry(
        "mho.backends.local_singularity.backend",
        "RemoteSingularityEnvironment",
        None,
    )
    log.info("EnvironmentType.SINGULARITY -> RemoteSingularityEnvironment (registry swap installed)")


def _install_sail_swap() -> None:
    """MHO_SANDBOX=sail: trials run in Sail cloud sandboxes instead of host singularity.

    The connector still builds EnvironmentConfig(type=singularity); repoint that registry
    entry at sail.harbor:SailboxEnvironment (the class `harbor run --environment-import-path`
    uses for SAIL evals). A task without docker_image is built from environment/Dockerfile."""
    from harbor.environments import factory
    from harbor.models.environment_type import EnvironmentType
    # Default: Sail with retried start on transient API errors (mho/backends/sail_retry.py).
    mod, _, cls = _os_env.get("MHO_SAIL_IMPORT",
                              "mho.backends.sail_retry:RetryingSailboxEnvironment").partition(":")
    factory._ENVIRONMENT_REGISTRY[EnvironmentType.SINGULARITY] = factory._EnvEntry(mod, cls, None)
    log.info("EnvironmentType.SINGULARITY -> %s:%s (MHO_SANDBOX=sail)", mod, cls)


def _relay_session_url(base_url: str) -> str:
    """Session URL for an agent inside a Sail sandbox. It cannot reach this node, so its
    model calls go to the Modal relay (MHO_RELAY_URL = https://<relay>/k/<secret>), whose
    bridge on this node forwards the path to the session server. base_url is
    http://<host>:<port>/sessions/<id>; the policy URL is that path + /v1."""
    from urllib.parse import urlparse
    # Keep host:port: each session lives on one of N session-server instances
    # (node IP, ports base..base+N-1); the bridge routes /_hp/<host>:<port> there.
    u = urlparse(base_url)
    return (_relay_base().rstrip("/") + f"/_hp/{u.hostname}:{u.port or 80}"
            + u.path.rstrip("/") + "/v1")


def _relay_base() -> str:
    """The relay to send a NEW trial to. train.sh rotates relays (a Sandbox lives <= 24 h) by
    rewriting MHO_RELAY_URL_FILE; trials already running keep the URL they started with, and the
    old relay stays up until they have finished."""
    f = _os_env.get("MHO_RELAY_URL_FILE")
    if f:
        try:
            url = Path(f).read_text().strip()
            if url:
                return url
        except OSError as e:
            log.warning("cannot read MHO_RELAY_URL_FILE (%s); using MHO_RELAY_URL", e)
    return _os_env["MHO_RELAY_URL"]


def _node_address(host: str) -> str:
    """An address of this node that a pod reaches over its egress (slirp4netns). A pod has its
    own /etc/hosts and resolves names through slirp's DNS forwarder, so the node's hostname
    (which Miles puts in session URLs, e.g. hnode657) does not resolve there; and slirp runs with
    --disable-host-loopback, so 127.x is never reachable. MHO_SESSION_HOST overrides."""
    import ipaddress
    import socket
    if override := _os_env.get("MHO_SESSION_HOST", "").strip():
        return override
    try:
        ipaddress.ip_address(host)
        addr = host
    except ValueError:
        try:
            addr = socket.gethostbyname(host)
        except OSError:
            addr = "127.0.0.1"
    if addr.startswith("127."):
        # The primary outbound address (no packet is sent for a UDP connect).
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            addr = s.getsockname()[0]
    return addr


def _pod_session_url(*args: Any, **kwargs: Any) -> str:
    """resolve_session_url for MHO_SANDBOX=hpcsb: Miles' URL with the host replaced by the
    node's IP address (see _node_address)."""
    from urllib.parse import urlparse
    url = _ORIG_RESOLVE_SESSION_URL(*args, **kwargs)
    u = urlparse(url)
    if not u.hostname:
        return url
    netloc = _node_address(u.hostname) + (f":{u.port}" if u.port else "")
    return u._replace(netloc=netloc).geturl()


def _install_hpcsb_swap() -> None:
    """MHO_SANDBOX=hpcsb: trials run in hpc-sandbox pods on this node, provisioned and driven by
    the host executor (service.py /hpcsb/*). The agent in a pod reaches the session servers at
    the node's IP address over the pod's egress (_pod_session_url rewrites the hostname)."""
    from harbor.environments import factory
    from harbor.models.environment_type import EnvironmentType
    factory._ENVIRONMENT_REGISTRY[EnvironmentType.SINGULARITY] = factory._EnvEntry(
        "mho.backends.hpcsb_remote", "RemoteHpcSandboxEnvironment", None)
    log.info("EnvironmentType.SINGULARITY -> RemoteHpcSandboxEnvironment (MHO_SANDBOX=hpcsb)")


import os as _os  # noqa: E402
_os_env = _os.environ
_MODE = _os_env.get("MHO_SANDBOX", "").strip().lower()
_SAIL = _MODE == "sail"
if _SAIL:
    if not _os_env.get("MHO_RELAY_URL"):
        raise RuntimeError("MHO_SANDBOX=sail needs MHO_RELAY_URL (train.sh deploys the relay)")
    _install_sail_swap()
elif _MODE == "hpcsb":
    _install_hpcsb_swap()
else:
    _install_registry_swap()
# Arm the AgentFactory swap (mini-swe-agent -> our MicroSweAgent, whose install adds
# --with mcp so litellm's mcp.types import resolves). Importing runs the idempotent swap.
import harness.micro_swe_agent  # noqa: E402,F401

# --------------------------------------------------------------------------- #
# Miles connector hook: re-export the agent function and be the --custom-agent-function-path.
# Importing backend (as the hook) arms the registry swap above, then re-exports
# harbor_agent_function.run unchanged -- so the connector's agent<->generate wiring and
# native token recording are untouched. run_training.py points --custom-agent-function-path
# at this module.
# --------------------------------------------------------------------------- #
import harbor_agent_function as _haf  # noqa: E402
if _SAIL:
    # run() resolves the agent's policy URL through this module-level name.
    _haf.resolve_session_url = _relay_session_url
elif _MODE == "hpcsb":
    _ORIG_RESOLVE_SESSION_URL = _haf.resolve_session_url
    _haf.resolve_session_url = _pod_session_url
run = _haf.run
