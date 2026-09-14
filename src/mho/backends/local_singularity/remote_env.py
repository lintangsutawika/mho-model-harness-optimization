"""Modal-style remote Singularity backend for harbor trials under Miles.

Unlike the whole-trial external executor (see .service `/trial`), this backend keeps
the harbor Trial -- agent (mini-swe-agent) and verifier -- running IN-PROCESS inside
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
``environment.import_path = mho.backends.local_singularity.remote_env:RemoteSingularityEnvironment``
with ``environment.type = singularity``.

Staging: the manager creates its staging dir under a host path that is bind-mounted into
BOTH the training SIF and the sandbox container at the same absolute path (put it under
BASE_DIR). The in-SIF client's inherited ``upload_file`` writes there; the container
reads it at ``/staging``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from harbor.environments.singularity.singularity import SingularityEnvironment

from . import client as _client

log = logging.getLogger("mho.backends.local_singularity.remote_env")


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
            "env_kwargs": {
                "override_cpus": self._override_cpus,
                "override_memory_mb": self._override_memory_mb,
                "override_storage_mb": self._override_storage_mb,
                "override_gpus": self._override_gpus,
                "singularity_image_cache_dir": str(self._image_cache_dir),
                "singularity_no_mount": self._singularity_no_mount,
                "singularity_force_pull": self._force_pull,
            },
        }

    async def start(self, force_build: bool) -> None:
        """Provision the sandbox on the host manager instead of spawning singularity."""
        info = await _client.provision_sandbox(self._provision_payload(force_build))
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
                await _client.stop_sandbox(sid, delete=delete)
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
        "mho.backends.local_singularity.remote_env",
        "RemoteSingularityEnvironment",
        None,
    )
    log.info("EnvironmentType.SINGULARITY -> RemoteSingularityEnvironment (registry swap installed)")


_install_registry_swap()
