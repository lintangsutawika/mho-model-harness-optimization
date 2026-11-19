"""Sail sandboxes with retried environment start and authenticated Docker Hub pulls.

1. Retry. SailboxEnvironment.start() fails the whole trial on a transient Sail API error. Under a
   burst of first-time image builds (e.g. TMax: one image per task) Sail's upload endpoint answers
   "local file upload failed: HTTP 503" for a large share of starts (28/80 trials in the TMax smoke
   test), and every such trial is lost. start() already tears down a half-created Sailbox before
   it raises, so starting again is safe: retry transient errors (ApiError marked retryable, HTTP
   429/5xx) with jittered exponential backoff.

2. Docker Hub login (sail >= 0.18). Sail's image builder pulls anonymously unless the image
   definition carries a registry login, and the anonymous quota (~100 pulls/h per builder IP)
   is gone within a minute on a dataset with one image per task (TMax sweep: 26.8k of 27.7k
   trials failed with TOOMANYREQUESTS). sail.harbor builds every task image through
   Image.from_dockerfile / Image.from_compose / Image.from_registry without logins, so this module
   swaps sail.harbor's Image for a proxy that adds a docker.io login to those calls. The password
   travels as Secret.ephemeral (sent with the build request only, never stored or logged).

    harbor run --environment-import-path mho.backends.sail_retry:RetryingSailboxEnvironment
    (eval.sh SAIL_IMPORT default; train.sh via backend.py's MHO_SANDBOX=sail swap)

Env: MHO_SAIL_START_RETRIES (default 5), MHO_SAIL_START_BACKOFF_S (first delay, default 15),
     MHO_DOCKER_USERNAME / MHO_DOCKER_PASSWORD, falling back to SINGULARITY_DOCKER_USERNAME /
     SINGULARITY_DOCKER_PASSWORD (repo .env). No credentials -> anonymous pulls, as before.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re

import sail
import sail.harbor as _sail_harbor
from sail.errors import ApiError
from sail.harbor import SailboxEnvironment

log = logging.getLogger("mho.backends.sail_retry")
_TRANSIENT = {429, 500, 502, 503, 504}
_HTTP = re.compile(r"\bHTTP (\d{3})\b")
DOCKER_HUB = "docker.io"


def is_transient(e: BaseException) -> bool:
    if not isinstance(e, ApiError):
        return False
    if getattr(e, "retryable", False) or getattr(e, "status_code", None) in _TRANSIENT:
        return True
    m = _HTTP.search(str(e))
    return bool(m) and int(m.group(1)) in _TRANSIENT


# --- Docker Hub login -------------------------------------------------------------------------


def docker_credentials() -> tuple[str, str] | None:
    user = os.environ.get("MHO_DOCKER_USERNAME") or os.environ.get("SINGULARITY_DOCKER_USERNAME")
    pw = os.environ.get("MHO_DOCKER_PASSWORD") or os.environ.get("SINGULARITY_DOCKER_PASSWORD")
    return (user, pw) if user and pw else None


def docker_logins() -> list:
    creds = docker_credentials()
    if creds is None:
        return []
    return [sail.RegistryLogin(DOCKER_HUB, username=creds[0], password=sail.Secret.ephemeral(creds[1]))]


def _is_docker_hub(ref: str) -> bool:
    """`docker pull` resolution: a first path segment with '.' or ':' (or 'localhost') is a
    registry host; anything else is Docker Hub."""
    first, sep, _ = ref.partition("/")
    if not sep:
        return True
    is_host = "." in first or ":" in first or first == "localhost"
    return not is_host or first in (DOCKER_HUB, "index.docker.io", "registry-1.docker.io")


class _WithLogins:
    """Callable with an .aio twin (like sail's dual methods) that adds registry_logins."""

    def __init__(self, fn):
        self._fn = fn

    @staticmethod
    def _kw(kwargs):
        if kwargs.get("registry_logins") is None:
            logins = docker_logins()
            if logins:
                kwargs["registry_logins"] = logins
        return kwargs

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **self._kw(kwargs))

    async def aio(self, *args, **kwargs):
        return await self._fn.aio(*args, **self._kw(kwargs))


class _FromRegistryWithLogin:
    """from_registry takes its login as username=/password= (Docker Hub refs only)."""

    def __init__(self, fn):
        self._fn = fn

    @staticmethod
    def _kw(ref, kwargs):
        creds = docker_credentials()
        if creds and "password" not in kwargs and _is_docker_hub(str(ref)):
            kwargs.update(username=creds[0], password=sail.Secret.ephemeral(creds[1]))
        return kwargs

    def __call__(self, ref, *args, **kwargs):
        return self._fn(ref, *args, **self._kw(ref, kwargs))

    async def aio(self, ref, *args, **kwargs):
        return await self._fn.aio(ref, *args, **self._kw(ref, kwargs))


class _ImageWithDockerLogin:
    """Stand-in for sail.Image inside sail.harbor; everything not wrapped passes through."""

    def __init__(self, image):
        self._image = image
        self.from_dockerfile = _WithLogins(image.from_dockerfile)
        self.from_compose = _WithLogins(image.from_compose)
        self.from_registry = _FromRegistryWithLogin(image.from_registry)

    def __getattr__(self, name):
        return getattr(self._image, name)


def install_docker_login() -> bool:
    if not hasattr(sail, "RegistryLogin"):
        log.warning("sail %s has no RegistryLogin (needs >= 0.18): Docker Hub pulls stay anonymous",
                    getattr(sail, "__version__", "?"))
        return False
    if not isinstance(_sail_harbor.Image, _ImageWithDockerLogin):
        _sail_harbor.Image = _ImageWithDockerLogin(_sail_harbor.Image)
    if docker_credentials() is None:
        log.warning("no Docker Hub credentials in env: Sail pulls stay anonymous (rate-limited)")
        return False
    return True


DOCKER_LOGIN_ACTIVE = install_docker_login()


class RetryingSailboxEnvironment(SailboxEnvironment):
    async def start(self, force_build: bool) -> None:
        retries = int(os.environ.get("MHO_SAIL_START_RETRIES", "5"))
        delay = float(os.environ.get("MHO_SAIL_START_BACKOFF_S", "15"))
        for attempt in range(retries + 1):
            try:
                return await super().start(force_build)
            except ApiError as e:
                if attempt == retries or not is_transient(e):
                    raise
                wait = delay * 2 ** attempt * random.uniform(0.75, 1.25)
                log.warning("sail start failed (%s); retry %d/%d in %.0fs", e, attempt + 1, retries, wait)
                await asyncio.sleep(wait)
