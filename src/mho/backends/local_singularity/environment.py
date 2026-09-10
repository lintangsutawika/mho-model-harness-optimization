"""Writable-sandbox Singularity backend for harbor trials on this HPC.

harbor's stock SingularityEnvironment runs `singularity exec --writable-tmpfs <sif>`. On
nodes without `user_allow_other` in /etc/fuse.conf that silently degrades to a read-only
rootfs (no overlayfs), so harbor's in-sandbox server can't create its venv. This backend
instead extracts the image to a per-session DIRECTORY (`singularity build --sandbox`) and
runs it with `--writable <dir>` (needs neither squashfuse nor overlayfs), by transparently
rewriting harbor's exec argv. Selected per trial via
`environment.import_path = mho.backends.local_singularity.environment:WritableSingularityEnvironment`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from harbor.environments.singularity.singularity import SingularityEnvironment

log = logging.getLogger("mho.backends.local_singularity.environment")

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
