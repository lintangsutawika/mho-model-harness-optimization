"""The sandbox host's wire format: framed JSON over TCP, HMAC challenge auth, file payloads, errors, tar helpers.

A frame is a 4-byte big-endian length, that many bytes of a JSON object (the header) and, when the header has
`nbytes`, that many raw bytes (the payload: an exec's stdin, a tar of an upload, a download or a mirrored mount).

Every connection starts with the server's challenge, `{"op": "challenge", "nonce": <hex>, "proto": 1}`, which the
client answers with `{"op": "auth", "mac": HMAC-SHA256(token, "hfsbh1:" + nonce)}`. The token never crosses the
network, and a fresh nonce per connection means a recorded answer cannot be replayed. The same handshake guards the
client's upstream relay (the model gateway's socket), with the roles reversed.

The token is a file of the fleet's state dir, mode 0600 (`read_token` refuses a file that others can read). Nothing
prints it.
"""
from __future__ import annotations

import asyncio
import builtins
import hashlib
import hmac
import json
import os
import secrets
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path
from typing import Any

PROTO = 1
MAC_PREFIX = b"hfsbh1:"
MAX_HEADER_BYTES = 256 * 1024 * 1024
CHUNK = 1024 * 1024
#: payloads up to this size are kept in memory; larger ones go to a temporary file
INLINE_PAYLOAD_BYTES = 8 * 1024 * 1024
AUTH_TIMEOUT_S = 20.0


class WireError(ConnectionError):
    """The peer broke the protocol or went away."""


class AuthError(WireError):
    """The peer failed the challenge."""


class HostFull(RuntimeError):
    """The sandbox host has no free slot."""


class RemoteError(RuntimeError):
    """An error raised on the sandbox host whose type has no local equivalent."""


# ---------------------------------------------------------------------------------------------------------- token
def new_token(path: str | os.PathLike) -> None:
    """Create the token file (0600) unless it exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(32) + "\n")


def read_token(path: str | os.PathLike) -> bytes:
    path = Path(path)
    st = path.stat()
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"token file {path} is readable by others (mode {oct(st.st_mode & 0o777)}); chmod 600")
    token = path.read_text().strip().encode()
    if len(token) < 32:
        raise ValueError(f"token file {path} is too short")
    return token


def mac(token: bytes, nonce: str) -> str:
    return hmac.new(token, MAC_PREFIX + nonce.encode(), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------------------------------------- frames
async def write_frame(writer: asyncio.StreamWriter, header: dict[str, Any],
                      payload: bytes | Path | None = None) -> None:
    header = dict(header)
    if payload is not None:
        header["nbytes"] = len(payload) if isinstance(payload, (bytes, bytearray)) else Path(payload).stat().st_size
    data = json.dumps(header, separators=(",", ":")).encode()
    writer.write(len(data).to_bytes(4, "big") + data)
    if isinstance(payload, (bytes, bytearray)):
        writer.write(payload)
    elif payload is not None:
        with open(payload, "rb") as f:
            while chunk := f.read(CHUNK):
                writer.write(chunk)
                await writer.drain()
    await writer.drain()


async def read_frame(reader: asyncio.StreamReader, *, spool_dir: str | os.PathLike | None = None,
                     timeout: float | None = None) -> tuple[dict[str, Any], bytes | Path | None]:
    """(header, payload). A payload larger than INLINE_PAYLOAD_BYTES is written to a temporary file in `spool_dir`
    (the caller removes it); smaller ones are bytes. EOF raises WireError."""
    async def _read() -> tuple[dict[str, Any], bytes | Path | None]:
        try:
            size = int.from_bytes(await reader.readexactly(4), "big")
            if size > MAX_HEADER_BYTES:
                raise WireError(f"frame header of {size} bytes")
            header = json.loads(await reader.readexactly(size))
        except asyncio.IncompleteReadError as exc:
            raise WireError("connection closed") from exc
        if not isinstance(header, dict):
            raise WireError("frame header is not an object")
        n = header.get("nbytes")
        if n is None:
            return header, None
        n = int(n)
        try:
            if n <= INLINE_PAYLOAD_BYTES:
                return header, await reader.readexactly(n)
            fd, name = tempfile.mkstemp(prefix="hfsbh-", suffix=".part", dir=spool_dir)
            try:
                with os.fdopen(fd, "wb") as f:
                    left = n
                    while left:
                        chunk = await reader.readexactly(min(CHUNK, left))
                        f.write(chunk)
                        left -= len(chunk)
            except BaseException:
                Path(name).unlink(missing_ok=True)
                raise
            return header, Path(name)
        except asyncio.IncompleteReadError as exc:
            raise WireError("connection closed in a payload") from exc

    return await (asyncio.wait_for(_read(), timeout) if timeout else _read())


def payload_bytes(payload: bytes | Path | None) -> bytes:
    if payload is None:
        return b""
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    try:
        return Path(payload).read_bytes()
    finally:
        Path(payload).unlink(missing_ok=True)


def payload_file(payload: bytes | Path | None, spool_dir: str | os.PathLike | None = None) -> Path:
    """The payload as a file (written out when it came inline); the caller removes it."""
    if isinstance(payload, Path):
        return payload
    fd, name = tempfile.mkstemp(prefix="hfsbh-", suffix=".part", dir=spool_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(payload or b"")
    return Path(name)


# ---------------------------------------------------------------------------------------------------------- auth
async def server_handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, token: bytes) -> dict:
    """Challenge the peer; returns its auth frame (which may carry more fields). Raises AuthError."""
    nonce = secrets.token_hex(16)
    await write_frame(writer, {"op": "challenge", "nonce": nonce, "proto": PROTO})
    header, _ = await read_frame(reader, timeout=AUTH_TIMEOUT_S)
    if header.get("op") != "auth" or not hmac.compare_digest(str(header.get("mac", "")), mac(token, nonce)):
        raise AuthError("authentication failed")
    return header


async def client_handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, token: bytes,
                           **extra: Any) -> None:
    header, _ = await read_frame(reader, timeout=AUTH_TIMEOUT_S)
    if header.get("op") != "challenge" or "nonce" not in header:
        raise WireError(f"expected a challenge, got {header.get('op')!r}")
    if int(header.get("proto", 0)) != PROTO:
        raise WireError(f"protocol {header.get('proto')} (this side speaks {PROTO})")
    await write_frame(writer, {"op": "auth", "mac": mac(token, str(header["nonce"])), **extra})


# ---------------------------------------------------------------------------------------------------------- errors
def error_header(exc: BaseException) -> dict[str, Any]:
    return {"error": {"type": type(exc).__name__, "module": type(exc).__module__, "message": str(exc)}}


_BUILTIN_ERRORS = ("FileNotFoundError", "FileExistsError", "IsADirectoryError", "NotADirectoryError",
                   "PermissionError", "ValueError", "TimeoutError", "NotImplementedError", "KeyError", "OSError",
                   "RuntimeError", "TypeError")


def rebuild_error(info: dict[str, Any]) -> BaseException:
    """The local exception for a remote one: builtins and hpc_sandbox/platoon errors keep their type."""
    name, module, message = str(info.get("type")), str(info.get("module")), str(info.get("message"))
    if name == "HostFull":
        return HostFull(message)
    if name in _BUILTIN_ERRORS:
        return getattr(builtins, name)(message)
    if module.startswith(("hpc_sandbox", "platoon")):
        try:
            import importlib
            cls = getattr(importlib.import_module(module), name)
            if isinstance(cls, type) and issubclass(cls, Exception):
                return cls(message)
        except Exception:  # noqa: BLE001 - fall through to a generic error
            pass
    return RemoteError(f"{name}: {message}")


def lost_error(message: str) -> BaseException:
    """hpc_sandbox.SandboxLostError when hpc-sandbox is installed (what platoon sees from a dead container)."""
    try:
        from hpc_sandbox import SandboxLostError
        return SandboxLostError(message)
    except ImportError:
        return ConnectionError(message)


# ---------------------------------------------------------------------------------------------------------- tar
def pack(src: str | os.PathLike, spool_dir: str | os.PathLike | None = None) -> tuple[Path, str]:
    """A tar of `src` (a file as member "f", a directory as "d" with its contents) in a temporary file, and its kind
    ("file" | "dir"). Symlinks are kept as links."""
    src = Path(src)
    kind = "dir" if src.is_dir() and not src.is_symlink() else "file"
    fd, name = tempfile.mkstemp(prefix="hfsbh-", suffix=".tar", dir=spool_dir)
    os.close(fd)
    with tarfile.open(name, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.add(str(src), arcname="d" if kind == "dir" else "f", recursive=True)
    return Path(name), kind


def _skip_unsafe(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    try:
        return tarfile.data_filter(member, path)
    except tarfile.FilterError:
        return None  # a link out of the copy: left out, as hpc-sandbox leaves them out


def unpack(tar_path: str | os.PathLike, dest: str | os.PathLike, kind: str, *, trusted: bool) -> Path:
    """Extract a `pack` tar into the directory `dest`; returns dest/"f" or dest/"d". `trusted` (our own client's
    upload) keeps what the tar says (the "tar" filter); otherwise unsafe members are skipped (data filter)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest, filter="tar" if trusted else _skip_unsafe)
    out = dest / ("d" if kind == "dir" else "f")
    if kind == "dir":
        out.mkdir(exist_ok=True)
    return out


def place(extracted: Path, dst: str | os.PathLike, kind: str) -> None:
    """Put a downloaded file at `dst`, or a downloaded directory's contents into `dst`."""
    dst = Path(dst)
    if kind == "dir":
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extracted, dst, symlinks=True, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_dir() and not dst.is_symlink():
            dst = dst / extracted.name
        shutil.move(str(extracted), str(dst))
