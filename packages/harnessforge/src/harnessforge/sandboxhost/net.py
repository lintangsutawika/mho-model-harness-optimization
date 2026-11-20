"""Addresses and byte relays shared by the sandbox host and its clients."""
from __future__ import annotations

import asyncio
import contextlib
import socket
import subprocess

#: TCP keepalive: a peer whose node vanished is noticed in about IDLE + INTERVAL * COUNT seconds.
KEEPALIVE_IDLE_S = 15
KEEPALIVE_INTERVAL_S = 5
KEEPALIVE_COUNT = 4


def advertise_ip(mode: str | None = None) -> str:
    """The address other nodes reach this one at. None/'auto': the first 10.0.x.y address (ABCI's compute network),
    else the hostname's address; a prefix ending in '.' picks the first local address with it; anything else is
    used as given."""
    prefix = "10.0." if not mode or mode == "auto" else mode if mode.endswith(".") else None
    if prefix is None:
        return mode  # type: ignore[return-value]
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > 3 and parts[3].startswith(prefix):
                return parts[3].split("/")[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def set_keepalive(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    if sock is None or sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    with contextlib.suppress(OSError, AttributeError):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, KEEPALIVE_IDLE_S)
        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, KEEPALIVE_INTERVAL_S)
        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, KEEPALIVE_COUNT)


async def close_writer(writer: asyncio.StreamWriter | None, timeout: float = 5.0) -> None:
    """Close ``writer`` and wait (at most ``timeout`` s) until its transport is gone; past that, abort it.

    The wait is shielded: ``writer.wait_closed()`` awaits the protocol's close future, which every later
    ``wait_closed()`` of the same writer shares. Timing it out unshielded cancels that future, and the next await
    on it (a second close of the same writer, as the relay's splice and its handler both do) raises a raw
    ``CancelledError`` in a task nobody cancelled. A writer that will not close in time (a peer that stopped
    reading: the transport waits to flush first) is aborted instead of left half-closed."""
    if writer is None:
        return
    with contextlib.suppress(Exception):
        writer.close()
    try:
        await asyncio.wait_for(asyncio.shield(writer.wait_closed()), timeout)
    except asyncio.CancelledError:
        raise                                      # this task's own cancellation goes on
    except Exception:  # noqa: BLE001 - a timeout or a broken connection: drop it hard
        transport = getattr(writer, "transport", None)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.abort()


async def splice(r1: asyncio.StreamReader, w1: asyncio.StreamWriter,
                 r2: asyncio.StreamReader, w2: asyncio.StreamWriter) -> None:
    """Copy bytes both ways until both directions ended (a half-close is passed on), then close both. A broken
    direction closes both, so the other one ends too."""
    async def pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while chunk := await reader.read(65536):
                writer.write(chunk)
                await writer.drain()
            if writer.can_write_eof():
                writer.write_eof()
        except (ConnectionError, OSError, RuntimeError):
            for w in (w1, w2):
                with contextlib.suppress(Exception):
                    w.close()

    try:
        await asyncio.gather(pump(r1, w2), pump(r2, w1))
    finally:
        await close_writer(w1)
        await close_writer(w2)
