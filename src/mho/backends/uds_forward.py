"""Forward a Unix socket to a TCP address: puts a host TCP server (e.g. the node's vLLM) behind a
socket that hpc-sandbox pods relay in as an "upstream" (see mho.backends.hpc_sandbox_env).

    python -m mho.backends.uds_forward /local/$PBS_JOBID/llm.sock 127.0.0.1:8000

The socket is created mode 0600 (only our uid connects; pod relays run as our uid) and removed on
exit. Prints "ready <path>" once listening. No dependencies beyond the standard library.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys

CHUNK = 1 << 16


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(CHUNK):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(Exception):
            if writer.can_write_eof():
                writer.write_eof()  # half-close: let the other direction finish


async def _handle(host: str, port: int, cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
    try:
        ur, uw = await asyncio.open_connection(host, port)
    except OSError:
        cw.close()
        return
    try:
        await asyncio.gather(_pipe(cr, uw), _pipe(ur, cw))
    finally:
        for w in (uw, cw):
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()


async def serve(path: str, host: str, port: int, ready: asyncio.Event | None = None) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    old = os.umask(0o177)
    try:
        server = await asyncio.start_unix_server(lambda r, w: _handle(host, port, r, w), path=path)
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    try:
        if ready is not None:
            ready.set()
        async with server:
            await server.serve_forever()
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or ":" not in argv[1]:
        print("usage: python -m mho.backends.uds_forward /abs/path.sock HOST:PORT", file=sys.stderr)
        return 2
    path, (host, _, port) = argv[0], argv[1].rpartition(":")

    async def run() -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)
        ready = asyncio.Event()
        srv = asyncio.create_task(serve(path, host, int(port), ready))
        waiter = asyncio.create_task(ready.wait())
        await asyncio.wait({srv, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if srv.done():  # failed to listen (bad path, path too long, ...): surface the error
            waiter.cancel()
            srv.result()
        print(f"ready {path}", flush=True)
        await srv

    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
