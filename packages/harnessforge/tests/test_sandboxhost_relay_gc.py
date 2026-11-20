"""Relay handler tasks: never garbage-collected while pending (keep_task), and splices always end.

The bug: a pending relay handler whose only references were its (gone) protocol and transport was collected by the
garbage collector inside an unrelated task; its cleanup (close_writer's 5 s wait_for) then armed a timeout against
that task, which cancelled it 5 s later: the rollouts "cancelled from within" and the 64kcap train-loop hangs."""
import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from harnessforge.sandboxhost import net

SRC = str(Path(net.__file__).resolve().parents[2])

GC_SCRIPT = r"""
import asyncio, gc, sys
from harnessforge.sandboxhost import net

async def relay_handler():                     # stuck in a splice whose peer went away
    ev = asyncio.Event()
    try:
        await ev.wait()
    finally:
        await net.close_writer(None)
        async with asyncio.timeout(0.2):        # what close_writer's wait_for does
            await asyncio.sleep(1)

async def innocent():
    gc.collect()                                # the collector happens to run in this task
    await asyncio.sleep(0.6)
    return "ok"

async def main(wrapped):
    loop = asyncio.get_running_loop()
    handler = net.keep_task(relay_handler) if wrapped else relay_handler
    stuck = loop.create_task(handler())         # nothing else references it
    del stuck
    await asyncio.sleep(0.05)
    try:
        print("innocent:", await asyncio.create_task(innocent()))
    except asyncio.CancelledError:
        print("innocent: CANCELLED")
    print("live handlers:", net.live_handlers())

asyncio.run(main(sys.argv[1] == "wrapped"))
"""


def _gc_run(mode: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, "-c", GC_SCRIPT, mode], env=env, capture_output=True, text=True,
                          timeout=60)


def test_unwrapped_pending_handler_cancels_an_innocent_task():   # the bug, reproduced
    out = _gc_run("bare")
    assert "innocent: CANCELLED" in out.stdout
    assert "Task was destroyed but it is pending" in out.stderr
    assert "coroutine ignored GeneratorExit" in out.stderr


def test_keep_task_handler_is_never_collected():                  # the fix
    out = _gc_run("wrapped")
    assert "innocent: ok" in out.stdout and "live handlers: 1" in out.stdout
    assert "Task was destroyed but it is pending" not in out.stderr
    assert "coroutine ignored GeneratorExit" not in out.stderr


def test_splice_ends_after_the_grace_when_one_direction_is_stuck():
    async def main():
        a1, b1 = socket.socketpair()
        a2, b2 = socket.socketpair()
        r1, w1 = await asyncio.open_connection(sock=a1)
        r2, w2 = await asyncio.open_connection(sock=a2)
        b1.close()                       # one side ends (EOF) ...
        # ... the other (b2) stays open and silent, like a peer that vanished without a FIN
        t0 = time.monotonic()
        await asyncio.wait_for(net.splice(r1, w1, r2, w2, grace=0.3), 5)
        assert time.monotonic() - t0 < 3
        assert w1.is_closing() and w2.is_closing()
        b2.close()
    asyncio.run(main())


def test_close_writer_without_a_task_closes_without_waiting():
    class W:
        closed = False

        def close(self):
            self.closed = True
    w = W()
    coro = net.close_writer(w)          # driven outside any task: must not arm a timeout
    try:
        coro.send(None)
    except StopIteration:
        pass
    assert w.closed
