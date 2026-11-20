"""net.close_writer: a timed-out close must not poison the writer's shared close future."""
import asyncio
import socket

from harnessforge.sandboxhost import net


async def _stuck_writer():
    """A client writer whose transport cannot flush (the peer never reads), so wait_closed() hangs."""
    accepted = []

    async def on_conn(r, w):
        accepted.append(w)
        await asyncio.sleep(3600)
    srv = await asyncio.start_server(on_conn, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    w.write(b"x" * (64 << 20))
    await asyncio.sleep(0.2)
    return srv, w, accepted


def test_second_close_after_a_timed_out_close_does_not_raise():
    async def main():
        srv, w, accepted = await _stuck_writer()
        try:
            await net.close_writer(w, timeout=0.3)
            assert not w._protocol._closed.cancelled()      # the shared close future is untouched
            assert w.transport.is_closing()                 # and the stuck transport was aborted
            await net.close_writer(w, timeout=0.3)          # the relay closes twice: no raw CancelledError
            assert asyncio.current_task().cancelling() == 0
        finally:
            for a in accepted:
                a.transport.abort()
            srv.close()
    asyncio.run(main())


def test_another_waiter_on_the_close_future_is_not_cancelled():
    async def main():
        srv, w, accepted = await _stuck_writer()
        try:
            w.close()
            other = asyncio.create_task(w.wait_closed())   # someone else waiting for the same close
            await asyncio.sleep(0)
            await net.close_writer(w, timeout=0.3)         # ours times out
            await asyncio.wait_for(other, 2)               # the other waiter ends normally (the abort closed it)
            assert not other.cancelled()
        finally:
            for a in accepted:
                a.transport.abort()
            srv.close()
    asyncio.run(main())


def test_close_writer_still_propagates_its_own_cancellation():
    async def main():
        srv, w, accepted = await _stuck_writer()
        try:
            t = asyncio.create_task(net.close_writer(w, timeout=30))
            await asyncio.sleep(0.05)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("close_writer swallowed its own cancellation")
        finally:
            w.transport.abort()
            for a in accepted:
                a.transport.abort()
            srv.close()
    asyncio.run(main())
