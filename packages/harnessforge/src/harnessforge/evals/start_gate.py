"""Cap how many hpc-sandbox sandboxes are being set up at once on a node (HPCSB_MAX_STARTING, default 16).

Starting a pod and setting up its agent is CPU-heavy (singularity, the rootfs, slirp4netns egress, the agent's
first imports). With every trial of a 96-wide evaluation starting at once on a 32-core node, each start took minutes
and many hit harbor's environment-start (600 s) or agent-setup (360 s) timeouts. A slot is held from the start of a
trial's sandbox until its agent is set up; trials beyond the cap wait for one, and then start quickly.

Slots are `flock`s on N files in a node-local directory (per batch job by default), so the cap holds across every
process of the job (harbor's, platoon's rollout workers) and a slot is freed when its holder dies. Waiting polls (0.1-0.3 s) instead of
blocking a thread; there is no FIFO order between waiters. mho.backends.start_gate (the harbor path) is the same gate (same
directory and files), so both eval paths on one node share the cap.

    gate = StartGate.from_env()          # HPCSB_MAX_STARTING (0 = no cap), HPCSB_START_SLOT_DIR
    async with gate.slot() as waited_s:
        ...                              # start the sandbox, set up the agent
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import random
import time
from pathlib import Path
from typing import AsyncIterator

DEFAULT_MAX_STARTING = 16


def default_slot_dir() -> Path:
    """Per batch job ($PBS_LOCALDIR: an rt_HC job is a CPU slice of a shared node, so the cap is per slice), else
    per user on this host."""
    if configured := os.environ.get("HPCSB_START_SLOT_DIR"):
        return Path(configured)
    if (local := os.environ.get("PBS_LOCALDIR")) and Path(local).is_dir():
        return Path(local) / "hpcsb-start-slots"
    return Path(f"/tmp/hpcsb-start-slots-{os.getuid()}")


class StartGate:
    def __init__(self, limit: int, slot_dir: str | Path | None = None, *, poll_s: tuple[float, float] = (0.1, 0.3)):
        self.limit = max(0, int(limit))
        self.slot_dir = Path(slot_dir) if slot_dir else default_slot_dir()
        self.poll_s = poll_s

    @classmethod
    def from_env(cls) -> "StartGate":
        raw = os.environ.get("HPCSB_MAX_STARTING", "").strip()
        return cls(int(raw) if raw else DEFAULT_MAX_STARTING)

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def _try_take(self) -> int | None:
        """An open, locked slot file descriptor, or None when all `limit` slots are held."""
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        for i in random.sample(range(self.limit), self.limit):  # spread waiters over the files
            fd = os.open(self.slot_dir / f"slot-{i}", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                os.close(fd)
            except BaseException:
                os.close(fd)
                raise
        return None

    async def acquire(self) -> tuple[int | None, float]:
        """(slot, seconds waited); slot None when the gate is off. Cancellation while waiting holds nothing."""
        if not self.enabled:
            return None, 0.0
        t0 = time.monotonic()
        while (fd := self._try_take()) is None:
            await asyncio.sleep(random.uniform(*self.poll_s))
        return fd, time.monotonic() - t0

    @staticmethod
    def release(slot: int | None) -> None:
        if slot is not None:
            with contextlib.suppress(OSError):
                os.close(slot)  # closing the descriptor drops the flock

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[float]:
        fd, waited = await self.acquire()
        try:
            yield waited
        finally:
            self.release(fd)

    def held(self) -> int:
        """How many slots are held right now (by any process); for logs and tests."""
        if not self.enabled:
            return 0
        n = 0
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.limit):
            fd = os.open(self.slot_dir / f"slot-{i}", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                n += 1
            finally:
                os.close(fd)
        return n
