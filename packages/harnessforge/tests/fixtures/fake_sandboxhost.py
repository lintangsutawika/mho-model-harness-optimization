"""A fake sandbox provider for sandbox host tests (no containers): a sandbox is a directory, its commands run as
local processes in it, and sandbox paths map into it. Run as a script it serves a SandboxHost with this provider:

    python fake_sandboxhost.py STATE_DIR INDEX SLOTS [LEASE_S]
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from platoon.sandbox.base import DownloadReport, ExecResult

STARTED: list = []      # specs the providers of this process started
STOPPED: list = []      # ids stopped


class FakeSandbox:
    def __init__(self, provider: "FakeProvider", spec) -> None:
        self.id = "fake-" + uuid.uuid4().hex[:10]
        self._provider = provider
        self.spec = spec
        self.root = Path(tempfile.mkdtemp(prefix="sbx-", dir=provider.base))
        self.killed = 0
        self.kept: frozenset = frozenset()
        self.stopped = False
        self.procs: set = set()
        self.routes = {name: f"http://127.0.0.1:{18080 + i}" for i, name in enumerate(sorted(spec.upstreams))}

    def path(self, p: str) -> Path:
        return self.root / p.lstrip("/")

    async def exec(self, cmd, *, timeout_s=None, env=None, cwd=None, stdin=None) -> ExecResult:
        if self.stopped:
            raise RuntimeError("stopped")
        argv = ["bash", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(self.root), env={**os.environ, "SBX": str(self.root), **(env or {})},
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.procs.add(proc)
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(exit_code=137, stdout="", stderr="", timed_out=True)
        except asyncio.CancelledError:
            proc.kill()
            raise
        finally:
            self.procs.discard(proc)
        if self.stopped:  # stopped under the command, as hpc-sandbox's stop kills it
            raise RuntimeError(f"sandbox {self.id} stopped during the command")
        return ExecResult(exit_code=proc.returncode, stdout=out.decode(), stderr=err.decode())

    async def upload(self, src, dst) -> None:
        target = self.path(dst)
        if Path(src).is_dir():
            shutil.copytree(src, target, dirs_exist_ok=True, symlinks=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    async def download(self, src, dst):
        source = self.path(src)
        if not source.exists():
            raise FileNotFoundError(f"no such file or directory in the sandbox: {src}")
        if source.is_dir():
            shutil.copytree(source, dst, dirs_exist_ok=True, symlinks=True)
        else:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dst)
        return DownloadReport()

    def keep_processes(self, entries) -> None:
        self.kept = self.kept | frozenset(entries)

    async def kill_processes(self) -> None:
        self.killed += 1

    async def ensure_routes(self):
        return dict(self.routes)

    async def stop(self) -> None:
        if not self.stopped:
            self.stopped = True
            for proc in list(self.procs):
                if proc.returncode is None:
                    proc.kill()
            STOPPED.append(self.id)
            shutil.rmtree(self.root, ignore_errors=True)


class FakeProvider:
    def __init__(self, options: dict) -> None:
        self.options = dict(options)
        self.base = Path(options.get("base") or tempfile.mkdtemp(prefix="fakehost-"))
        self.base.mkdir(parents=True, exist_ok=True)
        self.start_delay = float(options.get("start_delay", 0))

    async def start(self, spec) -> FakeSandbox:
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        if spec.image == "fail":
            raise RuntimeError("image fail does not start")
        for m in spec.mounts:
            if not os.path.exists(m.source):
                raise FileNotFoundError(f"mount source {m.source} is not on this host")
        sandbox = FakeSandbox(self, spec)
        STARTED.append(spec)
        return sandbox


def main() -> None:
    import logging
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from harnessforge.sandboxhost import fleet, wire
    from harnessforge.sandboxhost.daemon import SandboxHost
    state, index, slots = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    lease = float(sys.argv[4]) if len(sys.argv) > 4 else 900.0
    logging.basicConfig(level=logging.INFO)
    fd = fleet.FleetDir(state)
    host = SandboxHost(token=wire.read_token(fd.token_file), slots=slots,
                       scratch=Path(tempfile.mkdtemp(prefix=f"h{index}-")), name=f"fake/{index}",
                       provider_options={"start_delay": os.environ.get("FAKE_START_DELAY", "0")},
                       provider_factory=FakeProvider, lease_s=lease, status_file=fd.host_file(index),
                       stop_files=(fd.stop_file,), advertise="127.0.0.1",
                       info={"index": index, "job_id": f"pid{os.getpid()}"})

    async def run() -> None:
        await host.start_server("127.0.0.1", 0)
        await host.serve_forever()

    _ = time
    asyncio.run(run())


if __name__ == "__main__":
    main()
