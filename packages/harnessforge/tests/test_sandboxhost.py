"""Sandbox hosts (harnessforge.sandboxhost) with a fake provider: the wire, placement, relays, mirrors, failures."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

import fake_sandboxhost as fake  # noqa: E402
from platoon.sandbox import cleanup  # noqa: E402
from platoon.sandbox.base import Mount, SandboxSpec  # noqa: E402

from harnessforge.sandboxhost import fleet, provider_settings, wire  # noqa: E402
from harnessforge.sandboxhost.daemon import SandboxHost  # noqa: E402
from harnessforge.sandboxhost.provider import (REMOTE_PROVIDER, HostUnreachable, RemotePodProvider,  # noqa: E402
                                               close_clients, rpc)

FIXTURES = Path(__file__).parent / "fixtures"


def make_fleet(tmp_path: Path) -> fleet.FleetDir:
    fd = fleet.FleetDir(tmp_path / "fleet")
    fd.dir.mkdir(parents=True)
    wire.new_token(fd.token_file)
    fleet.write_json(fd.fleet_file, {"fleet_id": "t", "spec": {"fleet_id": "t"}, "jobs": {}})
    return fd


async def start_host(fd: fleet.FleetDir, tmp_path: Path, index: int, slots: int = 4, **kw) -> SandboxHost:
    host = SandboxHost(token=wire.read_token(fd.token_file), slots=slots, scratch=tmp_path / f"h{index}",
                       name=f"t/{index}", provider_factory=fake.FakeProvider, status_file=fd.host_file(index),
                       provider_options={"base": str(tmp_path / f"h{index}-boxes")},
                       info={"index": index, "job_id": f"job{index}"}, **kw)
    await host.start_server("127.0.0.1", 0)
    host._write_status("ready")
    return host


def provider(fd: fleet.FleetDir, tmp_path: Path, **kw) -> RemotePodProvider:
    kw.setdefault("shared_roots", [str(tmp_path / "shared")])
    return RemotePodProvider(hosts_spec=str(fd.dir), fakeroot=True, allow_network_none=True, **kw)


def spec(**kw) -> SandboxSpec:
    return SandboxSpec(image=kw.pop("image", "docker://ubuntu:22.04"), network="host", **kw)


def run(coro):
    async def wrapped():
        try:
            return await coro
        finally:
            await close_clients()
    return asyncio.run(wrapped())


# ---------------------------------------------------------------------------------------------------------- wire
def test_token_file_is_private(tmp_path):
    p = tmp_path / "token"
    wire.new_token(p)
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    assert len(wire.read_token(p)) == 64
    os.chmod(p, 0o644)
    with pytest.raises(PermissionError):
        wire.read_token(p)


def test_bad_token_is_refused(tmp_path):
    fd = make_fleet(tmp_path)

    async def go():
        host = await start_host(fd, tmp_path, 0)
        reply, _ = await rpc(host.address, wire.read_token(fd.token_file), {"op": "ping"})
        assert "pong" in reply
        with pytest.raises(Exception) as err:
            await rpc(host.address, b"x" * 64, {"op": "ping"})
        assert not isinstance(err.value, HostUnreachable)
        assert host.stats["auth_failed"] == 1
    run(go())


def test_error_types_survive_the_wire():
    err = wire.rebuild_error(wire.error_header(FileNotFoundError("nope"))["error"])
    assert isinstance(err, FileNotFoundError)
    from hpc_sandbox import SandboxLostError
    err = wire.rebuild_error(wire.error_header(SandboxLostError("dead"))["error"])
    assert isinstance(err, SandboxLostError)
    assert isinstance(wire.rebuild_error({"type": "HostFull", "module": "x", "message": "m"}), wire.HostFull)
    assert isinstance(wire.rebuild_error({"type": "Weird", "module": "x", "message": "m"}), wire.RemoteError)


# ---------------------------------------------------------------------------------------------------------- e2e
def test_remote_sandbox_round_trip(tmp_path):
    """exec (with stdin, env, timeout), upload/download of files and dirs, routes, kill_processes, collect, stop."""
    fd = make_fleet(tmp_path)

    async def go():
        host = await start_host(fd, tmp_path, 0)
        collect_dir = tmp_path / "collected"
        p = provider(fd, tmp_path, keep_agent_processes=False, collect_dir=str(collect_dir))
        sb = await p.start(spec(labels={"platoon.rollout_id": "r1"}))
        assert sb.routes == {}
        r = await sb.exec("echo hi; echo err >&2; exit 3")
        assert (r.exit_code, r.stdout, r.stderr) == (3, "hi\n", "err\n")
        r = await sb.exec(["cat"], stdin=b"from stdin")
        assert r.stdout == "from stdin"
        r = await sb.exec("echo $FOO", env={"FOO": "bar"})
        assert r.stdout == "bar\n"
        r = await sb.exec("sleep 5", timeout_s=0.5)
        assert r.timed_out
        # upload a file and a directory, download them back
        f = tmp_path / "a.txt"
        f.write_text("A")
        d = tmp_path / "tree"
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "x.sh").write_text("#!/bin/sh\necho x\n")
        os.chmod(d / "sub" / "x.sh", 0o755)
        await sb.upload(f, "/work/a.txt")
        await sb.upload(d, "/work/tree")
        r = await sb.exec("cat work/a.txt; sh work/tree/sub/x.sh; test -x work/tree/sub/x.sh && echo exe")
        assert r.stdout == "Ax\nexe\n"
        await sb.download("/work/a.txt", tmp_path / "back" / "a.txt")
        assert (tmp_path / "back" / "a.txt").read_text() == "A"
        await sb.download("/work/tree", tmp_path / "back" / "tree")
        assert (tmp_path / "back" / "tree" / "sub" / "x.sh").read_text().startswith("#!/bin/sh")
        big = os.urandom(12 * 1024 * 1024)  # larger than the inline payload limit: spooled to a file
        (tmp_path / "big.bin").write_bytes(big)
        await sb.upload(tmp_path / "big.bin", "/big.bin")
        await sb.download("/big.bin", tmp_path / "big2.bin")
        assert (tmp_path / "big2.bin").read_bytes() == big
        with pytest.raises(FileNotFoundError):
            await sb.download("/missing", tmp_path / "m")
        # kill_processes runs on the host (the provider does not keep the agents' processes)
        sb.keep_processes(["1:2"])
        await sb.kill_processes()
        inner = host.entries[sb.id].sandbox
        assert inner.killed == 1 and inner.kept == frozenset({"1:2"})
        assert host.status()["active"] == 1
        await sb.stop()
        await sb.stop()  # idempotent
        assert host.status()["active"] == 0
        assert inner.stopped
        meta = json.loads((collect_dir / "r1" / "sandbox.json").read_text())
        assert meta["labels"]["platoon.rollout_id"] == "r1" and meta["host"].startswith("host 0")
    run(go())


def test_keep_agent_processes_skips_kill(tmp_path):
    fd = make_fleet(tmp_path)

    async def go():
        host = await start_host(fd, tmp_path, 0)
        p = provider(fd, tmp_path, keep_agent_processes=True)
        sb = await p.start(spec())
        await sb.kill_processes()
        assert host.entries[sb.id].sandbox.killed == 0
        await sb.stop()
    run(go())


def test_gateway_upstream_is_relayed_back_to_the_client(tmp_path):
    """The host rewrites the unix upstream to a socket of its own whose connections reach the client's socket."""
    fd = make_fleet(tmp_path)

    async def go():
        sock_dir = Path("/tmp") / f"hfsbh-test-{os.getpid()}"
        sock_dir.mkdir(exist_ok=True)
        sock = sock_dir / "gw.sock"
        seen = []

        async def echo(reader, writer):
            data = await reader.read(100)
            seen.append(data)
            writer.write(b"pong:" + data)
            await writer.drain()
            writer.close()

        server = await asyncio.start_unix_server(echo, path=str(sock))
        try:
            await start_host(fd, tmp_path, 0)
            p = provider(fd, tmp_path)
            sb = await p.start(spec(upstreams={"gateway": f"unix://{sock}"}))
            assert sb.routes == {"gateway": "http://127.0.0.1:18080"}
            started = fake.STARTED[-1]
            host_sock = started.upstreams["gateway"][len("unix://"):]
            assert host_sock != str(sock) and host_sock.endswith("/gateway/gw.sock")
            for i in range(3):  # several connections through the same relay
                r, w = await asyncio.open_unix_connection(host_sock)
                w.write(f"ping{i}".encode())
                await w.drain()
                assert await asyncio.wait_for(r.read(100), 10) == f"pong:ping{i}".encode()
                w.close()
            # another key, or none, reaches nothing
            await sb.stop()
            assert not Path(host_sock).exists()
        finally:
            server.close()
    run(go())


def test_node_local_mounts_are_mirrored_once(tmp_path):
    fd = make_fleet(tmp_path)

    async def go():
        await start_host(fd, tmp_path, 0, slots=8)
        local = tmp_path / "local" / "toolchain"
        (local / "bin").mkdir(parents=True)
        (local / "bin" / "tool").write_text("t")
        shared = tmp_path / "shared" / "cand"
        shared.mkdir(parents=True)
        p = provider(fd, tmp_path)
        mounts = [Mount(source=str(local), target="/opt/tc"), Mount(source=str(shared), target="/opt/cand")]
        boxes = await asyncio.gather(*(p.start(spec(mounts=list(mounts))) for _ in range(4)))
        srcs = {s.mounts[0].source for s in fake.STARTED[-4:]}
        assert len(srcs) == 1
        (mirror,) = srcs
        assert mirror != str(local) and (Path(mirror) / "bin" / "tool").read_text() == "t"
        assert {s.mounts[1].source for s in fake.STARTED[-4:]} == {str(shared)}  # shared: as is
        with pytest.raises(ValueError):
            await p.start(spec(mounts=[Mount(source=str(local), target="/w", read_only=False)]))
        for b in boxes:
            await b.stop()
    run(go())


# ---------------------------------------------------------------------------------------------------------- placement
def test_placement_spreads_and_waits_when_full(tmp_path):
    fd = make_fleet(tmp_path)

    async def go():
        h0 = await start_host(fd, tmp_path, 0, slots=2)
        h1 = await start_host(fd, tmp_path, 1, slots=2)
        p = provider(fd, tmp_path, place_timeout_s=1.0)
        boxes = await asyncio.gather(*(p.start(spec()) for _ in range(4)))
        assert len(h0.entries) == 2 and len(h1.entries) == 2
        t0 = time.monotonic()
        with pytest.raises(wire.HostFull):
            await p.start(spec())
        assert time.monotonic() - t0 >= 1.0
        # a slot frees: the waiting start takes it
        p2 = provider(fd, tmp_path, place_timeout_s=30.0)
        waiting = asyncio.ensure_future(p2.start(spec()))
        await asyncio.sleep(0.5)
        assert not waiting.done()
        await boxes[0].stop()
        extra = await asyncio.wait_for(waiting, 20)
        for b in [*boxes[1:], extra]:
            await b.stop()
        assert not h0.entries and not h1.entries
    run(go())


def test_start_error_is_raised_not_retried(tmp_path):
    fd = make_fleet(tmp_path)

    async def go():
        h0 = await start_host(fd, tmp_path, 0)
        p = provider(fd, tmp_path)
        with pytest.raises(RuntimeError, match="does not start"):
            await p.start(spec(image="fail"))
        assert h0.stats["start_failed"] == 1 and not h0.entries and h0.starting == 0
        assert not cleanup.live_sandboxes()
    run(go())


def test_local_slots_first_then_remote(tmp_path, monkeypatch):
    fd = make_fleet(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "nodelocal"))
    (tmp_path / "nodelocal").mkdir()
    started_locally = []

    class LocalBox:
        id = "local"
        routes: dict = {}

        async def stop(self):
            started_locally.remove(self)

    async def fake_local_start(self, s):
        box = LocalBox()
        started_locally.append(box)
        return box

    from harnessforge.evals import pods
    monkeypatch.setattr(pods.PodSingularityProvider, "start", fake_local_start)

    async def go():
        h0 = await start_host(fd, tmp_path, 0)
        p = provider(fd, tmp_path, local_slots=2)
        a, b, c = [await p.start(spec()) for _ in range(3)]
        assert len(started_locally) == 2 and len(h0.entries) == 1
        await a.stop()
        d = await p.start(spec())  # the freed local slot is taken again
        assert len(started_locally) == 2 and len(h0.entries) == 1
        for x in (b, c, d):
            await x.stop()
        assert not started_locally and not h0.entries
    run(go())


# ---------------------------------------------------------------------------------------------------------- failures
def _spawn_host(fd: fleet.FleetDir, index: int, slots: int, lease: float = 900.0) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, str(FIXTURES / "fake_sandboxhost.py"), str(fd.dir), str(index),
                             str(slots), str(lease)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    deadline = time.time() + 30
    while time.time() < deadline:
        h = fd.hosts().get(index)
        if h and h.get("status") == "ready" and h.get("job_id") == f"pid{proc.pid}":
            return proc
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("fake host did not come up")


async def _start_on(p, index):
    while True:
        b = await p.start(spec())
        if b._host.index == index:
            return b
        await b.stop()


def test_host_killed_mid_exec_fails_fast_and_is_avoided(tmp_path):
    from hpc_sandbox import SandboxLostError
    fd = make_fleet(tmp_path)
    h0 = _spawn_host(fd, 0, 8)
    h1 = _spawn_host(fd, 1, 8)
    try:
        async def go():
            p = provider(fd, tmp_path)
            boxes = await asyncio.gather(*(p.start(spec()) for _ in range(6)))
            on0 = [b for b in boxes if b._host.index == 0]
            if not on0:  # placement is random: make sure host 0 has one
                boxes.append(await _start_on(p, 0))
                on0 = boxes[-1:]
            long = asyncio.ensure_future(on0[0].exec("sleep 60"))
            await asyncio.sleep(1.0)
            t0 = time.monotonic()
            h0.send_signal(signal.SIGKILL)  # the job dies under a running command
            with pytest.raises(SandboxLostError):
                await asyncio.wait_for(long, 30)
            assert time.monotonic() - t0 < 10
            with pytest.raises(SandboxLostError):  # later calls fail at once
                await on0[0].exec("true")
            for b in on0:
                await asyncio.wait_for(b.stop(), 30)  # forgets it, never hangs
            fresh = [await p.start(spec()) for _ in range(2)]  # host 0 is avoided
            assert all(b._host.index == 1 for b in fresh)
            for b in [*boxes, *fresh]:
                await b.stop()
            assert not cleanup.live_sandboxes()
        run(go())
    finally:
        for proc in (h0, h1):
            proc.kill()
            proc.wait()


def test_cancelled_start_is_stopped_on_the_host(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_START_DELAY", "1.5")
    fd = make_fleet(tmp_path)
    h0 = _spawn_host(fd, 0, 4)
    try:
        async def go():
            p = provider(fd, tmp_path)
            task = asyncio.ensure_future(p.start(spec()))
            await asyncio.sleep(0.7)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(3.0)
            from harnessforge.sandboxhost.provider import host_rpc_sync  # noqa: F401
            reply, _ = await rpc((fd.hosts()[0]["ip"], fd.hosts()[0]["port"]), wire.read_token(fd.token_file),
                                 {"op": "status"})
            assert reply["active"] == 0 and reply["starting"] == 0 and reply["stats"]["started"] == 1
            assert reply["stats"]["stopped"] == 1
        run(go())
    finally:
        h0.kill()
        h0.wait()


def test_leases_expire_and_renewals_keep_them(tmp_path):
    fd = make_fleet(tmp_path)

    async def go():
        host = await start_host(fd, tmp_path, 0, lease_s=100)
        p = provider(fd, tmp_path)
        a = await p.start(spec())
        b = await p.start(spec())
        now = time.time()
        reply, _ = await rpc(host.address, wire.read_token(fd.token_file), {"op": "renew", "ids": [a.id, "zzz"]})
        assert reply["unknown"] == ["zzz"]
        host.entries[b.id].last_seen = now - 500
        assert host.expire_leases(now=now) == [b.id]
        await asyncio.sleep(0.2)
        assert set(host.entries) == {a.id}
        with pytest.raises(Exception):
            await b.exec("true")
        await a.stop()
        await b.stop()
    run(go())


def test_reap_stops_labelled_remote_sandboxes(tmp_path):
    fd = make_fleet(tmp_path)

    async def go():
        h0 = await start_host(fd, tmp_path, 0)
        h1 = await start_host(fd, tmp_path, 1)
        p = provider(fd, tmp_path)
        with cleanup.labelled({cleanup.RUN_LABEL: "/runs/x"}):
            boxes = [await p.start(spec()) for _ in range(3)]
        other = await p.start(spec(labels={"platoon.rollout_id": "keep"}))
        p2 = provider(fd, tmp_path)  # as platoon's supervisor builds it from the mirror record
        n = await p2.reap({cleanup.RUN_LABEL: "/runs/x"})
        assert n == 3
        assert len(h0.entries) + len(h1.entries) == 1
        for b in [*boxes, other]:
            await b.stop()
    run(go())


def test_shutdown_stops_everything(tmp_path):
    fd = make_fleet(tmp_path)

    async def go():
        host = await start_host(fd, tmp_path, 0)
        p = provider(fd, tmp_path)
        boxes = [await p.start(spec()) for _ in range(3)]
        await host.shutdown()
        assert not host.entries and fd.hosts()[0]["status"] == "stopped"
        for b in boxes:
            with pytest.raises(Exception):
                await b.exec("true")
            await b.stop()
    run(go())


# ---------------------------------------------------------------------------------------------------------- config
def test_provider_settings():
    assert provider_settings(None) == (None, {})
    assert provider_settings({"mode": "local"}) == (None, {})
    prov, opts = provider_settings({"mode": "remote", "hosts_spec": "/f.json", "local_slots": 160})
    assert prov == REMOTE_PROVIDER and opts == {"hosts_spec": "/f.json", "local_slots": 160}
    with pytest.raises(ValueError):
        provider_settings({"mode": "remote"})
    with pytest.raises(ValueError):
        provider_settings({"mode": "remote", "hosts_spec": "x", "slots": 3})


def test_fleet_spec_and_host_job(tmp_path):
    spec_file = tmp_path / "fleet.json"
    spec_file.write_text(json.dumps({"fleet_id": "f1", "state_root": str(tmp_path / "state"), "hosts": 2,
                                     "python": "/venv/bin/python", "harnessforge_src": "/src",
                                     "launcher": {"kind": "pbs", "queue": "Q", "account": "A",
                                                  "resource_type": "rt_HC", "walltime": "12:00:00"}}))
    assert fleet.state_dir_of(spec_file) == tmp_path / "state" / "f1"
    spec = fleet.load_spec(spec_file)
    job = fleet.host_job(spec, fleet.FleetDir(tmp_path / "state" / "f1"), 1)
    assert job.command[:3] == ["/venv/bin/python", "-m", "harnessforge.sandboxhost"]
    assert job.env["PYTHONPATH"] == "/src" and job.walltime == "12:00:00"
    assert job.options["resource_type"] == "rt_HC"

    class FakeLauncher:
        submitted = []

        def submit(self, s):
            self.submitted.append(s)
            return f"{len(self.submitted)}.pbs"

        def status(self, j):
            from harnessforge.contracts import JobState, JobStatus
            return JobStatus(JobState.RUNNING)

    launcher = FakeLauncher()
    cfg = fleet.start(spec, launcher=launcher, wait=False, log=lambda *_: None)
    assert cfg["jobs"] == {"0": "1.pbs", "1": "2.pbs"}
    assert oct((tmp_path / "state" / "f1" / "token").stat().st_mode & 0o777) == "0o600"
    fleet.start(spec, launcher=launcher, wait=False, log=lambda *_: None)  # running jobs are not resubmitted
    assert len(launcher.submitted) == 2
