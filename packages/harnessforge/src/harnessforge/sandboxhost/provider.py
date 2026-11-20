"""RemotePodProvider: platoon's pod provider, with sandboxes on sandbox hosts (other batch jobs) as well as here.

    provider: harnessforge.sandboxhost.provider:RemotePodProvider
    provider_options: {hosts_spec: <fleet spec or state dir>, local_slots: 0, ...PodSingularityProvider options}

A start takes a local slot when one is free (`local_slots`, a node-wide cap shared by every process of the job, held
for the sandbox's life) and starts the pod here as `PodSingularityProvider` does; otherwise it asks the live hosts
how many slots each has free and starts the pod on one drawn in proportion to its free slots (`daemon.SandboxHost`), trying the next on HostFull or
when a host cannot be reached, and waiting (up to `place_timeout_s`) while every slot is taken. A remote sandbox is a
`RemoteSandbox`: exec, upload, download, kill_processes, ensure_routes and stop are calls to its host, and its routes
are the pod's own (http://127.0.0.1:18080 for the gateway).

What crosses to the host:

- **The spec**, with this process's cleanup labels, after the prepared-image map is applied here.
- **Unix-socket upstreams** (platoon's per-rollout model gateway socket): this process runs one TCP relay
  (`UpstreamRelay`, token-authenticated) per event loop; each upstream gets a key, and the host's end of the relay
  connects to it with that key. Nothing else is reachable through the relay.
- **Node-local paths.** Mount sources, a .sif image and the owner manifest that are not under `shared_roots`
  (default /groups, /home and $HOME; platoon's staged toolchain is on the job's local disk) are copied to each host
  once (keyed by path, inode, size and mtime: platoon names a staged toolchain by its tree hash and never changes
  it in place) and the spec points at the host's copy.
- **Sandbox options**: fakeroot, allow_network_none, pod_users, ... (`daemon.FORWARDED_OPTIONS`), and image_cache
  when it is shared; the host keeps its own scratch, start cap and defaults.

Failures. A host that cannot be reached when a start asks it is skipped (and avoided for `DEAD_FOR_S`). A host that
dies under a sandbox (its job ended, its node went away: a refused or broken connection, or no heartbeat for
`HEARTBEAT_TIMEOUT_S` during a long op) makes every call on that sandbox raise hpc_sandbox.SandboxLostError, which
platoon reports as an infrastructure failure of the rollout (untrainable; env_failed in evals), never a hang; its
stop then just forgets it. A keepalive renews each host's lease on this process's sandboxes every `RENEW_S`, and
learns of sandboxes a host no longer has (marked lost). `reap(labels)` reaps local sandboxes as the pod provider
does and asks every live host to stop the sandboxes with those labels; platoon's supervisor calls it for a rollout
worker that died, and a host stops on its own the sandboxes whose lease expired.

`collect` (the harbor-export hook) runs on remote sandboxes too, through exec. Stacks (multi-container tasks) and
restores run locally; remote sandboxes take no snapshots.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import random
import secrets
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

from platoon.sandbox import cleanup
from platoon.sandbox.base import DownloadReport, DroppedLink, ExecResult

from harnessforge.evals.pods import PodSingularityProvider, collect, mapped_image
from harnessforge.evals.start_gate import StartGate
from harnessforge.sandboxhost import fleet as fleetmod
from harnessforge.sandboxhost import net, wire
from harnessforge.sandboxhost.daemon import FORWARDED_OPTIONS, HEARTBEAT_S

logger = logging.getLogger(__name__)

from harnessforge.sandboxhost import REMOTE_PROVIDER  # noqa: E402,F401
CONNECT_TIMEOUT_S = 10.0
HEARTBEAT_TIMEOUT_S = 4 * HEARTBEAT_S
STATUS_TIMEOUT_S = 5.0
DEAD_FOR_S = 60.0
RENEW_S = 60.0
PLACE_POLL_S = (2.0, 5.0)
HOSTS_CACHE_S = 3.0


class HostUnreachable(ConnectionError):
    """The host could not be reached before the request was sent: nothing happened there."""


def default_shared_roots() -> list[str]:
    if raw := os.environ.get("HF_SANDBOXHOST_SHARED_ROOTS"):
        return [p for p in raw.split(os.pathsep) if p]
    roots = ["/groups", "/home", os.path.realpath(os.path.expanduser("~"))]
    return list(dict.fromkeys(roots))


def is_shared(path: str, roots: tuple[str, ...]) -> bool:
    real = os.path.realpath(path)
    return any(real == r.rstrip("/") or real.startswith(r.rstrip("/") + "/") for r in roots)


def _spool_dir() -> Path:
    d = Path(fleetmod.node_scratch()) / f"hfsbh-client-{os.getuid()}"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


# ---------------------------------------------------------------------------------------------------------- rpc
async def rpc(addr: tuple[str, int], token: bytes, header: dict[str, Any], payload: bytes | Path | None = None, *,
              ack: bool = False, spool_dir: Path | None = None,
              connect_timeout: float = CONNECT_TIMEOUT_S,
              heartbeat_timeout: float = HEARTBEAT_TIMEOUT_S) -> tuple[dict[str, Any], bytes | Path | None]:
    """One request on its own connection. HostUnreachable before the request went out; SandboxLostError when the
    connection broke or went silent after; the host's own error, rebuilt, when it raised one."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(addr[0], int(addr[1]), limit=2 ** 20),
                                                connect_timeout)
    except (OSError, asyncio.TimeoutError) as exc:
        raise HostUnreachable(f"sandbox host {addr[0]}:{addr[1]} unreachable: {type(exc).__name__}: {exc}") from exc
    net.set_keepalive(writer)
    sent = False
    remote_error: dict[str, Any] | None = None
    try:
        await wire.client_handshake(reader, writer, token)
        sent = True
        await wire.write_frame(writer, header, payload)
        while True:
            reply, data = await wire.read_frame(reader, spool_dir=spool_dir, timeout=heartbeat_timeout)
            if "hb" in reply:
                continue
            if "error" in reply:
                remote_error = reply["error"]
                break
            if ack:
                await wire.write_frame(writer, {"op": "ack"})
            return reply, data
    except (wire.WireError, ConnectionError, OSError, asyncio.TimeoutError) as exc:
        if isinstance(exc, wire.AuthError) or (isinstance(exc, wire.WireError) and "challenge" in str(exc)):
            raise
        if not sent:
            raise HostUnreachable(f"sandbox host {addr[0]}:{addr[1]}: {type(exc).__name__}: {exc}") from exc
        lost = wire.lost_error(f"sandbox host {addr[0]}:{addr[1]} lost during {header.get('op')}: "
                               f"{type(exc).__name__}: {exc}")
        lost.hf_connection_lost = True  # type: ignore[attr-defined]
        raise lost from exc
    finally:
        await net.close_writer(writer)
    raise wire.rebuild_error(remote_error)


def host_rpc_sync(fd: fleetmod.FleetDir, host: dict[str, Any], header: dict[str, Any], timeout: float = 10.0):
    token = wire.read_token(fd.token_file)
    reply, _ = asyncio.run(asyncio.wait_for(rpc((host["ip"], int(host["port"])), token, header), timeout))
    return reply


# ---------------------------------------------------------------------------------------------------------- relay
class UpstreamRelay:
    """A token-guarded TCP listener that connects each authenticated connection to the unix socket its key names."""

    def __init__(self, token: bytes, advertise: str | None = None) -> None:
        self.token = token
        self.advertise = advertise
        self.keys: dict[str, str] = {}
        self.server: asyncio.AbstractServer | None = None
        self.addr: tuple[str, int] | None = None
        self.connections = 0

    async def start(self, host: str = "0.0.0.0") -> tuple[str, int]:
        self.server = await asyncio.start_server(self._on_conn, host, 0, limit=2 ** 20, backlog=1024)
        port = self.server.sockets[0].getsockname()[1]
        self.addr = (net.advertise_ip(self.advertise) if host in ("0.0.0.0", "") else host, port)
        return self.addr

    def register(self, socket_path: str) -> str:
        key = secrets.token_hex(16)
        self.keys[key] = socket_path
        return key

    def unregister(self, key: str) -> None:
        self.keys.pop(key, None)

    async def _on_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        net.set_keepalive(writer)
        up_w = None
        try:
            auth = await wire.server_handshake(reader, writer, self.token)
            path = self.keys.get(str(auth.get("key")))
            if path is None:
                return
            self.connections += 1
            up_r, up_w = await asyncio.open_unix_connection(path, limit=2 ** 20)
            await net.splice(reader, writer, up_r, up_w)
        except (wire.WireError, ConnectionError, OSError, asyncio.TimeoutError):
            pass
        finally:
            await net.close_writer(writer)
            await net.close_writer(up_w)


# ---------------------------------------------------------------------------------------------------------- client
@dataclasses.dataclass(frozen=True)
class Host:
    index: int
    addr: tuple[str, int]
    job: str

    @property
    def label(self) -> str:
        return f"host {self.index} ({self.job})"


class _Client:
    """What every RemotePodProvider of this process shares for one fleet: the token, the host list, the relay and
    keepalive of each event loop, the mirrors made, the live sandboxes."""

    def __init__(self, state_dir: Path, advertise: str | None) -> None:
        self.fd = fleetmod.FleetDir(state_dir)
        self.token = wire.read_token(self.fd.token_file)
        self.advertise = advertise
        self.name = f"{socket.gethostname()}:{os.getpid()}"
        self.dead: dict[tuple[str, int], float] = {}
        self._hosts: tuple[float, list[Host]] = (0.0, [])
        self.relays: dict[int, UpstreamRelay] = {}
        self._relay_locks: dict[int, asyncio.Lock] = {}
        self.keepalives: dict[int, asyncio.Task] = {}
        self.mirrors: dict[tuple[tuple[str, int], str], str] = {}
        self._mirror_locks: dict[tuple[int, tuple[str, int], str], asyncio.Lock] = {}
        self.live: dict[str, "RemoteSandbox"] = {}
        self.spool = _spool_dir()

    # hosts
    def hosts(self) -> list[Host]:
        now = time.monotonic()
        if now - self._hosts[0] > HOSTS_CACHE_S:
            live = self.fd.live_hosts()
            self._hosts = (now, [Host(i, (h["ip"], int(h["port"])), str(h.get("job_id", "?")))
                                 for i, h in live.items()])
        return [h for h in self._hosts[1] if self.dead.get(h.addr, 0) < time.monotonic()]

    def mark_dead(self, host: Host, why: str) -> None:
        if self.dead.get(host.addr, 0) < time.monotonic():
            logger.warning("sandbox %s: %s; avoiding it for %.0fs", host.label, why, DEAD_FOR_S)
        self.dead[host.addr] = time.monotonic() + DEAD_FOR_S

    async def candidates(self) -> list[tuple[Host, int]]:
        """Live hosts with free slots, in a random order weighted by their free slots."""
        hosts = self.hosts()

        async def free(h: Host) -> tuple[Host, int]:
            try:
                reply, _ = await rpc(h.addr, self.token, {"op": "status"}, connect_timeout=STATUS_TIMEOUT_S,
                                     heartbeat_timeout=STATUS_TIMEOUT_S)
                return h, int(reply.get("free", 0))
            except HostUnreachable as exc:
                self.mark_dead(h, str(exc))
            except Exception as exc:  # noqa: BLE001
                logger.warning("status of sandbox %s failed: %r", h.label, exc)
            return h, -1

        got = [g for g in await asyncio.gather(*(free(h) for h in hosts)) if g[1] > 0]
        # an order drawn with probability proportional to free slots (Efraimidis-Spirakis), not strictly the
        # freest first: a burst of starts that all read the same counts spreads over the hosts
        return sorted(got, key=lambda g: random.random() ** (1.0 / g[1]), reverse=True)

    # relay and keepalive of the running loop
    async def relay(self) -> UpstreamRelay:
        loop = asyncio.get_running_loop()
        key = id(loop)
        lock = self._relay_locks.setdefault(key, asyncio.Lock())
        async with lock:
            relay = self.relays.get(key)
            if relay is None:
                relay = UpstreamRelay(self.token, self.advertise)
                await relay.start()
                self.relays[key] = relay
                logger.info("upstream relay of %s listening on %s:%d", self.name, *relay.addr)
            if key not in self.keepalives or self.keepalives[key].done():
                self.keepalives[key] = asyncio.ensure_future(self._keepalive())
        return relay

    async def _keepalive(self) -> None:
        loop_id = id(asyncio.get_running_loop())
        while True:
            await asyncio.sleep(RENEW_S)
            by_host: dict[tuple[str, int], list[RemoteSandbox]] = {}
            for sb in list(self.live.values()):
                if sb._loop_id == loop_id and not sb._lost:
                    by_host.setdefault(sb._host.addr, []).append(sb)
            for addr, boxes in by_host.items():
                try:
                    reply, _ = await rpc(addr, self.token, {"op": "renew", "ids": [b._sid for b in boxes]},
                                         connect_timeout=STATUS_TIMEOUT_S, heartbeat_timeout=30)
                except Exception as exc:  # noqa: BLE001 - its sandboxes' own calls will tell
                    logger.warning("renewing %d sandbox(es) on %s:%d failed: %r", len(boxes), *addr, exc)
                    continue
                for sid in reply.get("unknown", []):
                    for b in boxes:
                        if b._sid == sid:
                            b._mark_lost("its host no longer has it")

    # mirrors
    async def mirror(self, host: Host, source: str) -> str:
        """The host's copy of the local file or directory `source`."""
        st = os.stat(source)
        kind = "dir" if os.path.isdir(source) else "file"
        ident = f"{socket.gethostname()}|{os.path.realpath(source)}|{st.st_ino}|{st.st_size}|{st.st_mtime_ns}|{kind}"
        key = hashlib.sha256(ident.encode()).hexdigest()[:40]
        memo = (host.addr, key)
        if memo in self.mirrors:
            return self.mirrors[memo]
        lock = self._mirror_locks.setdefault((id(asyncio.get_running_loop()), host.addr, key), asyncio.Lock())
        async with lock:
            if memo in self.mirrors:
                return self.mirrors[memo]
            t0 = time.monotonic()
            while True:
                reply, _ = await rpc(host.addr, self.token, {"op": "mirror_begin", "key": key, "kind": kind})
                if reply["state"] == "present":
                    path = reply["path"]
                    break
                if reply["state"] == "upload":
                    tar, packed_kind = await asyncio.to_thread(wire.pack, source, self.spool)
                    try:
                        size = tar.stat().st_size
                        reply, _ = await rpc(host.addr, self.token, {"op": "mirror_put", "key": key,
                                                                     "kind": packed_kind, "source": source}, tar)
                    finally:
                        tar.unlink(missing_ok=True)
                    path = reply["path"]
                    logger.info("mirrored %s (%.1f MB) to sandbox %s in %.1fs", source, size / 1e6, host.label,
                                time.monotonic() - t0)
                    break
                await asyncio.sleep(2.0)  # another process is uploading it
            self.mirrors[memo] = path
            return path


_CLIENTS: dict[str, _Client] = {}


def client_for(state_dir: Path, advertise: str | None) -> _Client:
    key = str(state_dir)
    if key not in _CLIENTS:
        _CLIENTS[key] = _Client(state_dir, advertise)
    return _CLIENTS[key]


# ---------------------------------------------------------------------------------------------------------- sandbox
class RemoteSandbox:
    """A sandbox on a sandbox host (platoon's Sandbox contract, over `rpc`)."""

    def __init__(self, provider: "RemotePodProvider", client: _Client, host: Host, reply: dict[str, Any],
                 relay: UpstreamRelay | None, keys: list[str], token: int | None) -> None:
        self._provider = provider
        self._client = client
        self._host = host
        self._sid = str(reply["id"])
        self.id = self._sid
        self._routes = dict(reply.get("routes") or {})
        self._relay = relay
        self._keys = keys
        self._token = token
        self._kept: frozenset[str] = frozenset()
        self._lost: str | None = None
        self._stopped = False
        self._stopping: asyncio.Future | None = None
        self._loop_id = id(asyncio.get_running_loop())
        self.host = host.label

    def __repr__(self) -> str:
        return f"<RemoteSandbox {self._sid} on {self._host.label}>"

    @property
    def routes(self) -> dict[str, str]:
        return dict(self._routes)

    def _mark_lost(self, why: str) -> None:
        if self._lost is None:
            self._lost = why
            logger.warning("sandbox %s on %s lost: %s", self._sid, self._host.label, why)

    async def _call(self, header: dict[str, Any], payload: bytes | Path | None = None,
                    ) -> tuple[dict[str, Any], bytes | Path | None]:
        if self._lost is not None:
            raise wire.lost_error(f"sandbox {self._sid} on {self._host.label} is lost: {self._lost}")
        try:
            return await rpc(self._host.addr, self._client.token, {**header, "id": self._sid}, payload,
                             spool_dir=self._client.spool)
        except HostUnreachable as exc:
            self._client.mark_dead(self._host, str(exc))
            self._mark_lost(str(exc))
            raise wire.lost_error(f"sandbox {self._sid}: {exc}") from exc
        except Exception as exc:
            if getattr(exc, "hf_connection_lost", False):
                self._client.mark_dead(self._host, str(exc))
                self._mark_lost(str(exc))
            raise

    async def exec(self, cmd: str | list[str], *, timeout_s: float | None = None, env: dict[str, str] | None = None,
                   cwd: str | None = None, stdin: bytes | None = None) -> ExecResult:
        header = {"op": "exec", "cmd": cmd, "timeout_s": timeout_s, "env": env, "cwd": cwd,
                  "stdin": stdin is not None}
        reply, _ = await self._call(header, stdin if stdin is not None else None)
        return ExecResult(exit_code=int(reply["exit_code"]), stdout=reply["stdout"], stderr=reply["stderr"],
                          timed_out=bool(reply["timed_out"]))

    async def upload(self, src: str | Path, dst: str) -> None:
        if not os.path.exists(src):
            raise FileNotFoundError(f"no such file or directory: {src}")
        tar, kind = await asyncio.to_thread(wire.pack, src, self._client.spool)
        try:
            await self._call({"op": "upload", "dst": dst, "kind": kind}, tar)
        finally:
            tar.unlink(missing_ok=True)

    async def download(self, src: str, dst: str | Path) -> DownloadReport | None:
        reply, data = await self._call({"op": "download", "src": src})
        tar = wire.payload_file(data, self._client.spool)
        work = Path(tempfile.mkdtemp(prefix="down-", dir=self._client.spool))
        try:
            extracted = await asyncio.to_thread(wire.unpack, tar, work, reply["kind"], trusted=False)
            await asyncio.to_thread(wire.place, extracted, dst, reply["kind"])
        finally:
            tar.unlink(missing_ok=True)
            await asyncio.to_thread(shutil.rmtree, work, True)
        rep = reply.get("report")
        if rep is None:
            return None
        return DownloadReport(dropped_links=[DroppedLink(**d) for d in rep.get("dropped_links", [])],
                              dropped_total=int(rep.get("dropped_total", 0)))

    async def record_own_processes(self) -> None:
        return None  # the host recorded them when the sandbox started

    def keep_processes(self, entries) -> None:
        self._kept = self._kept | frozenset(entries)

    async def kill_processes(self) -> None:
        if getattr(self._provider, "keep_agent_processes", False):
            return
        await self._call({"op": "kill_processes", "keep": sorted(self._kept)})

    async def ensure_routes(self) -> dict[str, str]:
        reply, _ = await self._call({"op": "ensure_routes"})
        self._routes = dict(reply.get("routes") or {})
        return self.routes

    async def snapshot(self):
        raise NotImplementedError("remote sandboxes take no snapshots (fork shared or fresh)")

    async def stop(self) -> None:
        if self._stopped:
            return
        if self._stopping is None:
            self._stopping = asyncio.ensure_future(self._stop())
        await asyncio.shield(self._stopping)

    async def _stop(self) -> None:
        try:
            if self._lost is None:
                await collect(self)
                try:
                    await self._call({"op": "stop"})
                except Exception as exc:  # noqa: BLE001 - a lost host took its sandboxes with it
                    logger.warning("stopping sandbox %s on %s failed: %r", self._sid, self._host.label, exc)
        finally:
            for key in self._keys:
                if self._relay is not None:
                    self._relay.unregister(key)
            cleanup.unregister(self._token)
            self._token = None
            self._client.live.pop(self._sid, None)
            self._stopped = True


# ---------------------------------------------------------------------------------------------------------- provider
class RemotePodProvider(PodSingularityProvider):
    def __init__(self, *, hosts_spec: str | None = None, local_slots: int = 0,
                 shared_roots: list[str] | None = None, advertise: str | None = None,
                 place_timeout_s: float = 3600.0, **options: Any) -> None:
        super().__init__(**options)
        self.hosts_spec = str(hosts_spec) if hosts_spec else None
        self.state_dir = fleetmod.state_dir_of(self.hosts_spec) if self.hosts_spec else None
        self.local_slots = max(0, int(local_slots))
        self.shared_roots = tuple(shared_roots or default_shared_roots())
        self.advertise = advertise
        self.place_timeout_s = float(place_timeout_s)
        self.local_gate = StartGate(self.local_slots, slot_dir=Path(fleetmod.node_scratch())
                                    / f"hf-local-sandbox-slots-{os.getuid()}")
        if self.state_dir is None and self.local_slots == 0:
            raise ValueError("RemotePodProvider needs hosts_spec (a sandbox host fleet) or local_slots > 0")
        self.options = {**self.options, "hosts_spec": self.hosts_spec, "local_slots": self.local_slots,
                        "shared_roots": list(self.shared_roots), "advertise": advertise,
                        "place_timeout_s": self.place_timeout_s}
        forwarded = {k: v for k, v in options.items() if k in FORWARDED_OPTIONS}
        if "image_cache" in forwarded and not is_shared(str(forwarded["image_cache"]), self.shared_roots):
            forwarded.pop("image_cache")  # node-local here: the host uses its own
        self.forwarded_options = forwarded

    def _client(self) -> _Client:
        assert self.state_dir is not None
        return client_for(self.state_dir, self.advertise)

    async def start(self, spec: Any) -> Any:
        deadline = time.monotonic() + self.place_timeout_s
        while True:
            if (local := await self._start_local(spec)) is not None:
                return local
            if self.state_dir is not None:
                remote = await self._start_remote(spec)
                if remote is not None:
                    return remote
            if time.monotonic() > deadline:
                raise wire.HostFull(f"no free sandbox slot (local {self.local_slots}, hosts in {self.state_dir}) "
                                    f"within {self.place_timeout_s:.0f}s")
            await asyncio.sleep(random.uniform(*PLACE_POLL_S))

    async def _start_local(self, spec: Any) -> Any | None:
        if self.local_slots <= 0:
            return None
        slot = self.local_gate._try_take()
        if slot is None:
            return None
        try:
            sandbox = await super().start(spec)
        except BaseException:
            StartGate.release(slot)
            raise
        inner_stop = sandbox.stop
        released = False

        async def stop() -> None:
            nonlocal released
            try:
                await inner_stop()
            finally:
                if not released:
                    released = True
                    StartGate.release(slot)

        sandbox.stop = stop  # the slot is held for the sandbox's life
        return sandbox

    async def _start_remote(self, spec: Any) -> RemoteSandbox | None:
        """The sandbox on the freest host; None when every host is full or unreachable."""
        client = self._client()
        requested = time.time()
        if mapped := mapped_image(self.image_map, spec.image or ""):
            spec = dataclasses.replace(spec, image=mapped)
        labels = {**spec.labels, **cleanup.labels()}
        for host, _free in await client.candidates():
            token = cleanup.register(provider=cleanup.provider_ref(self), options=self.options, labels=labels)
            relay: UpstreamRelay | None = None
            keys: list[str] = []
            try:
                raw, relay, keys = await self._remote_spec(client, host, spec, labels)
                reply, _ = await rpc(host.addr, client.token,
                                     {"op": "start", "spec": raw, "relays": raw.pop("_relays"),
                                      "options": self.forwarded_options, "client": client.name},
                                     ack=True, spool_dir=client.spool)
            except BaseException as exc:
                cleanup.unregister(token)
                for key in keys:
                    relay.unregister(key)  # type: ignore[union-attr]
                if isinstance(exc, wire.HostFull):
                    continue
                if isinstance(exc, HostUnreachable) or getattr(exc, "hf_connection_lost", False):
                    # nothing started there, or the host stops what it started once no ack comes
                    client.mark_dead(host, str(exc))
                    continue
                raise
            sandbox = RemoteSandbox(self, client, host, reply, relay, keys, token)
            sandbox._hf_labels = dict(spec.labels)
            sandbox._hf_times = {"start_requested": requested, "started": time.time(), "host": host.label}
            client.live[sandbox._sid] = sandbox
            cleanup.attach(token, sandbox)
            return sandbox
        return None

    async def _remote_spec(self, client: _Client, host: Host, spec: Any, labels: dict[str, str],
                           ) -> tuple[dict[str, Any], UpstreamRelay | None, list[str]]:
        raw = dataclasses.asdict(spec)
        raw["labels"] = labels
        for m in raw["mounts"]:
            if not is_shared(m["source"], self.shared_roots):
                if not m.get("read_only", True):
                    raise ValueError(f"a writable mount of the node-local {m['source']} cannot reach a sandbox host")
                m["source"] = await client.mirror(host, m["source"])
        image = raw.get("image") or ""
        if image.startswith("/") and "://" not in image and not is_shared(image, self.shared_roots):
            raw["image"] = await client.mirror(host, image)
        if raw.get("file_owners") and not is_shared(raw["file_owners"], self.shared_roots):
            raw["file_owners"] = await client.mirror(host, raw["file_owners"])
        relay: UpstreamRelay | None = None
        keys: list[str] = []
        relays: dict[str, dict[str, Any]] = {}
        for name, url in sorted((raw.get("upstreams") or {}).items()):
            if str(url).startswith("unix://"):
                relay = relay or await client.relay()
                key = relay.register(str(url)[len("unix://"):])
                keys.append(key)
                relays[name] = {"addr": relay.addr[0], "port": relay.addr[1], "key": key}  # type: ignore[index]
        raw["_relays"] = relays
        return raw, relay, keys

    async def reap(self, labels: Any) -> int:
        n = await super().reap(labels)
        if self.state_dir is None:
            return n
        client = self._client()

        async def one(h: Host) -> int:
            try:
                reply, _ = await rpc(h.addr, client.token, {"op": "reap", "labels": dict(labels)},
                                     heartbeat_timeout=120)
                return int(reply.get("reaped", 0))
            except Exception as exc:  # noqa: BLE001
                logger.warning("reaping %s on sandbox %s failed: %r", dict(labels), h.label, exc)
                return 0

        got = await asyncio.gather(*(one(h) for h in client.hosts()))
        if sum(got):
            logger.info("reaped %d remote sandbox(es) labelled %s", sum(got), dict(labels))
        return n + sum(got)


async def close_clients() -> None:
    """Close this process's relays and keepalives (tests; a process that exits needs nothing)."""
    for client in _CLIENTS.values():
        for task in client.keepalives.values():
            task.cancel()
        for relay in client.relays.values():
            if relay.server is not None:
                relay.server.close()
        client.relays.clear()
        client.keepalives.clear()
    _CLIENTS.clear()
