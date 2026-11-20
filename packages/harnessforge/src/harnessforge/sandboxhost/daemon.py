"""The sandbox host: a daemon in a CPU batch job that starts hpc-sandbox pods for platoon clients on other nodes.

    python -m harnessforge.sandboxhost serve --state <fleet dir> --index <i>      (inside the job; `start` submits it)

It listens on TCP (every connection must pass the token's challenge, `wire`), and runs each request against a
`PodSingularityProvider` of its own (the same provider a local run uses, so prepared images, the start cap
`max_starting`, fakeroot pods and egress behave as they do locally). Ops: status, start, exec, upload, download,
kill_processes, ensure_routes, stop, renew, reap, mirror_begin / mirror_put.

- **Capacity.** At most `slots` sandboxes (started and starting); a start beyond that fails with HostFull and the
  client tries another host.
- **Upstreams.** A sandbox's unix-socket upstreams (platoon's model gateway, on the client's node) become sockets
  in this job's scratch, each relayed over TCP to the client's relay (`provider.UpstreamRelay`), which connects to
  the real socket. The pod relays that socket to its loopback as usual, so routes are unchanged (18080, ...).
- **Mirrors.** Mount sources (and SIFs, owner manifests) on the client's node-local disk (platoon's staged
  toolchain) are copied here once per host (`mirror_begin` / `mirror_put`), and the client rewrites the spec to
  the copy. Shared paths (/groups, $HOME) are used as they are.
- **Liveness.** A long op sends a heartbeat frame every `HEARTBEAT_S`, so a client tells a slow command from a dead
  host. A client that disconnects during an exec, download or upload cancels it. A start whose client is gone
  before it acknowledged the reply is stopped. Each sandbox has a lease (`lease_s`, default 900 s), renewed by any op
  on it and by the client's keepalive (`renew`); one that expires with nothing in flight is stopped (its client
  died without stopping or reaping it).
- **Shutdown.** SIGTERM, or STOP / STOP.<i> in the fleet dir: stop every sandbox, write status "stopped", exit.
- **Drain.** DRAIN / DRAIN.<i> in the fleet dir: place nothing new (free 0, starts refused as full), keep serving the
  sandboxes it has until they stop, then exit with status "drained" (`sandboxhost drain --replace` then starts a
  fresh host for that index): a host swapped without losing a rollout.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import shutil
import signal
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from harnessforge.sandboxhost import fleet as fleetmod
from harnessforge.sandboxhost import net, wire

logger = logging.getLogger("harnessforge.sandboxhost")

HEARTBEAT_S = 10.0
START_ACK_TIMEOUT_S = 60.0
STATUS_EVERY_S = 10.0
LEASE_CHECK_S = 30.0
STOP_ALL_TIMEOUT_S = 300.0
HOST_LABEL = "hf.sandboxhost"
#: client options a host applies on top of its own (the client's choice of sandbox behaviour)
FORWARDED_OPTIONS = ("fakeroot", "allow_network_none", "pod_users", "rootfs_layout", "extra_start_args",
                     "bind_resolv_conf", "default_timeout_s", "start_timeout_s", "stop_grace_s", "stop_timeout_s",
                     "output_limit_bytes", "image_cache", "gpu", "cgroup_limits", "rootfs_templates",
                     "service_stop_grace_s", "slirp4netns", "binary", "stage_images",
                     # the tools bundle replacing harbor's per-trial apt (evals.pods.SYSDEPS_EXTRACT): a remote
                     # sandbox must get it as a local one does, or the agent loses curl/git/gcc/make
                     "sysdeps")
#: ops that a client's disconnect cancels
CANCELLABLE = frozenset({"exec", "upload", "download", "ensure_routes", "kill_processes", "mirror_put"})


def default_provider_factory(options: dict[str, Any]) -> Any:
    from harnessforge.evals.pods import PodSingularityProvider
    return PodSingularityProvider(**options)


@dataclasses.dataclass
class _Entry:
    sandbox: Any
    labels: dict[str, str]
    relay_dir: Path | None
    relays: list[asyncio.AbstractServer]
    client: str
    started: float
    last_seen: float
    inflight: int = 0
    stopping: asyncio.Task | None = None


class SandboxHost:
    def __init__(self, *, token: bytes, slots: int, scratch: str | os.PathLike, name: str = "host",
                 provider_options: dict[str, Any] | None = None, lease_s: float = 900.0,
                 provider_factory: Callable[[dict[str, Any]], Any] | None = None,
                 status_file: str | os.PathLike | None = None, stop_files: tuple[Path, ...] = (),
                 advertise: str | None = None, info: dict[str, Any] | None = None,
                 drain_files: tuple[Path, ...] = ()) -> None:
        self.token = token
        self.slots = int(slots)
        self.name = name
        self.scratch = Path(scratch)
        self.provider_options = dict(provider_options or {})
        self.lease_s = float(lease_s)
        self.provider_factory = provider_factory or default_provider_factory
        self.status_file = Path(status_file) if status_file else None
        self.stop_files = tuple(stop_files)
        self.drain_files = tuple(drain_files)
        self.drained = False
        self.advertise = advertise
        self.info = dict(info or {})
        self.entries: dict[str, _Entry] = {}
        self.starting = 0
        self.stats = {"started": 0, "start_failed": 0, "stopped": 0, "lease_expired": 0, "full": 0, "auth_failed": 0}
        self._providers: dict[str, Any] = {}
        self._mirror_locks: dict[str, asyncio.Lock] = {}
        self._mirror_claims: dict[str, float] = {}
        self._server: asyncio.AbstractServer | None = None
        self._stopping = asyncio.Event()
        self._relay_seq = 0
        self.address: tuple[str, int] | None = None
        for sub in ("up", "xfer", "mirrors"):
            (self.scratch / sub).mkdir(parents=True, exist_ok=True)
        # relay sockets: <up_root>/<n>-xxxx/<upstream>/gw.sock must fit AF_UNIX's ~104-108 bytes
        self.up_root = self.scratch / "up"
        if len(os.fsencode(self.up_root)) > 56:
            self.up_root = Path(tempfile.mkdtemp(prefix="hfsbh-up-", dir="/tmp"))

    # ------------------------------------------------------------------ lifecycle
    async def start_server(self, host: str = "0.0.0.0", port: int = 0) -> tuple[str, int]:
        self._server = await asyncio.start_server(net.keep_task(self._handle), host, port, limit=2 ** 20,
                                                  backlog=1024)
        sock_port = self._server.sockets[0].getsockname()[1]
        self.address = (net.advertise_ip(self.advertise) if host in ("0.0.0.0", "") else host, sock_port)
        return self.address

    async def serve_forever(self, host: str = "0.0.0.0", port: int = 0) -> None:
        if self._server is None:
            await self.start_server(host, port)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, self._stopping.set)
        tasks = [asyncio.ensure_future(self._status_loop()), asyncio.ensure_future(self._lease_loop())]
        try:
            await self._stopping.wait()
        finally:
            for t in tasks:
                t.cancel()
            await self.shutdown()

    def request_stop(self) -> None:
        self._stopping.set()

    @property
    def draining(self) -> bool:
        return any(p.exists() for p in self.drain_files)

    async def shutdown(self) -> None:
        if self.drained:                           # nothing left to stop: say so, and keep the "drained" status
            if self._server is not None:
                self._server.close()
            self._write_status("drained")
            return
        self._write_status("stopping")
        if self._server is not None:
            self._server.close()
        n = await self.stop_all(timeout_s=STOP_ALL_TIMEOUT_S)
        logger.info("stopped %d sandbox(es) at shutdown", n)
        self._write_status("stopped")

    async def stop_all(self, timeout_s: float = STOP_ALL_TIMEOUT_S, labels: dict[str, str] | None = None) -> int:
        ids = [sid for sid, e in self.entries.items()
               if labels is None or all(e.labels.get(k) == v for k, v in labels.items())]
        if not ids:
            return 0
        tasks = [asyncio.ensure_future(self._stop(sid, reason="stop_all" if labels is None else "reap"))
                 for sid in ids]
        done, pending = await asyncio.wait(tasks, timeout=timeout_s)
        for t in pending:
            t.cancel()
        return sum(1 for t in done if not t.cancelled() and t.exception() is None and t.result())

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        try:
            load = os.getloadavg()
        except OSError:
            load = (0.0, 0.0, 0.0)
        return {"name": self.name, "slots": self.slots, "active": len(self.entries), "starting": self.starting,
                "free": 0 if self.draining else max(0, self.slots - len(self.entries) - self.starting),
                "draining": self.draining, "load": list(load),
                "cpus": os.cpu_count(), "stats": dict(self.stats), "pid": os.getpid(),
                "host": socket.gethostname(), "time": time.time(), **self.info}

    def _write_status(self, status: str) -> None:
        if self.status_file is None:
            return
        data = {**self.status(), "status": status, "updated_at": time.time()}
        if self.address:
            data.update(ip=self.address[0], port=self.address[1])
        fleetmod.write_json(self.status_file, data)

    async def _status_loop(self) -> None:
        while True:
            if any(p.exists() for p in self.stop_files):
                logger.info("STOP file present: shutting down")
                self._stopping.set()
                return
            if self.draining:
                if not self.entries and not self.starting:
                    logger.info("DRAIN file present and no sandbox left: exiting (drained)")
                    self.drained = True
                    self._write_status("drained")
                    self._stopping.set()
                    return
                self._write_status("draining")     # not "ready": clients place nothing here
            else:
                self._write_status("ready")
            await asyncio.sleep(STATUS_EVERY_S)

    async def _lease_loop(self) -> None:
        while True:
            await asyncio.sleep(LEASE_CHECK_S)
            self.expire_leases()

    def expire_leases(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        expired = [sid for sid, e in self.entries.items()
                   if e.inflight == 0 and e.stopping is None and now - e.last_seen > self.lease_s]
        for sid in expired:
            logger.warning("sandbox %s: lease expired (client %s silent for %.0fs); stopping", sid,
                           self.entries[sid].client, now - self.entries[sid].last_seen)
            self.stats["lease_expired"] += 1
            asyncio.ensure_future(self._stop(sid, reason="lease"))
        return expired

    # ------------------------------------------------------------------ connections
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        net.set_keepalive(writer)
        spooled: list[Path] = []
        torn_down = False
        try:
            try:
                await wire.server_handshake(reader, writer, self.token)
            except wire.AuthError:
                self.stats["auth_failed"] += 1
                peer = writer.get_extra_info("peername")
                logger.warning("refused a connection from %s: bad token", peer[0] if peer else "?")
                return
            header, payload = await wire.read_frame(reader, spool_dir=self.scratch / "xfer",
                                                    timeout=wire.AUTH_TIMEOUT_S)
            if isinstance(payload, Path):
                spooled.append(payload)
            op = str(header.get("op"))
            handler = getattr(self, f"op_{op}", None)
            if handler is None:
                await wire.write_frame(writer, wire.error_header(ValueError(f"unknown op {op!r}")))
                return
            await self._run(op, handler, header, payload, reader, writer, spooled)
        except (wire.WireError, ConnectionError, asyncio.TimeoutError):
            pass
        except Exception:  # noqa: BLE001 - one bad connection never ends the daemon
            logger.exception("connection failed")
        except GeneratorExit:                      # closed in place: close without awaiting anything
            torn_down = True
            net.close_now(writer)
            raise
        finally:
            for p in spooled:
                p.unlink(missing_ok=True)
            if not torn_down:
                await net.close_writer(writer)

    async def _run(self, op: str, handler: Any, header: dict, payload: Any, reader: asyncio.StreamReader,
                   writer: asyncio.StreamWriter, spooled: list[Path]) -> None:
        sid = header.get("id")
        entry = self.entries.get(sid) if sid else None
        if entry is not None:
            entry.inflight += 1
            entry.last_seen = time.time()
        task = asyncio.ensure_future(handler(header, payload))
        eof = asyncio.ensure_future(reader.read(1)) if op in CANCELLABLE else None
        try:
            while True:
                waits = {task} | ({eof} if eof is not None else set())
                done, _ = await asyncio.wait(waits, timeout=HEARTBEAT_S, return_when=asyncio.FIRST_COMPLETED)
                if task in done:
                    break
                if eof is not None and eof in done:  # the client went away: cancel what it asked for
                    task.cancel()
                    with contextlib.suppress(BaseException):
                        await task
                    return
                await wire.write_frame(writer, {"hb": time.time()})
            try:
                result = task.result()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - reported to the client
                if not isinstance(exc, (wire.HostFull, FileNotFoundError)):
                    logger.info("op %s on %s failed: %r", op, sid, exc)
                if entry is not None and (self._stopping.is_set() or entry.stopping is not None):
                    # the host stopped the sandbox under the op (shutdown, reap, lease): the client lost it
                    exc = wire.lost_error(f"sandbox {sid} was stopped by sandbox host {self.name} during {op}: "
                                          f"{type(exc).__name__}: {exc}")
                await wire.write_frame(writer, wire.error_header(exc))
                return
            reply, out_payload = result if isinstance(result, tuple) else (result, None)
            if op == "start":
                await self._reply_start(reader, writer, reply)
                return
            try:
                await wire.write_frame(writer, {"ok": True, **reply}, out_payload)
            finally:
                if isinstance(out_payload, Path):
                    out_payload.unlink(missing_ok=True)
        finally:
            if eof is not None and not eof.done():
                eof.cancel()
            if not task.done():  # the connection broke while the op ran
                if op in CANCELLABLE:
                    task.cancel()
                elif op == "start":
                    task.add_done_callback(self._stop_orphan_start)
            if entry is not None:
                entry.inflight -= 1
                entry.last_seen = time.time()

    def _stop_orphan_start(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None:
            return
        sid = task.result()["id"]
        logger.warning("sandbox %s: its client left during the start; stopping it", sid)
        asyncio.ensure_future(self._stop(sid, reason="orphan start"))

    async def _reply_start(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, reply: dict) -> None:
        """Send the start's reply and wait for the client's ack; a client that is gone by then (it gave the start
        up) gets its sandbox stopped."""
        sid = reply["id"]
        try:
            await wire.write_frame(writer, {"ok": True, **reply})
            header, _ = await wire.read_frame(reader, timeout=START_ACK_TIMEOUT_S)
            if header.get("op") == "ack":
                return
        except (wire.WireError, asyncio.TimeoutError, ConnectionError, OSError):
            pass
        logger.warning("sandbox %s: no ack of its start from the client; stopping it", sid)
        await self._stop(sid, reason="no ack")

    # ------------------------------------------------------------------ helpers
    def _entry(self, header: dict) -> _Entry:
        sid = header.get("id")
        entry = self.entries.get(sid)
        if entry is None or entry.stopping is not None:
            raise wire.lost_error(f"sandbox {sid} is not running on sandbox host {self.name}")
        return entry

    def _provider(self, client_options: dict[str, Any]) -> Any:
        options = {**self.provider_options,
                   **{k: v for k, v in (client_options or {}).items() if k in FORWARDED_OPTIONS}}
        options["keep_agent_processes"] = False  # the client decides whether kill_processes runs at all
        options.pop("collect_dir", None)         # the client collects, through exec
        key = json.dumps(options, sort_keys=True, default=str)
        if key not in self._providers:
            self._providers[key] = self.provider_factory(options)
        return self._providers[key]

    async def _stop(self, sid: str, *, reason: str) -> bool:
        entry = self.entries.get(sid)
        if entry is None:
            return False
        if entry.stopping is None:
            entry.stopping = asyncio.ensure_future(self._really_stop(sid, entry, reason))
        await asyncio.shield(entry.stopping)
        return True

    async def _really_stop(self, sid: str, entry: _Entry, reason: str) -> None:
        try:
            await entry.sandbox.stop()
        except Exception:  # noqa: BLE001
            logger.warning("sandbox %s: stop (%s) failed", sid, reason, exc_info=True)
        finally:
            for server in entry.relays:
                server.close()
            if entry.relay_dir is not None:
                shutil.rmtree(entry.relay_dir, ignore_errors=True)
            self.entries.pop(sid, None)
            self.stats["stopped"] += 1

    async def _upstream_relay(self, sock_path: Path, target: dict[str, Any]) -> asyncio.AbstractServer:
        """A unix socket here whose connections go to the client's relay, which connects them to its socket."""
        addr, port, key = str(target["addr"]), int(target["port"]), str(target["key"])

        async def on_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            up_w = None
            try:
                up_r, up_w = await asyncio.wait_for(asyncio.open_connection(addr, port, limit=2 ** 20), 15)
                net.set_keepalive(up_w)
                await wire.client_handshake(up_r, up_w, self.token, key=key)
                await net.splice(reader, writer, up_r, up_w)
            except Exception as exc:  # noqa: BLE001 - the agent sees a closed connection
                logger.warning("upstream relay to %s:%d failed: %r", addr, port, exc)
                await net.close_writer(writer)
                await net.close_writer(up_w)

        return await asyncio.start_unix_server(net.keep_task(on_conn), path=str(sock_path))

    # ------------------------------------------------------------------ ops
    async def op_ping(self, header: dict, payload: Any) -> dict:
        return {"pong": time.time()}

    async def op_status(self, header: dict, payload: Any) -> dict:
        return self.status()

    async def op_start(self, header: dict, payload: Any) -> dict:
        from platoon.sandbox.base import Mount, SandboxSpec
        if self._stopping.is_set():
            raise wire.HostFull(f"sandbox host {self.name} is shutting down")
        if self.draining:
            raise wire.HostFull(f"sandbox host {self.name} is draining")
        if len(self.entries) + self.starting >= self.slots:
            self.stats["full"] += 1
            raise wire.HostFull(f"sandbox host {self.name} is full ({self.slots} slots)")
        self.starting += 1
        relay_dir: Path | None = None
        relays: list[asyncio.AbstractServer] = []
        try:
            raw = dict(header["spec"])
            raw["mounts"] = [Mount(**m) for m in raw.get("mounts", [])]
            upstreams = dict(raw.get("upstreams") or {})
            targets = header.get("relays") or {}
            for name, url in sorted(upstreams.items()):
                if not str(url).startswith("unix://"):
                    continue
                if name not in targets:
                    raise ValueError(f"upstream {name!r} is a unix socket on the client, and no relay was given")
                if relay_dir is None:
                    self._relay_seq += 1
                    relay_dir = Path(tempfile.mkdtemp(prefix=f"{self._relay_seq}-", dir=self.up_root))
                sub = relay_dir / name
                sub.mkdir(mode=0o755)
                sock = sub / os.path.basename(str(url)[len("unix://"):])
                relays.append(await self._upstream_relay(sock, targets[name]))
                upstreams[name] = f"unix://{sock}"
            raw["upstreams"] = upstreams
            raw["labels"] = {**(raw.get("labels") or {}), HOST_LABEL: self.name}
            spec = SandboxSpec(**raw)
            provider = self._provider(header.get("options") or {})
            sandbox = await provider.start(spec)
        except BaseException:
            self.stats["start_failed"] += 1
            for server in relays:
                server.close()
            if relay_dir is not None:
                shutil.rmtree(relay_dir, ignore_errors=True)
            raise
        finally:
            self.starting -= 1
        sid = str(getattr(sandbox, "id", None) or uuid.uuid4().hex)
        while sid in self.entries:
            sid = f"{sid}-{uuid.uuid4().hex[:6]}"
        now = time.time()
        self.entries[sid] = _Entry(sandbox, dict(spec.labels), relay_dir, relays, str(header.get("client", "?")),
                                   now, now)
        self.stats["started"] += 1
        return {"id": sid, "routes": dict(getattr(sandbox, "routes", {}) or {}),
                "host": self.name, "started": now}

    async def op_exec(self, header: dict, payload: Any) -> dict:
        entry = self._entry(header)
        stdin = wire.payload_bytes(payload) if header.get("stdin") else None
        result = await entry.sandbox.exec(header["cmd"], timeout_s=header.get("timeout_s"), env=header.get("env"),
                                          cwd=header.get("cwd"), stdin=stdin)
        return {"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr,
                "timed_out": bool(result.timed_out)}

    async def op_upload(self, header: dict, payload: Any) -> dict:
        entry = self._entry(header)
        work = Path(tempfile.mkdtemp(prefix="up-", dir=self.scratch / "xfer"))
        try:
            tar = wire.payload_file(payload, self.scratch / "xfer")
            try:
                src = await asyncio.to_thread(wire.unpack, tar, work, header["kind"], trusted=True)
            finally:
                tar.unlink(missing_ok=True)
            await entry.sandbox.upload(src, header["dst"])
        finally:
            await asyncio.to_thread(shutil.rmtree, work, True)
        return {}

    async def op_download(self, header: dict, payload: Any) -> tuple[dict, Path]:
        entry = self._entry(header)
        work = Path(tempfile.mkdtemp(prefix="down-", dir=self.scratch / "xfer"))
        try:
            out = work / "out"
            report = await entry.sandbox.download(header["src"], out)
            tar, kind = await asyncio.to_thread(wire.pack, out, self.scratch / "xfer")
        finally:
            await asyncio.to_thread(shutil.rmtree, work, True)
        rep = None
        if report is not None:
            rep = {"dropped_links": [dataclasses.asdict(d) for d in report.dropped_links],
                   "dropped_total": report.dropped_total}
        return {"kind": kind, "report": rep}, tar

    async def op_kill_processes(self, header: dict, payload: Any) -> dict:
        entry = self._entry(header)
        keep = header.get("keep") or []
        if keep and hasattr(entry.sandbox, "keep_processes"):
            entry.sandbox.keep_processes(keep)
        await entry.sandbox.kill_processes()
        return {}

    async def op_ensure_routes(self, header: dict, payload: Any) -> dict:
        entry = self._entry(header)
        ensure = getattr(entry.sandbox, "ensure_routes", None)
        routes = await ensure() if ensure is not None else dict(getattr(entry.sandbox, "routes", {}) or {})
        return {"routes": dict(routes)}

    async def op_stop(self, header: dict, payload: Any) -> dict:
        return {"stopped": await self._stop(str(header.get("id")), reason="client")}

    async def op_renew(self, header: dict, payload: Any) -> dict:
        now = time.time()
        unknown = []
        for sid in header.get("ids") or []:
            entry = self.entries.get(sid)
            if entry is None:
                unknown.append(sid)
            else:
                entry.last_seen = now
        return {"unknown": unknown}

    async def op_reap(self, header: dict, payload: Any) -> dict:
        labels = {str(k): str(v) for k, v in (header.get("labels") or {}).items()}
        if not labels:
            raise ValueError("reap needs labels")
        return {"reaped": await self.stop_all(labels=labels)}

    async def op_mirror_begin(self, header: dict, payload: Any) -> dict:
        key = _key(header["key"])
        final = self.scratch / "mirrors" / key
        if final.exists():
            return {"state": "present", "path": str(final / ("d" if header.get("kind") == "dir" else "f"))}
        claimed = self._mirror_claims.get(key)
        if claimed is not None and time.time() - claimed < 1800:
            return {"state": "pending"}
        self._mirror_claims[key] = time.time()
        return {"state": "upload"}

    async def op_mirror_put(self, header: dict, payload: Any) -> dict:
        key = _key(header["key"])
        kind = header["kind"]
        final = self.scratch / "mirrors" / key
        lock = self._mirror_locks.setdefault(key, asyncio.Lock())
        async with lock:
            try:
                if not final.exists():
                    tmp = final.with_name(f".{key}.{uuid.uuid4().hex[:8]}")
                    tar = wire.payload_file(payload, self.scratch / "xfer")
                    try:
                        await asyncio.to_thread(wire.unpack, tar, tmp, kind, trusted=True)
                    finally:
                        tar.unlink(missing_ok=True)
                    os.chmod(tmp, 0o755)
                    tmp.rename(final)
                    logger.info("mirrored %s (%s) into %s", header.get("source", "?"), kind, final)
            finally:
                self._mirror_claims.pop(key, None)
        return {"path": str(final / ("d" if kind == "dir" else "f"))}


def _key(raw: Any) -> str:
    key = str(raw)
    if not key or not all(c in "0123456789abcdef" for c in key) or len(key) > 64:
        raise ValueError("bad mirror key")
    return key


# ---------------------------------------------------------------------------------------------------------- job entry
def serve(state: str | os.PathLike, index: int, *, port: int | None = None) -> int:
    """Run host `index` of the fleet whose state dir is `state` (inside its batch job)."""
    fd = fleetmod.FleetDir(state)
    cfg = fd.fleet()
    spec = cfg["spec"]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scratch = Path(os.environ.get("HF_SANDBOXHOST_SCRATCH") or fleetmod.node_scratch()) / "hfsbh"
    scratch.mkdir(parents=True, exist_ok=True)
    host = SandboxHost(
        token=wire.read_token(fd.token_file), slots=int(spec.get("slots_per_host", 96)), scratch=scratch,
        name=f"{cfg['fleet_id']}/{index}", provider_options=spec.get("provider_options") or {},
        lease_s=float(spec.get("lease_s", 900)), status_file=fd.host_file(index),
        stop_files=(fd.stop_file, fd.dir / f"STOP.{index}"), advertise=spec.get("advertise"),
        drain_files=(fd.dir / "DRAIN", fd.dir / f"DRAIN.{index}"),
        info={"index": index, "job_id": os.environ.get("PBS_JOBID", "local")})

    async def main() -> None:
        address = await host.start_server("0.0.0.0", int(spec.get("port", 0)) if port is None else port)
        logger.info("sandbox host %s listening on %s:%d with %d slots", host.name, address[0], address[1],
                    host.slots)
        await host.serve_forever()

    asyncio.run(main())
    return 0
