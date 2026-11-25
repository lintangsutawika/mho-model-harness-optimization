"""A fleet of sandbox hosts: its spec, its shared state dir, and launching / inspecting / stopping its jobs.

Spec (JSON; every key but fleet_id optional):

    {"fleet_id": "sbh-grpo",
     "state_root": "~/hf-dev/sandboxhosts/state",     # <state_root>/<fleet_id> is the fleet's state dir
     "hosts": 4,                                      # host jobs `start` keeps submitted
     "slots_per_host": 96,                            # sandboxes per host (started + starting)
     "lease_s": 900,                                  # a sandbox whose client is silent this long is stopped
     "port": 0,                                       # 0: any free port (the host file says which)
     "advertise": "auto",                             # address clients reach a host at (10.0.x.y)
     "python": "~/hf-dev/evals/venv/bin/python",      # has platoon[hpc] + hpc-sandbox
     "harnessforge_src": "~/hf-dev/.../harnessforge/src",
     "provider_options": {"image_cache": "/groups/.../hpc-sandbox-images", "max_starting": 16,
                          "fakeroot": true, "allow_network_none": true},
     "env": {"PLATOON_SITE": "..."},                  # extra job environment
     "launcher": {"kind": "pbs", "queue": "R9920261300", "account": "gai51740", "resource_type": "rt_HC",
                  "walltime": "72:00:00", "name_prefix": "hfsbh"}}

State dir (on a filesystem every node sees):

    fleet.json        the spec, the job of each host index, history
    token             the shared secret (0600; never printed)
    hosts/<i>.json    host i: ip, port, slots, active, starting, free, status, updated_at (every 10 s)
    logs/host<i>.log  host i's job log
    STOP, STOP.<i>    stop every host / host i
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from harnessforge.contracts import JobSpec, JobState

DEFAULT_ROOT = Path(os.environ.get("HF_SANDBOXHOST_ROOT", Path.home() / "hf-dev" / "sandboxhosts" / "state"))
SRC_DIR = Path(__file__).resolve().parents[2]
#: a host file older than this is a dead host (it is rewritten every 10 s)
STALE_S = 60.0


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def node_scratch() -> str:
    """This job's node-local scratch: $PBS_LOCALDIR, else /local/<jobid>, else $TMPDIR or /tmp."""
    if (local := os.environ.get("PBS_LOCALDIR")) and Path(local).is_dir():
        return local
    if (job := os.environ.get("PBS_JOBID")) and Path("/local", job).is_dir():
        return str(Path("/local", job))
    return os.environ.get("TMPDIR") or "/tmp"


def _expand(v: Any) -> Any:
    if isinstance(v, str) and (v.startswith("~") or "$" in v):
        return os.path.expanduser(os.path.expandvars(v))
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_spec(path: str | os.PathLike) -> dict[str, Any]:
    spec = _expand(json.loads(Path(path).expanduser().read_text()))
    if "fleet_id" not in spec:
        raise ValueError(f"fleet spec {path} needs fleet_id")
    return spec


def state_dir_of(hosts_spec: str | os.PathLike) -> Path:
    """The state dir a client option names: the state dir itself (it holds fleet.json), or a spec file."""
    p = Path(os.path.expanduser(os.path.expandvars(str(hosts_spec))))
    if p.is_dir():
        return p
    spec = load_spec(p)
    return Path(spec.get("state_root") or DEFAULT_ROOT).expanduser() / spec["fleet_id"]


class FleetDir:
    def __init__(self, path: str | os.PathLike) -> None:
        self.dir = Path(path).expanduser()

    @property
    def fleet_file(self) -> Path:
        return self.dir / "fleet.json"

    @property
    def token_file(self) -> Path:
        return self.dir / "token"

    @property
    def hosts_dir(self) -> Path:
        return self.dir / "hosts"

    @property
    def logs(self) -> Path:
        return self.dir / "logs"

    @property
    def stop_file(self) -> Path:
        return self.dir / "STOP"

    def host_file(self, i: int) -> Path:
        return self.hosts_dir / f"{i}.json"

    def fleet(self) -> dict[str, Any]:
        d = read_json(self.fleet_file)
        if d is None:
            raise FileNotFoundError(f"no fleet state at {self.fleet_file}")
        return d

    def hosts(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        if not self.hosts_dir.is_dir():
            return out
        for p in self.hosts_dir.glob("*.json"):
            try:
                i = int(p.stem)
            except ValueError:
                continue
            d = read_json(p)
            if isinstance(d, dict):
                out[i] = d
        return dict(sorted(out.items()))

    def live_hosts(self, now: float | None = None, stale_s: float = STALE_S) -> dict[int, dict[str, Any]]:
        """Hosts whose file says ready and was refreshed in the last `stale_s` seconds."""
        now = time.time() if now is None else now
        return {i: h for i, h in self.hosts().items()
                if h.get("status") == "ready" and now - float(h.get("updated_at", 0)) <= stale_s
                and h.get("ip") and h.get("port")}


# ---------------------------------------------------------------------------------------------------------- launch
def _launcher(spec: dict[str, Any]):
    from harnessforge.launch import make_launcher
    L = dict(spec.get("launcher") or {})
    return make_launcher(L.get("kind"), defaults={k: v for k, v in L.items()
                                                   if k in ("queue", "account", "resource_type", "select",
                                                            "extra_args")})


def host_job(spec: dict[str, Any], fd: FleetDir, i: int) -> JobSpec:
    L = dict(spec.get("launcher") or {})
    src = str(spec.get("harnessforge_src") or SRC_DIR)
    env = {"PYTHONPATH": src, "PYTHONUNBUFFERED": "1", **{k: str(v) for k, v in (spec.get("env") or {}).items()}}
    # batch jobs lack the PBS clients and /usr/sbin on PATH (ABCI)
    env.setdefault("PATH", os.pathsep.join(["/opt/pbs/bin", "/usr/sbin", os.environ.get("PATH", "/usr/bin:/bin")]))
    python = str(spec.get("python") or sys.executable)
    return JobSpec(
        name=f"{L.get('name_prefix', 'hfsbh')}-{spec['fleet_id']}-{i}"[:63],
        command=[python, "-m", "harnessforge.sandboxhost", "serve", "--state", str(fd.dir), "--index", str(i)],
        nodes=1, walltime=str(L.get("walltime", "24:00:00")), env=env, log=fd.logs / f"host{i}.log",
        options={k: v for k, v in L.items() if k in ("queue", "account", "resource_type", "select", "host",
                                                      "extra_args")})


def start(spec: dict[str, Any], *, hosts: int | None = None, launcher=None, wait: bool = True,
          timeout: float = 1800.0, log=print) -> dict[str, Any]:
    """Create the state dir and token (once) and submit a job for every host index below `hosts` that has no live
    job. Returns fleet.json."""
    from harnessforge.sandboxhost import wire
    root = Path(spec.get("state_root") or DEFAULT_ROOT).expanduser()
    fd = FleetDir(root / spec["fleet_id"])
    fd.dir.mkdir(parents=True, exist_ok=True)
    os.chmod(fd.dir, 0o700)
    wire.new_token(fd.token_file)
    launcher = launcher or _launcher(spec)
    cfg = read_json(fd.fleet_file) or {"fleet_id": spec["fleet_id"], "jobs": {}, "history": [],
                                       "created_at": time.time()}
    cfg["spec"] = spec
    n = int(spec.get("hosts", 1) if hosts is None else hosts)
    for name in ("STOP",):
        if (fd.dir / name).exists():
            (fd.dir / name).rename(fd.dir / f"{name}.{int(time.time())}")
    for i in range(n):
        old = cfg["jobs"].get(str(i))
        if old and launcher.status(old).state != JobState.DONE:
            continue
        for marker in (f"STOP.{i}", f"DRAIN.{i}"):  # a fresh host starts neither stopping nor draining
            m = fd.dir / marker
            if m.exists():
                m.rename(fd.dir / f"{marker}.{int(time.time())}")
        job = launcher.submit(host_job(spec, fd, i))
        cfg["jobs"][str(i)] = job
        cfg["history"].append({"at": time.time(), "event": "submit", "index": i, "job": job})
        log(f"host {i}: job {job}")
    write_json(fd.fleet_file, cfg)
    if wait:
        deadline = time.time() + timeout
        while len(fd.live_hosts()) < n:
            if time.time() > deadline:
                raise TimeoutError(f"{len(fd.live_hosts())}/{n} sandbox hosts ready after {timeout:.0f}s")
            time.sleep(5)
        log(f"{n} sandbox host(s) ready in {fd.dir}")
    return cfg


def status(fd: FleetDir, *, launcher=None, ping: bool = True) -> dict[str, Any]:
    cfg = fd.fleet()
    launcher = launcher or _launcher(cfg["spec"])
    now = time.time()
    hosts = fd.hosts()
    out = {"fleet_id": cfg["fleet_id"], "dir": str(fd.dir), "hosts": []}
    for i_s, job in sorted(cfg.get("jobs", {}).items(), key=lambda kv: int(kv[0])):
        i = int(i_s)
        h = hosts.get(i, {})
        st = launcher.status(job)
        row = {"index": i, "job": job, "job_state": st.state.value, "status": h.get("status"),
               "addr": f"{h.get('ip')}:{h.get('port')}" if h.get("ip") else None,
               "age_s": round(now - h["updated_at"], 1) if h.get("updated_at") else None,
               "slots": h.get("slots"), "active": h.get("active"), "starting": h.get("starting"),
               "load": h.get("load"), "stats": h.get("stats")}
        if ping and h.get("ip") and h.get("status") == "ready":
            from harnessforge.sandboxhost.provider import host_rpc_sync
            try:
                live = host_rpc_sync(fd, h, {"op": "status"}, timeout=10)
                row.update(active=live["active"], starting=live["starting"], free=live["free"], ping="ok",
                           load=live.get("load"))
            except Exception as exc:  # noqa: BLE001
                row["ping"] = f"failed: {type(exc).__name__}: {exc}"
        out["hosts"].append(row)
    return out


def stop(fd: FleetDir, *, index: int | None = None, launcher=None, grace_s: float = 330.0, log=print) -> None:
    """Ask the host(s) to stop their sandboxes and exit (STOP / STOP.<i>), then cancel jobs still running after
    `grace_s`."""
    cfg = fd.fleet()
    launcher = launcher or _launcher(cfg["spec"])
    target = fd.dir / ("STOP" if index is None else f"STOP.{index}")
    target.write_text(f"{time.time()}\n")
    jobs = {int(k): v for k, v in cfg.get("jobs", {}).items() if index is None or int(k) == index}
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if all(launcher.status(j).state == JobState.DONE for j in jobs.values()):
            break
        time.sleep(5)
    for i, j in jobs.items():
        if launcher.status(j).state != JobState.DONE:
            log(f"host {i}: job {j} still running after {grace_s:.0f}s; cancelling")
            launcher.cancel(j)
    cfg.setdefault("history", []).append({"at": time.time(), "event": "stop", "index": index})
    write_json(fd.fleet_file, cfg)
    log("stopped")


def drain(fd: FleetDir, index: int, *, launcher=None, timeout: float = 4 * 3600.0, poll: float = 15.0,
          replace: bool = False, spec: dict[str, Any] | None = None, log=print) -> bool:
    """Drain host `index` (DRAIN.<index>): it places nothing new and exits once its sandboxes have stopped. Waits up
    to `timeout` for its job to end; with `replace`, then starts a fresh host for that index (with `spec`, else the
    fleet's own). Returns whether the host drained in time (it keeps draining otherwise)."""
    cfg = fd.fleet()
    launcher = launcher or _launcher(cfg["spec"])
    job = cfg.get("jobs", {}).get(str(index))
    if job is None:
        raise KeyError(f"fleet {cfg['fleet_id']} has no host {index}")
    (fd.dir / f"DRAIN.{index}").write_text(f"{time.time()}\n")
    cfg.setdefault("history", []).append({"at": time.time(), "event": "drain", "index": index})
    write_json(fd.fleet_file, cfg)
    log(f"host {index}: draining (job {job})")
    deadline = time.time() + timeout
    while launcher.status(job).state != JobState.DONE:
        if time.time() > deadline:
            h = fd.hosts().get(index, {})
            log(f"host {index}: still {h.get('active')} sandbox(es) after {timeout:.0f}s; it keeps draining")
            return False
        time.sleep(poll)
    log(f"host {index}: drained")
    if replace:
        new_spec = dict(spec or cfg["spec"])
        n = max(int(new_spec.get("hosts", 1)), index + 1, len(cfg.get("jobs", {})))
        start(new_spec, hosts=n, launcher=launcher, wait=False, log=log)
    return True
