"""The platoon client side of a run: `platoon train` against the run's gateway, the sandboxes it
starts (on this node: platoon runs Harbor sandboxes on the client's node), and the HF exporter.

    python -m harnessforge.train driver-job --run-dir <run>

runs in an rt_HC job (driver.where = "hc", ~28 concurrent sandboxes on 32 cores) or on the
gateway's head node (driver.where = "gateway", started by the gateway job; 192 cores). It

  1. waits for <run>/gateway.json status=ready;
  2. selects and materializes the tasks, renders the harness spec, stages sandbox images to
     node-local disk (driver.seed_image_cache) or uses a shared cache (driver.image_cache);
  3. writes <run>/platoon-train.json and runs `platoon train` (log: logs/platoon.log);
  4. runs the HF exporter beside it (in the miles SIF: numpy/zstandard/xxhash), one final export
     at the end;
  5. writes <run>/driver.json and, when it runs in its own job, <run>/STOP so the gateway ends.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from harnessforge.train.config import TrainRun
from harnessforge.train.gateway import _write_json, container_argv
from harnessforge.train.guard import (STACKS_DIR, WATCHDOG_RC, StallGuard, collect_stacks, dump_stacks, log_tail,
                                      note_wandb, record_restart, restarts, stop_platoon, wandb_run_id)

#: marker files in the run dir: KEEP_GATEWAY = the gateway outlives this driver (restarts it when
#: the driver runs on its head node); DRIVER_STOP = ask the driver (any placement) to stop
KEEP_GATEWAY = "KEEP_GATEWAY"
DRIVER_STOP = "DRIVER_STOP"

TINKER_API_KEY = "tml-local"          # miles-tinker has no auth: the key only names the tenant
#: read from driver.wandb_env_file (only these names); passed to the platoon process env only
WANDB_ENV_NAMES = ("WANDB_API_KEY", "WANDB_ENTITY")
DEFAULT_WANDB_ENV_FILE = "~/mho-model-harness-optimization/mho-platoon-port/.env"


def read_env_file(path: str | Path, names: tuple[str, ...] = WANDB_ENV_NAMES) -> dict[str, str]:
    """`NAME=value` / `export NAME="value"` lines of a .env file, for `names` only (other lines,
    comments and values of other variables are never looked at beyond the name). Missing file: {}."""
    out: dict[str, str] = {}
    try:
        text = Path(path).expanduser().read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or name not in names:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if value:
            out[name] = value
    return out


def wandb_secrets(run: TrainRun, environ: dict[str, str] | None = None) -> tuple[dict[str, str], str]:
    """(W&B variables for the platoon process, a log line naming only their sources). The
    process env wins over driver.wandb_env_file (default ~/mho-model-harness-optimization/mho-platoon-port/.env). Nothing when W&B is
    not online."""
    environ = os.environ if environ is None else environ
    mode = (run.get("stats", {}).get("wandb") or {}).get("mode", "disabled")
    if mode != "online":
        return {}, f"wandb mode {mode}: no W&B credentials needed"
    path = run.driver.get("wandb_env_file", DEFAULT_WANDB_ENV_FILE)
    from_file = read_env_file(path)
    out, src = {}, []
    for name in WANDB_ENV_NAMES:
        if environ.get(name):
            out[name] = environ[name]
            src.append(f"{name} from the environment")
        elif name in from_file:
            out[name] = from_file[name]
            src.append(f"{name} from {path}")
    return out, "; ".join(src) or f"no W&B credentials in the environment or {path}"


def _log(run: TrainRun, msg: str) -> None:
    line = f"[driver {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    run.paths.logs.mkdir(parents=True, exist_ok=True)
    with open(run.paths.logs / "driver.events.log", "a") as f:
        f.write(line + "\n")


def wait_gateway(run: TrainRun, timeout: float = 4 * 3600, poll: float = 15) -> dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        g = json.loads(run.paths.gateway_file.read_text()) if run.paths.gateway_file.exists() else {}
        if g.get("status") == "ready":
            return g
        if g.get("status") == "stopped" or run.paths.stop_file.exists():
            raise RuntimeError(f"gateway ended before it was ready: {g}")
        time.sleep(poll)
    raise TimeoutError("gateway not ready")


def exporter_argv(run: TrainRun, *, follow: bool) -> list[str]:
    from harnessforge.train.gateway import node_local_dir
    node_local = node_local_dir()
    py = ["python3", "-m", "harnessforge.train", "export", "--run-dir", str(run.dir)]
    return container_argv(run, node_local=node_local) + py + (["--follow"] if follow else [])


def platoon_env(run: TrainRun, url: str, secrets: dict[str, str] | None = None) -> dict[str, str]:
    d = run.driver
    hf_src = run.get("harnessforge_src") or str(Path(__file__).resolve().parents[2])
    from harnessforge.train.gateway import node_local_dir
    node_local = node_local_dir()
    env = dict(os.environ)
    env.update({
        # the stack-dump sitecustomize first (it chains to any other), then this package
        "PYTHONPATH": os.pathsep.join(p for p in (str(Path(hf_src) / "harnessforge" / "train" / "stackdump_site"),
                                                  hf_src, env.get("PYTHONPATH")) if p),
        "TINKER_BASE_URL": url, "TINKER_API_KEY": TINKER_API_KEY,
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PLATOON_NODE_SCRATCH": node_local, "PYTHONUNBUFFERED": "1",
        "HF_STACKDUMP_DIR": str(run.paths.logs / STACKS_DIR),
        "no_proxy": ",".join(filter(None, [url.split("//")[-1].split(":")[0], "127.0.0.1",
                                           env.get("no_proxy", "")])),
    })
    if d.get("site"):
        env["PLATOON_SITE"] = str(d["site"])
    env.update({k: str(v) for k, v in (d.get("env") or {}).items()})
    env.update(secrets or {})          # this process env only: never logged, never written to disk
    return env


def prepare(run: TrainRun, url: str) -> Path:
    """Tasks, dataset copy, harness spec, images, and the platoon config. Returns its path."""
    from harnessforge.train import platoon as P
    tasks = P.select_tasks(run)
    (run.dir / "tasks.json").write_text(json.dumps({"train": tasks}, indent=0))
    prep = P.materialize(run, tasks)
    spec = P.render_harness(run, prep)
    d = run.driver
    env = run.raw.setdefault("env", {})
    popts = env.setdefault("provider_options", {})
    if d.get("seed_image_cache"):
        from harnessforge.evals.__main__ import stage_images
        from harnessforge.train.gateway import node_local_dir
        node_cache = Path(node_local_dir()) / "images"
        staged, missing = stage_images(run.paths.dataset_dir, tasks, d["seed_image_cache"], node_cache)
        _log(run, f"staged {staged} images to {node_cache}; {missing} not cached (pulled on first use)")
        popts.setdefault("image_cache", str(node_cache))
    elif d.get("image_cache"):
        popts.setdefault("image_cache", str(d["image_cache"]))
    cfg = P.train_config(run, tasks, url, spec)
    run.paths.platoon_config.write_text(json.dumps(cfg, indent=1))
    _log(run, f"{len(tasks)} tasks, platoon config {run.paths.platoon_config}")
    return run.paths.platoon_config


def _cpu_times() -> tuple[int, int]:
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v), idle


def node_sample(prev: tuple[int, int] | None) -> tuple[dict, tuple[int, int]]:
    """One row of this node's load: CPU busy % since `prev`, load average, memory (stdlib, /proc)."""
    cur = _cpu_times()
    busy = None
    if prev is not None and cur[0] > prev[0]:
        busy = round(100.0 * (1 - (cur[1] - prev[1]) / (cur[0] - prev[0])), 1)
    with open("/proc/loadavg") as f:
        load1 = float(f.read().split()[0])
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0])
    total, avail = mem.get("MemTotal", 0), mem.get("MemAvailable", 0)
    return {"time": round(time.time()), "cpu_busy_pct": busy, "load1": load1, "cores": os.cpu_count(),
            "mem_used_gb": round((total - avail) / 2**20, 1), "mem_total_gb": round(total / 2**20, 1)}, cur


def node_monitor(path: Path, stop: "threading.Event", interval: float = 30.0) -> None:
    prev = None
    with open(path, "a") as f:
        while not stop.wait(interval if prev is not None else 1.0):
            try:
                row, prev = node_sample(prev)
            except OSError:
                continue
            if row["cpu_busy_pct"] is not None:
                f.write(json.dumps(row) + "\n")
                f.flush()


def previous_model_ids(run: TrainRun) -> list[str]:
    """Tinker model ids a previous platoon process of this run trained (from its checkpoints)."""
    import re
    ids: list[str] = []
    for f in run.paths.platoon_dir.glob("*/*/checkpoints.jsonl"):
        for m in re.finditer(r"tinker://([A-Za-z0-9_.-]+)/", f.read_text()):
            if m.group(1) not in ids:
                ids.append(m.group(1))
    return ids


def release_previous_models(run: TrainRun, url: str, *, post=None, wait_s: float = 20.0) -> list[str]:
    """The full-fine-tuning gateway serves one model at a time and holds a client's lease for an hour:
    a restarted driver's platoon (resume = a new training client) is refused until the previous model is
    unloaded. Unload every model this run trained; ids the gateway does not know (a fresh gateway) are
    ignored. Returns the ids it unloaded."""
    import urllib.error
    import urllib.request

    def _post(u: str, body: dict) -> int:
        req = urllib.request.Request(u, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "X-API-Key": TINKER_API_KEY})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except OSError:
            return 0
    post = post or _post
    done = [m for m in previous_model_ids(run)
            if post(url.rstrip("/") + "/api/v1/unload_model", {"model_id": m}) == 200]
    if done:
        time.sleep(wait_s)                  # the unload completes behind a future
    return done


def resume_batch(run: TrainRun) -> int:
    """The batch platoon resumes from: its newest checkpoint in log_path (0 for a fresh run)."""
    last = 0
    for f in run.paths.platoon_dir.glob("*/*/checkpoints.jsonl"):
        for line in f.read_text().splitlines():
            try:
                last = max(last, int(json.loads(line).get("batch", 0)))
            except (ValueError, TypeError):
                pass
    return last


def record_stream_base(run: TrainRun) -> None:
    """Tell the HF exporter which step this gateway's delta stream starts after (resume)."""
    from harnessforge.train.export import DeltaExporter
    d = run.paths.delta_dir
    if not d.is_dir():
        return
    sid = DeltaExporter(base=".", deltas=d, work=".", out=".").stream_id()
    _write_json(d.parent / "stream_base.json", {"stream": sid, "batch": resume_batch(run),
                                                "at": time.time()})


def supervise(run: TrainRun, proc: subprocess.Popen, guard: StallGuard, term, stop: dict,
              *, poll: float = 10.0, sleep=time.sleep) -> dict[str, Any] | None:
    """Wait for platoon to exit; pass on stop requests (STOP / DRIVER_STOP); on the guard's events dump
    the stacks, and on a stall stop platoon. Returns the incident (None: platoon exited on its own or
    was asked to stop): {"reason", "step", "silent_s", "stacks"}."""
    P = run.paths
    incident: dict[str, Any] | None = None
    watchdog_stacks: str | None = None
    while proc.poll() is None:
        if (P.stop_file.exists() or (P.dir / DRIVER_STOP).exists()) and not stop["flag"]:
            stop["user"] = True
            term()
        event = None if stop["flag"] else guard.poll()
        if event:
            n = len(guard_restarts(run)) + 1
            t = time.time()
            pids = dump_stacks(P.logs, proc.pid, sleep=sleep)
            dest = collect_stacks(P.logs, P.logs / f"stall.{n}.{time.strftime('%Y%m%d-%H%M%S')}.txt", pids, t)
            if event == "watchdog":
                watchdog_stacks = str(dest)
                _log(run, f"platoon's watchdog is warning (no heartbeat); stacks of pids {pids} in {dest}")
            else:
                _log(run, f"STALL: no training step for {guard.silent_for():.0f}s (last step "
                          f"{guard.last_step}); stacks of pids {pids} in {dest}; stopping platoon")
                incident = {"reason": "stall", "step": guard.last_step,
                            "silent_s": round(guard.silent_for()), "stacks": str(dest)}
                stop_platoon(proc, term, stop_wait=float(run.driver.get("stall_stop_wait", 600)), sleep=sleep)
                break
        sleep(poll)
    if incident is None:
        guard.timeout = 0                   # read platoon's last lines (its last step) without judging them
        guard.poll()
    if incident is None and proc.returncode not in (0, None) and not stop["user"]:
        reason = "watchdog" if proc.returncode == WATCHDOG_RC else f"exit rc={proc.returncode}"
        stacks = watchdog_stacks
        if stacks is None:                  # a crash: the end of platoon's log (its traceback) is the record
            n = len(guard_restarts(run)) + 1
            dest = P.logs / f"stall.{n}.{time.strftime('%Y%m%d-%H%M%S')}.txt"
            dest.write_text(log_tail(P.logs / "platoon.log", guard.start_offset))
            stacks = str(dest)
        incident = {"reason": reason, "step": guard.last_step, "silent_s": round(guard.silent_for()),
                    "stacks": stacks}
    return incident


def guard_restarts(run: TrainRun) -> list[dict[str, Any]]:
    return restarts(run.dir)


def auto_restart(run: TrainRun, incident: dict[str, Any] | None, rc: int, env: dict[str, str]) -> bool:
    """After a stall, a watchdog exit or a crash (any non-zero exit nobody asked for), ask the gateway for a driver restart (KEEP_GATEWAY) while the
    budget (driver.max_auto_restarts) lasts; record it (auto_restarts.json, driver.events.log, W&B).
    Over budget, or with a separate driver job, the run stops and waits for a person. Whether it did."""
    if incident is None:
        return False
    budget = int(run.driver.get("max_auto_restarts", 8))
    done = len(guard_restarts(run))
    can = run.driver.get("where", "hc") == "gateway"
    if done >= budget or not can:
        why = (f"auto-restart budget used ({done}/{budget})" if can
               else "auto-restart needs driver.where = gateway")
        _log(run, f"{incident['reason'].upper()} (rc={rc}) at step {incident['step']}: {why}; "
                  f"the run stops here and waits for a person")
        return False
    n = record_restart(run.dir, {**incident, "rc": rc, "at": time.time(),
                                 "at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    # the resumed platoon re-sends its config's W&B tags: keep the restart's tag in them too
    try:
        cfg = json.loads(run.paths.config.read_text())
        tags = cfg.setdefault("stats", {}).setdefault("wandb", {}).setdefault(
            "tags", list((run.get("stats", {}).get("wandb") or {}).get("tags") or []))
        if f"auto-restart-{n}" not in tags:
            tags.append(f"auto-restart-{n}")
        _write_json(run.paths.config, cfg)
    except (OSError, ValueError) as e:
        _log(run, f"could not add the auto-restart tag to {run.paths.config}: {e}")
    (run.dir / KEEP_GATEWAY).write_text(f"auto-restart {n}: {incident['reason']}\n")
    _log(run, f"AUTO-RESTART {n}/{budget}: {incident['reason']} (rc={rc}, last step {incident['step']}, "
              f"silent {incident['silent_s']}s, stacks {incident['stacks']}); the gateway restarts the "
              f"driver, which resumes from the latest recovery checkpoint")
    rid = wandb_run_id(run.paths.platoon_dir)
    wb = run.get("stats", {}).get("wandb") or {}
    if rid and wb.get("mode", "online") == "online" and env.get("WANDB_API_KEY"):
        python = str(Path(str(run.driver.get("platoon", "platoon"))).with_name("python"))
        note = (f"[{time.strftime('%Y-%m-%d %H:%M %Z')}] auto-restart {n}: {incident['reason']} after step "
                f"{incident['step']} (silent {incident['silent_s']}s, rc={rc}); stacks: {incident['stacks']}")
        _log(run, note_wandb(python, str(wb.get("project", "")), rid, f"auto-restart-{n}", note, env))
    return True


def run_driver_job(run_dir: str | Path) -> int:
    run = TrainRun.load(Path(run_dir) / "run.json")
    P = run.paths
    job_id = os.environ.get("PBS_JOBID", "local")
    _write_json(P.driver_file, {"status": "waiting", "job_id": job_id, "host": os.uname().nodename,
                                "started_at": time.time()})
    platoon_proc: subprocess.Popen | None = None
    exporter: subprocess.Popen | None = None
    rc = 1
    stop = {"flag": False, "user": False}

    def _user_term(*_: Any) -> None:            # a stop someone asked for: never auto-restarted
        stop["user"] = True
        _term()

    def _term(*_: Any) -> None:
        # once: platoon takes a second SIGTERM as "force exit" (no interrupted checkpoint, no reaping)
        if stop["flag"]:
            return
        stop["flag"] = True
        if platoon_proc is not None and platoon_proc.poll() is None:
            platoon_proc.send_signal(signal.SIGTERM)       # platoon checkpoints, stops its sandboxes
    signal.signal(signal.SIGTERM, _user_term)
    signal.signal(signal.SIGINT, _user_term)
    try:
        g = wait_gateway(run)
        url = g["url"]
        _log(run, f"gateway ready at {url}")
        secrets, how = wandb_secrets(run)
        _log(run, how)                                   # variable names and sources only
        if (run.get("stats", {}).get("wandb") or {}).get("mode") == "online" and \
                "WANDB_API_KEY" not in secrets:
            _log(run, "WARNING: no WANDB_API_KEY; logging to W&B offline instead (wandb sync later)")
            run.raw.setdefault("stats", {}).setdefault("wandb", {})["mode"] = "offline"
        freed = release_previous_models(run, url)
        if freed:
            _log(run, f"unloaded the previous platoon process's model(s) {freed} from the gateway")
        cfg = prepare(run, url)
        env = platoon_env(run, url, secrets)
        record_stream_base(run)
        final = P.dir / "export" / "FINAL"
        if final.exists():                                  # a previous driver's end marker
            final.rename(final.with_name(f"FINAL.{int(time.time())}"))
        if int(run.get("export", {}).get("every", 0)) > 0 and \
                run.gateway.get("weight_sync", "disk-delta") == "disk-delta":
            with open(P.logs / "export.log", "ab") as out:
                exporter = subprocess.Popen(exporter_argv(run, follow=True), stdout=out,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            start_new_session=True)
        import threading
        mon_stop = threading.Event()
        threading.Thread(target=node_monitor, daemon=True,
                         args=(P.logs / f"node.{os.uname().nodename}.jsonl", mon_stop)).start()
        platoon = str(run.driver.get("platoon", "platoon"))
        _write_json(P.driver_file, {"status": "training", "job_id": job_id, "host": os.uname().nodename,
                                    "gateway": url, "started_at": time.time()})
        t0 = time.time()
        with open(P.logs / "platoon.log", "ab") as out:
            guard = StallGuard(P.logs / "platoon.log", float(run.driver.get("stall_timeout", 1200)),
                               float(run.driver.get("stall_startup_grace", 3600)))
            platoon_proc = subprocess.Popen([platoon, "train", str(cfg)], env=env, stdout=out,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            cwd=str(run.dir), start_new_session=True)
            incident = supervise(run, platoon_proc, guard, _term, stop)
        rc = platoon_proc.returncode
        _log(run, f"platoon train exited rc={rc} after {time.time() - t0:.0f}s")
        if not stop["user"]:
            auto_restart(run, incident, rc, env)
    except Exception as e:                                  # report, then clean up below
        _log(run, f"ERROR: {type(e).__name__}: {e}")
        rc = 1
    finally:
        if exporter is not None:
            # the exporter takes a final snapshot when it sees STOP-for-export
            (P.dir / "export" ).mkdir(parents=True, exist_ok=True)
            (P.dir / "export" / "FINAL").write_text(str(time.time()))
            try:
                exporter.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                os.killpg(exporter.pid, signal.SIGKILL)
        d = json.loads(P.driver_file.read_text()) if P.driver_file.exists() else {}
        d.update(status="done", returncode=rc, finished_at=time.time())
        _write_json(P.driver_file, d)
        ds = P.dir / DRIVER_STOP
        if ds.exists():
            ds.rename(ds.with_name(f"{DRIVER_STOP}.{int(time.time())}"))
        keep = P.dir / KEEP_GATEWAY
        if keep.exists():
            if run.driver.get("where", "hc") != "gateway":
                # separate driver job: consumed here; on the head node the gateway consumes it
                keep.rename(keep.with_name(f"{KEEP_GATEWAY}.{int(time.time())}"))
        elif run.driver.get("where", "hc") != "gateway" and run.driver.get("stop_gateway", True):
            P.stop_file.write_text(f"driver finished rc={rc}\n")
    return rc
