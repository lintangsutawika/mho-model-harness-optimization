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

TINKER_API_KEY = "tml-local"          # miles-tinker has no auth: the key only names the tenant


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


def platoon_env(run: TrainRun, url: str) -> dict[str, str]:
    d = run.driver
    hf_src = run.get("harnessforge_src") or str(Path(__file__).resolve().parents[2])
    from harnessforge.train.gateway import node_local_dir
    node_local = node_local_dir()
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join(p for p in (hf_src, env.get("PYTHONPATH")) if p),
        "TINKER_BASE_URL": url, "TINKER_API_KEY": TINKER_API_KEY,
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PLATOON_NODE_SCRATCH": node_local, "PYTHONUNBUFFERED": "1",
        "no_proxy": ",".join(filter(None, [url.split("//")[-1].split(":")[0], "127.0.0.1",
                                           env.get("no_proxy", "")])),
    })
    if d.get("site"):
        env["PLATOON_SITE"] = str(d["site"])
    env.update({k: str(v) for k, v in (d.get("env") or {}).items()})
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


def run_driver_job(run_dir: str | Path) -> int:
    run = TrainRun.load(Path(run_dir) / "run.json")
    P = run.paths
    job_id = os.environ.get("PBS_JOBID", "local")
    _write_json(P.driver_file, {"status": "waiting", "job_id": job_id, "host": os.uname().nodename,
                                "started_at": time.time()})
    platoon_proc: subprocess.Popen | None = None
    exporter: subprocess.Popen | None = None
    rc = 1
    stop = {"flag": False}

    def _term(*_: Any) -> None:
        # once: platoon takes a second SIGTERM as "force exit" (no interrupted checkpoint, no reaping)
        if stop["flag"]:
            return
        stop["flag"] = True
        if platoon_proc is not None and platoon_proc.poll() is None:
            platoon_proc.send_signal(signal.SIGTERM)       # platoon checkpoints, stops its sandboxes
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    try:
        g = wait_gateway(run)
        url = g["url"]
        _log(run, f"gateway ready at {url}")
        cfg = prepare(run, url)
        env = platoon_env(run, url)
        final = P.dir / "export" / "FINAL"
        if final.exists():                                  # a previous driver's end marker
            final.rename(final.with_name(f"FINAL.{int(time.time())}"))
        if int(run.get("export", {}).get("every", 0)) > 0 and \
                run.gateway.get("weight_sync", "disk-delta") == "disk-delta":
            with open(P.logs / "export.log", "ab") as out:
                exporter = subprocess.Popen(exporter_argv(run, follow=True), stdout=out,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            start_new_session=True)
        platoon = str(run.driver.get("platoon", "platoon"))
        _write_json(P.driver_file, {"status": "training", "job_id": job_id, "host": os.uname().nodename,
                                    "gateway": url, "started_at": time.time()})
        t0 = time.time()
        with open(P.logs / "platoon.log", "ab") as out:
            platoon_proc = subprocess.Popen([platoon, "train", str(cfg)], env=env, stdout=out,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            cwd=str(run.dir), start_new_session=True)
            while platoon_proc.poll() is None:
                if P.stop_file.exists() and not stop["flag"]:
                    _term()
                time.sleep(10)
        rc = platoon_proc.returncode
        _log(run, f"platoon train exited rc={rc} after {time.time() - t0:.0f}s")
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
        keep = P.dir / "KEEP_GATEWAY"
        if keep.exists():
            keep.rename(keep.with_name(f"KEEP_GATEWAY.{int(time.time())}"))
        elif run.driver.get("where", "hc") != "gateway" and run.driver.get("stop_gateway", True):
            P.stop_file.write_text(f"driver finished rc={rc}\n")
    return rc
