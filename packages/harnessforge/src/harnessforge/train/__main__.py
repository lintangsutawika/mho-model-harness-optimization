"""Training CLI.

    python -m harnessforge.train render --config configs/train/smoke.json     # resolved run + job specs
    python -m harnessforge.train launch --config configs/train/smoke.json     # gateway job (+ driver job)
    python -m harnessforge.train status --run-dir <run>
    python -m harnessforge.train stop   --run-dir <run> [--grace 600]
    python -m harnessforge.train export --run-dir <run> [--follow] [--version N]
    python -m harnessforge.train unlend --run-dir <run>        # external mode: return lent pool nodes
    python -m harnessforge.train probe  --run-dir <run> --tokens 131072   # one long datum: fwd/bwd + optim
    python -m harnessforge.train mapping                                        # old run -> new settings

Inside jobs (launched by `launch`): gateway-job, driver-job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from harnessforge.contracts import JobSpec, JobState
from harnessforge.train.config import OLD_RUN_MAPPING, TrainRun


def driver_job(run: TrainRun) -> JobSpec:
    from harnessforge.evals.jobs import ABCI_CPUS
    d, L = run.driver, run.get("launcher", {})
    py = run.get("python") or sys.executable
    hf_src = run.get("harnessforge_src") or str(Path(__file__).resolve().parents[2])
    res = d.get("resource", "rt_HC")
    opts = {"resource_type": res, **{k: L[k] for k in ("queue", "account") if L.get(k)}}
    return JobSpec(name=f"hftrain-drv-{run.run_id}"[:63],
                   command=[py, "-m", "harnessforge.train", "driver-job", "--run-dir", str(run.dir)],
                   nodes=1, cpus=ABCI_CPUS.get(res, 0), walltime=str(d.get("walltime", "4:00:00")),
                   env={"PYTHONPATH": hf_src, "PYTHONUNBUFFERED": "1"},
                   log=run.paths.logs / "driver.job.log", options=opts)


def _launcher(run: TrainRun):
    from harnessforge.launch import make_launcher
    L = run.get("launcher", {})
    return make_launcher(L.get("kind", "pbs"))


def _jobs(run: TrainRun) -> dict:
    p = run.dir / "jobs.json"
    return json.loads(p.read_text()) if p.exists() else {}


def cmd_render(a) -> int:
    from harnessforge.train.gateway import gateway_job, serve_args
    from harnessforge.train.external import is_external, rollout_cfg
    run = TrainRun.load(a.config)
    addrs = ["<engines of lent pool nodes %s>" % rollout_cfg(run).get("lend_nodes")] \
        if is_external(run) else None
    out = {"run_dir": str(run.dir),
           "rollout": rollout_cfg(run) or {"mode": "managed"},
           "gateway_job": _spec(gateway_job(run)),
           "serve_args": serve_args(run, port=0, local_ckpt="$PBS_LOCALDIR/hf-local", addrs=addrs),
           "driver_job": _spec(driver_job(run)) if run.driver.get("where", "hc") == "hc" else "on the gateway head",
           "train": run["train"]}
    print(json.dumps(out, indent=1))
    return 0


def _spec(s: JobSpec) -> dict:
    return {"name": s.name, "nodes": s.nodes, "walltime": s.walltime, "command": list(s.command),
            "options": s.options, "log": str(s.log)}


def cmd_launch(a) -> int:
    from harnessforge.train.gateway import gateway_job
    run = TrainRun.load(a.config)
    if run.paths.config.exists() and not a.force:
        print(f"error: {run.dir} already has a run (use a new run_id, or --force to relaunch it)",
              file=sys.stderr)
        return 2
    if run.paths.stop_file.exists():
        run.paths.stop_file.rename(run.paths.stop_file.with_name(f"STOP.{int(time.time())}"))
    for f in (run.paths.gateway_file, run.paths.driver_file):
        if f.exists():
            f.rename(f.with_name(f"{f.name}.{int(time.time())}"))
    run.write()
    run.paths.logs.mkdir(parents=True, exist_ok=True)
    L = _launcher(run)
    from harnessforge.train import external as X
    if X.is_external(run):
        X.lend_for_run(run)                    # raises (nodes returned) if the engines don't fit
    try:
        jobs = {"gateway": L.submit(gateway_job(run))}
        if run.driver.get("where", "hc") == "hc" and not a.no_driver:
            jobs["driver"] = L.submit(driver_job(run))
    except Exception:
        if X.is_external(run):
            X.unlend_for_run(run)
        raise
    (run.dir / "jobs.json").write_text(json.dumps({**jobs, "submitted_at": time.time()}, indent=1))
    print(json.dumps({"run_dir": str(run.dir), **jobs}))
    return 0


def cmd_driver(a) -> int:
    """(Re)submit only the driver job of an existing run, against its running gateway (e.g. after
    changing run.json's env/train sections). The previous driver must have ended."""
    run = TrainRun.load(Path(a.run_dir) / "run.json")
    if a.set:
        from harnessforge.train.config import merge
        for kv in a.set:
            key, _, val = kv.partition("=")
            node: dict = {}
            cur = node
            parts = key.split(".")
            for k in parts[:-1]:
                cur = cur.setdefault(k, {})
            cur[parts[-1]] = json.loads(val)
            run.raw = merge(run.raw, node)
        run.write()
    jobs = _jobs(run)
    if run.driver.get("where", "hc") == "gateway":
        if not run.paths.driver_file.exists():
            # held driver (driver.hold): the gateway starts it when START_DRIVER appears
            (run.dir / "START_DRIVER").write_text(str(time.time()))
            print("START_DRIVER written: the gateway starts the held driver")
            return 0
        return _stop_driver_only(run, _launcher(run), jobs, grace=a.grace, restart=True)
    f = run.paths.driver_file
    if f.exists():
        f.rename(f.with_name(f"{f.name}.{int(time.time())}"))
    jobs["driver"] = _launcher(run).submit(driver_job(run))
    (run.dir / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print(json.dumps({"driver": jobs["driver"]}))
    return 0


def cmd_status(a) -> int:
    from harnessforge.train.export import versions
    run = TrainRun.load(Path(a.run_dir) / "run.json")
    P = run.paths
    L = _launcher(run)
    out = {"run_id": run.run_id, "jobs": {}}
    for k, jid in _jobs(run).items():
        if k == "submitted_at":
            continue
        try:
            st = L.status(jid)
            out["jobs"][k] = {"id": jid, "state": st.state.value, "exit": st.exit_code}
        except Exception as e:
            out["jobs"][k] = {"id": jid, "error": str(e)}
    for name, f in (("gateway", P.gateway_file), ("driver", P.driver_file)):
        out[name] = json.loads(f.read_text()) if f.exists() else None
    out["weight_versions"] = (versions(P.delta_dir) or [None])[-1] if P.delta_dir.exists() else None
    out["hf_exports"] = sorted(p.name for p in P.hf_dir.glob("step_*")) if P.hf_dir.exists() else []
    out["stopping"] = P.stop_file.exists()
    from harnessforge.train.external import read_rollout
    out["rollout"] = read_rollout(run) or {"mode": "managed"}
    print(json.dumps(out, indent=1))
    return 0


def cmd_stop(a) -> int:
    run = TrainRun.load(Path(a.run_dir) / "run.json")
    L = _launcher(run)
    if a.grace is None:
        a.grace = float(run.driver.get("stop_timeout", 900)) + 900
    jobs = {k: v for k, v in _jobs(run).items() if k != "submitted_at"}
    if a.driver_only:
        return _stop_driver_only(run, L, jobs, grace=a.grace, restart=False)
    else:
        run.paths.stop_file.write_text(f"stop requested {time.time()}\n")
    deadline = time.time() + a.grace
    while time.time() < deadline:
        if all(L.status(j).state == JobState.DONE for j in jobs.values()):
            print("stopped")
            return _return_nodes(run, L, jobs)
        time.sleep(10)
    for k, j in jobs.items():
        if L.status(j).state != JobState.DONE:
            print(f"cancelling {k} {j}")
            L.cancel(j)
    return _return_nodes(run, L, jobs)


def _stop_driver_only(run: TrainRun, L, jobs: dict, *, grace: float, restart: bool) -> int:
    """Stop the platoon driver, keep the gateway (trainer + engines). A driver on the gateway's
    head node is stopped through DRIVER_STOP and restarted by the gateway (KEEP_GATEWAY) with
    the current run.json and code; a separate driver job is cancelled (resubmit: `train driver`)."""
    from harnessforge.train.driver import DRIVER_STOP, KEEP_GATEWAY
    (run.dir / KEEP_GATEWAY).write_text(str(time.time()))
    if run.driver.get("where", "hc") == "gateway":
        before = (json.loads(run.paths.driver_file.read_text()) if run.paths.driver_file.exists()
                  else {}).get("started_at")
        (run.dir / DRIVER_STOP).write_text(str(time.time()))
        print("asked the head-node driver to stop; the gateway restarts it")
        deadline = time.time() + grace
        while time.time() < deadline:
            d = json.loads(run.paths.driver_file.read_text()) if run.paths.driver_file.exists() else {}
            if d.get("started_at") != before and d.get("status") in ("waiting", "training"):
                print(f"driver restarted ({d.get('status')})")
                return 0
            time.sleep(10)
        print("driver not restarted yet; see logs/driver.events.log and logs/gateway.log", file=sys.stderr)
        return 1
    for j in [v for k, v in jobs.items() if k == "driver"]:
        L.cancel(j)
    return 0


def _return_nodes(run: TrainRun, L, jobs: dict) -> int:
    """External mode: once the gateway job is gone, make sure the lent nodes went back (its
    teardown normally did it; a hard kill skips that)."""
    from harnessforge.train import external as X
    if not X.is_external(run) or X.read_rollout(run).get("status") != "lent":
        return 0
    if "gateway" not in jobs:                  # stop --driver-only: the gateway keeps its engines
        return 0
    gw = jobs.get("gateway")
    deadline = time.time() + 300
    while gw and L.status(gw).state != JobState.DONE and time.time() < deadline:
        time.sleep(10)
    if gw and L.status(gw).state != JobState.DONE:
        print(f"gateway job {gw} still running: not returning its engines", file=sys.stderr)
        return 1
    return 0 if X.unlend_for_run(run) else 1


def cmd_unlend(a) -> int:
    run = TrainRun.load(Path(a.run_dir) / "run.json")
    return _return_nodes(run, _launcher(run), {k: v for k, v in _jobs(run).items() if k == "gateway"})


def cmd_export(a) -> int:
    from harnessforge.train.export import DeltaExporter
    run = TrainRun.load(Path(a.run_dir) / "run.json")
    P = run.paths
    m = run["model"]
    ex = DeltaExporter(base=m["hf_checkpoint"], deltas=P.delta_dir, work=P.export_work, out=P.hf_dir,
                       every=int(a.every if a.every is not None else run.get("export", {}).get("every", 10)),
                       policy_file=P.latest_policy, run_id=run.run_id, base_model=m.get("name"),
                       version_offset=int(run.get("export", {}).get("version_offset", 1)),
                       log=lambda s: print(s, flush=True))
    if a.follow:
        final = P.dir / "export" / "FINAL"
        ex.run(stop=lambda: final.exists() or P.stop_file.exists(), interval=a.interval)
    else:
        ex.poll(final=True)
        if a.version is not None and ex.version != a.version:
            print(f"error: latest published version is {ex.version}, not {a.version}", file=sys.stderr)
            return 1
    return 0


def scan_for(values: list[bytes], roots: list[Path], max_bytes: int = 512 << 20) -> tuple[int, list[str]]:
    """(files scanned, paths of files containing any of `values`). Values are never printed."""
    hits, n = [], 0
    for root in roots:
        files = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
        for f in files:
            try:
                if f.stat().st_size > max_bytes:
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            n += 1
            if any(v in data for v in values):
                hits.append(str(f))
    return n, hits


def cmd_scan_secrets(a) -> int:
    """Check that no W&B credential value (from driver.wandb_env_file / the env) appears in the run
    dir or extra paths; prints counts and file paths only."""
    from harnessforge.train.driver import DEFAULT_WANDB_ENV_FILE, WANDB_ENV_NAMES, read_env_file
    import os
    run = TrainRun.load(Path(a.run_dir) / "run.json")
    vals = read_env_file(run.driver.get("wandb_env_file", DEFAULT_WANDB_ENV_FILE))
    vals.update({k: os.environ[k] for k in WANDB_ENV_NAMES if os.environ.get(k)})
    secrets = [v.encode() for k, v in vals.items() if k == "WANDB_API_KEY" and len(v) >= 16]
    if not secrets:
        print("no WANDB_API_KEY value found to scan for")
        return 2
    n, hits = scan_for(secrets, [run.dir, *map(Path, a.also)])
    print(json.dumps({"scanned_files": n, "files_containing_key": hits}, indent=1))
    return 1 if hits else 0


def cmd_probe(a) -> int:
    """Run harnessforge.train.probe with the platoon venv's python (it has the tinker SDK)."""
    import os
    import subprocess
    run = TrainRun.load(Path(a.run_dir) / "run.json")
    g = json.loads(run.paths.gateway_file.read_text()) if run.paths.gateway_file.exists() else {}
    if g.get("status") != "ready":
        print(f"gateway not ready: {g.get('status')}", file=sys.stderr)
        return 2
    py = str(Path(run.driver.get("platoon", "platoon")).with_name("python"))
    hf_src = run.get("harnessforge_src") or str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": hf_src, "TINKER_API_KEY": "tml-local",
           "no_proxy": f"{g['ip']},127.0.0.1"}
    out = run.paths.logs / f"probe-{a.tokens}.json"
    return subprocess.call([py, "-m", "harnessforge.train.probe", "--url", g["url"],
                            "--model", run["model"]["name"], "--tokens", str(a.tokens),
                            "--steps", str(a.steps), "--out", str(out)], env=env)


def cmd_mapping(a) -> int:
    print("| old setting | old value | platoon / miles-tinker | status |")
    print("|---|---|---|---|")
    for row in OLD_RUN_MAPPING:
        print("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessforge.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "launch"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        if name == "launch":
            s.add_argument("--force", action="store_true")
            s.add_argument("--no-driver", action="store_true")
    for name in ("status", "stop", "export", "gateway-job", "driver-job", "driver", "unlend",
                 "scan-secrets"):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", required=True)
        if name == "stop":
            s.add_argument("--grace", type=float, default=None,
                           help="seconds before the jobs are cancelled (default: driver.stop_timeout + 900)")
            s.add_argument("--driver-only", action="store_true",
                           help="stop the driver (platoon) and keep the gateway serving")
        if name == "scan-secrets":
            s.add_argument("--also", action="append", default=[], help="more files/dirs to scan")
        if name == "driver":
            s.add_argument("--grace", type=float, default=900,
                           help="driver on the gateway head: seconds to wait for its restart")
            s.add_argument("--set", action="append", default=[],
                           help="run.json override, dotted key = JSON value, e.g. env.harbor_system_deps=false")
        if name == "export":
            s.add_argument("--follow", action="store_true")
            s.add_argument("--every", type=int)
            s.add_argument("--interval", type=float, default=30)
            s.add_argument("--version", type=int)
    pr = sub.add_parser("probe")
    pr.add_argument("--run-dir", required=True)
    pr.add_argument("--tokens", type=int, default=131072)
    pr.add_argument("--steps", type=int, default=2)
    sub.add_parser("mapping")
    a = ap.parse_args(argv)
    if a.cmd == "gateway-job":
        from harnessforge.train.gateway import run_gateway_job
        return run_gateway_job(a.run_dir)
    if a.cmd == "driver-job":
        from harnessforge.train.driver import run_driver_job
        return run_driver_job(a.run_dir)
    return {"render": cmd_render, "launch": cmd_launch, "status": cmd_status, "stop": cmd_stop,
            "driver": cmd_driver, "unlend": cmd_unlend, "scan-secrets": cmd_scan_secrets,
            "probe": cmd_probe,
            "export": cmd_export, "mapping": cmd_mapping}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
