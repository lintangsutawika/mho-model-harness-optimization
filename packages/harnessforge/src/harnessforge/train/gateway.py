"""The miles-tinker gateway job: a Tinker API server for full fine-tuning on 1 trainer node plus R
rollout nodes, as ONE batch job (one Ray cluster over all its nodes).

Two rollout modes:

* managed (default): Ray places the Megatron actor on the head node (8 GPUs) and 8*R TP1 SGLang
  engines (plus miles' router) on the other nodes;
* external (`rollout.mode: external`, harnessforge.train.external): a 1-node job; miles attaches
  pool engines lent to the run (`--rollout-external --rollout-external-engine-addrs`, needs
  miles-tinker `external-engines-fixes`), and the job's teardown returns them to the pool.

Weight sync is disk-delta through the run dir (or NCCL broadcast).

    job = gateway_job(run)               # JobSpec: nodes = 1 + R, rt_HF
    launcher.submit(job)

Inside the job, on the head node (host python, stdlib only):

    python -m harnessforge.train gateway-job --run-dir <run>

  1. starts the head container: `python -m harnessforge.train.gateway inner-head` (ray head, wait
     for every node's GPUs, then `ray job submit ... serve_tinker.py <args>`), and through pbsdsh
     one worker container per other node (`inner-worker`: ray start --block);
  2. waits for `GET /api/v1/healthz`, writes <run>/gateway.json {url, ...};
  3. optionally runs the platoon driver on the head node (driver.where == "gateway");
  4. tears down on <run>/STOP, on the driver's exit, or when the server dies.

Pure builders (`serve_args`, `container_argv`, `gateway_job`) are unit-tested; the rest runs on
the cluster.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from harnessforge.contracts import JobSpec
from harnessforge.train.config import TrainRun

RAY_PORT = 6379
DASH_PORT = 8265
HEALTH = "/api/v1/healthz"


# --------------------------------------------------------------------------- pure builders
def tinker_port(run: TrainRun, job_id: str = "") -> int:
    p = int(run.gateway.get("port") or 0)
    if p:
        return p
    digits = "".join(c for c in job_id.split(".")[0] if c.isdigit()) or "0"
    return 20000 + int(digits) % 20000


def rollout_nodes(run: TrainRun) -> int:
    """Nodes the gateway job runs miles-managed engines on (0 in external mode)."""
    from harnessforge.train.external import is_external
    return 0 if is_external(run) else int(run.gateway.get("rollout_nodes", 1))


def serve_args(run: TrainRun, *, port: int, local_ckpt: str,
               addrs: list[str] | None = None) -> list[str]:
    """serve_tinker.py's miles/Megatron/tinker flags (the model architecture flags come from
    miles' scripts/models/<megatron_model_type>.py, added in the container)."""
    g, m = run.gateway, run["model"]
    from harnessforge.train.external import engine_addrs, engine_local_dir, is_external
    external = is_external(run)
    R = rollout_nodes(run)
    if external:
        # attached pool engines: the host-local checkpoint dir is on the ENGINE hosts, run-unique
        local_ckpt = engine_local_dir(run)
    # device: miles renders SGLang's argv on a GPU-less manager actor, where this SGLang build
    # auto-detects device=None and the argv round-trip check fails; name it explicitly
    sg = {"device": "cuda", "mem-fraction-static": 0.85, "context-length": 65536,
          "chunked-prefill-size": 8192, "max-running-requests": 256, **(g.get("sglang") or {})}
    args = [
        "--hf-checkpoint", m["hf_checkpoint"], "--ref-load", m["ref_load"],
        "--tinker-train-mode", "full", "--tinker-server-port", str(port),
        "--tinker-checkpoint-root", str(run.paths.tinker_root),
        "--tinker-base-model", m["name"],
        "--tinker-lease-timeout-s", str(int(g.get("lease_timeout_s", 3600))),
        "--tinker-max-concurrent-samples", str(int(g.get("max_concurrent_samples", 512))),
        # longest training datum; miles-tinker's default (32768) is below our 64k context
        # (patches/miles-tinker-max-tokens-per-datum.patch adds the flag)
        "--tinker-max-tokens-per-datum", str(int(g.get("max_tokens_per_datum",
                                                       run["train"].get("context_window_length", 32768)))),
        # initial optimizer config only: platoon's AdamParams arrive with every optim_step
        "--optimizer", "adam", "--lr", str(run["train"].get("optimizer", {}).get("learning_rate", 1e-6)),
        "--tensor-model-parallel-size", str(int(g.get("tp", 2))),
        *(["--sequence-parallel"] if int(g.get("tp", 2)) > 1 else []),
        "--pipeline-model-parallel-size", "1", "--context-parallel-size", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", str(int(g.get("max_tokens_per_gpu", 65536))),
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--rollout-temperature", "1.0",
        "--rollout-num-gpus-per-engine", "1",
        "--pause-generation-mode", "retract",
        "--update-weight-buffer-size", str(int(g.get("update_weight_buffer_size", 1 << 30))),
        "--num-gpus-per-node", "8", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "8",
        # external: miles derives --rollout-num-gpus from the engine count (and refuses another)
        *([] if external else ["--rollout-num-gpus", str(8 * R)]),
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
    ]
    if g.get("log_probs_chunk_size"):
        args += ["--log-probs-chunk-size", str(int(g["log_probs_chunk_size"]))]
    for k, v in sg.items():
        args += [f"--sglang-{k}", str(v)]
    if external:
        args += ["--rollout-external", "--rollout-external-engine-addrs",
                 *(engine_addrs(run) if addrs is None else addrs)]
    mode = g.get("weight_sync", "disk-delta")
    if mode == "disk-delta":
        args += ["--update-weight-transfer-mode", "disk-delta",
                 "--update-weight-disk-dir", str(run.paths.delta_dir),
                 "--update-weight-local-checkpoint-dir", local_ckpt,
                 "--update-weight-delta-encoding", g.get("delta_encoding", "xor")]
    elif mode != "broadcast":
        raise ValueError(f"gateway.weight_sync must be disk-delta or broadcast, not {mode!r}")
    return args + [str(a) for a in g.get("extra_args", [])]


def hf_cache_binds() -> list[str]:
    """~/.cache/huggingface may be a symlink off /home; the container only sees bound paths."""
    p = Path.home() / ".cache" / "huggingface"
    real = p.resolve() if p.exists() else p
    return [] if str(real).startswith("/home/") else [f"{real}:{real}"]


def container_argv(run: TrainRun, *, node_local: str, ifname: str = "", master: str = "",
                   extra_env: dict[str, str] | None = None) -> list[str]:
    """`singularity exec ... <sif>` (append the command)."""
    g = run.gateway
    hf_src = run.get("harnessforge_src") or str(Path(__file__).resolve().parents[2])
    binds = [f"{g['miles_src']}:/root/miles", g["extra_site"], str(run.dir), node_local,
             *hf_cache_binds(), hf_src, *g.get("binds", ["/groups"])]
    env = {
        "PYTHONPATH": ":".join(["/root/miles", "/root/Megatron-LM", g["extra_site"], hf_src]),
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TMPDIR": f"{node_local}/tmp", "RAY_TMPDIR": f"{node_local}/ray",
        "TRITON_CACHE_DIR": f"{node_local}/triton", "TORCHINDUCTOR_CACHE_DIR": f"{node_local}/inductor",
        "FLASHINFER_WORKSPACE_DIR": f"{node_local}/flashinfer", "XDG_CACHE_HOME": f"{node_local}/xdg",
        "RAY_memory_monitor_refresh_ms": "0", "RAY_raylet_startup_timeout": "300",
        "RAY_AUTH_MODE": "token", "RAY_AUTH_TOKEN_PATH": str(run.paths.secrets / "ray_token"),
        "MILES_SCRIPT_EXTERNAL_RAY": "1", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        # ABCI jobs set CUDA_VISIBLE_DEVICES to GPU UUIDs; miles parses Ray's GPU ids as ints
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(8))),
        "WANDB_MODE": "disabled",
        **({"NCCL_SOCKET_IFNAME": ifname, "GLOO_SOCKET_IFNAME": ifname} if ifname else {}),
        **({"MASTER_ADDR": master} if master else {}),
        **{k: str(v) for k, v in (g.get("env") or {}).items()},
        **(extra_env or {}),
    }
    argv = ["singularity", "exec", "--nv"]
    for b in dict.fromkeys(binds):
        argv += ["--bind", b]
    for k, v in env.items():
        argv += ["--env", f"{k}={v}"]
    return argv + [g["sif"]]


def gateway_job(run: TrainRun) -> JobSpec:
    g = run.gateway
    L = run.get("launcher", {})
    py = run.get("python") or sys.executable
    hf_src = run.get("harnessforge_src") or str(Path(__file__).resolve().parents[2])
    opts = {"resource_type": "rt_HF", **{k: L[k] for k in ("queue", "account") if L.get(k)}}
    return JobSpec(
        name=f"hftrain-gw-{run.run_id}"[:63],
        command=[py, "-m", "harnessforge.train", "gateway-job", "--run-dir", str(run.dir)],
        nodes=1 + rollout_nodes(run), gpus=8,
        walltime=str(g.get("walltime", "4:00:00")),
        env={"PYTHONPATH": hf_src, "PYTHONUNBUFFERED": "1"},
        log=run.paths.logs / "gateway.job.log", options=opts)


# --------------------------------------------------------------------------- cluster runtime
def _log(path: Path, msg: str) -> None:
    line = f"[gateway {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def node_ip_ifname() -> tuple[str, str]:
    """This node's 10.0.x.y address and interface (ABCI compute network)."""
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True, text=True,
                             timeout=10).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > 3 and parts[3].startswith("10.0."):
                return parts[3].split("/")[0], parts[1]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return socket.gethostbyname(socket.gethostname()), ""


def node_local_dir(min_free_gb: float = 40.0) -> str:
    """Node-local scratch for this job: $PBS_LOCALDIR, else /local/<jobid> (ABCI does not always
    export the variable), else the roomiest of $TMPDIR and /tmp/<user>-<job>. The disk-delta
    local checkpoint (~9 GB for 4B) and Ray's object store live here."""
    import shutil
    job = os.environ.get("PBS_JOBID", "local")
    cands = [os.environ.get("PBS_LOCALDIR", ""), f"/local/{job}"]
    for c in cands:
        if c and os.path.isdir(c) and os.access(c, os.W_OK):
            return c
    fallback = [os.environ.get("TMPDIR", ""), f"/tmp/{os.environ.get('USER', 'u')}-{job}"]
    best, free = fallback[-1], -1.0
    for c in fallback:
        if not c:
            continue
        try:
            os.makedirs(c, exist_ok=True)
            f = shutil.disk_usage(c).free / 2**30
        except OSError:
            continue
        if f > free:
            best, free = c, f
    if free < min_free_gb:
        print(f"WARNING: node-local dir {best} has only {free:.0f} GB free", flush=True)
    return best


def job_hosts() -> list[tuple[str, int]]:
    """(host, pbsdsh vnode index) of each node of the job, head first."""
    nf = os.environ.get("PBS_NODEFILE")
    if not nf:
        return [(socket.gethostname(), 0)]
    hosts: dict[str, int] = {}
    for i, h in enumerate(Path(nf).read_text().split()):
        hosts.setdefault(h, i)
    return list(hosts.items())


WARMUP_FAILED = "backend warmup failed"
WARMUP_DONE = ("warmup done", "backend warmup ok", "Tinker gateway ready")


def warmup_failed(head_log: Path) -> str | None:
    """serve_tinker answers /healthz even when its backend warmup (first weight sync) failed;
    the head log says so."""
    try:
        with open(head_log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - (8 << 20)))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    i = tail.rfind(WARMUP_FAILED)
    return tail[i:i + 300].splitlines()[0] if i >= 0 else None


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + HEALTH, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def _pbsdsh() -> str:
    for p in (os.path.join(os.environ.get("PBS_EXEC", ""), "bin", "pbsdsh"), "/opt/pbs/bin/pbsdsh"):
        if p and os.path.exists(p):
            return p
    return "pbsdsh"


def run_gateway_job(run_dir: str | Path, *, health_timeout: float = 3600) -> int:
    run = TrainRun.load(Path(run_dir) / "run.json")
    P = run.paths
    P.logs.mkdir(parents=True, exist_ok=True)
    log = P.logs / "gateway.log"
    job_id = os.environ.get("PBS_JOBID", "local")
    node_local = node_local_dir()
    os.makedirs(f"{node_local}/tmp", exist_ok=True)
    ip, ifname = node_ip_ifname()
    nodes = job_hosts()
    hosts = [h for h, _ in nodes]
    want = 1 + rollout_nodes(run)
    if len(hosts) != want:
        _log(log, f"ERROR: job has {len(hosts)} nodes {hosts}, config wants {want}")
        return 2
    port = tinker_port(run, job_id)
    url = f"http://{ip}:{port}"
    P.secrets.mkdir(mode=0o700, parents=True, exist_ok=True)
    tok = P.secrets / "ray_token"
    if not tok.exists():
        fd = os.open(tok, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, os.urandom(32).hex().encode())
        os.close(fd)
    if run.gateway.get("weight_sync", "disk-delta") == "disk-delta":
        P.delta_dir.mkdir(parents=True, exist_ok=True)
    P.tinker_root.mkdir(parents=True, exist_ok=True)
    local_ckpt = f"{node_local}/hf-local"
    _write_json(P.gateway_file, {"status": "starting", "job_id": job_id, "host": hosts[0], "ip": ip,
                                 "port": port, "url": url, "hosts": hosts, "started_at": time.time()})
    _log(log, f"job {job_id} hosts {hosts} head {ip} ({ifname}) port {port} node-local {node_local}")
    cont = container_argv(run, node_local=node_local, ifname=ifname, master=ip)
    py = "python3"
    head_cmd = cont + [py, "-m", "harnessforge.train.gateway", "inner-head", "--run-dir", str(run.dir),
                       "--ip", ip, "--port", str(port), "--gpus", str(8 * want),
                       "--local-ckpt", local_ckpt]
    procs: list[subprocess.Popen] = []
    with open(P.logs / "gateway.head.log", "ab") as out:
        head = subprocess.Popen(head_cmd, stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    procs.append(head)
    for k, (h, vnode) in enumerate(nodes[1:], start=1):
        # pbsdsh gives a bare environment: everything is in the argv; node-local dirs first
        wcmd = (f"mkdir -p {shlex.quote(node_local)}/tmp && exec env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin "
                f"HOME={shlex.quote(str(Path.home()))} NVIDIA_VISIBLE_DEVICES=all "
                + shlex.join(cont + [py, "-m", "harnessforge.train.gateway", "inner-worker",
                                     "--head", ip])
                + f" >> {shlex.quote(str(P.logs / f'gateway.worker{k}.log'))} 2>&1")
        procs.append(subprocess.Popen([_pbsdsh(), "-n", str(vnode), "--", "/bin/bash", "-c", wcmd],
                                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, start_new_session=True))
    driver: subprocess.Popen | None = None
    rc = 0
    stopping = {"flag": False}

    def _term(*_: Any) -> None:
        stopping["flag"] = True
    signal.signal(signal.SIGTERM, _term)
    try:
        t0 = time.time()
        while not _healthy(url):
            if head.poll() is not None:
                _log(log, f"ERROR: head container exited rc={head.returncode} before the server "
                          f"answered; see logs/gateway.head.log")
                rc = 1
                return rc
            if time.time() - t0 > health_timeout or P.stop_file.exists() or stopping["flag"]:
                _log(log, "ERROR: gateway not healthy (timeout or STOP)")
                rc = 1
                return rc
            time.sleep(10)
        # the server answers before (and regardless of) its warmup weight sync
        time.sleep(30)
        if (why := warmup_failed(P.logs / "gateway.head.log")):
            _log(log, f"ERROR: {why}")
            rc = 1
            return rc
        ready_s = time.time() - t0
        _write_json(P.gateway_file, {"status": "ready", "job_id": job_id, "host": hosts[0], "ip": ip,
                                     "port": port, "url": url, "hosts": hosts,
                                     "ready_after_s": round(ready_s), "ready_at": time.time()})
        _log(log, f"gateway healthy at {url} after {ready_s:.0f}s")
        def start_driver() -> subprocess.Popen:
            # a fresh process: it reads run.json and this package's code as they are now
            r = TrainRun.load(P.config)
            hf_src = r.get("harnessforge_src") or str(Path(__file__).resolve().parents[2])
            env = {**os.environ, "PYTHONPATH": hf_src}
            with open(P.logs / "driver.log", "ab") as out:
                return subprocess.Popen([r.get("python") or sys.executable, "-m", "harnessforge.train",
                                         "driver-job", "--run-dir", str(run.dir)], env=env, stdout=out,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)

        if run.driver.get("where") == "gateway":
            driver = start_driver()
        while True:
            if P.stop_file.exists() or stopping["flag"]:
                _log(log, "STOP requested")
                break
            if head.poll() is not None:
                _log(log, f"ERROR: head container exited rc={head.returncode}")
                rc = 1
                break
            if driver is not None and driver.poll() is not None:
                _log(log, f"driver exited rc={driver.returncode}")
                if driver_exit_action(P.dir) == "restart":
                    _log(log, "KEEP_GATEWAY: restarting the driver; trainer and engines untouched")
                    driver = start_driver()
                    continue
                rc = driver.returncode
                break
            time.sleep(15)
    finally:
        if driver is not None and driver.poll() is None:
            driver.send_signal(signal.SIGTERM)
            try:
                driver.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(driver.pid, signal.SIGKILL)
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(10)
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        collect_ray_logs(Path(node_local) / "ray", P.logs / f"ray-head-{job_id.split('.')[0]}")
        from harnessforge.train.external import is_external, unlend_for_run
        if is_external(run):
            # the server (miles) is down: hand the lent engines back to the pool
            try:
                unlend_for_run(run, log=lambda m: _log(log, m))
            except Exception as e:                       # never lose the job's teardown over it
                _log(log, f"ERROR: unlend failed ({type(e).__name__}: {e}); "
                          f"run `python -m harnessforge.train unlend --run-dir {run.dir}`")
        g = json.loads(P.gateway_file.read_text()) if P.gateway_file.exists() else {}
        g.update(status="stopped", stopped_at=time.time(), returncode=rc)
        _write_json(P.gateway_file, g)
        _log(log, f"gateway job done rc={rc} (PBS reaps what is left on the job's nodes)")
    return rc


def driver_exit_action(run_dir: Path) -> str:
    """What the gateway does when its head-node driver exits: "restart" when a driver-only stop
    asked for it (KEEP_GATEWAY, consumed here), else "stop" (the run is over)."""
    keep = Path(run_dir) / "KEEP_GATEWAY"
    if keep.exists():
        keep.rename(keep.with_name(f"KEEP_GATEWAY.{int(time.time())}"))
        return "restart"
    return "stop"


def collect_ray_logs(ray_tmp: Path, dest: Path, max_bytes: int = 50 << 20) -> None:
    """Copy the head node's Ray session logs (node-local, gone with the job) into the run dir:
    worker stderr is where a crashed actor's last words are."""
    import shutil
    try:
        logs = sorted(ray_tmp.glob("session_*/logs"))
        if not logs:
            return
        dest.mkdir(parents=True, exist_ok=True)
        for f in logs[-1].iterdir():
            if f.is_file() and f.suffix in (".err", ".out", ".log") and 0 < f.stat().st_size <= max_bytes:
                shutil.copy2(f, dest / f.name)
    except OSError:
        pass


# --------------------------------------------------------------------------- inside the container
def _ray(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["ray", *args], check=check, text=True, capture_output=not check)


def inner_worker(head: str) -> int:
    ip, _ = node_ip_ifname()
    for attempt in range(60):
        r = subprocess.run(["ray", "start", f"--address={head}:{RAY_PORT}", f"--node-ip-address={ip}",
                            "--num-gpus", "8", "--temp-dir", os.environ.get("RAY_TMPDIR", "/tmp/ray"),
                            "--disable-usage-stats", "--block"])
        if r.returncode == 0:
            return 0
        print(f"worker join retry {attempt}", flush=True)
        time.sleep(10)
    return 1


def inner_head(run_dir: str, ip: str, port: int, gpus: int, local_ckpt: str) -> int:
    run = TrainRun.load(Path(run_dir) / "run.json")
    _ray("start", "--head", f"--node-ip-address={ip}", f"--port={RAY_PORT}", "--dashboard-host=127.0.0.1",
         f"--dashboard-port={DASH_PORT}", "--num-gpus", "8",
         "--temp-dir", os.environ.get("RAY_TMPDIR", "/tmp/ray"), "--disable-usage-stats")
    import ray  # noqa: E402  (in the container)
    ray.init(address=f"{ip}:{RAY_PORT}", logging_level="ERROR")
    t0 = time.time()
    while ray.cluster_resources().get("GPU", 0) < gpus:
        if time.time() - t0 > 900:
            print(f"only {ray.cluster_resources().get('GPU', 0)} of {gpus} GPUs joined", flush=True)
            return 1
        time.sleep(10)
    ray.shutdown()
    print(f"ray cluster up: {gpus} GPUs after {time.time() - t0:.0f}s", flush=True)
    from miles.utils.external_utils.command_utils import shell_safe_model_args
    model_args = shell_safe_model_args(run["model"].get("megatron_model_type", "qwen3.5-4B"))
    args = serve_args(run, port=port, local_ckpt=local_ckpt)
    env_vars = {
        "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "1",
        "no_proxy": f"127.0.0.1,{ip}", "MASTER_ADDR": ip, "PYTHONFAULTHANDLER": "1",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        **{k: os.environ[k] for k in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_DEBUG",
                                      "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TMPDIR",
                                      "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                                      "FLASHINFER_WORKSPACE_DIR", "XDG_CACHE_HOME", "WANDB_MODE")
           if k in os.environ},
    }
    cmd = (f"ray job submit --address=http://127.0.0.1:{DASH_PORT} "
           f"--runtime-env-json={shlex.quote(json.dumps({'env_vars': env_vars}))} "
           f"-- python3 /root/miles/serve_tinker.py {model_args} {shlex.join(args)}")
    print("EXEC: " + cmd.split(" -- ", 1)[1][:4000], flush=True)
    return subprocess.run(cmd, shell=True, cwd="/root/miles").returncode


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m harnessforge.train.gateway")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("inner-head")
    h.add_argument("--run-dir", required=True)
    h.add_argument("--ip", required=True)
    h.add_argument("--port", type=int, required=True)
    h.add_argument("--gpus", type=int, required=True)
    h.add_argument("--local-ckpt", required=True)
    w = sub.add_parser("inner-worker")
    w.add_argument("--head", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "inner-head":
        return inner_head(a.run_dir, a.ip, a.port, a.gpus, a.local_ckpt)
    return inner_worker(a.head)


if __name__ == "__main__":
    sys.exit(main())
