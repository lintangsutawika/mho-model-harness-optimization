import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from harnessforge import JobSpec, JobState, JobStatus, ModelPool, PolicyRef, PoolSpec
from harnessforge.launch import LocalLauncher
from harnessforge.pool import (PoolDir, PoolError, SGLangPool, endpoint_of, from_config,
                               load_config)
from harnessforge.pool import http
from harnessforge.pool.__main__ import main as cli
from harnessforge.pool.node import advertise_ip, render_args
from harnessforge.pool.state import read_json, write_json

FAKE = str(Path(__file__).with_name("fake_sglang.py"))
BASE = PolicyRef("base", "/ckpt/base")


class RecordingLauncher:
    def __init__(self):
        self.jobs: dict[str, JobSpec] = {}
        self.cancelled: list[str] = []
        self.states: dict[str, JobStatus] = {}

    def submit(self, spec):
        jid = f"{len(self.jobs) + 1}.pbs1"
        self.jobs[jid] = spec
        return jid

    def status(self, jid):
        return self.states.get(jid, JobStatus(JobState.RUNNING))

    def cancel(self, jid):
        self.cancelled.append(jid)
        self.states[jid] = JobStatus(JobState.DONE)


# ----------------------------------------------------------------------------- pure pieces
def test_render_args():
    assert render_args({"tool-call-parser": "qwen3_coder", "enable_metrics": True,
                        "x": None, "y": False, "--mem-fraction-static": 0.8,
                        "cuda-graph-bs": [1, 2]}) == [
        "--tool-call-parser", "qwen3_coder", "--enable-metrics", "--mem-fraction-static", "0.8",
        "--cuda-graph-bs", "1", "2"]


def test_advertise_ip_modes():
    assert advertise_ip("10.1.2.3") == "10.1.2.3"
    assert advertise_ip("auto")


def test_state_atomic_json_and_endpoint(tmp_path):
    d = PoolDir(tmp_path, "p1")
    with pytest.raises(FileNotFoundError):
        d.pool()
    d.write_pool({"pool_id": "p1", "policy": {"id": "a", "path": "/a"}})
    assert d.pool()["policy"]["id"] == "a" and d.exists()
    assert not list(d.dir.glob(".*tmp"))                   # temp files renamed away
    with pytest.raises(RuntimeError):
        endpoint_of("p1", tmp_path)
    write_json(d.router_file, {"url": "http://10.0.0.1:30000"})
    assert endpoint_of("p1", tmp_path) == "http://10.0.0.1:30000/v1"
    now = time.time()
    write_json(d.node_file(0), {"status": "ready", "updated_at": now, "engines": [
        {"url": "http://a:1", "ready": True}, {"url": "http://a:2", "ready": False}]})
    write_json(d.node_file(1), {"status": "starting", "updated_at": now,
                                "engines": [{"url": "http://b:1", "ready": False}]})
    write_json(d.node_file(2), {"status": "ready", "updated_at": now - 1000,
                                "engines": [{"url": "http://c:1", "ready": True}]})   # stale
    assert d.engine_urls() == ["http://a:1"]
    assert sorted(d.engine_urls(ready_only=False)) == ["http://a:1", "http://a:2",
                                                       "http://b:1", "http://c:1"]
    (d.nodes_dir / "junk.json").write_text("{")
    assert sorted(d.nodes()) == [0, 1, 2]
    assert read_json(d.dir / "missing.json", 5) == 5


def test_node_jobs_and_pool_file(tmp_path):
    L = RecordingLauncher()
    spec = PoolSpec(nodes=3, gpus_per_node=8, tp=2, max_context=8192,
                    engine_args={"tool-call-parser": "qwen3_coder"},
                    launcher={"queue": "R99", "account": "gai", "resource_type": "rt_HF",
                              "walltime": "12:00:00", "kind": "pbs"})
    p = SGLangPool(spec, BASE, pool_id="t", state_root=tmp_path, sif="/s/miles.sif",
                   launcher=L, binds=["/groups"])
    assert isinstance(p, ModelPool)
    p.start(wait=False)
    assert len(L.jobs) == 3
    job = L.jobs["2.pbs1"]
    assert job.name == "hfpool-t-1" and job.gpus == 8 and job.walltime == "12:00:00"
    assert job.options == {"queue": "R99", "account": "gai", "resource_type": "rt_HF"}
    assert job.command[0].endswith(("singularity", "apptainer"))
    assert list(job.command[1:6]) == ["exec", "--nv", "--bind", "/groups",
                                     "/s/miles.sif"]
    assert list(job.command[-4:]) == ["--pool-dir", str(tmp_path / "t"), "--node", "1"]
    assert job.env["HF_HUB_OFFLINE"] == "1" and job.env["PYTHONPATH"].endswith("src")
    cfg = p.dir.pool()
    assert cfg["jobs"] == {"0": "1.pbs1", "1": "2.pbs1", "2": "3.pbs1"}
    assert cfg["policy"] == {"id": "base", "path": "/ckpt/base", "base_model": None}
    # a second start on a live pool is refused; attach sees the same pool
    with pytest.raises(PoolError):
        SGLangPool(spec, BASE, pool_id="t", state_root=tmp_path, launcher=L).start(wait=False)
    q = SGLangPool.attach("t", tmp_path, launcher=L)
    assert q.spec == spec and q.policy() == BASE and q.sif == "/s/miles.sif"
    # stop: cancel every job, move the dir aside
    q.stop()
    assert sorted(L.cancelled) == ["1.pbs1", "2.pbs1", "3.pbs1"]
    assert not (tmp_path / "t").exists()
    assert len(list((tmp_path / ".stopped").iterdir())) == 1


def test_repair_restarts_dead_and_stale_nodes(tmp_path):
    L = RecordingLauncher()
    p = SGLangPool(PoolSpec(nodes=3, gpus_per_node=1), BASE, pool_id="r", state_root=tmp_path,
                   launcher=L)
    p.start(wait=False)
    now = time.time()
    write_json(p.dir.node_file(0), {"status": "ready", "updated_at": now, "engines": []})
    write_json(p.dir.node_file(1), {"status": "ready", "updated_at": now - 9999, "engines": []})
    L.states["3.pbs1"] = JobStatus(JobState.DONE, 1)
    assert p.repair() == [1, 2]
    jobs = p.dir.pool()["jobs"]
    assert jobs == {"0": "1.pbs1", "1": "4.pbs1", "2": "5.pbs1"}
    assert "2.pbs1" in L.cancelled
    assert not p.dir.node_file(1).exists()


# ----------------------------------------------------------------------------- load() vs fakes
class Engines:
    """N fake engines in threads, for load()/health() orchestration."""

    def __init__(self, n, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("fake_sglang", FAKE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.servers = []
        self.urls = []
        for _ in range(n):
            srv = self._engine()
            self.servers.append(srv)
            self.urls.append(f"http://127.0.0.1:{srv.server_address[1]}")

    def _engine(self):
        import argparse
        holder = {}
        orig = ThreadingHTTPServer.serve_forever

        a = argparse.Namespace(host="127.0.0.1", port=0, model_path="/ckpt/base",
                               weight_version="base", served_model_name="policy")

        class Capture(ThreadingHTTPServer):
            def serve_forever(self, *x, **k):
                holder["srv"] = self
                threading.Thread(target=orig, args=(self,), daemon=True).start()

        self.mod.ThreadingHTTPServer = Capture
        self.mod.engine(a, [])
        return holder["srv"]

    def close(self):
        for s in self.servers:
            s.shutdown()


@pytest.fixture
def engines(tmp_path):
    e = Engines(3, tmp_path)
    yield e
    e.close()


def _pool_with_engines(tmp_path, urls):
    p = SGLangPool(PoolSpec(nodes=1, gpus_per_node=len(urls)), BASE, pool_id="l",
                   state_root=tmp_path, launcher=RecordingLauncher())
    p.start(wait=False)
    write_json(p.dir.node_file(0), {"status": "ready", "updated_at": time.time(),
                                    "engines": [{"url": u, "ready": True} for u in urls]})
    return p


def test_load_swaps_every_engine_and_records_policy(tmp_path, engines):
    p = _pool_with_engines(tmp_path, engines.urls)
    new = PolicyRef("step10", "/ckpt/step10", base_model="Qwen/Qwen3.5-4B")
    p.load(new)
    assert p.policy() == new
    for u in engines.urls:
        info = http.get_json(u + "/model_info")
        assert info["weight_version"] == "step10" and info["model_path"] == "/ckpt/step10"
    cfg = p.dir.pool()
    assert cfg["status"] == "ready" and cfg["previous_policy"]["id"] == "base"
    assert cfg["history"][-1]["engines"] == 3 and cfg["history"][-1]["errors"] == {}
    h = p.health()
    assert h["engines_ok"] == 3
    assert {e["weight_version"] for e in h["nodes"][0]["engines"]} == {"step10"}
    assert h["nodes"][0]["engines"][0]["running"] == 0


def test_load_failure_marks_degraded(tmp_path, engines):
    p = _pool_with_engines(tmp_path, engines.urls)
    with pytest.raises(PoolError, match="3/3"):
        p.load(PolicyRef("bad", "/ckpt/BAD"))
    assert p.dir.pool()["status"] == "degraded"


def test_load_needs_engines(tmp_path):
    p = SGLangPool(PoolSpec(nodes=1), BASE, pool_id="e", state_root=tmp_path,
                   launcher=RecordingLauncher())
    p.start(wait=False)
    with pytest.raises(PoolError, match="no ready engines"):
        p.load(PolicyRef("x", "/x"))


# ----------------------------------------------------------------------------- end to end, local
def _chat(endpoint, text):
    return http.post_json(endpoint + "/chat/completions",
                          {"model": "policy", "messages": [{"role": "user", "content": text}]},
                          timeout=30)


def test_local_pool_end_to_end(tmp_path):
    """Two 'nodes' x 2 'GPUs' of fake engines + fake router via LocalLauncher: start, chat
    through the router (both nodes serve), load, node restart, CLI status/load/stop."""
    cfg = {
        "pool_id": "e2e", "state_root": str(tmp_path / "state"),
        "policy": {"id": "base", "path": "/ckpt/base"},
        "spec": {"nodes": 2, "gpus_per_node": 2, "tp": 1,
                 "launcher": {"kind": "local", "engine_port": 42100, "router_port": 42000}},
        "engine_cmd": [sys.executable, FAKE, "engine"],
        "router_cmd": [sys.executable, FAKE, "router"],
        "advertise": "127.0.0.1", "job_env": {"HF_POOL_POLL": "0.2"},
        "startup_timeout": 60,
    }
    (tmp_path / "pool.json").write_text(json.dumps(cfg))
    L = LocalLauncher(tmp_path / "jobs")
    p = from_config(load_config(tmp_path / "pool.json"), launcher=L)
    try:
        p.start(wait=False)
        p.wait_ready(timeout=60, poll=0.2)
        ep = p.endpoint()
        assert ep == endpoint_of("e2e", tmp_path / "state")
        served = {_chat(ep, "hi")["served_by"] for _ in range(8)}
        assert len(served) == 4                                # 2 nodes x 2 engines
        assert _chat(ep, "hi")["choices"][0]["message"]["content"] == "base: hi"

        p.load(PolicyRef("v2", "/ckpt/v2"))
        assert {_chat(ep, "x")["choices"][0]["message"]["content"] for _ in range(8)} \
            == {"v2: x"}

        # node 1 dies: repair brings it back on the ACTIVE policy and the router re-learns it
        L.cancel(p.dir.pool()["jobs"]["1"])
        time.sleep(0.5)
        assert p.repair() == [1]
        p.wait_ready(timeout=60, poll=0.2)
        h = p.health()
        assert h["engines_ok"] == 4 and len(h["router"]["workers"]) == 4
        assert {e["weight_version"] for n in h["nodes"] for e in n["engines"]} == {"v2"}

        spec_arg = ["--spec", str(tmp_path / "pool.json")]
        os.environ["HF_LAUNCHER"] = "local"
        assert cli(["status", *spec_arg]) == 0
        assert cli(["load", *spec_arg, "--id", "v3", "--path", "/ckpt/v3"]) == 0
        assert p.policy().id == "v3"
    finally:
        os.environ.pop("HF_LAUNCHER", None)
        p.stop(wait=20)
    assert not (tmp_path / "state" / "e2e").exists()
