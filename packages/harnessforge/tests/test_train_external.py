import json
import time
from pathlib import Path

import pytest

from harnessforge import JobState, JobStatus, PolicyRef, PoolSpec
from harnessforge.pool import SGLangPool, http
from harnessforge.pool.state import write_json
from harnessforge.train import __main__ as cli
from harnessforge.train import external as X
from harnessforge.train.config import TrainRun
from harnessforge.train.gateway import gateway_job, serve_args

from test_pool import Engines, RecordingLauncher

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"
BASE = PolicyRef("base", "/ckpt/base")


@pytest.fixture
def engines(tmp_path):
    e = Engines(4, tmp_path)
    yield e
    e.close()


def make_pool(tmp_path, urls, nodes=2):
    """A pool with `nodes` nodes of len(urls)//nodes fake engines each, agents that can lend."""
    per = len(urls) // nodes
    p = SGLangPool(PoolSpec(nodes=nodes, gpus_per_node=per), BASE, pool_id="ext",
                   state_root=tmp_path / "state", launcher=RecordingLauncher())
    p.start(wait=False)
    for i in range(nodes):
        write_json(p.dir.node_file(i), {"status": "ready", "updated_at": time.time(), "features": ["lend"],
                                        "engines": [{"url": u, "ready": True, "k": k}
                                                    for k, u in enumerate(urls[i * per:(i + 1) * per])]})
    return p


def make_run(tmp_path, *, nodes=(1,), hf="/ckpt/base", **rollout):
    cfg = json.loads((CONFIGS / "naive_ext.json").read_text())
    cfg.update(extends=str(CONFIGS / "naive.json"), runs_root=str(tmp_path / "runs"))
    cfg["model"] = {"name": "Qwen/Qwen3.5-4B", "hf_checkpoint": hf, "ref_load": "/ckpt/ref",
                    "megatron_model_type": "qwen3.5-4B"}
    cfg["rollout"] = {"mode": "external", "pool_id": "ext", "state_root": str(tmp_path / "state"),
                      "lend_nodes": list(nodes), **rollout}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return TrainRun.load(p), p


def good(url, *, hf_checkpoint, policy_id):
    return []


def test_check_engine_flags():
    def fake(info, mi=None):
        return lambda u: info if u.endswith("/server_info") else (mi or {})
    ok = {"model_path": "/ckpt/base", "tp_size": 1}
    assert X.check_engine("http://e:1", hf_checkpoint="/ckpt/base", policy_id="base",
                          get_json=fake(ok, {"weight_version": "base"})) == []
    assert X.check_engine("http://e:1", hf_checkpoint="/ckpt/base", policy_id="base",
                          get_json=fake(ok, {"weight_version": "default"})) == []
    hub = X.check_engine("http://e:1", hf_checkpoint="/ckpt/base", policy_id=None,
                         get_json=fake({"model_path": "Qwen/Qwen3.5-4B", "tp_size": 1}))
    assert "hub ids" in hub[0]
    other = X.check_engine("http://e:1", hf_checkpoint="/ckpt/base", policy_id="base",
                           get_json=fake({"model_path": "/ckpt/x", "tp_size": 2}, {"weight_version": "s7"}))
    assert len(other) == 3 and "previous trainer" in other[2]

    def dead(u):
        raise http.HTTPError(u, 0, "refused")
    assert "failed" in X.check_engine("http://e:1", hf_checkpoint="/ckpt/base", policy_id=None, get_json=dead)[0]
    assert X.engine_addr("http://10.0.82.8:31600/") == "10.0.82.8:31600"
    with pytest.raises(ValueError):
        X.engine_addr("http://h:1/v1")


def test_lend_for_run_and_serve_args(tmp_path, engines):
    pool = make_pool(tmp_path, engines.urls)
    run, _ = make_run(tmp_path, nodes=[1])
    addrs = X.lend_for_run(run, pool=pool, check=good, log=lambda m: None)
    assert addrs == [X.engine_addr(u) for u in engines.urls[2:]]
    lent = pool.lent()[1]
    assert lent["to"] == run.run_id and lent["on_restart"] == "latest"
    # a dead engine restarts on the BASE checkpoint (disk-delta replays every delta from the base)
    assert json.loads(Path(lent["policy_file"]).read_text())["path"] == "/ckpt/base"
    assert pool.engines() == engines.urls[:2]                       # out of the router / load()
    info = X.read_rollout(run)
    assert info["status"] == "lent" and info["nodes"] == [1] and info["engines"] == addrs
    a = serve_args(run, port=1, local_ckpt="/local/ignored")
    i = a.index("--rollout-external-engine-addrs")
    assert a[i - 1] == "--rollout-external" and a[i + 1:i + 3] == addrs
    assert "--rollout-num-gpus" not in a                              # miles derives it
    local = a[a.index("--update-weight-local-checkpoint-dir") + 1]
    assert run.run_id in local and local.startswith("/dev/shm/")      # run-unique, host-local
    assert local == info["engine_local_dir"]                          # fixed at lend time
    assert a[a.index("--update-weight-transfer-mode") + 1] == "disk-delta"
    assert gateway_job(run).nodes == 1                                # only the trainer node
    # returned: reloaded to the pool policy, back in the router
    assert X.unlend_for_run(run, pool=pool, log=lambda m: None)
    assert pool.lent() == {} and X.read_rollout(run)["status"] == "returned"
    assert X.unlend_for_run(run, pool=pool)                           # idempotent
    # a relaunch of the same run id gets a different engine-local dir
    later = X.engine_local_dir(run, launch="29990101-000000")
    assert later != local and "29990101-000000" in later


def test_lend_refusals(tmp_path, engines):
    pool = make_pool(tmp_path, engines.urls)
    run, _ = make_run(tmp_path, nodes=[1], hf="/ckpt/other")
    with pytest.raises(X.ExternalRolloutError, match="serves"):        # pool policy != hf_checkpoint
        X.lend_for_run(run, pool=pool, check=good, log=lambda m: None)
    assert pool.lent() == {}
    run, _ = make_run(tmp_path, nodes=[1])
    pool.lend([1], to="someone-else")
    with pytest.raises(X.ExternalRolloutError, match="someone-else"):
        X.lend_for_run(run, pool=pool, check=good, log=lambda m: None)
    pool.unlend([1])
    # engine checks fail after lending: the nodes go back, rollout.json says why
    bad = lambda u, **k: [f"{u}: tp_size 2 != 1"]
    with pytest.raises(X.ExternalRolloutError, match="tp_size"):
        X.lend_for_run(run, pool=pool, check=bad, log=lambda m: None)
    assert pool.lent() == {} and X.read_rollout(run)["status"] == "refused"


def test_lend_needs_healthy_engines(tmp_path, engines):
    pool = make_pool(tmp_path, engines.urls)
    run, _ = make_run(tmp_path, nodes=[1])
    engines.servers[3].shutdown()
    engines.servers[3].server_close()
    with pytest.raises(X.ExternalRolloutError, match="1/2 engines healthy"):
        X.lend_for_run(run, pool=pool, check=good, log=lambda m: None)
    assert pool.lent() == {}


def test_failed_unlend_keeps_nodes_lent(tmp_path, engines):
    pool = make_pool(tmp_path, engines.urls)
    run, _ = make_run(tmp_path, nodes=[1])
    X.lend_for_run(run, pool=pool, check=good, log=lambda m: None)
    pool.dir.update_pool(policy={"id": "bad", "path": "/ckpt/BAD", "base_model": None})
    assert not X.unlend_for_run(run, pool=pool, log=lambda m: None)
    assert 1 in pool.lent()
    info = X.read_rollout(run)
    assert info["status"] == "lent" and "unlend_error" in info


class RecLauncher:
    def __init__(self, fail_submit=False):
        self.jobs, self.states, self.fail = [], {}, fail_submit

    def submit(self, spec):
        if self.fail:
            raise RuntimeError("qsub: under maintenance")
        self.jobs.append(spec)
        return f"{len(self.jobs)}.pbs"

    def status(self, jid):
        return self.states.get(jid, JobStatus(JobState.RUNNING))

    def cancel(self, jid):
        self.states[jid] = JobStatus(JobState.DONE)


def test_cli_launch_stop_unlend(tmp_path, engines, monkeypatch):
    pool = make_pool(tmp_path, engines.urls)
    monkeypatch.setattr(X, "check_engine", good)
    run, cfgp = make_run(tmp_path, nodes=[1])
    # a failed submission returns the nodes
    monkeypatch.setattr(cli, "_launcher", lambda r: RecLauncher(fail_submit=True))
    orig = X.lend_for_run
    monkeypatch.setattr(X, "lend_for_run", lambda r, **k: orig(r, check=good, log=lambda m: None))
    with pytest.raises(RuntimeError, match="maintenance"):
        cli.main(["launch", "--config", str(cfgp)])
    assert SGLangPool.attach("ext", tmp_path / "state").lent() == {}
    # launch: lend, then ONE gateway job (driver on its head node)
    L = RecLauncher()
    monkeypatch.setattr(cli, "_launcher", lambda r: L)
    assert cli.main(["launch", "--force", "--config", str(cfgp)]) == 0
    assert [j.nodes for j in L.jobs] == [1]
    assert SGLangPool.attach("ext", tmp_path / "state").lent()[1]["to"] == run.run_id
    # stop --driver-only never returns the gateway's engines
    # (head-node driver: asks it to stop and the gateway to restart it; no restart seen within grace 0)
    assert cli.main(["stop", "--driver-only", "--grace", "0", "--run-dir", str(run.dir)]) == 1
    assert (run.dir / "DRIVER_STOP").exists() and (run.dir / "KEEP_GATEWAY").exists()
    assert 1 in SGLangPool.attach("ext", tmp_path / "state").lent()
    # gateway still running: unlend refuses; once it ended, the nodes go back
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    t = [0.0]
    monkeypatch.setattr(cli.time, "time", lambda: t.__setitem__(0, t[0] + 100) or t[0])
    assert cli.main(["unlend", "--run-dir", str(run.dir)]) == 1
    L.states["1.pbs"] = JobStatus(JobState.DONE)
    assert cli.main(["unlend", "--run-dir", str(run.dir)]) == 0
    assert SGLangPool.attach("ext", tmp_path / "state").lent() == {}


def test_naive_ext_config():
    ext = TrainRun.load(CONFIGS / "naive_ext.json")
    naive = TrainRun.load(CONFIGS / "naive.json")
    assert X.is_external(ext) and not X.is_external(naive)
    assert ext.run_id != naive.run_id and "noKL" in ext.run_id and ext.dir != naive.dir
    assert "no-KL" in ext["labels"]["kl"] and "no-KL" in ext["stats"]["wandb"]["tags"]
    assert X.rollout_cfg(ext)["lend_nodes"] == [1, 2, 3]
    assert X.rollout_cfg(ext)["pool_spec"].endswith("pool-n4hf72.json")
    assert ext.gateway["miles_src"].endswith("miles-tinker-ext-run")
    assert gateway_job(ext).nodes == 1 and gateway_job(naive).nodes == 4
    # same training recipe as naive.json
    assert ext["train"] == naive["train"] and ext["env"] == naive["env"]
