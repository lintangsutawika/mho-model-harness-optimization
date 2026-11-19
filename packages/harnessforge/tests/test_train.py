import json
from pathlib import Path

import pytest

from harnessforge.train import __main__ as cli
from harnessforge.train.config import OLD_RUN_MAPPING, TrainRun, merge
from harnessforge.train.gateway import container_argv, gateway_job, serve_args, tinker_port
from harnessforge.train.platoon import (TRAIN_DEFAULTS, environment_entry, materialize, render_harness,
                                        select_tasks, train_config)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"


def make_dataset(root, n=5, dockerfile=True):
    for i in range(n):
        t = root / f"task_{i:03d}"
        (t / "environment").mkdir(parents=True)
        (t / "task.toml").write_text('[agent]\ntimeout_sec = 600.0\n[environment]\n')
        if dockerfile:
            (t / "environment" / "Dockerfile").write_text("FROM ghcr.io/x/img:1\nRUN mkdir -p /workspace\n")
    (root / "seed.jsonl").write_text("")
    return root


def make_candidate(root):
    (root / "src" / "micro_swe").mkdir(parents=True)
    (root / "src" / "micro_swe" / "__main__.py").write_text("def main(argv=None):\n    pass\n")
    return root


@pytest.fixture
def run(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    cand = make_candidate(tmp_path / "cand")
    cfg = json.loads((CONFIGS / "smoke.json").read_text())
    cfg.update(extends=str(CONFIGS / "abci-base.json"), runs_root=str(tmp_path / "runs"),
               harnessforge_src=str(tmp_path / "hfsrc"))
    cfg["data"] = {"dataset": str(ds), "tasks": None, "limit": 3, "seed": 1}
    cfg["harness"] = {"dir": str(cand)}
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps(cfg))
    return TrainRun.load(p)


def test_shipped_configs_resolve():
    smoke = TrainRun.load(CONFIGS / "smoke.json")
    naive = TrainRun.load(CONFIGS / "naive.json")
    pr = TrainRun.load(CONFIGS / "pass_rate.json")
    assert smoke.gateway["rollout_nodes"] == 1 and naive.gateway["rollout_nodes"] == 3
    # the comparison run's batch shape = the old run's (8 prompts x 32, lr 1e-6, 200 steps, staleness 4)
    t = naive["train"]
    assert (t["batch_size"], t["workflow_config"]["group_size"], t["max_training_steps"],
            t["max_staleness"], t["task_sampler"]) == (8, 32, 200, 4, "uniform")
    assert t["optimizer"] == {"learning_rate": 1e-6, "beta1": 0.9, "beta2": 0.98, "eps": 1e-8,
                              "weight_decay": 0.1, "grad_clip_norm": 1.0}
    assert t["workflow_config"]["rollout_config"]["inference_params"]["max_completion_tokens"] == 16384
    assert naive["env"]["agent_timeout_multiplier"] == 6.0                # 600 s x 6 = old 3600 s
    assert pr["train"]["task_sampler"] == "pass_rate" and pr["train"]["batch_size"] == 8
    assert pr.run_id != naive.run_id and naive.driver["where"] == "gateway"
    assert all(r.run_id.startswith("platoon-") for r in (smoke, naive, pr))
    assert str(naive.dir).startswith("/groups/")                          # never /home
    assert "_comment" not in naive.raw
    assert len(OLD_RUN_MAPPING) > 20 and all(len(r) == 4 for r in OLD_RUN_MAPPING)


def test_merge():
    assert merge({"a": {"b": 1, "c": 2}, "l": [1]}, {"a": {"c": 3}, "l": [2]}) == {"a": {"b": 1, "c": 3}, "l": [2]}


def test_serve_args_disk_delta_and_topology():
    naive = TrainRun.load(CONFIGS / "naive.json")
    a = serve_args(naive, port=21234, local_ckpt="/local/x/hf-local")
    def val(flag):
        return a[a.index(flag) + 1]
    assert val("--tinker-train-mode") == "full" and val("--tinker-server-port") == "21234"
    assert val("--rollout-num-gpus") == "24" and val("--actor-num-gpus-per-node") == "8"
    assert val("--update-weight-transfer-mode") == "disk-delta"
    assert val("--update-weight-disk-dir").endswith("/weights/delta")
    assert val("--update-weight-local-checkpoint-dir") == "/local/x/hf-local"
    assert val("--rollout-temperature") == "1.0" and val("--context-parallel-size") == "1"
    assert val("--sglang-context-length") == "65536"
    # miles-tinker full mode refuses these
    for bad in ("--use-precision-aware-optimizer", "--offload-train", "--calculate-per-token-loss",
                "--colocate", "--lora-rank"):
        assert bad not in a
    naive.raw["gateway"]["weight_sync"] = "broadcast"
    assert "--update-weight-transfer-mode" not in serve_args(naive, port=1, local_ckpt="/l")
    naive.raw["gateway"]["weight_sync"] = "nccl?"
    with pytest.raises(ValueError):
        serve_args(naive, port=1, local_ckpt="/l")


def test_gateway_job_and_container():
    smoke = TrainRun.load(CONFIGS / "smoke.json")
    job = gateway_job(smoke)
    assert job.nodes == 2 and job.options["resource_type"] == "rt_HF" and job.walltime == "2:30:00"
    assert list(job.command[1:4]) == ["-m", "harnessforge.train", "gateway-job"]
    assert tinker_port(smoke, "2991193.pbs1") == 20000 + 2991193 % 20000
    argv = container_argv(smoke, node_local="/local/j", ifname="ib0", master="10.0.0.1")
    assert argv[:3] == ["singularity", "exec", "--nv"] and argv[-1] == smoke.gateway["sif"]
    binds = [argv[i + 1] for i, x in enumerate(argv) if x == "--bind"]
    assert f"{smoke.gateway['miles_src']}:/root/miles" in binds and "/groups" in binds
    envs = dict(argv[i + 1].split("=", 1) for i, x in enumerate(argv) if x == "--env")
    assert envs["NCCL_SOCKET_IFNAME"] == "ib0" and envs["MASTER_ADDR"] == "10.0.0.1"
    assert envs["PYTHONPATH"].startswith("/root/miles:/root/Megatron-LM:")
    assert envs["RAY_AUTH_MODE"] == "token" and "ray_token" in envs["RAY_AUTH_TOKEN_PATH"]


def test_platoon_config(run):
    tasks = select_tasks(run)
    assert len(tasks) == 3 and tasks == select_tasks(run)                  # seeded shuffle
    prep = materialize(run, tasks)
    assert "mkdir -p /workspace" in prep[0] and len(prep) == 1      # smoke: replay, no system deps
    run.raw["env"]["harbor_system_deps"] = True
    assert len(materialize(run, tasks)) == 2                        # + harbor's apt step
    for t in tasks:
        assert 'docker_image = "ghcr.io/x/img:1"' in (run.paths.dataset_dir / t / "task.toml").read_text()
    spec = render_harness(run, prep)
    cfg = train_config(run, tasks, "http://10.0.0.1:21234", spec)
    assert cfg["tinker_base_url"] == "http://10.0.0.1:21234"
    assert cfg["log_path"] == str(run.paths.platoon_dir)
    tr = cfg["train"]
    assert tr["model_name"] == "Qwen/Qwen3.5-4B" and tr["renderer_name"] == TRAIN_DEFAULTS["renderer_name"]
    assert tr["batch_size"] == 2 and tr["workflow_config"]["group_size"] == 4
    rc = tr["workflow_config"]["rollout_config"]
    assert rc["inference_params"] == {"temperature": 1.0, "top_p": 1.0, "max_completion_tokens": 4096}
    assert rc["sampling_precedence"] == "config" and rc["max_steps"] == 30
    assert tr["loss_fn"] == "ppo" and tr["loss_fn_config"]["clip_high_threshold"] == 1.28
    env = cfg["environments"][0]
    sk = env["service_kwargs"]
    assert sk["splits"] == {"train": tasks} and sk["agent_user"] == "ignore"
    assert sk["provider"].endswith("PodSingularityProvider")
    assert sk["provider_options"]["keep_agent_processes"] is True
    assert env["on_harness_exit"] == {"timeout": "grade", "crash": "grade"}
    assert cfg["stats"]["experiment_name"] == run.run_id and cfg["stats"]["wandb"]["mode"] == "disabled"


def test_select_tasks_from_file_and_exclude(run, tmp_path):
    f = tmp_path / "tasks.json"
    f.write_text(json.dumps({"train": ["task_000", "task_001", "task_002"]}))
    run.raw["data"].update(tasks=str(f), exclude=["task_001"], shuffle=False, limit=None)
    assert select_tasks(run) == ["task_000", "task_002"]
    run.raw["data"]["exclude"] = ["task_000", "task_001", "task_002"]
    with pytest.raises(ValueError):
        select_tasks(run)


class RecLauncher:
    def __init__(self):
        self.jobs = []

    def submit(self, spec):
        self.jobs.append(spec)
        return f"{len(self.jobs)}.pbs"


def test_launch_writes_run_and_submits(run, tmp_path, monkeypatch, capsys):
    L = RecLauncher()
    monkeypatch.setattr(cli, "_launcher", lambda r: L)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(run.raw))
    assert cli.main(["launch", "--config", str(p)]) == 0
    assert [j.name for j in L.jobs] == [f"hftrain-gw-{run.run_id}", f"hftrain-drv-{run.run_id}"]
    assert L.jobs[1].options["resource_type"] == "rt_HC"
    assert json.loads((run.dir / "jobs.json").read_text())["gateway"] == "1.pbs"
    assert TrainRun.load(run.paths.config).run_id == run.run_id
    assert cli.main(["launch", "--config", str(p)]) == 2                    # refuses to clobber
    # gateway placement: no separate driver job
    run.raw["driver"]["where"] = "gateway"
    p.write_text(json.dumps(run.raw))
    L.jobs.clear()
    assert cli.main(["launch", "--config", str(p), "--force"]) == 0
    assert len(L.jobs) == 1


def test_mapping_table(capsys):
    assert cli.main(["mapping"]) == 0
    out = capsys.readouterr().out
    assert "--kl-loss-coef" in out and "differs" in out
