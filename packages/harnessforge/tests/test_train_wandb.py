import json
from pathlib import Path

from harnessforge.train import driver as D
from harnessforge.train.config import TrainRun
from harnessforge.train.platoon import train_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"
KEY = "0123456789abcdef" * 2 + "fakekey00"          # not a real key


def test_read_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text(f"# comment\nOTHER_SECRET=nope\nexport WANDB_API_KEY=\"{KEY}\"\n"
                 "WANDB_ENTITY=team # trailing comment\nWANDB_PROJECT=ignored\nnot a line\n")
    assert D.read_env_file(f) == {"WANDB_API_KEY": KEY, "WANDB_ENTITY": "team"}
    assert D.read_env_file(tmp_path / "missing") == {}


def _run(tmp_path, mode="online", env_file=None):
    cfg = json.loads((CONFIGS / "naive_ext.json").read_text())
    cfg.update(extends=str(CONFIGS / "naive.json"), runs_root=str(tmp_path / "runs"))
    cfg.setdefault("driver", {})["wandb_env_file"] = str(env_file or tmp_path / "none.env")
    cfg["stats"]["wandb"]["mode"] = mode
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return TrainRun.load(p)


def test_wandb_secrets_sources_and_no_leak(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"WANDB_API_KEY={KEY}\n")
    run = _run(tmp_path, env_file=env)
    secrets, how = D.wandb_secrets(run, environ={})
    assert secrets == {"WANDB_API_KEY": KEY} and KEY not in how and str(env) in how
    # the process env wins
    secrets, how = D.wandb_secrets(run, environ={"WANDB_API_KEY": "fromenv", "WANDB_ENTITY": "e"})
    assert secrets == {"WANDB_API_KEY": "fromenv", "WANDB_ENTITY": "e"} and "environment" in how
    # not online: nothing read
    assert D.wandb_secrets(_run(tmp_path, mode="offline", env_file=env), environ={})[0] == {}
    # the key reaches only the platoon process env, never the run's files or rendered config
    run.write()
    pe = D.platoon_env(run, "http://10.0.0.1:1", {"WANDB_API_KEY": KEY})
    assert pe["WANDB_API_KEY"] == KEY
    cfg = train_config(run, ["t1"], "http://10.0.0.1:1", tmp_path / "spec.toml")
    assert cfg["stats"]["wandb"] == {"mode": "online", "project": "mho-harness",
                                     "name": run.run_id, "tags": ["no-KL", "external-engines", "platoon"]}
    blob = json.dumps(cfg) + run.paths.config.read_text()
    assert KEY not in blob
    assert "WANDB_API_KEY" not in D.platoon_env(run, "http://10.0.0.1:1")      # without secrets


def test_base_config_is_online_with_run_name(tmp_path):
    base = TrainRun.load(CONFIGS / "naive.json")
    assert base["stats"]["wandb"]["mode"] == "online" and base["stats"]["wandb"]["project"] == "mho-harness"
    assert base.driver["wandb_env_file"].endswith("mho-platoon-port/.env")
    cfg = train_config(base, ["t"], "http://x:1", tmp_path / "s.toml")
    assert cfg["stats"]["wandb"]["name"] == base.run_id
    assert TrainRun.load(CONFIGS / "smoke.json")["stats"]["wandb"]["mode"] == "disabled"


def test_scan_secrets(tmp_path, capsys):
    from harnessforge.train import __main__ as cli
    env = tmp_path / ".env"
    env.write_text(f"WANDB_API_KEY={KEY}\n")
    run = _run(tmp_path, env_file=env)
    run.write()
    (run.dir / "logs").mkdir()
    (run.dir / "logs" / "a.log").write_text("WANDB_MODE=disabled\n")
    assert cli.main(["scan-secrets", "--run-dir", str(run.dir)]) == 0
    (run.dir / "logs" / "leak.log").write_text(f"env WANDB_API_KEY={KEY}\n")
    assert cli.main(["scan-secrets", "--run-dir", str(run.dir)]) == 1
    out = capsys.readouterr().out
    assert "leak.log" in out and KEY not in out
