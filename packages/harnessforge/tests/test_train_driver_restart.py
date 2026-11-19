import json
import threading
import time
from pathlib import Path

from harnessforge.train import __main__ as cli
from harnessforge.train.config import TrainRun
from harnessforge.train.gateway import driver_exit_action

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"


def test_driver_exit_action_consumes_keep(tmp_path):
    assert driver_exit_action(tmp_path) == "stop"
    (tmp_path / "KEEP_GATEWAY").write_text("1")
    assert driver_exit_action(tmp_path) == "restart"
    assert not (tmp_path / "KEEP_GATEWAY").exists()          # one restart per request
    assert driver_exit_action(tmp_path) == "stop"


def test_train_driver_on_gateway_head_restarts_through_gateway(tmp_path, monkeypatch):
    cfg = json.loads((CONFIGS / "naive.json").read_text())
    cfg.update(extends=str(CONFIGS / "abci-base.json"), runs_root=str(tmp_path / "runs"))
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    run = TrainRun.load(p)
    assert run.driver["where"] == "gateway"
    run.write()
    (run.dir / "jobs.json").write_text(json.dumps({"gateway": "1.pbs"}))
    run.paths.driver_file.write_text(json.dumps({"status": "training", "started_at": 1.0}))
    monkeypatch.setattr(cli, "_launcher", lambda r: None)

    real_sleep = time.sleep

    def gateway():                              # what the gateway + new driver do
        while not (run.dir / "DRIVER_STOP").exists():
            real_sleep(0.01)
        assert driver_exit_action(run.dir) == "restart"
        run.paths.driver_file.write_text(json.dumps({"status": "waiting", "started_at": 2.0}))
    real_sleep = time.sleep
    t = threading.Thread(target=gateway)
    t.start()
    monkeypatch.setattr(cli.time, "sleep", lambda s: real_sleep(0.01))
    assert cli.main(["driver", "--run-dir", str(run.dir), "--grace", "5",
                     "--set", "stats.wandb.mode=\"online\""]) == 0
    t.join()
    assert TrainRun.load(run.paths.config)["stats"]["wandb"]["mode"] == "online"
