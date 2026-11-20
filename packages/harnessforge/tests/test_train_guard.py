import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harnessforge.train import driver as D
from harnessforge.train import guard as G
from harnessforge.train.config import TrainRun

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"

FAKE_PLATOON = r"""
import sys, time
print("2026-11-20 18:13:20 | INFO | platoon.train.step.step | Step 33/200 | Epoch 0 | Batch time: 458.3s", flush=True)
print("2026-11-20 18:14:00 | INFO | platoon.gateway.core | rollouts keep logging", flush=True)
time.sleep(3600)          # the trainer goes silent
"""

WATCHDOG_PLATOON = r"""
import sys, time
print("x | INFO | platoon.train.step.step | Step 1/200 | Epoch 0", flush=True)
print("x | WARNING | platoon.train.trainer.signals | WATCHDOG: No heartbeat for 461s (timeout in 139s)", flush=True)
time.sleep(1.0)
sys.exit(2)
"""


def _run(tmp_path, **driver) -> TrainRun:
    cfg = json.loads((CONFIGS / "naive.json").read_text())
    cfg.update(extends=str(CONFIGS / "abci-base.json"), runs_root=str(tmp_path / "runs"))
    cfg.setdefault("driver", {}).update(driver)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    run = TrainRun.load(p)
    run.write()
    run.paths.logs.mkdir(parents=True, exist_ok=True)
    return run


def test_v2_config_has_guard_and_recovery_every_5():
    run = TrainRun.load(CONFIGS / "naive_ext_64kcap_v2.json")
    assert run.driver["stall_timeout"] == 2700 and run.driver["max_auto_restarts"] == 3
    assert run.get("checkpoint")["recovery_every"] == 5


def test_stall_guard_counts_steps_not_other_lines(tmp_path):
    log = tmp_path / "platoon.log"
    log.write_bytes(b"old | Step 9/200 | from a previous process\n")
    now = [1000.0]
    g = G.StallGuard(log, timeout=100, startup_grace=50, clock=lambda: now[0])
    now[0] += 140
    assert g.poll() is None                           # before the first step: timeout + grace
    with open(log, "ab") as f:
        f.write(b"t | INFO | platoon.train.step.step | Step 1/200 | Epoch 0\n")
    assert g.poll() is None and g.last_step == 1
    for _ in range(5):                                # rollouts keep logging: no progress
        now[0] += 20
        with open(log, "ab") as f:
            f.write(b"t | WARNING | platoon.train.dispatch.runner | Rollout took 3720.0s (very slow)\n")
        assert g.poll() is None
    now[0] += 1
    assert g.poll() == "stall" and g.last_step == 1
    with open(log, "ab") as f:                        # a partial line is not a step yet
        f.write(b"t | Step 2/2")
    g.poll()
    assert g.last_step == 1
    with open(log, "ab") as f:
        f.write(b"00 | Epoch 0\n")
    assert g.poll() is None and g.last_step == 2


def test_stall_guard_reports_the_watchdog_warning_once(tmp_path):
    log = tmp_path / "platoon.log"
    log.write_text("")
    g = G.StallGuard(log, timeout=0)
    with open(log, "a") as f:
        f.write("x | WARNING | platoon.train.trainer.signals | WATCHDOG: No heartbeat for 461s\n")
    assert g.poll() == "watchdog"
    with open(log, "a") as f:
        f.write("x | WARNING | platoon.train.trainer.signals | WATCHDOG: No heartbeat for 521s\n")
    assert g.poll() is None


def _fake(run, script):
    out = open(run.paths.logs / "platoon.log", "ab")
    return subprocess.Popen([sys.executable, "-c", script], stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)


def test_silent_fake_platoon_is_stopped_dumped_and_restarted(tmp_path, monkeypatch):
    run = _run(tmp_path, where="gateway", max_auto_restarts=2, stall_stop_wait=5)
    (run.paths.logs / "platoon.log").write_text("")
    guard = G.StallGuard(run.paths.logs / "platoon.log", timeout=1.0, startup_grace=0.0)
    proc = _fake(run, FAKE_PLATOON)
    signalled = []

    def fake_dump(logs, pgid, **kw):                  # portable: the real signals are tested on Linux
        signalled.append(pgid)
        (logs / G.STACKS_DIR).mkdir(exist_ok=True)
        (logs / G.STACKS_DIR / f"stacks.{pgid}.txt").write_text("Thread 0x1 (most recent call first):\n")
        return [pgid]
    monkeypatch.setattr(D, "dump_stacks", fake_dump)
    stop = {"flag": False, "user": False}

    def term():
        stop["flag"] = True
        proc.send_signal(signal.SIGTERM)
    t0 = time.time()
    incident = D.supervise(run, proc, guard, term, stop, poll=0.1)
    assert time.time() - t0 < 30 and proc.returncode is not None
    assert incident["reason"] == "stall" and incident["step"] == 33 and signalled == [proc.pid]
    assert "Thread 0x1" in Path(incident["stacks"]).read_text()
    assert "STALL: no training step" in (run.paths.logs / "driver.events.log").read_text()

    notes = []
    monkeypatch.setattr(D, "note_wandb", lambda *a, **k: notes.append(a) or "W&B noted")
    (run.paths.platoon_dir / "x" / "train").mkdir(parents=True)
    (run.paths.platoon_dir / "x" / "train" / "checkpoints.jsonl").write_text(
        json.dumps({"batch": 30, "wandb_run_id": "abc123"}) + "\n")
    env = {"WANDB_API_KEY": "k"}
    assert D.auto_restart(run, incident, proc.returncode, env)
    assert (run.dir / D.KEEP_GATEWAY).exists()
    ev = G.restarts(run.dir)
    assert len(ev) == 1 and ev[0]["reason"] == "stall" and ev[0]["stacks"] == incident["stacks"]
    assert "auto-restart-1" in json.loads(run.paths.config.read_text())["stats"]["wandb"]["tags"]
    assert notes and notes[0][2] == "abc123" and notes[0][3] == "auto-restart-1"
    log = (run.paths.logs / "driver.events.log").read_text()
    assert "AUTO-RESTART 1/2: stall" in log and "W&B noted" in log

    (run.dir / D.KEEP_GATEWAY).unlink()               # the gateway consumed it
    assert D.auto_restart(run, incident, 0, env)       # 2/2
    (run.dir / D.KEEP_GATEWAY).unlink()
    assert not D.auto_restart(run, incident, 0, env)   # budget used: stop, leave it for a person
    assert not (run.dir / D.KEEP_GATEWAY).exists()
    assert "budget used (2/2)" in (run.paths.logs / "driver.events.log").read_text()


def test_watchdog_exit_is_an_incident_with_stacks(tmp_path, monkeypatch):
    run = _run(tmp_path, where="gateway")
    (run.paths.logs / "platoon.log").write_text("")
    guard = G.StallGuard(run.paths.logs / "platoon.log", timeout=2700)
    proc = _fake(run, WATCHDOG_PLATOON)
    monkeypatch.setattr(D, "dump_stacks", lambda logs, pgid, **kw: [pgid])
    incident = D.supervise(run, proc, guard, lambda: None, {"flag": False, "user": False}, poll=0.1)
    assert proc.returncode == 2 and incident["reason"] == "watchdog" and incident["step"] == 1
    assert incident["stacks"] and "watchdog is warning" in (run.paths.logs / "driver.events.log").read_text()


def test_user_stop_is_never_auto_restarted(tmp_path, monkeypatch):
    run = _run(tmp_path, where="gateway")
    (run.paths.logs / "platoon.log").write_text("")
    guard = G.StallGuard(run.paths.logs / "platoon.log", timeout=1.0, startup_grace=0.0)
    proc = _fake(run, FAKE_PLATOON)
    (run.dir / D.DRIVER_STOP).write_text("1")
    stop = {"flag": False, "user": False}

    def term():
        stop["flag"] = True
        proc.send_signal(signal.SIGTERM)
    assert D.supervise(run, proc, guard, term, stop, poll=0.1) is None and stop["user"]


def test_separate_driver_job_does_not_auto_restart(tmp_path):
    run = _run(tmp_path, where="hc")
    assert not D.auto_restart(run, {"reason": "stall", "step": 3, "silent_s": 2800, "stacks": None}, 0, {})
    assert not (run.dir / D.KEEP_GATEWAY).exists()


def test_note_wandb_reports_without_secrets():
    calls = []

    def run_cmd(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, "noted", "")
    msg = G.note_wandb("/venv/bin/python", "mho-harness", "abc", "auto-restart-1", "note", {"WANDB_API_KEY": "s"},
                       run_cmd=run_cmd)
    assert "tag auto-restart-1 added" in msg and "s" not in msg.split()
    assert json.loads(calls[0][0][3]) == {"project": "mho-harness", "id": "abc", "tag": "auto-restart-1",
                                          "note": "note"}


def test_platoon_env_puts_the_stackdump_site_first(tmp_path, monkeypatch):
    run = _run(tmp_path)
    monkeypatch.setattr("harnessforge.train.gateway.node_local_dir", lambda: str(tmp_path))
    env = D.platoon_env(run, "http://10.0.0.1:1")
    first = env["PYTHONPATH"].split(os.pathsep)[0]
    hf_src = run.get("harnessforge_src")
    assert first == str(Path(hf_src) / "harnessforge" / "train" / "stackdump_site")
    assert (G.SITE_DIR / "sitecustomize.py").is_file()          # what that path holds in a deploy
    assert env["HF_STACKDUMP_DIR"] == str(run.paths.logs / G.STACKS_DIR)


@pytest.mark.skipif(not hasattr(signal, "SIGRTMIN") or not Path("/proc").is_dir(), reason="Linux only")
def test_real_stack_dump_of_a_hung_await(tmp_path):
    """A process with the sitecustomize, stuck in an await: both dumps land in its stacks file."""
    script = r"""
import asyncio
async def wedged_optim_step():
    await asyncio.Event().wait()
async def main():
    t = asyncio.create_task(wedged_optim_step())
    print("ready", flush=True)
    await t
asyncio.run(main())
"""
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {**os.environ, "HF_STACKDUMP_DIR": str(logs / G.STACKS_DIR),
           "PYTHONPATH": os.pathsep.join([str(G.SITE_DIR), os.environ.get("PYTHONPATH", "")])}
    p = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        assert p.stdout.readline().strip() == "ready"
        pids = G.dump_stacks(logs, p.pid, wait=2.0)
        assert pids == [p.pid]
        out = G.collect_stacks(logs, tmp_path / "stall.txt", pids, time.time() - 30).read_text()
        assert "most recent call first" in out          # faulthandler's thread stacks
        assert "wedged_optim_step" in out                 # the asyncio task stack
        assert p.poll() is None                           # the dump does not kill it
    finally:
        p.kill()
