#!/usr/bin/env python3
"""A stand-in for `platoon eval CONFIG` in tests: reads the generated config, then writes rollout directories as
platoon does (task-major, num_concurrent_workers at a time, each trial `HF_FAKE_TRIAL_S` seconds, reward 1 for
even rollout indices), and exits 143 on SIGTERM leaving in-flight trials without metadata.json."""
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

assert sys.argv[1] == "eval"
config = json.loads(Path(sys.argv[2]).read_text())
inference = config["inference"]
out = Path(inference["output_dir"]) / "rollouts"
tasks = config["environments"][0]["service_kwargs"]["splits"]["eval"]
trials = inference["workflow"]["num_rollouts_per_task"]
slots = inference["workflow"]["num_concurrent_workers"]
duration = float(os.environ.get("HF_FAKE_TRIAL_S", "0.2"))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
queue = [(t, k) for t in tasks for k in range(trials)]
running = []
while queue or running:
    while queue and len(running) < slots:
        task, k = queue.pop(0)
        d = out / task / f"rollout_{k}"
        d.mkdir(parents=True, exist_ok=True)
        running.append((task, k, d, time.time()))
    time.sleep(duration / 4)
    for item in list(running):
        task, k, d, t0 = item
        if time.time() - t0 >= duration:
            reward = 1.0 if k % 2 == 0 else 0.0
            root = {"reward": reward, "steps": [{}], "misc": {"outcome": {"status": "scored", "rewards": {"reward": reward}}}}
            (d / "trajectory_collection.json").write_text(json.dumps({"trajectories": {"root": root}}))
            meta = {"task_id": task, "rollout_index": k, "wall_time_seconds": time.time() - t0, "error": None,
                    "outcome_status": "scored", "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "completed", "usage": {"policy": {"calls": 3, "prompt_tokens": 300, "completion_tokens": 30}}}
            (d / "metadata.json").write_text(json.dumps(meta))
            running.remove(item)
