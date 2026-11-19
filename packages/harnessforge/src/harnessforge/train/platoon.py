"""The `platoon train` config of a run: the evals backend's Harbor environment (micro-swe-agent
candidate in hpc-sandbox pods, harbor parity shims) plus platoon's `train:` section, pointed at
the run's miles-tinker gateway.

    tasks = select_tasks(run)                        # data.tasks / every task dir, shuffled, limited
    prepare = materialize(run, tasks)                # copy tasks into the run dir, Dockerfile replays
    cfg = train_config(run, tasks, url, spec, prepare)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from harnessforge.contracts import Candidate
from harnessforge.evals import dataset as datasets
from harnessforge.evals.backend import DEFAULT_PROVIDER, HARBOR_SYSTEM_DEPS
from harnessforge.evals.harness import CANDIDATE_MOUNT, TOOLCHAIN, check_candidate, render_spec
from harnessforge.train.config import TrainRun

#: platoon `train:` defaults for Qwen3.5 on miles-tinker full fine-tuning (a run's `train`
#: section overrides any of them)
TRAIN_DEFAULTS: dict[str, Any] = {
    "renderer_name": "chat_template",
    "renderer_kwargs": {"tool_call_parser": "qwen3_coder"},
    "lora_rank": 32,                       # ignored by miles-tinker in full mode
    "num_minibatches": 1,
    "task_sampler": "uniform",
    "sample_timeout_seconds": 3600,
    "forward_backward_timeout_seconds": 5400,
    "optim_step_timeout_seconds": 1800,
}


def task_dirs(dataset: Path) -> list[str]:
    return sorted(p.name for p in Path(dataset).iterdir() if (p / "task.toml").is_file())


def select_tasks(run: TrainRun) -> list[str]:
    """data.tasks: a list, a JSON file (a list, or {"train": [...]}), or null (every task dir of
    data.dataset). Then data.exclude is removed, shuffled with data.seed (unless shuffle false),
    and cut to data.limit."""
    d = run["data"]
    tasks = d.get("tasks")
    if tasks is None:
        tasks = task_dirs(Path(d["dataset"]))
    elif isinstance(tasks, str):
        loaded = json.loads(Path(tasks).read_text())
        tasks = loaded.get("train", loaded.get("tasks")) if isinstance(loaded, dict) else loaded
    tasks = [t for t in tasks if t not in set(d.get("exclude", []))]
    if d.get("shuffle", True):
        random.Random(int(d.get("seed", 0))).shuffle(tasks)
    if d.get("limit"):
        tasks = tasks[: int(d["limit"])]
    if not tasks:
        raise ValueError("no training tasks selected")
    return list(tasks)


def materialize(run: TrainRun, tasks: Sequence[str]) -> list[str]:
    """Copy the tasks into <run>/dataset (Dockerfile-only tasks get their FROM image) and return the
    harness `[install] commands`: the Dockerfile RUN replay and harbor's system deps (as the evals
    backend). Returns the prepare commands."""
    env = run.get("env", {})
    replays = datasets.materialize(Path(run["data"]["dataset"]), tasks, run.paths.dataset_dir)
    commands = {datasets.replay_command(r) for r in replays.values()} - {None}
    if len(commands) > 1:
        raise ValueError("tasks replay different Dockerfile steps: " + "; ".join(sorted(commands))[:300])
    return [*commands, *([HARBOR_SYSTEM_DEPS] if env.get("harbor_system_deps", True) else [])]


def render_harness(run: TrainRun, prepare: Sequence[str]) -> Path:
    h = run["harness"]
    cand = Candidate(h.get("id", "base"), Path(h["dir"]))
    check_candidate(cand.harness_dir)
    return render_spec(cand, run.dir / "harness", agent_config=h.get("agent_config"),
                       toolchain=h.get("toolchain", TOOLCHAIN), prepare=prepare,
                       keep_children=bool(run.get("env", {}).get("harbor_keep_processes", True)))


def environment_entry(run: TrainRun, tasks: Sequence[str], spec: Path) -> dict[str, Any]:
    env = run.get("env", {})
    provider = env.get("provider", DEFAULT_PROVIDER)
    provider_options = {"fakeroot": True, "allow_network_none": True, **env.get("provider_options", {})}
    if provider == DEFAULT_PROVIDER:
        provider_options.setdefault("keep_agent_processes", bool(env.get("harbor_keep_processes", True)))
    return {
        "name": "harnessforge",
        "service": "harbor",
        "service_kwargs": {
            "dataset_dir": str(run.paths.dataset_dir),
            "splits": {"train": list(tasks)},
            "provider": provider,
            "provider_options": provider_options,
            "mounts": [{"source": str(Path(run["harness"]["dir"]).resolve()), "target": CANDIDATE_MOUNT,
                        "read_only": True}],
            "agent_timeout_multiplier": float(env.get("agent_timeout_multiplier", 1.0)),
            "verifier_timeout_multiplier": float(env.get("verifier_timeout_multiplier", 1.0)),
            "agent_user": "ignore",
            **env.get("service_kwargs", {}),
        },
        "harness": str(spec),
        "on_harness_exit": env.get("on_harness_exit", {"timeout": "grade", "crash": "grade"}),
    }


def train_config(run: TrainRun, tasks: Sequence[str], url: str, spec: Path) -> dict[str, Any]:
    from harnessforge.train.config import merge
    train = merge(TRAIN_DEFAULTS, {"model_name": run["model"]["name"], **run["train"]})
    stats = merge({"experiment_name": run.run_id, "trial_name": "train",
                   "wandb": {"mode": "disabled", "name": run.run_id}}, run.get("stats", {}))
    # credentials go through the platoon process env (driver.wandb_secrets), never the config
    stats.get("wandb", {}).pop("api_key", None)
    return {
        "log_path": str(run.paths.platoon_dir),
        "tinker_base_url": url,
        "train": train,
        "checkpoint": run.get("checkpoint", {"every": 0}),
        "stats": stats,
        "environments": [environment_entry(run, tasks, spec)],
    }
