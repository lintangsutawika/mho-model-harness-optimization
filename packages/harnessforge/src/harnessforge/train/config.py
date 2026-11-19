"""A training run's settings (one JSON file) and where its files live.

    run = TrainRun.load("configs/train/smoke.json")
    run.dir                      # <runs_root>/<run_id>
    run.paths.gateway_file       # written by the gateway job once the Tinker API answers

Sections (see configs/train/*.json and docs/train.md):

    run_id, runs_root, python, harnessforge_src, launcher {kind, queue, account}
    model    {name, hf_checkpoint, ref_load, megatron_model_type}
    gateway  {sif, miles_src, extra_site, rollout_nodes, walltime, port, weight_sync, tp,
              max_tokens_per_gpu, log_probs_chunk_size, sglang {...}, binds, env, extra_args}
    driver   {where: hc|gateway, walltime, platoon, site, image_cache, seed_image_cache, env}
    data     {dataset, tasks (list | file | null = every task dir), limit, seed, exclude}
    harness  {dir, agent_config}
    env      {agent_timeout_multiplier, verifier_timeout_multiplier, harbor_system_deps,
              harbor_keep_processes, provider, provider_options, service_kwargs, on_harness_exit}
    sandboxes {mode: local | remote, hosts_spec, local_slots}: remote = pods on sandbox host jobs too
              (harnessforge.sandboxhost, docs/sandboxhost.md)
    train    platoon's `train:` section (batch_size, max_training_steps, optimizer, loss_fn, ...)
    checkpoint  platoon's `checkpoint:` section (resumable Tinker state; heavy: model + Adam)
    export   {every}: HF checkpoints from the disk-delta stream every N steps
    stats    platoon's `stats:` section (wandb)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RUNS_ROOT = "/groups/gai51740/aci18914wh/mho/runs"


def _expand(v: Any) -> Any:
    if isinstance(v, str) and (v.startswith("~") or "$" in v):
        return os.path.expanduser(os.path.expandvars(v))
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge (over wins; lists and scalars are replaced)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(d: dict[str, Any], base_dir: Path | None) -> dict[str, Any]:
    """Expand ~/$VARS and merge `extends` parents (paths relative to the file); `_comment`
    keys are dropped."""
    d = _expand(dict(d))
    parent = d.pop("extends", None)
    if parent:
        pp = Path(parent)
        if not pp.is_absolute() and base_dir is not None:
            pp = base_dir / pp
        d = merge(_resolve(json.loads(pp.read_text()), pp.parent), d)
    d.pop("_comment", None)
    return d


@dataclass(frozen=True)
class RunPaths:
    dir: Path

    @property
    def config(self) -> Path:           # the resolved TrainRun, what every job reads
        return self.dir / "run.json"

    @property
    def gateway_file(self) -> Path:     # {url, host, ip, port, job_id, status}
        return self.dir / "gateway.json"

    @property
    def driver_file(self) -> Path:      # {status, job_id, returncode, ...}
        return self.dir / "driver.json"

    @property
    def stop_file(self) -> Path:        # present => the gateway (and driver) shut down
        return self.dir / "STOP"

    @property
    def logs(self) -> Path:
        return self.dir / "logs"

    @property
    def secrets(self) -> Path:          # 0700: ray token
        return self.dir / ".secrets"

    @property
    def tinker_root(self) -> Path:      # tinker:// checkpoints (save_state: torch_dist + Adam)
        return self.dir / "tinker"

    @property
    def delta_dir(self) -> Path:        # disk-delta publication (weight_vNNNNNN/), never pruned
        return self.dir / "weights" / "delta"

    @property
    def hf_dir(self) -> Path:           # HF exports step_<N>/ + latest.json
        return self.dir / "hf"

    @property
    def export_work(self) -> Path:      # the exporter's working full HF copy
        return self.dir / "export" / "work"

    @property
    def latest_policy(self) -> Path:
        return self.hf_dir / "latest.json"

    @property
    def platoon_dir(self) -> Path:      # platoon log_path
        return self.dir / "platoon"

    @property
    def platoon_config(self) -> Path:
        return self.dir / "platoon-train.json"

    @property
    def dataset_dir(self) -> Path:
        return self.dir / "dataset"


@dataclass
class TrainRun:
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, base_dir: Path | None = None) -> "TrainRun":
        d = _resolve(d, base_dir)
        for key in ("run_id", "model", "gateway", "data", "harness", "train"):
            if key not in d:
                raise ValueError(f"train run config needs {key!r}")
        d.setdefault("runs_root", DEFAULT_RUNS_ROOT)
        return cls(d)

    @classmethod
    def load(cls, path: str | Path) -> "TrainRun":
        p = Path(path).expanduser()
        return cls.from_dict(json.loads(p.read_text()), base_dir=p.parent)

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self.raw.get(k, default)

    @property
    def run_id(self) -> str:
        return self.raw["run_id"]

    @property
    def dir(self) -> Path:
        return Path(self.raw["runs_root"]) / self.run_id

    @property
    def paths(self) -> RunPaths:
        return RunPaths(self.dir)

    @property
    def gateway(self) -> dict[str, Any]:
        return self.raw["gateway"]

    @property
    def driver(self) -> dict[str, Any]:
        return self.raw.get("driver", {})

    def write(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.paths.config.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.raw, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.paths.config)
        return self.paths.config


# --------------------------------------------------------------------------- old -> new mapping
#: The old naive GRPO run (mho-rsft scripts/optimize/train.sh + src/run_training.py, miles
#: train_async.py with in-process harbor trials) mapped onto platoon + miles-tinker. Status:
#: "same", "mapped" (same intent, different mechanism), "differs" (cannot be matched today).
OLD_RUN_MAPPING: list[tuple[str, str, str, str]] = [
    # (old setting, old value, new setting, status/notes)
    ("--hf-checkpoint", "Qwen/Qwen3.5-4B", "model.name / model.hf_checkpoint (local snapshot)", "same"),
    ("--prompt-data", "tmax-v11-train-naive/seed.jsonl (6725 tasks)", "data.dataset = tmax-v11-train-naive", "same"),
    ("--rollout-shuffle", "on", "train.task_sampler = uniform", "mapped: uniform sampling vs epoch shuffle"),
    ("--num-rollout", "200", "train.max_training_steps = 200", "same"),
    ("--rollout-batch-size", "8 prompts", "train.batch_size = 8", "same"),
    ("--n-samples-per-prompt", "32", "train.workflow_config.group_size = 32", "same"),
    ("--global-batch-size", "256 (1 optimizer step / rollout)", "train.num_minibatches = 1", "same"),
    ("--lr / --lr-decay-style", "1e-6 constant", "train.optimizer.learning_rate = 1e-6 (per optim_step)", "same"),
    ("--adam-beta1/2, --weight-decay", "0.9 / 0.98 / 0.1", "train.optimizer {beta1 .9, beta2 .98, weight_decay .1}", "same"),
    ("--clip-grad (miles default)", "1.0", "train.optimizer.grad_clip_norm = 1.0", "same"),
    ("adam eps (megatron default)", "1e-8", "train.optimizer.eps = 1e-8", "same (miles-tinker default would be 1e-12)"),
    ("--advantage-estimator grpo + --disable-grpo-std-normalization", "r - mean(group)",
     "platoon group advantages (r - group mean, no std)", "same"),
    ("drgrpo pg loss reducer", "constant divisor", "platoon: sum loss x 1/(action tokens per microbatch)",
     "differs: per-token mean over the batch vs Dr.GRPO constant divisor (scale only, not direction)"),
    ("--eps-clip / --eps-clip-high", "0.2 / 0.28 (PPO)", "train.loss_fn = ppo, loss_fn_config {clip_low_threshold .8, clip_high_threshold 1.28}",
     "mapped (miles-tinker ppo ratio clip)"),
    ("--use-tis (fully async)", "TIS against rollout logprobs", "ppo ratio vs sampled logprobs (behaviour policy)",
     "differs: no separate truncated IS weight; the PPO ratio is taken against the sampler's logprobs"),
    ("--use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl", "KL to frozen reference", "none",
     "differs: platoon/miles-tinker have no reference-KL term (only logged); needs work, see docs/train.md"),
    ("--entropy-coef", "0", "none", "same (no entropy bonus)"),
    ("dynamic filter apply_reward_nonzero_std_filter", "drop all-0/all-1 groups, resample",
     "train.group_admission = informative, train.refill.max_groups_per_step = 8",
     "mapped: rejected (uniform-reward) groups are refilled until 8 informative groups, capped at 64 groups/step "
     "(old: uncapped resampling)"),
    ("--fully-async --max-weight-staleness 4", "async, staleness <= 4", "train.max_staleness = 4", "mapped"),
    ("--async-max-concurrent-samples", "256", "train.max_concurrent_rollouts", "mapped (limited by sandbox cores)"),
    ("--rollout-temperature", "1.0", "rollout_config.inference_params.temperature = 1.0 (miles-tinker requires 1.0)", "same"),
    ("AGENT_MAX_OUTPUT_TOKENS / --rollout-max-response-len", "16384 per model call",
     "inference_params.max_completion_tokens = 16384, sampling_precedence = config", "same"),
    ("AGENT_MAX_INPUT_TOKENS / --max-seq-len", "65536", "train.context_window_length = 65536; sglang context 65536", "same"),
    ("AGENT_TIMEOUT x HARBOR_TIMEOUT_MULTIPLIER", "1800 s override x 2.0 = 3600 s", "env.agent_timeout_multiplier = 6 (every tmax task says 600 s)",
     "same (3600 s)"),
    ("AGENT_TRIAL_TIMEOUT", "7200 s wall cap, scores 0", "rollout_config.timeout = 7200", "mapped"),
    ("harness", "micro-swe-agent (base scaffold) via harbor MiniSweAgent", "harness.dir rendered as the evals backend's spec (same runner/toolchain)", "same code"),
    ("chat template / parsers", "SGLang qwen3_coder + qwen3 reasoning parser, tito qwen35", "renderer chat_template, tool_call_parser qwen3_coder",
     "mapped: platoon renders/parses itself; token-in/out preserved"),
    ("sandbox", "hpc-sandbox pods via host executors on the 4 GPU nodes", "harnessforge.evals.pods provider on the driver node", "mapped"),
    ("rollout engines", "24 GPUs (3 nodes) miles SGLang", "gateway.rollout_nodes = 3 (miles-managed SGLang)", "same"),
    ("trainer", "1 node, TP2, max-tokens-per-gpu 65536, log-probs chunk 1024, CPU-offloaded precision-aware Adam",
     "gateway tp 2, same tokens/chunk; Adam on GPU (miles-tinker full mode forbids precision-aware optimizer)", "mapped"),
    ("--save-interval 10 --save-hf", "HF every 10 steps", "export.every = 10 (from the disk-delta stream)", "same"),
]
