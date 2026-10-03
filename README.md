# mho-model-harness-optimization

Model + Harness joint-optimization for agentic RL. The optimizer loop
co-evolves a model and its agent harness,
running each train/eval phase as a harbor job on the cluster.

- **Model**: Qwen3.5-4B, trained with Miles (Megatron-LM RL, harbor trials in-process).
- **Harness**: `micro-swe-agent` scaffold, mutated per iteration by `propose.sh`.
- **Sandbox**: Modal or local singularity on the node via `harbor-singularity-hpc` (writable-rootfs,
  no Docker/cloud).

## Setup

Create `.env` in the repo root (gitignored; it holds storage roots, docker creds, WANDB key,
model/serving config)

Required `.env`/env keys:
- `BASE_DIR` — node-local tmp/cache (`$PBS_LOCALDIR`); `HF_DIR`/`TMP_DIR`/`SINGULARITY_*` derive from it.
- `USER_DATA` — persistent run/checkpoint/data root.
- `BASE_SIF` — the single image knob (e.g. `skyrl-megatron.sif` for serving, `miles.sif` for training).
- `HARBOR_ENV_TYPE=singularity`, `EXECUTOR_URL=http://127.0.0.1:8900` (local sandbox service).
- `SINGULARITY_DOCKER_USERNAME`/`PASSWORD`, `WANDB_API_KEY`.
- `OPTIMIZER_MODEL`/`OPTIMIZER_LLM_BASE_URL` (propose-phase LLM).

## Running

### Optimization loop (choose a mode)

```bash
# harness-only: propose -> eval (fixed model)
bash scripts/run/harness-only.sh --model Qwen/Qwen3.5-4B --iterations 10 --run-name test

# model-only: train -> eval (fixed harness)
bash scripts/run/model-only.sh     --model Qwen/Qwen3.5-4B --iterations 10 --run-name test

# model-harness: propose -> train -> eval
bash scripts/run/model-harness.sh  --model Qwen/Qwen3.5-4B --iterations 10 --run-name test
```

Each phase is a `qsub` job; the driver polls `qstat`. See `src/optimize_loop.py` for the
env interface (`MHO_MODEL`, `MHO_TASK`, `MHO_OUT`, `MHO_TRAINER`, ...).

### Train directly (Miles)

```bash
bash scripts/optimize/train.sh
```

### Eval directly

```bash
MHO_MODEL=Qwen/Qwen3.5-4B \
MHO_DATA_PATH=data-harbor/<task> \
EVAL_JOB_NAME=baseline \
bash scripts/optimize/eval.sh
```

### Prepare data / generate tasks

```bash
uv run --with datasets python -m tasks.dapo_math_17k --limit 10 --output-dir out/dapo
uv run --with datasets python -m tasks.hmmt_feb_2025 --limit 10 --output-dir out/hmmt
```

### Build images

```bash
bash scripts/build/build_train.sh      # skyrl-megatron.sif
bash scripts/build/build-miles.sh      # miles.sif (from radixark/miles:latest)
```

## Notes / constraints (ABCI)

- Singularity-only: no Docker. Sandboxes run on the node writable-rootfs via
  `harbor-singularity-hpc`.
- `BASE_SIF` is the only image knob — it serves both training (Miles) and vLLM serving
  (the train image ships `vllm`). No separate `vllm-cuda.sif`.
- `tasks/*/data/` (generated dataset JSONL) is gitignored — never commit it.
- The legacy SkyRL path (nested-apptainer) lives under `scripts/train/train_math_dapo.sh`.

## Scheduler

Auto-detected PBS/SLURM (`MHO_SCHEDULER=pbs|slurm`); per-scheduler wrappers in `scripts/hpc/`.
