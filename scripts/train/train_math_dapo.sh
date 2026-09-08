#!/usr/bin/env bash


set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

source .env

# Two storage roots, both from .env:
#   BASE_DIR  = ephemeral tmp/cache (/scratch or $PBS_LOCALDIR) -> TMP_DIR / HF_DIR / caches
#   USER_DATA = persistent run data (/data/user_data/lsutawik/... or /home/aci18914wh/...)
#               -> data-harbor, runs_output (checkpoints/exports), scaffolds, run.log.
USER_DATA="${USER_DATA:-/data/user_data/lsutawik}"

UV_CACHE_PERSIST="${UV_CACHE_PERSIST:-${USER_DATA}/uv_cache}"
mkdir -p "$UV_CACHE_PERSIST"

# Host-writable SIF image cache for the external executor's docker->sif sandbox conversion.
# PERSISTENT (next to BASE_SIF, not node-local) so a pre-seeded python_3.11-slim.sif -- built
# from scripts/train/agent_sandbox.def (adds /usr/bin/python3 + tools harbor's bootstrap
# needs) -- survives across jobs and is reused instead of re-pulling vanilla python:3.11-slim.
# Passed into the SIF so train_entrypoint writes it into each trial's environment config.
SIF_IMAGE_CACHE_DIR="${SIF_IMAGE_CACHE_DIR:-$(dirname "${BASE_SIF}")/sif_cache}"
mkdir -p "$SIF_IMAGE_CACHE_DIR"

MATH_DATA_DIR="${MATH_DATA_DIR:-${USER_DATA}/mho-model-harness-optimization/data-harbor/DAPO-Math-17k}"
MATH_TRAIN_DIR="${MATH_TRAIN_DIR:-${MATH_DATA_DIR}/train}"

MAX_TRAIN="${MAX_TRAIN:-500}"
if [ ! -d "$MATH_TRAIN_DIR" ]; then
  # Host-side data prep only needs `datasets`. Use --no-project so uv does NOT resolve
  # this repo's pyproject (its megatron/vllm-router pins are for the SIF and fail to
  # install on a host with a different glibc); --with datasets supplies the one real dep.
  uv run --no-project --with datasets src/harbor/prepare_math_tasks.py --out "$MATH_TRAIN_DIR" --split train --max-tasks "$MAX_TRAIN"
fi

# Held-out eval set: 50 tasks disjoint from train (seed-0 rows 500-549 of DAPO-Math-17k).
MATH_VAL_DIR="${MATH_VAL_DIR:-${MATH_DATA_DIR}/test}"
MAX_VAL="${MAX_VAL:-50}"
if [ ! -d "$MATH_VAL_DIR" ]; then
  # Disjoint from train: same seed-0 shuffle, rows [MAX_TRAIN, MAX_TRAIN+MAX_VAL).
  uv run --no-project --with datasets src/harbor/prepare_math_tasks.py --out "$MATH_VAL_DIR" --split train --max-tasks "$MAX_VAL" --start "$MAX_TRAIN"
fi

RUN_ID="${RUN_ID:-math}"
CAND_ID="${CAND_ID:-0}"
RUNS_ROOT="${RUNS_ROOT:-${USER_DATA}/mho-model-harness-optimization/runs}"
# The harness (micro-swe-agent scaffold) deployed into each trial sandbox. Point it at any
# scaffold dir directly (MINI_FORK_LOCAL=/path/to/micro-swe-agent), or leave it to the
# RUN_ID/CAND_ID convention below. If the dir is missing it's materialized from
# MICRO_SCAFFOLD_BASE; if you pass an existing dir, materialize is skipped and it's used as-is.
export MINI_FORK_LOCAL="${MINI_FORK_LOCAL:-$RUNS_ROOT/run_${RUN_ID}/candidate_${CAND_ID}/micro-swe-agent}"
if [ ! -d "$MINI_FORK_LOCAL" ]; then
MICRO_SCAFFOLD_BASE="${MICRO_SCAFFOLD_BASE:-$REPO_DIR/../micro-swe-agent}" \
    PYTHONPATH=src .venv/bin/python -m harness.scaffold materialize \
      --run-id "$RUN_ID" --candidate-id "$CAND_ID" \
      --runs-dir "$RUNS_ROOT"
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
# One run directory holds everything for this run together:
#   ${RUN_DIR}/checkpoints/global_step_N/policy/*.distcp   (weights + optimizer)
#   ${RUN_DIR}/exports/dumped_evals/global_step_N_evals     (eval traces)
#   ${RUN_DIR}/exports/...                                  (data dumps, HF exports)
# Persistent USER_DATA by default (off quota-limited $HOME; traces survive across jobs).
# Checkpoints are the big item -- MAX_CKPTS_TO_KEEP bounds them (default keep the last 2).
# If they still outgrow USER_DATA, override CKPT_PATH alone to /scratch (2.6T free).
RUN_NAME="${RUN_NAME:-mho_qwen3.5-4b_math}"
RUN_DIR="${RUN_DIR:-${USER_DATA}/mho-model-harness-optimization/runs_output/${RUN_NAME}}"
CKPT_PATH="${CKPT_PATH:-${RUN_DIR}/checkpoints}"
EXPORT_PATH="${EXPORT_PATH:-${RUN_DIR}/exports}"
MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-2}"
# HF-format (portable, transformers-loadable) checkpoints, saved every N steps + at epoch
# end to ${EXPORT_PATH}/global_step_N/policy/ (config.json + *.safetensors + tokenizer).
# These are for eval/deployment/sharing; the .distcp checkpoints above are for RESUME
# (they carry optimizer state). -1 disables. 10 matches the distcp ckpt_interval.
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-10}"
mkdir -p "$EXPORT_PATH"

# Mirror all output to the terminal AND a canonical log at ${RUN_DIR}/run.log (fresh each
# run) via tee: an interactive `bash ...` still prints live, while batch (qsub/sbatch) both
# captures stdout to the job file and leaves run.log in the canonical spot.
# rm -f first so a stale symlink can't redirect the write to an old file.
# Set RUN_LOG= (empty) to disable the file entirely (terminal only).
mkdir -p "$RUN_DIR"
RUN_LOG="${RUN_LOG-${RUN_DIR}/run.log}"
if [ -n "$RUN_LOG" ]; then
    rm -f "$RUN_LOG"
    exec > >(tee "$RUN_LOG") 2>&1
fi

NUM_GPUS="${NUM_GPUS:-8}"          # total GPUs visible on the node
NUM_NODES="${NUM_NODES:-1}"

# Non-colocated placement: policy+ref on their own GPUs, inference engines on the
# rest. Avoids the colocated CUDA-IPC weight-sync path (which deadlocked) -- sync
# now uses NCCL broadcast. Requires POLICY_NUM_GPUS + NUM_INFERENCE_ENGINES*TP == NUM_GPUS.
COLOCATE_ALL="${COLOCATE_ALL:-false}"
POLICY_NUM_GPUS="${POLICY_NUM_GPUS:-4}"

GPU_LIST="${GPU_LIST:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
MEGATRON_TP="${MEGATRON_TP:-1}"    # policy on POLICY_NUM_GPUS (=4): one replica, TP=4 PP=1
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-1}"
MEGATRON_ETP="${MEGATRON_ETP:-null}"

NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-4}"   # 4 engines x TP1 = 4 inference GPUs
INFERENCE_ENGINE_TP="${INFERENCE_ENGINE_TP:-1}"

# Fully-async (off-policy) training: sampler runs ahead of the trainer by up to
# max_staleness_steps. num_parallel_generation_workers caps concurrent trajectories
# -- each worker is a harbor sandbox (local docker or modal), so keep it modest,
# not the 768 default.
FULLY_ASYNC="${FULLY_ASYNC:-true}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-2}"
# Each parallel worker = one harbor sandbox (local docker or modal). Too many at
# once drove tmpfs/memory pressure -> SIGBUS. Start conservative; raise once stable.
NUM_PARALLEL_GEN_WORKERS="${NUM_PARALLEL_GEN_WORKERS:-32}"
# Non-colocated: GPUs 0-3 are DEDICATED to inference (policy is on 4-7), so the
# engines can use most of the GPU for KV cache. 0.4 was a colocated-mode leftover
# that starved the KV cache (~6 GiB) and choked under concurrent long generations.
# (If you go back to colocate_all=true, drop this to ~0.4 to leave room for the policy.)
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# Skip vLLM torch.compile + CUDA-graph capture. The Qwen3.5 compile path on this
# cu13/torch-2.11/vLLM-0.28 stack segfaults the engine core during profile_run
# (this is the instability the original DAPO recipe's enforce_eager guarded against).
# NOTE: SkyRL force-disables this when LoRA weight sync is active (config.py ~1767).
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
# Qwen3.5 is a vision-language model; this is a text-only math task, so load only the
# LLM backbone on BOTH sides. Inference (vLLM): otherwise inits the vision encoder +
# multimodal processor (vit/MMEncoder/shm mm-cache) that kills the engine core.
# Training (Megatron policy/ref): the Qwen3.5-VL model packs sequences inside its own
# forward, which double-packs against SkyRL's sample packing and corrupts the
# GatedDeltaNet cu_seqlens -- language_model_only routes it to the native GPTModel GDN
# packing path. Drives generator + trainer.policy + trainer.ref.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-true}"
# Cap vLLM context. Qwen3.5-4B's default max_model_len is 262144 (256K), whose KV
# cache (~8 GiB for a single request) overflows the colocated budget and makes the
# engine core raise ValueError at startup ("larger than available KV cache memory").
# 64K is ample for prompt(4096)+generate(4096) plus long agent rollouts, and its KV
# (~2 GiB) fits easily. Passed via engine_init_kwargs (overrides config-derived args).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# The micro-swe-agent scaffold drives the model via OpenAI tool/function calling
# (tools=[BASH_TOOL], tool_choice=auto). vLLM rejects that unless the OpenAI server
# is started with tool-call support, so enable it on the engines. Parser must match
# how Qwen3.5 emits tool calls: `hermes` (classic <tool_call>{json}</tool_call>) is the
# broadly-compatible default; try `qwen3_xml` if tool calls don't parse.
ENABLE_TOOL_CHOICE="${ENABLE_TOOL_CHOICE:-true}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
APPLY_OVERLONG_FILTERING="${APPLY_OVERLONG_FILTERING:-true}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
LR="${LR:-1e-6}"

# Trainer
# 20-step run: train_batch_size=25 over 500 tasks x epochs=1 = ceil(500/25)=20 steps
# (the RL trainer has no max_steps knob; total steps = ceil(N_train/batch) x epochs).
EPOCHS="${EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-25}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-25}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
# Pre-train eval gate. Set EVAL_BEFORE_TRAIN=false to skip the (slow) eval-before-train
# and go straight to the first training step. Shrink it instead with MAX_VAL + EVAL_BATCH_SIZE
# (e.g. MAX_VAL=10 EVAL_BATCH_SIZE=10). EVAL_INTERVAL controls periodic eval during training.
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-true}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
LOGGER="${LOGGER:-wandb}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-4B}"

# PyTorch expandable_segments (CUDA VMM) on the training workers. SkyRL defaults it
# True, but only the FSDP worker auto-disables it around weight sync; the Megatron
# worker does not. Colocated with vLLM's sleep-mode CuMemAllocator (also VMM), it
# clobbers the VA range vLLM reserved, so vLLM's wake-for-weight-sync dies with
# `CUDA Error: invalid argument at cumem_allocator.cpp:258`. Off for colocation.
USE_EXPANDABLE_SEGMENTS="${USE_EXPANDABLE_SEGMENTS:-false}"

# Harbor sandbox backend. "remote" model: the harbor SDK runs in this training
# container but dispatches each trial sandbox to an external executor. modal =
# Modal cloud (default). Other remote clouds (e2b, daytona, ...) also selectable;
# a same-host docker/singularity backend would need that runtime reachable from
# inside this frozen SIF, so prefer a network-remote one here.
HARBOR_ENV_TYPE="${HARBOR_ENV_TYPE:-singularity}"
# Local external sandbox executor (Modal-like, but local): when EXECUTOR_URL is
# set, harbor trials are POSTed to a separate executor process on the node (see
# scripts/executor/executor_service.py) instead of running in-process. Leave
# unset to run trials in-process (original behavior).
EXECUTOR_URL="${EXECUTOR_URL:-}"

# How the trainer runs INSIDE the SIF -- auto-selected from the image itself, so
# BASE_SIF is the only knob you need to switch paths:
#   * skyrl-megatron SIF  (built by scripts/build/build_train.sh; baked env at /opt/SkyRL/.venv):
#     run that interpreter directly -- no uv, no --extra, no runtime resolve.
#   * base SIF (system-only): build the env at runtime from pyproject.toml via uv.
# Export TRAIN_LAUNCHER yourself to override the auto-detection.
BAKED_PY=/opt/SkyRL/.venv/bin/python
if [ -z "${TRAIN_LAUNCHER:-}" ]; then
  if singularity exec "${BASE_SIF}" test -x "${BAKED_PY}" 2>/dev/null; then
    TRAIN_LAUNCHER="${BAKED_PY} -m src.train_entrypoint"
    echo "[launcher] skyrl-megatron SIF detected -> ${BAKED_PY} (env baked, no runtime resolve)"
  else
    TRAIN_LAUNCHER="uv run --isolated --python 3.12 --extra megatron -m src.train_entrypoint"
    echo "[launcher] base SIF -> uv run --isolated (runtime env build from pyproject.toml)"
  fi
fi

echo ${BASE_SIF}

singularity exec --nv --writable-tmpfs \
    --workdir "${TMP_DIR}" \
    --bind "${BASE_DIR}:${BASE_DIR}" \
    --bind "${USER_DATA}:${USER_DATA}" \
    --env SIF_IMAGE_CACHE_DIR="${SIF_IMAGE_CACHE_DIR}" \
    --bind "${HF_DIR}:/root/.cache/huggingface" \
    --bind "${TMP_DIR}:/tmp_work" \
    --env TMPDIR=/tmp_work \
    --env HF_HOME=/root/.cache/huggingface \
    --env CPATH= \
    --env PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts/executor" \
    --env MICRO_SCAFFOLD_DIR="${MINI_FORK_LOCAL}" \
    --env MSWEA_API_KEY="${MSWEA_API_KEY:-dummy}" \
    --env AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-16384}" \
    --env AGENT_EXEC_TIMEOUT_SEC="${AGENT_EXEC_TIMEOUT_SEC:-1800}" \
    --env RAY_worker_register_timeout_seconds=600 \
    --env CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    --bind "${UV_CACHE_PERSIST}:/root/.cache/uv" \
    --env UV_CACHE_DIR=/root/.cache/uv \
    --env TRITON_CACHE_DIR=/tmp_work/triton_cache \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp_work/torchinductor_cache \
    --env VLLM_CACHE_ROOT=/tmp_work/vllm_cache \
    --env XDG_CACHE_HOME=/tmp_work/xdg_cache \
    --env SINGULARITY_CACHEDIR=/tmp_work/singularity_cache \
    --env HARBOR_ENV_TYPE="${HARBOR_ENV_TYPE}" \
    --env EXECUTOR_URL="${EXECUTOR_URL}" \
    --env SINGULARITY_TMPDIR=/tmp_work \
    --env SINGULARITY_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME:-}" \
    --env SINGULARITY_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD:-}" \
    "${BASE_SIF}" \
        ${TRAIN_LAUNCHER} \
                data.train_data="['${MATH_TRAIN_DIR}']" \
                data.val_data="['${MATH_VAL_DIR}']" \
                data.dataloader.num_workers=0 \
                trainer.algorithm.advantage_estimator="grpo" \
                trainer.policy.model.path="${MODEL}" \
                trainer.placement.colocate_all=${COLOCATE_ALL} \
                trainer.use_expandable_segments=${USE_EXPANDABLE_SEGMENTS} \
                trainer.fully_async.enabled=${FULLY_ASYNC} \
                trainer.fully_async.max_staleness_steps=${MAX_STALENESS_STEPS} \
                trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GEN_WORKERS} \
                trainer.strategy=megatron \
                trainer.placement.policy_num_gpus_per_node=${POLICY_NUM_GPUS} \
                trainer.placement.ref_num_gpus_per_node=${POLICY_NUM_GPUS} \
                trainer.placement.policy_num_nodes=${NUM_NODES} \
                trainer.placement.ref_num_nodes=${NUM_NODES} \
                trainer.policy.megatron_config.tensor_model_parallel_size=${MEGATRON_TP} \
                trainer.policy.megatron_config.pipeline_model_parallel_size=${MEGATRON_PP} \
                trainer.policy.megatron_config.context_parallel_size=${MEGATRON_CP} \
                trainer.policy.megatron_config.expert_model_parallel_size=${MEGATRON_EP} \
                trainer.policy.megatron_config.expert_tensor_parallel_size=${MEGATRON_ETP} \
                trainer.policy.language_model_only=${LANGUAGE_MODEL_ONLY} \
                trainer.ref.language_model_only=${LANGUAGE_MODEL_ONLY} \
                generator.inference_engine.backend=vllm \
                generator.inference_engine.run_engines_locally=True \
                generator.inference_engine.num_engines=${NUM_INFERENCE_ENGINES} \
                generator.inference_engine.tensor_parallel_size=${INFERENCE_ENGINE_TP} \
                generator.inference_engine.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
                generator.inference_engine.enforce_eager=${ENFORCE_EAGER} \
                generator.inference_engine.language_model_only=${LANGUAGE_MODEL_ONLY} \
                generator.inference_engine.engine_init_kwargs.max_model_len=${MAX_MODEL_LEN} \
                generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=${ENABLE_TOOL_CHOICE} \
                generator.inference_engine.engine_init_kwargs.tool_call_parser=${TOOL_CALL_PARSER} \
                generator.inference_engine.served_model_name=${SERVED_MODEL_NAME} \
                generator.step_wise_trajectories=true \
                generator.merge_stepwise_output=true \
                generator.batched=true \
                generator.apply_overlong_filtering=${APPLY_OVERLONG_FILTERING} \
                generator.n_samples_per_prompt=${N_SAMPLES_PER_PROMPT} \
                generator.sampling_params.max_generate_length=${MAX_GENERATE_LENGTH} \
                generator.sampling_params.temperature=${TEMPERATURE} \
                trainer.epochs=${EPOCHS} \
                trainer.eval_batch_size=${EVAL_BATCH_SIZE} \
                trainer.eval_before_train=${EVAL_BEFORE_TRAIN} \
                trainer.eval_interval=${EVAL_INTERVAL} \
                trainer.train_batch_size=${TRAIN_BATCH_SIZE} \
                trainer.policy_mini_batch_size=${POLICY_MINI_BATCH_SIZE} \
                trainer.max_prompt_length=${MAX_PROMPT_LENGTH} \
                trainer.algorithm.max_seq_len=${MAX_SEQ_LEN} \
                trainer.policy.optimizer_config.lr=${LR} \
                trainer.algorithm.use_kl_loss=${USE_KL_LOSS} \
                trainer.logger=${LOGGER} \
                trainer.project_name="mho-harness" \
                trainer.run_name="${RUN_NAME}" \
                trainer.ckpt_path="$CKPT_PATH" \
                trainer.export_path="$EXPORT_PATH" \
                trainer.max_ckpts_to_keep=${MAX_CKPTS_TO_KEEP} \
                trainer.hf_save_interval=${HF_SAVE_INTERVAL} \
                $@

    # --env NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
    # --env NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" \
    # --env NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    # --env NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH,ENV}" \
    # --env TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}" \
    # --env SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-600}" \
