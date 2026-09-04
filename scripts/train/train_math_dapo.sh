#!/usr/bin/env bash


set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

source .env

# USER_DATA is defined in .env (sourced above); fall back only if unset.
USER_DATA="${USER_DATA:-/data/user_data/lsutawik}"

UV_CACHE_PERSIST="${UV_CACHE_PERSIST:-${USER_DATA}/uv_cache}"
mkdir -p "$UV_CACHE_PERSIST"

MATH_DATA_DIR="${MATH_DATA_DIR:-${USER_DATA}/mho-model-harness-optimization/data-harbor/DAPO-Math-17k}"
MATH_TRAIN_DIR="${MATH_TRAIN_DIR:-${MATH_DATA_DIR}/train}"

MAX_TRAIN="${MAX_TRAIN:-500}"
if [ ! -d "$MATH_TRAIN_DIR" ]; then
  .venv/bin/python src/harbor/prepare_math_tasks.py --out "$MATH_TRAIN_DIR" --split train --max-tasks "$MAX_TRAIN"
fi

MATH_VAL_DIR="${MATH_VAL_DIR:-$MATH_TRAIN_DIR}"

RUN_ID="${RUN_ID:-math}"
CAND_ID="${CAND_ID:-0}"
RUNS_ROOT="${RUNS_ROOT:-${USER_DATA}/mho-model-harness-optimization/runs}"
export MINI_FORK_LOCAL="$RUNS_ROOT/run_${RUN_ID}/candidate_${CAND_ID}/micro-swe-agent"
if [ ! -d "$MINI_FORK_LOCAL" ]; then
MICRO_SCAFFOLD_BASE="${MICRO_SCAFFOLD_BASE:-$REPO_DIR/../micro-swe-agent}" \
    PYTHONPATH=src .venv/bin/python -m harness.scaffold materialize \
      --run-id "$RUN_ID" --candidate-id "$CAND_ID" \
      --runs-dir "$RUNS_ROOT"
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/mho_qwen3.5-4b_math}"

NUM_GPUS="${NUM_GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"

GPU_LIST="${GPU_LIST:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
MEGATRON_TP="${MEGATRON_TP:-2}"
MEGATRON_PP="${MEGATRON_PP:-2}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-1}"
MEGATRON_ETP="${MEGATRON_ETP:-null}"

NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-2}"
INFERENCE_ENGINE_TP="${INFERENCE_ENGINE_TP:-2}"
# Colocated with Megatron training on the same GPUs, so vLLM must leave headroom
# for policy weights/optimizer/activations -- 0.8 OOMs the engine core at startup.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
# Skip vLLM torch.compile + CUDA-graph capture. The Qwen3.5 compile path on this
# cu13/torch-2.11/vLLM-0.28 stack segfaults the engine core during profile_run
# (this is the instability the original DAPO recipe's enforce_eager guarded against).
# NOTE: SkyRL force-disables this when LoRA weight sync is active (config.py ~1767).
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"

N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
APPLY_OVERLONG_FILTERING="${APPLY_OVERLONG_FILTERING:-true}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
LR="${LR:-1e-6}"

# Trainer
EPOCHS="${EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
LOGGER="${LOGGER:-wandb}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-4B}"

echo ${BASE_SIF}

singularity exec --nv --writable-tmpfs \
    --workdir "${TMP_DIR}" \
    --bind "${HF_DIR}:/root/.cache/huggingface" \
    --bind "${TMP_DIR}:/tmp_work" \
    --env TMPDIR=/tmp_work \
    --env HF_HOME=/root/.cache/huggingface \
    --env CPATH= \
    --env RAY_worker_register_timeout_seconds=600 \
    --env CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    --bind "${UV_CACHE_PERSIST}:/root/.cache/uv" \
    --env UV_CACHE_DIR=/root/.cache/uv \
    "${BASE_SIF}" \
        uv run \
        --isolated --python 3.12 --extra megatron \
            -m src.train_entrypoint \
                data.train_data="['${MATH_TRAIN_DIR}']" \
                data.val_data="['${MATH_VAL_DIR}']" \
                trainer.algorithm.advantage_estimator="grpo" \
                trainer.policy.model.path="${MODEL}" \
                trainer.placement.colocate_all=true \
                trainer.strategy=megatron \
                trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
                trainer.placement.ref_num_gpus_per_node=${NUM_GPUS} \
                trainer.placement.policy_num_nodes=${NUM_NODES} \
                trainer.placement.ref_num_nodes=${NUM_NODES} \
                trainer.policy.megatron_config.tensor_model_parallel_size=${MEGATRON_TP} \
                trainer.policy.megatron_config.pipeline_model_parallel_size=${MEGATRON_PP} \
                trainer.policy.megatron_config.context_parallel_size=${MEGATRON_CP} \
                trainer.policy.megatron_config.expert_model_parallel_size=${MEGATRON_EP} \
                trainer.policy.megatron_config.expert_tensor_parallel_size=${MEGATRON_ETP} \
                generator.inference_engine.backend=vllm \
                generator.inference_engine.run_engines_locally=True \
                generator.inference_engine.num_engines=${NUM_INFERENCE_ENGINES} \
                generator.inference_engine.tensor_parallel_size=${INFERENCE_ENGINE_TP} \
                generator.inference_engine.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
                generator.inference_engine.enforce_eager=${ENFORCE_EAGER} \
                generator.inference_engine.served_model_name=${SERVED_MODEL_NAME} \
                generator.batched=true \
                generator.apply_overlong_filtering=${APPLY_OVERLONG_FILTERING} \
                generator.n_samples_per_prompt=${N_SAMPLES_PER_PROMPT} \
                generator.sampling_params.max_generate_length=${MAX_GENERATE_LENGTH} \
                generator.sampling_params.temperature=${TEMPERATURE} \
                trainer.epochs=${EPOCHS} \
                trainer.eval_batch_size=${EVAL_BATCH_SIZE} \
                trainer.eval_before_train=true \
                trainer.eval_interval=${EVAL_INTERVAL} \
                trainer.train_batch_size=${TRAIN_BATCH_SIZE} \
                trainer.policy_mini_batch_size=${POLICY_MINI_BATCH_SIZE} \
                trainer.max_prompt_length=${MAX_PROMPT_LENGTH} \
                trainer.algorithm.max_seq_len=${MAX_SEQ_LEN} \
                trainer.policy.optimizer_config.lr=${LR} \
                trainer.algorithm.use_kl_loss=${USE_KL_LOSS} \
                trainer.logger=${LOGGER} \
                trainer.project_name="mho-harness" \
                trainer.run_name="${RUN_ID}_qwen3.5-4b_math" \
                trainer.ckpt_path="$CKPT_PATH" \
                $@
