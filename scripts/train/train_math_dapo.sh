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

NUM_GPUS="${NUM_GPUS:-8}"          # total GPUs visible on the node
NUM_NODES="${NUM_NODES:-1}"

# Non-colocated placement: policy+ref on their own GPUs, inference engines on the
# rest. Avoids the colocated CUDA-IPC weight-sync path (which deadlocked) -- sync
# now uses NCCL broadcast. Requires POLICY_NUM_GPUS + NUM_INFERENCE_ENGINES*TP == NUM_GPUS.
COLOCATE_ALL="${COLOCATE_ALL:-false}"
POLICY_NUM_GPUS="${POLICY_NUM_GPUS:-4}"

GPU_LIST="${GPU_LIST:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
MEGATRON_TP="${MEGATRON_TP:-4}"    # policy on POLICY_NUM_GPUS (=4): one replica, TP=4 PP=1
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-1}"
MEGATRON_ETP="${MEGATRON_ETP:-null}"

NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-4}"   # 4 engines x TP1 = 4 inference GPUs
INFERENCE_ENGINE_TP="${INFERENCE_ENGINE_TP:-1}"

# Fully-async (off-policy) training: sampler runs ahead of the trainer by up to
# max_staleness_steps. num_parallel_generation_workers caps concurrent trajectories
# -- each is a harbor sandbox (nested apptainer), so keep it modest, not the 768 default.
FULLY_ASYNC="${FULLY_ASYNC:-true}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-2}"
# Each parallel worker = one nested apptainer sandbox with a RAM-backed --writable-tmpfs
# overlay. Too many at once (16) drove tmpfs/memory pressure -> SIGBUS on the fuse/sandbox
# processes and an unreachable engine mid-eval. Start conservative; raise once stable.
NUM_PARALLEL_GEN_WORKERS="${NUM_PARALLEL_GEN_WORKERS:-8}"
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

# PyTorch expandable_segments (CUDA VMM) on the training workers. SkyRL defaults it
# True, but only the FSDP worker auto-disables it around weight sync; the Megatron
# worker does not. Colocated with vLLM's sleep-mode CuMemAllocator (also VMM), it
# clobbers the VA range vLLM reserved, so vLLM's wake-for-weight-sync dies with
# `CUDA Error: invalid argument at cumem_allocator.cpp:258`. Off for colocation.
USE_EXPANDABLE_SEGMENTS="${USE_EXPANDABLE_SEGMENTS:-false}"

# --- Nested-apptainer support for harbor's `singularity` trial environment ------
# Each rollout trial runs the agent in an apptainer sandbox NESTED inside this
# training container. The SIF has no apptainer binary/libs, so we bind the host's
# apptainer in (below) and stage here the shared libs the image lacks. Proven
# minimal set (see _container_run.sh for why ldconfig, not LD_LIBRARY_PATH).
# Under /scratch (like HF_DIR/TMP_DIR) -- storage outside /scratch is limited, and
# these (host-specific libs + a re-pullable sif) are cheap to regenerate if purged.
NESTED_DIR="${NESTED_DIR:-${BASE_DIR%/}/lsutawik/nested_apptainer}"
HOSTLIBS_DIR_HOST="${NESTED_DIR}/hostlibs"
SIF_CACHE_HOST="${NESTED_DIR}/sif_cache"
mkdir -p "$HOSTLIBS_DIR_HOST" "$SIF_CACHE_HOST"
_stage_lib() {  # <soname>: copy newest matching host lib -> hostlibs/<soname>
  local soname="$1" src
  [ -e "$HOSTLIBS_DIR_HOST/$soname" ] && return 0
  src=$(find /usr/lib64 /lib64 -maxdepth 1 -name "${soname}*" 2>/dev/null | sort | tail -1)
  [ -n "$src" ] && cp -f "$src" "$HOSTLIBS_DIR_HOST/$soname"
}
for _lib in libsubid.so.3 libseccomp.so.2 libcrypt.so.2 libfuse3.so.3 \
            liblz4.so.1 liblzo2.so.2 libzstd.so.1 liblzma.so.5; do
  _stage_lib "$_lib"
done
# Pre-seed the agent sandbox image so harbor skips a per-image docker pull inside
# the container. harbor caches as <docker_image, / and : -> _>.sif, i.e. the
# python:3.11-slim in each task.toml maps to python_3.11-slim.sif. We BUILD (not
# pull) so we can bake in /app -- harbor launches the sandbox with `--pwd /app`
# (its default workdir when the task has no Dockerfile), and stock python:3.11-slim
# has no /app, so a plain pull dies with "chdir /app: no such file or directory".
AGENT_SIF="${SIF_CACHE_HOST}/python_3.11-slim.sif"
if [ ! -f "$AGENT_SIF" ]; then
  echo "Pre-building agent sandbox image -> $AGENT_SIF"
  APPTAINER_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME:-}" \
  APPTAINER_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD:-}" \
    apptainer build --fakeroot --force "$AGENT_SIF" "${REPO_DIR}/scripts/train/agent_sandbox.def"
fi
# Prebuild harbor's server venv (uvicorn+fastapi) at /opt/harbor-server, bind-mounted
# into each trial sandbox (see harbor_trial_config/default.yaml environment.mounts).
# /opt is a read-only volume in the sandbox, so bootstrap.sh can't create it there;
# a prebuilt bind lets it skip venv creation + the per-trial pip install. Built via
# the sandbox's own python so shebangs resolve to /opt/harbor-server/bin/python3.
HARBOR_OPT_HOST="${NESTED_DIR}/harbor_server_opt"
# Guard on a real host-side file: harbor-server/bin/python3 is a symlink to the
# sandbox's /usr/local/bin/python3, which doesn't exist on the host, so `-x` on it
# is always false and would rebuild every launch. pyvenv.cfg is a plain file that
# only exists once the venv (and its pip installs) completed.
if [ ! -f "${HARBOR_OPT_HOST}/harbor-server/pyvenv.cfg" ]; then
  echo "Pre-building harbor server venv -> ${HARBOR_OPT_HOST}/harbor-server"
  mkdir -p "$HARBOR_OPT_HOST"
  apptainer exec --writable-tmpfs --fakeroot -B "${HARBOR_OPT_HOST}:/opt" "$AGENT_SIF" \
    bash -c 'python3 -m venv /opt/harbor-server && /opt/harbor-server/bin/pip install --no-cache-dir uvicorn fastapi'
fi

echo ${BASE_SIF}

singularity exec --nv --writable-tmpfs \
    --workdir "${TMP_DIR}" \
    --bind "${HF_DIR}:/root/.cache/huggingface" \
    --bind "${TMP_DIR}:/tmp_work" \
    --env TMPDIR=/tmp_work \
    --env HF_HOME=/root/.cache/huggingface \
    --env CPATH= \
    --env PYTHONPATH="${REPO_DIR}/src" \
    --env MICRO_SCAFFOLD_DIR="${MINI_FORK_LOCAL}" \
    --env MSWEA_API_KEY="${MSWEA_API_KEY:-dummy}" \
    --env AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-32768}" \
    --env RAY_worker_register_timeout_seconds=600 \
    --env NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
    --env NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" \
    --env NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    --env NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH,ENV}" \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}" \
    --env SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-600}" \
    --env CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    --bind "${UV_CACHE_PERSIST}:/root/.cache/uv" \
    --env UV_CACHE_DIR=/root/.cache/uv \
    --bind /usr/bin/apptainer:/usr/bin/apptainer \
    --bind /usr/bin/apptainer:/usr/bin/singularity \
    --bind /usr/libexec/apptainer:/usr/libexec/apptainer \
    --bind /etc/apptainer:/etc/apptainer \
    --bind /var/lib/apptainer:/var/lib/apptainer \
    --bind "${NESTED_DIR}:/mnt" \
    --env HOSTLIBS_DIR=/mnt/hostlibs \
    --env TRITON_CACHE_DIR=/tmp_work/triton_cache \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp_work/torchinductor_cache \
    --env VLLM_CACHE_ROOT=/tmp_work/vllm_cache \
    --env XDG_CACHE_HOME=/tmp_work/xdg_cache \
    --env APPTAINER_CACHEDIR=/tmp_work/nested_apptainer_cache \
    --env APPTAINER_TMPDIR=/tmp_work \
    --env APPTAINER_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME:-}" \
    --env APPTAINER_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD:-}" \
    "${BASE_SIF}" \
        bash "${REPO_DIR}/scripts/train/_container_run.sh" \
        uv run \
        --isolated --python 3.12 --extra megatron \
            -m src.train_entrypoint \
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
                trainer.run_name="${RUN_NAME}" \
                trainer.ckpt_path="$CKPT_PATH" \
                trainer.export_path="$EXPORT_PATH" \
                trainer.max_ckpts_to_keep=${MAX_CKPTS_TO_KEEP} \
                trainer.hf_save_interval=${HF_SAVE_INTERVAL} \
                $@
