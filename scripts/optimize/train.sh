#!/usr/bin/env bash
# MHO training (Miles) -- the single main training script. Env-only; run directly, via the
# optimizer loop (scripts/optimize/train.sh is what optimize_loop submits), or through the
# hpc submitters (bash scripts/hpc/submit_pbs.sh scripts/optimize/train.sh).
#
# Miles runs Harbor trials IN-PROCESS in the rollout worker; the sandbox container is
# provisioned on the host via the local_singularity backend (HARBOR_ENV_TYPE=singularity +
# the remote_env registry swap -> RemoteSingularityEnvironment -> POST /sandbox). Token-in/out
# is recorded natively (--tito-model). HF->Megatron conversion is done by run.py prepare().
# The Miles launcher is src/run_training.py, bind-mounted over the example run.py.
#
# Optimizer interface (env): MHO_MODEL -> HF_MODEL, MHO_HARNESS -> MICRO_SCAFFOLD_DIR/MINI_FORK_LOCAL,
#   MHO_TRAIN_DATA -> MATH_TRAIN_DIR, MHO_VAL_DATA -> MATH_VAL_DIR, MHO_OUT -> RUN_DIR.
#   (SkyRL training lives separately in scripts/train/train_math_dapo.sh.)
set -euo pipefail
REPO_DIR="${MHO_REPO_DIR:-${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}}"
cd "$REPO_DIR"
source .env

# --- optimizer interface: map MHO_* (set by optimize_loop / hpc submitters) to the
# internal knobs below; each also honors its own env name for manual runs. ---
[ -n "${MHO_MODEL:-}" ]      && HF_MODEL="$MHO_MODEL"
[ -n "${MHO_HARNESS:-}" ]    && { MICRO_SCAFFOLD_DIR="$MHO_HARNESS"; MINI_FORK_LOCAL="$MHO_HARNESS"; }
[ -n "${MHO_TRAIN_DATA:-}" ] && MATH_TRAIN_DIR="$MHO_TRAIN_DATA"
[ -n "${MHO_VAL_DATA:-}" ]   && MATH_VAL_DIR="$MHO_VAL_DATA"
[ -n "${MHO_OUT:-}" ]        && RUN_DIR="$MHO_OUT"

# --- storage / data (mirror train_math_dapo.sh) ---------------------------------
USER_DATA="${USER_DATA:-/data/user_data/lsutawik}"
# BASE_SIF (from .env) is the only sif knob: the docker->sif cache is its sibling.
[ -n "${BASE_SIF:-}" ] || { echo "ERROR: BASE_SIF is not set (put it in ${REPO_DIR}/.env)" >&2; exit 2; }
SIF_IMAGE_CACHE_DIR="$(dirname "${BASE_SIF}")/sif_cache"
mkdir -p "$SIF_IMAGE_CACHE_DIR"
MATH_DATA_DIR="${MATH_DATA_DIR:-${USER_DATA}/mho-model-harness-optimization/data-harbor/DAPO-Math-17k}"
MATH_TRAIN_DIR="${MATH_TRAIN_DIR:-${MATH_DATA_DIR}/train}"
MAX_TRAIN="${MAX_TRAIN:-500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
# Prepare training task dirs if absent (prepare.py writes both the harbor task dirs
# AND the Miles rollout.jsonl). If the dirs already exist from a pre-jsonl prep, the
# --emit-rollout-only backfill below (re)writes rollout.jsonl from them -- idempotent,
# no dataset download, and single-sourced with prepare() so the two never drift.
if [ ! -d "$MATH_TRAIN_DIR" ]; then
  uv run --no-project --with datasets tasks/dapo_math_17k/prepare.py --out "$MATH_TRAIN_DIR" --split train --max-tasks "$MAX_TRAIN"
fi
if [ ! -f "${MATH_TRAIN_DIR}/rollout.jsonl" ]; then
  uv run --no-project tasks/dapo_math_17k/prepare.py --emit-rollout-only --out "$MATH_TRAIN_DIR"
fi
MATH_VAL_DIR="${MATH_VAL_DIR:-${MATH_DATA_DIR}/test_${EVAL_BATCH_SIZE}}"
if [ ! -d "$MATH_VAL_DIR" ]; then
  uv run --no-project --with datasets tasks/dapo_math_17k/prepare.py --out "$MATH_VAL_DIR" --split train --max-tasks "$((MAX_TRAIN + EVAL_BATCH_SIZE))" --start "$MAX_TRAIN"
fi

# Harbor task dir consumed by the connector (one task dir per instance_id).
HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-${MATH_TRAIN_DIR}}"

# --- run dir / logging ----------------------------------------------------------
RUN_NAME="${RUN_NAME:-mho_qwen3.5-4b_math_miles}"
RUN_DIR="${RUN_DIR:-${USER_DATA}/mho-model-harness-optimization/runs_output/${RUN_NAME}}"
mkdir -p "$RUN_DIR"
RUN_LOG="${RUN_LOG-${RUN_DIR}/run.log}"
if [ -n "$RUN_LOG" ]; then
  rm -f "$RUN_LOG"
  exec > >(tee "$RUN_LOG") 2>&1
fi
SAVE_DIR="${SAVE_DIR:-${RUN_DIR}/checkpoints}"
mkdir -p "$SAVE_DIR"
# HF-format checkpoint export. "{}" is filled with the rollout id (e.g. .../hf/step_100).
# Saved at --save-interval; HF checkpoints are never auto-deleted. Set SAVE_HF="" to disable.
SAVE_HF="${SAVE_HF:-${SAVE_DIR}/hf/step_{}}"
[ -n "$SAVE_HF" ] && mkdir -p "$(dirname "$SAVE_HF")"

# --- knobs (mirror train_math_dapo.sh) ------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-4B}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-4B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-true}"
LOGGER="${LOGGER:-wandb}"

# --- local_singularity executor auto-start (same pattern as SkyRL) ---------------
# HARBOR_ENV_TYPE=singularity -> the connector builds EnvironmentType.SINGULARITY,
# which (via the registry swap in remote_env.py) resolves to RemoteSingularityEnvironment,
# which POSTs /sandbox to this host service. It must be running before training.
if [ "${HARBOR_ENV_TYPE:-singularity}" = "singularity" ] && [ -n "${EXECUTOR_URL:-}" ]; then
  EXECUTOR_PYTHON="${EXECUTOR_PYTHON:-/home/aci18914wh/executor_env/.venv/bin/python}"
  EXECUTOR_LOG="${EXECUTOR_LOG:-${HOME}/executor_svc.log}"
  EXECUTOR_PORT="${EXECUTOR_PORT:-${EXECUTOR_URL##*:}}"; EXECUTOR_PORT="${EXECUTOR_PORT%%/*}"
  echo "[executor] (re)starting local_singularity service on ${EXECUTOR_URL} (log: ${EXECUTOR_LOG})"
  pkill -9 -f 'mho.backends.local_singularity.service' 2>/dev/null || true
  pkill -9 -f 'executor_service' 2>/dev/null || true
  sleep 2
  PYTHONPATH="${REPO_DIR}/src" EXECUTOR_PORT="${EXECUTOR_PORT}" \
    HB_STAGING_ROOT="${BASE_DIR:-${TMP_DIR}}/hbstaging" SIF_IMAGE_CACHE_DIR="${SIF_IMAGE_CACHE_DIR}" \
    nohup "${EXECUTOR_PYTHON}" -m mho.backends.local_singularity.service > "${EXECUTOR_LOG}" 2>&1 &
  for _i in $(seq 1 20); do
    curl -sf --max-time 2 "${EXECUTOR_URL%/}/health" >/dev/null 2>&1 && break
    sleep 1
  done
  if curl -sf --max-time 3 "${EXECUTOR_URL%/}/health" >/dev/null 2>&1; then
    echo "[executor] healthy at ${EXECUTOR_URL}"
  else
    echo "[executor] ERROR: did not become healthy; see ${EXECUTOR_LOG}" >&2
    exit 1
  fi
fi

# --- launch via miles harbor run.py inside miles.sif -----------------------------
MILES_SIF="${MILES_SIF:-$(dirname "${BASE_SIF}")/miles.sif}"
MILES_RUN_PY=/root/miles/examples/experimental/harbor/run.py
[ -f "$MILES_SIF" ] || { echo "miles.sif not found: $MILES_SIF" >&2; exit 1; }
# --- full HF model pre-download ---------------------------------------------------
# The first conversion run raced an unauthenticated, streaming model download, leaving
# the Qwen3.5-4B snapshot partial in the bound HF cache -- so convert_hf_to_torch_dist
# could not find later-shard weights (e.g. model.language_model.layers.24...). Force the
# whole snapshot into ${HF_DIR} (bound at /root/.cache/huggingface) up front.
echo "[miles] ensuring full HF snapshot for ${HF_MODEL}"
singularity exec \
  --bind "${HF_DIR}:/root/.cache/huggingface" \
  --env HF_HOME=/root/.cache/huggingface \
  "$MILES_SIF" \
    /opt/sglang/bin/python -c \
      "from huggingface_hub import snapshot_download; \
       p=snapshot_download('${HF_MODEL}'); \
       print('HF snapshot complete:', p)" \
  || echo "[miles] WARN: pre-download failed -- converter may hit shard gaps"


singularity exec --nv \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --bind "${USER_DATA}:${USER_DATA}" \
  --bind "${HF_DIR}:/root/.cache/huggingface" \
  --bind "${TMP_DIR}:/tmp_work" \
  --bind "${HARBOR_TASKS_DIR}:${HARBOR_TASKS_DIR}" \
  --bind "${SAVE_DIR}:${SAVE_DIR}" \
  --bind "${REPO_DIR}/src/run_training.py:/root/miles/examples/experimental/harbor/run.py" \
  --env TMPDIR=/tmp_work \
  --env HF_HOME=/root/.cache/huggingface \
  --env PYTHONPATH="${REPO_DIR}/src" \
  --env MICRO_SCAFFOLD_DIR="${MICRO_SCAFFOLD_DIR:-}" \
  --env MINI_FORK_LOCAL="${MINI_FORK_LOCAL:-}" \
  --env MILES_SAVE_HF="${SAVE_HF}" \
  --env HARBOR_ENV_TYPE="${HARBOR_ENV_TYPE:-singularity}" \
  --env HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR}" \
  --env HARBOR_TRIALS_DIR="${RUN_DIR}/harbor_trials" \
  --env EXECUTOR_URL="${EXECUTOR_URL}" \
  --env MSWEA_API_KEY="${MSWEA_API_KEY:-dummy}" \
  --env AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-16384}" \
  --env AGENT_EXEC_TIMEOUT_SEC="${AGENT_EXEC_TIMEOUT_SEC:-1800}" \
  --env MILES_ROUTER_EXTERNAL_HOST="${MILES_ROUTER_EXTERNAL_HOST:-$(hostname)}" \
  --env CUDA_VISIBLE_DEVICES="${GPU_LIST:-$(seq -s, 0 $((NUM_GPUS - 1)))}" \
  "$MILES_SIF" \
    /opt/sglang/bin/python "$MILES_RUN_PY" \
      --megatron-model-type "$MODEL_KEY" \
      --model-name "$MODEL_KEY" \
      --base-dir "$SAVE_DIR" \
      --hf-checkpoint "$HF_MODEL" \
      --ref-load "$SAVE_DIR/${MODEL_KEY}_torch_dist" \
      --save-dir "$SAVE_DIR" \
      --prompt-data "${MATH_TRAIN_DIR}/rollout.jsonl" \
      --harbor-tasks-dir "$HARBOR_TASKS_DIR" \
      --num-gpus-per-node "$NUM_GPUS" \
      --num-nodes "$NUM_NODES" \
      --max-seq-len "$MAX_MODEL_LEN" \
      --num-rollout "$NUM_ROLLOUT" \
      --rollout-batch-size "$TRAIN_BATCH_SIZE" \
      --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
      --global-batch-size "$GLOBAL_BATCH_SIZE" \
      --wandb-key "${WANDB_API_KEY:-}" \
      --wandb-project mho-harness \
      --wandb-run-name "$RUN_NAME" \
      $@
