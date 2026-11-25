#!/usr/bin/env bash
# MHO training (Miles) -- the single main training script. Env-only; run directly, via the
# optimizer loop (scripts/optimize/train.sh is what optimize_loop submits), or through the
# hpc submitters (bash scripts/hpc/submit_pbs.sh scripts/optimize/train.sh).
#
# Miles runs Harbor trials IN-PROCESS in the rollout worker; the sandbox container is
# provisioned on the host via the local_singularity backend (HARBOR_ENV_TYPE=singularity +
# the backend registry swap -> RemoteSingularityEnvironment -> POST /sandbox). Token-in/out
# is recorded natively (--tito-model). HF->Megatron conversion is done by run.py prepare().
# The Miles launcher is src/run_training.py, bind-mounted over the example run.py.
#
# Optimizer interface (env): MHO_MODEL -> HF_MODEL, MHO_HARNESS -> MICRO_SCAFFOLD_DIR/MINI_FORK_LOCAL,
#   MHO_TRAIN_DATA -> MATH_TRAIN_DIR, MHO_VAL_DATA -> MATH_VAL_DIR, MHO_OUT -> RUN_DIR.
#   (SkyRL training lives separately in scripts/train/train_math_dapo.sh.)
set -euo pipefail
REPO_DIR="${MHO_REPO_DIR:-${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}}"
cd "$REPO_DIR"
. scripts/utils/load_env.sh && load_env .env

# --- optimizer interface: map MHO_* (set by optimize_loop / hpc submitters) to the
# internal knobs below; each also honors its own env name for manual runs. ---
[ -n "${MHO_MODEL:-}" ]      && HF_MODEL="$MHO_MODEL"
[ -n "${MHO_HARNESS:-}" ]    && { MICRO_SCAFFOLD_DIR="$MHO_HARNESS"; MINI_FORK_LOCAL="$MHO_HARNESS"; }
[ -n "${MHO_TRAIN_DATA:-}" ] && MATH_TRAIN_DIR="$MHO_TRAIN_DATA"
[ -n "${MHO_VAL_DATA:-}" ]   && MATH_VAL_DIR="$MHO_VAL_DATA"
[ -n "${MHO_OUT:-}" ]        && RUN_DIR="$MHO_OUT"
# No harness passed (a model candidate on the base harness) -> roll out with the BASE scaffold,
# as eval.sh does. The agent hard-requires a snapshot (harness/scaffold.py:materialize); without
# one every rollout fails at agent install ("No scaffold snapshot set"), scores 0, and GRPO
# resamples forever since no group has reward variance.
if [ -z "${MICRO_SCAFFOLD_DIR:-}" ]; then
  MICRO_SCAFFOLD_DIR="$(PYTHONPATH="${REPO_DIR}/src" "${REPO_DIR}/.venv/bin/python" -c \
    "from harness import scaffold; print(scaffold.DEFAULT_BASE)" 2>/dev/null || true)"
  [ -n "${MICRO_SCAFFOLD_DIR}" ] && [ -d "${MICRO_SCAFFOLD_DIR}" ] \
    || { echo "ERROR: no harness set and base scaffold not found (set MHO_HARNESS or MICRO_SCAFFOLD_BASE)" >&2; exit 2; }
  MINI_FORK_LOCAL="${MINI_FORK_LOCAL:-$MICRO_SCAFFOLD_DIR}"
  echo "[train] no harness passed -> using base scaffold: ${MICRO_SCAFFOLD_DIR}"
fi

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
# The train/val task dirs + their seed.jsonl are produced by tasks/<task>/main.py
# (e.g. `uv run --with datasets python -m tasks.arxivmath_training --val-size 1000
# --output-dir out/x` -> out/x-train-split + out/x-val-split). Each split dir carries a
# seed.jsonl: one row per task dir {prompt, metadata.instance_id} that SEEDS Miles'
# rollout dataset (the RL loop generates the rollouts from it). Point MHO_TRAIN_DATA /
# MHO_VAL_DATA at those split dirs; require seed.jsonl (--prompt-data consumes it).
[ -f "${MATH_TRAIN_DIR}/seed.jsonl" ] || {   echo "ERROR: no ${MATH_TRAIN_DIR}/seed.jsonl -- prepare the task first (tasks/<task>/main.py), then set MHO_TRAIN_DATA." >&2;   exit 2; }
MATH_VAL_DIR="${MATH_VAL_DIR:-}"

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

# --- sigma reward (model candidates) ----------------------------------------------
# MHO_SIGMA_REWARD = <cand>/model/sigma_rewards/<sigma>.json (set by optimize_loop for a model
# candidate). Stage a copy of the train split whose tests/ run the task's own verifier and then
# the sigma rule (mho.data.reward_synthesis stage), and train on that copy. The base split is
# never modified. Unset -> train on the plain split (correctness reward).
if [ -n "${MHO_SIGMA_REWARD:-}" ]; then
  [ -f "$MHO_SIGMA_REWARD" ] || { echo "ERROR: MHO_SIGMA_REWARD=$MHO_SIGMA_REWARD not found" >&2; exit 2; }
  SIGMA_TASKS_DIR="${RUN_DIR}/sigma_tasks"
  _PY="${REPO_DIR}/.venv/bin/python"; [ -x "$_PY" ] || _PY=python3
  PYTHONPATH="${REPO_DIR}/src" "$_PY" -m mho.data.reward_synthesis stage \
    --train-dir "$MATH_TRAIN_DIR" --sigma "$MHO_SIGMA_REWARD" --out "$SIGMA_TASKS_DIR"
  MATH_TRAIN_DIR="$SIGMA_TASKS_DIR"
  HARBOR_TASKS_DIR="$SIGMA_TASKS_DIR"
fi
SAVE_DIR="${SAVE_DIR:-${RUN_DIR}/checkpoints}"
mkdir -p "$SAVE_DIR"
# HF-format checkpoint export. Miles fills "{rollout_id}" (save_hf.format(rollout_id=...)), e.g. .../hf/step_100.
# Saved at --save-interval; HF checkpoints are never auto-deleted. Set SAVE_HF="" to disable.
SAVE_HF="${SAVE_HF:-${SAVE_DIR}/hf/step_{rollout_id}}"
[ -n "$SAVE_HF" ] && mkdir -p "$(dirname "$SAVE_HF")"

# --- knobs (mirror train_math_dapo.sh) ------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-4B}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-4B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# The rollout agent must call the SERVED model by its served name. run_training.py reads
# AGENT_MODEL_NAME and defaults to a bogus "model"; without this the trial's agent config has
# model_name="openai/model" and the agent makes zero model calls (No model calls recorded).
export AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-${MODEL_KEY}}"
# Batch shape follows WHALE's verl script for both objectives, so they compare at matched
# rollouts: 8 rollouts per prompt, updates in minibatches of 64 trajectories
# (GLOBAL_BATCH_SIZE), lr 1e-6, temperature 1.0 (mho/train/objectives.py).
#   rsft: 256 prompts per step (WHALE's RSFT batch); SFT on the kept ones only.
#   grpo:  64 prompts per step (WHALE's GRPO script default); groups without reward variance
#          are resampled, so a step can generate more than 64 x 8 rollouts.
# A 0.2-epoch phase on a ~2.4k-task train split is then ~2 rsft steps or ~8 grpo steps
# (~4k rollouts each). Explicit env values win.
if [ "${MHO_TRAIN_OBJECTIVE:-grpo}" = "rsft" ]; then
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-true}"
LOGGER="${LOGGER:-wandb}"

# --- local_singularity executor auto-start (same pattern as SkyRL) ---------------
# HARBOR_ENV_TYPE=singularity -> the connector builds EnvironmentType.SINGULARITY,
# which (via the registry swap in backend.py) resolves to RemoteSingularityEnvironment,
# which POSTs /sandbox to this host service. It must be running before training.
# EXECUTOR_URL is auto-inferred (defaults to 127.0.0.1:8900, matching backend.py's
# _executor_url fallback) so the executor starts without the caller providing it.
# Disable with MHO_SKIP_EXECUTOR=1.
EXECUTOR_URL="${EXECUTOR_URL:-http://127.0.0.1:8900}"
# MHO_SANDBOX=sail: trials run in Sail cloud sandboxes (no host executor; see the relay below).
# MHO_SANDBOX=hpcsb: trials run in hpc-sandbox pods on THIS node (own network namespace per trial,
# egress via slirp4netns), driven by the host executor's /hpcsb routes (mho.backends.hpcsb_remote
# in the trainer). No relay: pods reach the session servers at the node's address. Knobs (read by
# the executor): HPCSB_IMAGE_CACHE, HPCSB_NETWORK, HPCSB_RUNTIME, HPCSB_DOCKERFILE_STEPS,
# HPCSB_POD_USERS; the executor runs from HPCSB_VENV (needs harbor + hpc-sandbox).
MHO_SANDBOX="${MHO_SANDBOX:-singularity}"
case "${MHO_SANDBOX}" in singularity|sail|hpcsb) ;; *) echo "ERROR: MHO_SANDBOX must be singularity|sail|hpcsb" >&2; exit 2 ;; esac
if [ "${MHO_SANDBOX}" = "hpcsb" ]; then
  HPCSB_VENV="${HPCSB_VENV:-/home/aci18914wh/mho-model-harness-optimization/.venv-sail018}"
  EXECUTOR_PYTHON="${EXECUTOR_PYTHON:-${HPCSB_VENV}/bin/python}"
  # Sandbox images on node-local scratch ($PBS_LOCALDIR, same path on every node of the job);
  # the shared /groups cache is only the fallback when there is no node-local dir.
  _HPCSB_IMG_DEFAULT="/groups/gai51740/aci18914wh/mho/hpc-sandbox-images"
  [ -n "${PBS_LOCALDIR:-}" ] && [ -d "${PBS_LOCALDIR}" ] && _HPCSB_IMG_DEFAULT="${PBS_LOCALDIR}/hpcsb-images"
  export HPCSB_IMAGE_CACHE="${HPCSB_IMAGE_CACHE:-${_HPCSB_IMG_DEFAULT}}"
  export HPCSB_NETWORK="${HPCSB_NETWORK:-pod}" HPCSB_RUNTIME="${HPCSB_RUNTIME:-}" \
         HPCSB_DOCKERFILE_STEPS="${HPCSB_DOCKERFILE_STEPS:-}" HPCSB_POD_USERS="${HPCSB_POD_USERS:-}"
  mkdir -p "${HPCSB_IMAGE_CACHE}"
fi
if [ "${HARBOR_ENV_TYPE:-singularity}" = "singularity" ] && [ "${MHO_SKIP_EXECUTOR:-0}" != "1" ] \
   && [ "${MHO_SANDBOX}" != "sail" ] && [ "${NUM_NODES}" -le 1 ]; then
  EXECUTOR_PYTHON="${EXECUTOR_PYTHON:-/home/aci18914wh/executor_env/.venv/bin/python}"
  EXECUTOR_LOG="${EXECUTOR_LOG:-${HOME}/executor_svc.log}"
  EXECUTOR_PORT="${EXECUTOR_PORT:-${EXECUTOR_URL##*:}}"; EXECUTOR_PORT="${EXECUTOR_PORT%%/*}"
  echo "[executor] (re)starting local_singularity service on ${EXECUTOR_URL} (log: ${EXECUTOR_LOG})"
  pkill -9 -f "mho.backends.local_singularity.service" 2>/dev/null || true
  pkill -9 -f "executor_service" 2>/dev/null || true
  sleep 2
  # Stage sandboxes on the job's node-local scratch ($PBS_LOCALDIR, wiped when the job ends,
  # bound into the trainer as BASE_DIR). Not the run dir: on /groups each sandbox costs files
  # against the group's shared file-count quota, and at the limit every provision fails
  # (Errno 122). The service deletes each sandbox's staging dirs when it stops, so a long job
  # does not fill the scratch either. Override with MHO_STAGING_DIR.
  STAGING_DEFAULT="${RUN_DIR}/hbstaging"
  [ -n "${PBS_LOCALDIR:-}" ] && [ -d "${PBS_LOCALDIR}" ] && STAGING_DEFAULT="${PBS_LOCALDIR}/hbstaging"
  PYTHONPATH="${REPO_DIR}/src" EXECUTOR_PORT="${EXECUTOR_PORT}" \
    HB_STAGING_ROOT="${MHO_STAGING_DIR:-${STAGING_DEFAULT}}" SIF_IMAGE_CACHE_DIR="${SIF_IMAGE_CACHE_DIR}" \
    nohup "${EXECUTOR_PYTHON}" -m mho.backends.local_singularity.service > "${EXECUTOR_LOG}" 2>&1 &
  for _i in $(seq 1 20); do
    curl -sf --max-time 2 "${EXECUTOR_URL%/}/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf --max-time 3 "${EXECUTOR_URL%/}/health" >/dev/null 2>&1 \
    || { echo "[executor] ERROR: did not become healthy; see ${EXECUTOR_LOG}" >&2; exit 1; }
  echo "[executor] healthy at ${EXECUTOR_URL}"
fi

# --- Sail sandboxes: reverse relay to the session server --------------------------------
# The agent runs INSIDE the sandbox and calls the policy at its session URL; a Sail sandbox
# cannot reach this node, so a relay is the public rendezvous and a bridge on this node dials
# out to it. backend.py rewrites each session URL to <relay>/k/<secret>/_hp/<host:port>/... .
# The relay is a Modal Sandbox (src/harness/relay/sandbox_relay.py) under the shared app
# "mho-relay": nothing is deployed per job, and Modal reaps it on its own once no bridge is
# connected (idle timeout), so a killed job leaks nothing. A Sandbox lives at most 24 h, so a
# job that runs longer ROTATES: every RELAY_ROTATE_S a new relay + bridge comes up, new trials
# read its URL from MHO_RELAY_URL_FILE, and the old pair is retired RELAY_DRAIN_S later (after
# the trials that started on it are done). The relay replays unanswered requests when its bridge
# reconnects (relay_server.py / bridge.py), so a dropped socket no longer loses responses.
MHO_RELAY_URL=""; MHO_RELAY_URL_FILE=""; RELAY_ROTATE_PID=""; RELAY_STATE=""
_relay_down_all() {
  [ -n "${RELAY_STATE}" ] && [ -f "${RELAY_STATE}" ] || return 0
  while read -r _sb _pid; do
    kill "${_pid}" 2>/dev/null
    (cd "${REPO_DIR}" && timeout 120 uv run python src/harness/relay/sandbox_relay.py down "${_sb}" \
      </dev/null >/dev/null 2>&1) || true
  done < "${RELAY_STATE}"
  echo "[relay] relays stopped"
}
_train_cleanup() {
  [ -n "${RELAY_ROTATE_PID}" ] && { pkill -P "${RELAY_ROTATE_PID}" 2>/dev/null; kill "${RELAY_ROTATE_PID}" 2>/dev/null; }
  _relay_down_all
  return 0
}
trap _train_cleanup EXIT
trap 'exit 143' TERM INT
# relay_up <n>: start relay #n and its bridge; sets RELAY_SB, RELAY_HOST, RELAY_BRIDGE_PID.
relay_up() {
  local out
  out="$(cd "${REPO_DIR}" && RELAY_SECRET="${RELAY_SECRET}" RELAY_REQUEST_TIMEOUT="${RELAY_REQUEST_TIMEOUT:-1800}" \
      timeout 600 uv run python src/harness/relay/sandbox_relay.py up \
        --label "${RUN_NAME}-${PBS_JOBID%%.*}-r$1" --idle-timeout "${RELAY_IDLE_TIMEOUT:-900}" \
        --cpu "${RELAY_CPU:-4}" 2>&1 | tail -1)" || { echo "[relay] up #$1 failed: ${out}" >&2; return 1; }
  RELAY_SB="$(printf '%s' "${out}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sandbox_id"])' 2>/dev/null)"
  RELAY_HOST="$(printf '%s' "${out}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["url"])' 2>/dev/null)"
  [ -n "${RELAY_SB}" ] && [ -n "${RELAY_HOST}" ] || { echo "[relay] up #$1: bad output: ${out}" >&2; return 1; }
  (cd "${REPO_DIR}" && RELAY_WS_URL="${RELAY_HOST/https:/wss:}/bridge" RELAY_SECRET="${RELAY_SECRET}" \
    VLLM_LOCAL_URL="http://127.0.0.1:${SESSION_SERVER_PORT}" BRIDGE_TIMEOUT="${RELAY_REQUEST_TIMEOUT:-1800}" \
    exec uv run --with websockets --with httpx python src/harness/relay/bridge.py) &
  RELAY_BRIDGE_PID=$!
  echo "${RELAY_SB} ${RELAY_BRIDGE_PID}" >> "${RELAY_STATE}"
  local i
  for i in $(seq 1 60); do
    curl -sf --max-time 10 "${RELAY_HOST}/health" 2>/dev/null | grep -q '"bridge_connected":true' && {
      echo "[relay] #$1 up: sandbox ${RELAY_SB}, ${RELAY_HOST}"; return 0; }
    sleep 5
  done
  echo "[relay] #$1: bridge never connected" >&2; return 1
}
# Write the relay URL for new trials (atomic; holds the secret, so owner-only).
relay_publish() {
  ( umask 077; printf '%s/k/%s\n' "${RELAY_HOST}" "${RELAY_SECRET}" > "${MHO_RELAY_URL_FILE}.tmp" )
  mv -f "${MHO_RELAY_URL_FILE}.tmp" "${MHO_RELAY_URL_FILE}"
}
relay_rotate_loop() {
  local n=1 old_sb="${RELAY_SB}" old_pid="${RELAY_BRIDGE_PID}" wait_s="${RELAY_ROTATE_S:-75600}"
  while sleep "${wait_s}"; do
    if relay_up $((n + 1)); then
      n=$((n + 1)); relay_publish
      echo "[relay] rotated: new trials use relay #${n}; retiring ${old_sb} in ${RELAY_DRAIN_S:-5400}s"
      ( sleep "${RELAY_DRAIN_S:-5400}"; kill "${old_pid}" 2>/dev/null
        cd "${REPO_DIR}" && timeout 120 uv run python src/harness/relay/sandbox_relay.py down "${old_sb}" \
          </dev/null >/dev/null 2>&1; echo "[relay] retired ${old_sb}" ) &
      old_sb="${RELAY_SB}"; old_pid="${RELAY_BRIDGE_PID}"; wait_s="${RELAY_ROTATE_S:-75600}"
    else
      echo "[relay] rotation attempt failed; retrying in 600s" >&2; wait_s=600
    fi
  done
}
if [ "${MHO_SANDBOX}" = "sail" ]; then
  export SAIL_API_KEY="${SAIL_API_KEY:-${SAIL_KEY:-}}"
  [ -n "${SAIL_API_KEY}" ] || { echo "ERROR: MHO_SANDBOX=sail needs SAIL_API_KEY (or SAIL_KEY in .env)" >&2; exit 2; }
  SESSION_SERVER_PORT="${SESSION_SERVER_PORT:-30000}"   # run_training.py: agentic_train_args default
  RELAY_SECRET="$(openssl rand -hex 24)"
  RELAY_STATE="${RUN_DIR}/.relays"; : > "${RELAY_STATE}"
  MHO_RELAY_URL_FILE="${RUN_DIR}/.relay_url"
  echo "[relay] starting sandbox relay (-> session servers from :${SESSION_SERVER_PORT}) ..."
  relay_up 1 || exit 1
  relay_publish
  MHO_RELAY_URL="${RELAY_HOST}/k/${RELAY_SECRET}"
  relay_rotate_loop &
  RELAY_ROTATE_PID=$!
  echo "[relay] trials run on Sail; rotation every ${RELAY_ROTATE_S:-75600}s"
fi

# --- launch via miles harbor run.py inside miles.sif -----------------------------
# Trainer image: MHO_TRAINER_SIF is the explicit override (we use miles only), else MILES_SIF,
# else the miles.sif sibling of BASE_SIF.
MILES_SIF="${MHO_TRAINER_SIF:-${MILES_SIF:-$(dirname "${BASE_SIF}")/miles.sif}}"
MILES_RUN_PY=/root/miles/examples/experimental/harbor/run.py
[ -f "$MILES_SIF" ] || { echo "miles.sif not found: $MILES_SIF" >&2; exit 1; }
# --- full HF model pre-download ---------------------------------------------------
# LOCAL CHECKPOINT: when HF_MODEL is a local dir (a prior candidate's saved HF checkpoint),
# the weights are already on disk -- snapshot_download would try to hit the hub with a local
# path and fail. Skip the hub download; the local dir is used directly as --hf-checkpoint.
if [ -d "${HF_MODEL}" ]; then
  echo "[miles] HF_MODEL is a LOCAL dir (prior checkpoint), skipping hub snapshot_download: ${HF_MODEL}"
  HF_DIR="$(dirname "${HF_MODEL}")"
else
  echo "[miles] ensuring full HF snapshot for ${HF_MODEL}"
  echo [train-env] HARBOR_ENV_TYPE="${HARBOR_ENV_TYPE:-(unset)}" MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-(unset)}"
  singularity exec \
    --bind "${HF_DIR}:/root/.cache/huggingface" \
    --env HF_HOME=/root/.cache/huggingface \
    "$MILES_SIF" \
      /opt/sglang/bin/python -c \
        "from huggingface_hub import snapshot_download; \
         p=snapshot_download('${HF_MODEL}'); \
         print('HF snapshot complete:', p)" \
    || echo "[miles] WARN: pre-download failed -- converter may hit shard gaps"
fi


# --- host python packages for the container --------------------------------------
# The rollout side needs a few packages that live only in host venvs (executor_env,
# harbor_singularity_hpc, ...). Putting those venvs' site-packages on PYTHONPATH would let them
# SHADOW the container's own stack (e.g. the repo venv's tokenizers 0.23.3 / numpy 2.5 broke
# the container transformers and the HF->Megatron converter). Instead link only the modules the
# container cannot already import into an overlay dir: the container's packages always win.
HOST_SITE_PACKAGES="${HOST_SITE_PACKAGES:-/home/aci18914wh/executor_env/.venv/lib/python3.12/site-packages ${REPO_DIR}/.venv/lib/python3.12/site-packages}"
PYOVERLAY="${RUN_DIR}/pyoverlay"
# The run dir (and a local start checkpoint) may live outside /home, e.g. on /groups, which
# singularity does not bind by default: bind them explicitly into the trainer container.
RUN_BINDS=(--bind "${RUN_DIR}:${RUN_DIR}")
[ -d "${HF_MODEL}" ] && RUN_BINDS+=(--bind "${HF_MODEL}:${HF_MODEL}")
singularity exec "${RUN_BINDS[@]}" "$MILES_SIF" env -u PYTHONPATH /opt/sglang/bin/python -I \
  "${REPO_DIR}/scripts/optimize/container_pyoverlay.py" "$PYOVERLAY" $HOST_SITE_PACKAGES

# Trainer container: binds + env shared by the launcher and (multi-node) every Ray worker node.
SING_ARGS=(--nv
  "${RUN_BINDS[@]}"
  --bind "${BASE_DIR}:${BASE_DIR}"
  --bind "${USER_DATA}:${USER_DATA}"
  --bind "${HF_DIR}:/root/.cache/huggingface"
  --bind "${TMP_DIR}:/tmp_work"
  --bind "${HARBOR_TASKS_DIR}:${HARBOR_TASKS_DIR}"
  --bind "${SAVE_DIR}:${SAVE_DIR}"
  --bind "${REPO_DIR}/src/run_training.py:/root/miles/examples/experimental/harbor/run.py"
  --bind "${REPO_DIR}/src/launch_common.py:/root/miles/examples/experimental/harbor/launch_common.py"
  --bind "${REPO_DIR}/src/mho/train/harbor_agent_function.py:/root/miles/examples/experimental/harbor/harbor_agent_function.py"
  --env TMPDIR=/tmp_work
  --env HF_HOME=/root/.cache/huggingface
  --env PYTHONPATH="${REPO_DIR}/src:${PYOVERLAY}"
  --env MICRO_SCAFFOLD_DIR="${MICRO_SCAFFOLD_DIR:-}"
  --env MINI_FORK_LOCAL="${MINI_FORK_LOCAL:-}"
  --env MILES_SAVE_HF="${SAVE_HF}"
  --env MILES_SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
  --env HARBOR_ENV_TYPE="${HARBOR_ENV_TYPE:-singularity}"
  --env MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-/home/aci18914wh/.modal.toml}"
  --env HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR}"
  --env HARBOR_TRIALS_DIR="${RUN_DIR}/harbor_trials"
  --env EXECUTOR_URL="${EXECUTOR_URL}"
  --env MHO_SANDBOX="${MHO_SANDBOX}"
  --env MHO_RELAY_URL="${MHO_RELAY_URL}"
  --env MHO_RELAY_URL_FILE="${MHO_RELAY_URL_FILE}"
  --env SAIL_API_KEY="${SAIL_API_KEY:-}"
  --env MSWEA_API_KEY="${MSWEA_API_KEY:-dummy}"
  --env AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-16384}"
  --env AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-qwen3.5-4B}"
  --env AGENT_EXEC_TIMEOUT_SEC="${AGENT_EXEC_TIMEOUT_SEC:-1800}"
  --env MILES_ROUTER_EXTERNAL_HOST="${MILES_ROUTER_EXTERNAL_HOST:-$(hostname)}"
  --env CUDA_VISIBLE_DEVICES="${GPU_LIST:-$(seq -s, 0 $((NUM_GPUS - 1)))}")

# --- multi-node (NUM_NODES>1, hpcsb only): executors + Ray cluster across the PBS job's nodes ----
# Node 0 (this one) is the Ray head: session servers are pinned there (miles pin_to_head), so the
# session URL keeps this node's hostname. Every node runs a sandbox executor bound to its cluster
# IP (token-protected); the trainer's RemoteHpcSandboxEnvironment spreads pods over them
# (EXECUTOR_URLS, least-loaded). Transfer dirs live on the shared run dir (HPCSB_XFER_ROOT) since
# the trainer and a pod's executor can be on different nodes. ssh between job nodes is refused
# on ABCI: remote processes start with pbsdsh. Split GPUs with ACTOR_NUM_NODES /
# ACTOR_GPUS_PER_NODE / ROLLOUT_NUM_GPUS (run_training.py).
MN_PIDS=()
MN_ENV=()
if [ "${NUM_NODES}" -gt 1 ]; then
  [ "${MHO_SANDBOX}" = "hpcsb" ] || { echo "ERROR: NUM_NODES>1 needs MHO_SANDBOX=hpcsb" >&2; exit 2; }
  MN_DIR="${RUN_DIR}/multinode"; rm -rf "${MN_DIR}"; mkdir -p "${MN_DIR}/xfer" "${MN_DIR}/logs"; chmod 700 "${MN_DIR}"
  HEAD_IP="$(hostname -i | awk '{print $1}')"
  NODE_IPS=("${HEAD_IP}")
  for k in $(seq 1 $((NUM_NODES - 1))); do
    pbsdsh -n "$k" -- /bin/bash -c "hostname -i | awk '{print \$1}' > ${MN_DIR}/ip.$k" || true
    for _i in $(seq 1 30); do [ -s "${MN_DIR}/ip.$k" ] && break; sleep 1; done
    ip="$(head -1 "${MN_DIR}/ip.$k" 2>/dev/null)"
    [ -n "$ip" ] && [ "$ip" != "${HEAD_IP}" ] || { echo "ERROR: node $k unreachable via pbsdsh (got '$ip')" >&2; exit 1; }
    NODE_IPS+=("$ip")
  done
  echo "[multinode] nodes: ${NODE_IPS[*]} (head ${HEAD_IP})"
  ( umask 077; openssl rand -hex 32 > "${MN_DIR}/executor_token"; openssl rand -hex 32 > "${MN_DIR}/ray_token" )
  EXEC_PORT="${EXECUTOR_PORT:-8900}"
  LOCAL_STAGE="${PBS_LOCALDIR:-/tmp/${USER}-${PBS_JOBID%%.*}}/hbstaging"
  # one launcher script per role, every value baked in (pbsdsh gives a bare environment)
  {
    echo '#!/bin/bash'
    printf 'IP=$1; mkdir -p %q %q 2>/dev/null || true\n' "${LOCAL_STAGE}" "${HPCSB_IMAGE_CACHE:-${LOCAL_STAGE}}"
    printf 'export PYTHONPATH=%q EXECUTOR_PORT=%q EXECUTOR_HOST="$IP" EXECUTOR_TOKEN_FILE=%q\n' \
      "${REPO_DIR}/src" "${EXEC_PORT}" "${MN_DIR}/executor_token"
    printf 'export HB_STAGING_ROOT=%q HPCSB_XFER_ROOT=%q SIF_IMAGE_CACHE_DIR=%q\n' \
      "${LOCAL_STAGE}" "${MN_DIR}/xfer" "${SIF_IMAGE_CACHE_DIR}"
    printf 'export HPCSB_IMAGE_CACHE=%q HPCSB_NETWORK=%q HPCSB_RUNTIME=%q HPCSB_DOCKERFILE_STEPS=%q HPCSB_POD_USERS=%q\n' \
      "${HPCSB_IMAGE_CACHE}" "${HPCSB_NETWORK}" "${HPCSB_RUNTIME}" "${HPCSB_DOCKERFILE_STEPS}" "${HPCSB_POD_USERS}"
    for v in SINGULARITY_DOCKER_USERNAME SINGULARITY_DOCKER_PASSWORD APPTAINER_DOCKER_USERNAME APPTAINER_DOCKER_PASSWORD; do
      [ -n "${!v:-}" ] && printf 'export %s=%q\n' "$v" "${!v}"
    done
    printf 'cd %q && exec %q -m mho.backends.local_singularity.service\n' "${REPO_DIR}" "${EXECUTOR_PYTHON}"
  } > "${MN_DIR}/executor.sh"
  chmod 700 "${MN_DIR}/executor.sh"
  RAY_TMP="/dev/shm/mho-r${PBS_JOBID%%.*}"
  EXECUTOR_URLS="$(printf "http://%s:${EXEC_PORT}," "${NODE_IPS[@]}")"; EXECUTOR_URLS="${EXECUTOR_URLS%,}"
  RAY_ENV=(--env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --env RAY_TMPDIR="${RAY_TMP}"
           --env RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 --env RAY_raylet_startup_timeout=300
           --env RAY_AUTH_MODE=token --env RAY_AUTH_TOKEN_PATH="${MN_DIR}/ray_token"
           --env EXECUTOR_URLS="${EXECUTOR_URLS}" --env EXECUTOR_TOKEN_FILE="${MN_DIR}/executor_token")
  # Bind sources on node-local scratch (BASE_DIR=$PBS_LOCALDIR from .env: HF_DIR, TMP_DIR) exist
  # only where train.sh ran: create them on each worker, and give its SGLang engines the model
  # snapshot the head downloaded (same revision as the trainer's converted weights) via the run dir.
  HF_STAGE=""
  if [ ! -d "${HF_MODEL}" ]; then
    HF_STAGE="${MN_DIR}/hf"; mkdir -p "${HF_STAGE}/hub"
    cp -a "${HF_DIR%/}/hub/models--${HF_MODEL//\//--}" "${HF_STAGE}/hub/" \
      || { echo "ERROR: could not stage ${HF_MODEL} for the worker nodes" >&2; exit 1; }
  fi
  {
    echo '#!/bin/bash'
    printf 'IP=$1; mkdir -p %q %q %q %q\n' "${RAY_TMP}" "${HF_DIR}" "${TMP_DIR}" "${BASE_DIR}"
    [ -n "${HF_STAGE}" ] && printf 'cp -a %q/. %q/ || exit 1\n' "${HF_STAGE}" "${HF_DIR%/}"
    printf 'exec singularity exec'; printf ' %q' "${SING_ARGS[@]}" "${RAY_ENV[@]}"; printf ' %q' "${MILES_SIF}"
    printf ' ray start --address %q --node-ip-address "$IP" --num-gpus 8 --disable-usage-stats --block\n' "${HEAD_IP}:6379"
  } > "${MN_DIR}/ray_worker.sh"
  chmod 700 "${MN_DIR}/ray_worker.sh"
  _mn_cleanup() {
    for k in $(seq 1 $((NUM_NODES - 1))); do
      pbsdsh -n "$k" -- /bin/bash -c "pkill -f mho.backends.local_singularity.service; singularity exec ${MILES_SIF} ray stop --force; rm -rf ${RAY_TMP}" >/dev/null 2>&1 || true
    done
    pkill -f mho.backends.local_singularity.service 2>/dev/null || true
    singularity exec "${MILES_SIF}" ray stop --force >/dev/null 2>&1 || true
    rm -rf "${RAY_TMP}" "${MN_DIR}/xfer" "${MN_DIR}/executor.sh" "${MN_DIR}/hf" 2>/dev/null || true
    for p in "${MN_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  }
  trap '_mn_cleanup; _train_cleanup' EXIT
  # executors: this node directly, the others through pbsdsh
  pkill -9 -f "mho.backends.local_singularity.service" 2>/dev/null || true
  bash "${MN_DIR}/executor.sh" "${HEAD_IP}" > "${MN_DIR}/logs/executor.0.log" 2>&1 &
  MN_PIDS+=("$!")
  for k in $(seq 1 $((NUM_NODES - 1))); do
    pbsdsh -n "$k" -- /bin/bash -c "bash ${MN_DIR}/executor.sh ${NODE_IPS[$k]} > ${MN_DIR}/logs/executor.$k.log 2>&1" &
    MN_PIDS+=("$!")
  done
  TOKEN_HDR="X-Executor-Token: $(cat "${MN_DIR}/executor_token")"
  for k in $(seq 0 $((NUM_NODES - 1))); do
    u="http://${NODE_IPS[$k]}:${EXEC_PORT}/health"
    for _i in $(seq 1 60); do curl -sf --max-time 3 -H "@-" "$u" <<<"${TOKEN_HDR}" >/dev/null 2>&1 && break; sleep 2; done
    curl -sf --max-time 3 -H "@-" "$u" <<<"${TOKEN_HDR}" >/dev/null 2>&1 \
      || { echo "ERROR: executor on node $k (${NODE_IPS[$k]}) not healthy; see ${MN_DIR}/logs/executor.$k.log" >&2; exit 1; }
  done
  unset TOKEN_HDR
  echo "[multinode] executors healthy on ${NUM_NODES} nodes"
  # Ray: head here, a worker per other node; the dashboard listens on the head's cluster IP
  # (worker-node actors call it), so Ray token auth guards its job API.
  mkdir -p "${RAY_TMP}"
  singularity exec "${SING_ARGS[@]}" "${RAY_ENV[@]}" "${MILES_SIF}" ray start --head --node-ip-address "${HEAD_IP}" \
    --dashboard-host "${HEAD_IP}" --num-gpus 8 --disable-usage-stats --block > "${MN_DIR}/logs/ray_head.log" 2>&1 &
  MN_PIDS+=("$!")
  for _i in $(seq 1 60); do
    singularity exec "${RUN_BINDS[@]}" "${RAY_ENV[@]}" "${MILES_SIF}" ray status --address "${HEAD_IP}:6379" >/dev/null 2>&1 && break; sleep 5
  done
  for k in $(seq 1 $((NUM_NODES - 1))); do
    pbsdsh -n "$k" -- /bin/bash -c "bash ${MN_DIR}/ray_worker.sh ${NODE_IPS[$k]} > ${MN_DIR}/logs/ray_worker.$k.log 2>&1" &
    MN_PIDS+=("$!")
  done
  want=$((8 * NUM_NODES)); GPUS=0
  for _i in $(seq 1 90); do
    GPUS="$(singularity exec "${RUN_BINDS[@]}" "${RAY_ENV[@]}" "${MILES_SIF}" python3 -c "import ray; ray.init(address='${HEAD_IP}:6379', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null | tail -1)"
    [ "${GPUS:-0}" -ge "$want" ] && break; sleep 5
  done
  [ "${GPUS:-0}" -ge "$want" ] || { echo "ERROR: Ray cluster has ${GPUS:-0} GPUs, expected $want" >&2; exit 1; }
  echo "[multinode] Ray cluster up: ${GPUS} GPUs"
  MN_ENV=("${RAY_ENV[@]}" --env MILES_SCRIPT_EXTERNAL_RAY=1 --env RAY_ADDRESS="http://${HEAD_IP}:8265"
          --env MASTER_ADDR="${HEAD_IP}")
fi

singularity exec "${SING_ARGS[@]}" "${MN_ENV[@]}" \
  "$MILES_SIF" \
    /opt/sglang/bin/python "$MILES_RUN_PY" \
      --megatron-model-type "$MODEL_KEY" \
      --model-name "$MODEL_KEY" \
      --base-dir "$SAVE_DIR" \
      --hf-checkpoint "$HF_MODEL" \
      --ref-load "$SAVE_DIR/${MODEL_KEY}_torch_dist" \
      --save-dir "$SAVE_DIR" \
      --prompt-data "${MATH_TRAIN_DIR}/seed.jsonl" \
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

# --- complete the HF exports -------------------------------------------------------
# miles exports only the Megatron (language) weights, and copies config/tokenizer only when
# --hf-checkpoint is a local dir: for Qwen3.5 a step_N has no config.json, vision tower or MTP
# head. Fill those from the base (training never touches them) so the loop finds the checkpoint
# and vLLM can serve it.
if [ -n "$SAVE_HF" ]; then
  HF_BASE_BIND=()
  [ -d "${HF_MODEL}" ] && HF_BASE_BIND=(--bind "${HF_MODEL}:${HF_MODEL}")
  singularity exec \
    --bind "${HF_DIR}:/root/.cache/huggingface" \
    --bind "${SAVE_DIR}:${SAVE_DIR}" \
    "${HF_BASE_BIND[@]}" \
    --env HF_HOME=/root/.cache/huggingface \
    "$MILES_SIF" env -u PYTHONPATH /opt/sglang/bin/python -I \
      "${REPO_DIR}/scripts/optimize/complete_hf_export.py" "$(dirname "$SAVE_HF")" "$HF_MODEL"
fi
