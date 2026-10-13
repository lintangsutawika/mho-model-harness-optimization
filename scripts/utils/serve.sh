#!/usr/bin/env bash
# Serve vLLM on this node's GPUs, from BASE_SIF (the single image knob, which ships vllm).
# A standalone, reusable serving entrypoint. eval.sh calls `bash scripts/utils/serve.sh &`
# and waits on /health; you can also run it directly.
#
# Two modes:
#   USE_RELAY=1  serve locally AND expose it through the Modal reverse relay
#                (a public https://<app>.modal.run OpenAI-compatible endpoint, reachable
#                from ANYWHERE -- the node dials OUT to the relay, no inbound needed).
#   (default)    serve locally on :PORT only; callers reach http://127.0.0.1:PORT/v1.
#
# Usage (from the repo root):
#   MODEL=Qwen/Qwen3.5-4B PORT=8000 bash scripts/utils/serve.sh            # local only
#   MODEL=Qwen/Qwen3.5-4B USE_RELAY=1 bash scripts/utils/serve.sh          # + Modal gateway
#   USE_RELAY=1 RELAY_WS_URL=wss://.../bridge RELAY_SECRET=... \
#       bash scripts/utils/serve.sh                                        # reuse a deployed relay
#
# Knobs (env):
#   MODEL            HF repo to serve (default Qwen/Qwen3.5-4B).
#   PORT             serve port (default 8000).
#   TENSOR_PARALLEL  TP (default 1); DATA_PARALLEL DP (default #visible GPUs).
#   MAX_MODEL_LEN    context length (default from configs/serve/<MODEL>.json, else 32768).
#   GPU_MEM_UTIL     gpu-memory-utilization (default 0.85).
#   VLLM_ARGS_EXTRA  extra vLLM flags; else loaded from configs/serve/<MODEL>.json extra_args.
#   BASE_SIF         the serving image (default $BASE_SIF env, else <repo>/skyrl-megatron.sif).
#   USE_RELAY=1      expose via Modal reverse relay (src/harness/relay). Default 0.
#   RELAY_WS_URL     wss://.../bridge (when USE_RELAY=1); deploy once via modal if unset.
#   RELAY_SECRET     shared secret (matches the Modal relay secret).
#   VLLM_LOCAL_URL   local vLLM base for the bridge (default http://127.0.0.1:$PORT).
#   APPTAINER_BIN    container tool (auto: apptainer > singularity).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}}"
cd "$REPO_ROOT"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo singularity)}"
: "${BASE_SIF:=${PWD%/}/skyrl-megatron.sif}"   # allow unset BASE_SIF in standalone use
[ -f "${BASE_SIF}" ] || { echo "ERROR: serving sif not found: ${BASE_SIF}" >&2; exit 2; }
# Visible GPUs -> DP default (small model, many concurrent agents -> data parallelism).
_NGPU="$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr "," "\n" | grep -c . || true)"
[ "${_NGPU:-0}" -ge 1 ] 2>/dev/null || _NGPU="$(nvidia-smi -L 2>/dev/null | grep -c . || echo 1)"
DATA_PARALLEL="${DATA_PARALLEL:-${_NGPU:-1}}"
# Per-model vLLM serve params (configs/serve/<MODEL>.json): model-intrinsic flags travel
# WITH the model; node knobs stay in env. Explicit VLLM_ARGS_EXTRA wins.
VLLM_ARGS_EXTRA="${VLLM_ARGS_EXTRA:-}"
SERVE_CONFIG="${SERVE_CONFIG:-configs/serve/${MODEL}.json}"
if [ -z "${VLLM_ARGS_EXTRA:-}" ] && [ -f "${REPO_ROOT}/${SERVE_CONFIG}" ]; then
    echo "[serve] loading vLLM serve params from ${SERVE_CONFIG}" >&2
    eval "$(python3 - "${REPO_ROOT}/${SERVE_CONFIG}" <<'PY'
import json, os, shlex, sys
c = json.load(open(sys.argv[1]))
def emit(var, val):
    if os.environ.get(var):
        return
    print(f"{var}={shlex.quote(val)}")
if isinstance(c.get("extra_args"), list):
    emit("VLLM_ARGS_EXTRA", " ".join(str(a) for a in c["extra_args"]))
if c.get("max_model_len"):
    emit("MAX_MODEL_LEN", str(c["max_model_len"]))
PY
)"
fi
[ -z "${MAX_MODEL_LEN:-}" ] && MAX_MODEL_LEN=32768
command -v "${APPTAINER_BIN}" >/dev/null 2>&1 || { echo "ERROR: container tool not found: ${APPTAINER_BIN}" >&2; exit 1; }
# Normalize UUID GPU ids to integer indices (vLLM needs ints).
case "${CUDA_VISIBLE_DEVICES:-}" in
    *[!0-9,]*)
        _ng=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_ng - 1)))"
        echo "[serve] normalized CUDA_VISIBLE_DEVICES -> ${CUDA_VISIBLE_DEVICES}" >&2 ;;
esac
echo "[serve] launching vLLM ${MODEL} :${PORT} (TP=${TENSOR_PARALLEL} DP=${DATA_PARALLEL}) in ${BASE_SIF}" >&2
BASE_DIR="${BASE_DIR:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}"
CACHE_DIR="${CACHE_DIR:-${XDG_CACHE_HOME:-${BASE_DIR}}/.cache}"
mkdir -p "${CACHE_DIR}"
export APPTAINER_TMPDIR="${BASE_DIR}" SINGULARITY_TMPDIR="${BASE_DIR}"
vllm_argv=(
    "${APPTAINER_BIN}" exec --nv --writable-tmpfs
    --workdir "${BASE_DIR}"
    --bind "${HF_DIR:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
    "${BASE_SIF}"
    vllm serve "${MODEL}"
        --host 0.0.0.0
        --port "${PORT}"
        --tensor-parallel-size "${TENSOR_PARALLEL}"
        --data-parallel-size "${DATA_PARALLEL}"
        --max-model-len "${MAX_MODEL_LEN}"
        --gpu-memory-utilization "${GPU_MEM_UTIL}"
)
# shellcheck disable=SC2206   # VLLM_ARGS_EXTRA is deliberately word-split into flags.
vllm_argv+=( ${VLLM_ARGS_EXTRA:-} )
echo "[serve] flags: ${VLLM_ARGS_EXTRA:-<config>}" >&2
# --- Modal reverse relay (USE_RELAY=1) -------------------------------------------
USE_RELAY="${USE_RELAY:-0}"
if [ "${USE_RELAY}" = "1" ]; then
    RELAY_DIR="${RELAY_DIR:-${REPO_ROOT}/src/harness/relay}"
    [ -f "${RELAY_DIR}/bridge.py" ] || { echo "ERROR: ${RELAY_DIR}/bridge.py missing" >&2; exit 2; }
    RELAY_WS_URL="${RELAY_WS_URL:-}"
    RELAY_SECRET="${RELAY_SECRET:-}"
    VLLM_LOCAL_URL="${VLLM_LOCAL_URL:-http://127.0.0.1:${PORT}}"
    if [ -z "${RELAY_WS_URL}" ]; then
        RELAY_APP_NAME="${RELAY_APP_NAME:-mho-vllm-relay}"
        RELAY_SECRET_NAME="${RELAY_SECRET_NAME:-mho-relay}"
        if [ -z "${RELAY_SECRET}" ]; then
            RELAY_SECRET="$(openssl rand -hex 16)"
            echo "[serve] creating/updating Modal secret ${RELAY_SECRET_NAME} (--force)" >&2
            uv run modal secret create --force "${RELAY_SECRET_NAME}" RELAY_SECRET="${RELAY_SECRET}" >&2
        fi
        echo "[serve] deploying Modal relay app ${RELAY_APP_NAME}" >&2
        _DEPLOY_OUT="$(RELAY_APP_NAME="${RELAY_APP_NAME}" RELAY_SECRET_NAME="${RELAY_SECRET_NAME}" \
            uv run modal deploy "${RELAY_DIR}/modal_relay.py" 2>&1)" || { echo "${_DEPLOY_OUT}" >&2; exit 2; }
        echo "${_DEPLOY_OUT}" >&2
        _RELAY_URL="$(printf '%s' "${_DEPLOY_OUT}" | grep -oE 'https://[a-zA-Z0-9._-]+\.modal\.run' | head -1)"
        [ -n "${_RELAY_URL}" ] || { echo "ERROR: could not parse relay URL from deploy output" >&2; exit 2; }
        RELAY_WS_URL="$(printf '%s' "${_RELAY_URL%/}" | sed "s#^https://#wss://#")/bridge"
        echo "[serve] relay deployed -> ${_RELAY_URL} ; bridge ws -> ${RELAY_WS_URL}" >&2
    fi
    echo "[serve] starting vLLM in background, then Modal relay bridge -> ${RELAY_WS_URL}" >&2
    "${vllm_argv[@]}" &
    VLLM_PID=$!
    trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT TERM INT
    (cd "${RELAY_DIR}" \
        && RELAY_WS_URL="${RELAY_WS_URL}" RELAY_SECRET="${RELAY_SECRET}" \
            VLLM_LOCAL_URL="${VLLM_LOCAL_URL}" \
            uv run --with websockets --with httpx python bridge.py)
else
    # Local-only: run in the FOREGROUND; caller backgrounds + polls /health.
    exec "${vllm_argv[@]}"
fi