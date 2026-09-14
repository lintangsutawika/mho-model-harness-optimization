#!/usr/bin/env bash
# Direct (no-scaffold) eval: self-host MHO_MODEL with vLLM, then send one completion per
# problem (no harbor, no agent) and grade with math_verify. A model-capability baseline /
# sanity check to sit beside the harnessed harbor eval.
#
# Env:
#   MHO_MODEL           HF name or local dir to serve (required)
#   MHO_DATA_PATH       local harbor dataset dir (same problems as harness eval), OR
#   MHO_DATASET         HF dataset id (columns problem/answer), e.g. MathArena/hmmt_feb_2025
#   MHO_OUT             output dir (default runs_output/direct)
#   EVAL_JOB_NAME       output filename stem (default direct_<ts>)
#   N_ATTEMPTS          samples per problem (default 1)
#   N_CONCURRENT        parallel requests (default 32)
#   DIRECT_TEMPERATURE  sampling temp (default 0.7)
#   DIRECT_MAX_TOKENS   max generation tokens per call (default 30000)
#   VLLM_MAX_MODEL_LEN  context window (default 32768; raise for more thinking room)
#   MAX_TASKS           limit #problems (default 0 = all)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This is a manual tool (not spooled by PBS), so the script's own location is the reliable repo
# root. Prefer it over PBS_O_WORKDIR, which in an interactive `qsub -I` job points at wherever
# qsub was launched, not the repo. Override with MHO_REPO_DIR if ever run as a batch job.
REPO_DIR="${MHO_REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONSAFEPATH=1
VENV_BIN="${MHO_VENV_BIN:-${REPO_DIR}/.venv/bin}"
# Source .env with `set -u` relaxed: .env lines like BASE_SIF=${USER_DATA}/... reference vars
# that may be unset in a bare shell, which would otherwise abort sourcing under `set -u`.
[ -f "${REPO_DIR}/.env" ] && { set -a; set +u; . "${REPO_DIR}/.env" || true; set -u; set +a; }

[ -n "${MHO_MODEL:-}" ] || { echo "ERROR: MHO_MODEL is required" >&2; exit 2; }
[ -n "${MHO_DATA_PATH:-}${MHO_DATASET:-}" ] || { echo "ERROR: set MHO_DATA_PATH or MHO_DATASET" >&2; exit 2; }
# Anchor a relative MHO_DATA_PATH to the repo (the eval's cwd may differ) and fail early if absent.
if [ -n "${MHO_DATA_PATH:-}" ]; then
  case "$MHO_DATA_PATH" in /*) ;; *) MHO_DATA_PATH="$REPO_DIR/$MHO_DATA_PATH" ;; esac
  [ -d "$MHO_DATA_PATH" ] || { echo "ERROR: MHO_DATA_PATH not found: $MHO_DATA_PATH" >&2; exit 2; }
fi

# Resolve the vLLM sif without depending on .env being sourced (BASE_SIF may be unset under
# `set -u`). It lives next to the repo checkout; also try BASE_SIF / SIF_IMAGE_CACHE_DIR dirs.
# The ${VAR:+...} guards expand to empty (not an error) when the var is unset.
if [ -z "${VLLM_SIF:-}" ]; then
  for _cand in \
    "$(dirname "$REPO_DIR")/vllm-cuda.sif" \
    "${BASE_SIF:+$(dirname "${BASE_SIF}")/vllm-cuda.sif}" \
    "${SIF_IMAGE_CACHE_DIR:+$(dirname "${SIF_IMAGE_CACHE_DIR}")/vllm-cuda.sif}"; do
    [ -n "$_cand" ] && [ -f "$_cand" ] && { VLLM_SIF="$_cand"; break; }
  done
fi
VLLM_PORT="${VLLM_PORT:-8000}"; SERVED_NAME="${SERVED_NAME:-$(basename "${MHO_MODEL}")}"
[ -n "${VLLM_SIF:-}" ] && [ -f "$VLLM_SIF" ] \
  || { echo "ERROR: no vLLM sif found (set VLLM_SIF=/path/to/vllm-cuda.sif)" >&2; exit 1; }
OUT_DIR="${MHO_OUT:-${REPO_DIR}/runs_output/direct}"; mkdir -p "$OUT_DIR"
JOB="${EVAL_JOB_NAME:-direct_$(date +%Y%m%d_%H%M%S)}"

VLLM_PID=""
cleanup(){ [ -n "$VLLM_PID" ] && { echo "[direct] stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null; }; }
trap cleanup EXIT

# GPUs -> data-parallel replicas (small model, many concurrent requests).
_NGPU="$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr "," "\n" | grep -c . || true)"
[ "${_NGPU:-0}" -ge 1 ] 2>/dev/null || _NGPU="$(nvidia-smi -L 2>/dev/null | grep -c . || echo 1)"
DP="${VLLM_DATA_PARALLEL:-${_NGPU:-1}}"
SERVE_ARGS=(--port "$VLLM_PORT" --served-model-name "$SERVED_NAME"
            --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}")
[ "$DP" -gt 1 ] 2>/dev/null && SERVE_ARGS+=(--data-parallel-size "$DP")
MODEL_BIND=(); [ -d "${MHO_MODEL}" ] && MODEL_BIND=(--bind "${MHO_MODEL}:${MHO_MODEL}")

echo "[direct] serving ${MHO_MODEL} as '${SERVED_NAME}' on :${VLLM_PORT} (dp=${DP}, ctx=${VLLM_MAX_MODEL_LEN:-32768})"
singularity exec --nv \
  --bind "${HF_DIR:-$HOME/.cache/huggingface}:/root/.cache/huggingface" "${MODEL_BIND[@]}" \
  "$VLLM_SIF" vllm serve "${MHO_MODEL}" "${SERVE_ARGS[@]}" ${VLLM_EXTRA_ARGS:-} \
    >"${OUT_DIR}/${JOB}.vllm.log" 2>&1 &
VLLM_PID=$!
for _ in $(seq 1 120); do curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && break; sleep 5; done
curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 \
  || { echo "[direct] vLLM not healthy; see ${OUT_DIR}/${JOB}.vllm.log" >&2; exit 1; }

DATA_ARG=(); [ -n "${MHO_DATA_PATH:-}" ] && DATA_ARG=(--data-path "${MHO_DATA_PATH}") || DATA_ARG=(--dataset "${MHO_DATASET}")
echo "[direct] running eval -> ${OUT_DIR}/${JOB}.json"
uv run --no-project --with openai --with math-verify --with datasets \
  python "${SCRIPT_DIR}/direct_eval.py" \
    --model "${SERVED_NAME}" --base-url "http://localhost:${VLLM_PORT}/v1" \
    "${DATA_ARG[@]}" \
    --n-attempts "${N_ATTEMPTS:-1}" --concurrency "${N_CONCURRENT:-32}" \
    --temperature "${DIRECT_TEMPERATURE:-0.7}" --max-tokens "${DIRECT_MAX_TOKENS:-30000}" \
    --max-tasks "${MAX_TASKS:-0}" --out "${OUT_DIR}/${JOB}.json"
