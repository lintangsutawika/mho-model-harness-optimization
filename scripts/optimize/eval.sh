#!/usr/bin/env bash
# Optimizer eval phase: consolidated harbor eval of a (model, harness), driven ENTIRELY by env
# (the loop exports these; set them by hand to run manually). Adapted from
# scripts/eval/eval_terminal_bench_2.sh -- that copy stays for standalone/paper/system evals.
#
# Sandbox environment (HARBOR_ENV, default 'singularity'):
#   singularity -> run trials in LOCAL apptainer sandboxes on THIS node via our
#                  SingularityWritableEnvironment. harbor run executes on the host (not nested),
#                  and the sandbox shares the host net namespace, so the agent reaches the local
#                  vLLM at localhost directly -- NO Modal, NO relay.
#   modal       -> cloud sandboxes; the local vLLM is exposed to them via the Modal reverse relay.
#
# Model handling:
#   MHO_MODEL set  -> SELF-HOST it with vLLM (HF name OR local hf-compatible dir). singularity:
#                     agent hits localhost:VLLM_PORT. modal: exposed via the relay.
#   MHO_MODEL unset-> use MODEL (a litellm id) directly (API model; no self-serve).
#
# Env: HARBOR_ENV, MHO_MODEL, MHO_HARNESS (->MINI_FORK_LOCAL), MHO_DATA / MHO_DATA_PATH, MHO_OUT,
#   AGENT_IMPORT, MODEL, N_ATTEMPTS, N_CONCURRENT, TASK_SET, AGENT_TEMPERATURE,
#   AGENT_TIMEOUT_MULTIPLIER, HARBOR_TIMEOUT_SECONDS, BASE_SIF, SINGULARITY_NO_MOUNT,
#   MODAL_*, VLLM_PORT/SERVED_NAME/VLLM_EXTRA_ARGS, RELAY/VLLM_LOCAL_URL/RELAY_APP_NAME.
#
# Images: BASE_SIF is the ONLY sif knob -- it serves training AND vLLM serving (the train
# image ships vLLM). The docker->sif cache (sif_cache/) is its sibling, so one path picks
# the whole set; no separate vllm-cuda.sif image.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MHO_REPO_DIR (set by the loop) selects WHICH checkout is evaluated. That is the point of the
# eval -- the harness edits live in that checkout -- so it is set explicitly or not at all; the
# only fallback is this script's own repo. PBS_O_WORKDIR / SLURM_SUBMIT_DIR are deliberately NOT
# consulted: under srun/sbatch they point at the submit dir, which silently evaluates a
# different checkout and never reads this one's .env.
REPO_DIR="${MHO_REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$REPO_DIR"
echo "[eval] repo: ${REPO_DIR}${MHO_REPO_DIR:+ (MHO_REPO_DIR)}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Do not let cwd (the repo root) prepend to sys.path: the stale repo-root harness/ package
# would shadow src/harness/ (agent_harness/math_verifier live only in src). src stays on
# PYTHONPATH, so this makes src win. See also the sys.path fixes in service.py/miles_agent_fn.
export PYTHONSAFEPATH=1
# Use the prebuilt project .venv directly (harbor + deps already installed, shared home) rather
# than `uv run`, which re-syncs the git harbor dep per job on a fresh node-local uv cache and
# left harbor missing in the batch eval. Override with MHO_VENV_BIN.
VENV_BIN="${MHO_VENV_BIN:-${REPO_DIR}/.venv/bin}"
if [ -f "${REPO_DIR}/.env" ]; then set -a; . "${REPO_DIR}/.env"; set +a; fi
# Every sif this script needs sits next to BASE_SIF. Resolve the dir now (empty if BASE_SIF is
# unset) and demand it only on the paths that use one -- a modal run needs no image at all.
SIF_DIR="${BASE_SIF:+$(dirname "${BASE_SIF}")}"
require_sif_dir() {
  [ -n "${SIF_DIR}" ] && return 0
  echo "ERROR: BASE_SIF is not set; it resolves training, vLLM serving, and the sif cache." >&2
  echo "       Put it in ${REPO_DIR}/.env or export BASE_SIF=/path/to/<train>.sif" >&2
  exit 2
}

HARBOR_ENV="${HARBOR_ENV:-singularity}"   # local singularity by default; 'modal' for cloud
AGENT_IMPORT="${AGENT_IMPORT:-harness.agent_harness:AgentHarness}"
# Custom harbor-singularity-hpc env class, selected by --environment-import-path (harbor's
# -e is a fixed enum and cannot name a custom class). Override to test a fork.
HARBOR_SING_IMPORT="${HARBOR_SING_IMPORT:-harbor_singularity_hpc.environment:SingularityWritableEnvironment}"
# Singularity (on-node) knobs. The .sif cache is a PERSISTENT SHARED dir for converted task
# images, decoupled from the sif-in-use (BASE_SIF); node-local dirs are wiped per job, causing
# re-pulls. Sandbox dir defaults to live $PBS_LOCALDIR.
SINGULARITY_CACHE_DIR="${SINGULARITY_CACHE_DIR:-$(dirname "${BASE_SIF}")/sif_cache}"
SINGULARITY_FORCE_PULL="${SINGULARITY_FORCE_PULL:-}"
SINGULARITY_NO_MOUNT="${SINGULARITY_NO_MOUNT:-home,tmp}"
SINGULARITY_WRITABLE_SANDBOX="${SINGULARITY_WRITABLE_SANDBOX:-}"     # empty=on (adapter default)
SINGULARITY_SANDBOX_DIR="${SINGULARITY_SANDBOX_DIR:-}"               # empty=node-local scratch
SINGULARITY_MEMORY_MB="${SINGULARITY_MEMORY_MB:-}"
SINGULARITY_MEMORY_ENFORCEMENT="${SINGULARITY_MEMORY_ENFORCEMENT:-}"
# Modal sandbox knobs.
ALLOW_HOST="${ALLOW_HOST:-}"
TASK_SET="${TASK_SET:-full}"
RUNS="${N_ATTEMPTS:-1}"
N_CONCURRENT="${N_CONCURRENT:-16}"
HARBOR_PATH="${MHO_DATA_PATH:-${HARBOR_PATH:-}}"        # -p: local task/dataset dir (priority)
HARBOR_DATASET="${MHO_DATA:-${HARBOR_DATASET:-}}"        # -d: registry name@version
MINI_FORK_LOCAL="${MHO_HARNESS:-${MINI_FORK_LOCAL:-${MICRO_SCAFFOLD_DIR:-}}}"
# No harness passed (baseline / model-only) -> evaluate the BASE scaffold, not "no scaffold".
# The agent hard-requires a snapshot (harness/scaffold.py:materialize); without one every trial
# errors with "No scaffold snapshot set" and the whole eval scores 0. Resolve scaffold.DEFAULT_BASE
# (honors MICRO_SCAFFOLD_BASE) so baseline evals the same base scaffold the manual baseline used.
if [ -z "${MINI_FORK_LOCAL:-}" ]; then
  MINI_FORK_LOCAL="$("${VENV_BIN}/python" -c "from harness import scaffold; print(scaffold.DEFAULT_BASE)" 2>/dev/null || true)"
  [ -n "${MINI_FORK_LOCAL}" ] && [ -d "${MINI_FORK_LOCAL}" ] \
    || { echo "ERROR: no harness set and base scaffold not found (set MHO_HARNESS or MICRO_SCAFFOLD_BASE)" >&2; exit 2; }
  echo "[eval] no harness passed -> using base scaffold: ${MINI_FORK_LOCAL}"
fi
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.7}"
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-}"
HARBOR_TIMEOUT_SECONDS="${HARBOR_TIMEOUT_SECONDS:-28800}"
MODAL_APP_NAME="${MODAL_APP_NAME:-mho_opt_eval}"
MODAL_SANDBOX_TIMEOUT_SEC="${MODAL_SANDBOX_TIMEOUT_SEC:-7200}"
MODAL_SANDBOX_IDLE_TIMEOUT_SEC="${MODAL_SANDBOX_IDLE_TIMEOUT_SEC:-3600}"

# --- cleanup (vLLM serve + relay), one EXIT trap -----------------------------
VLLM_PID=""; RELAY_BRIDGE_PID=""; RELAY_APP_STOP=""
cleanup() {
  [ -n "${RELAY_BRIDGE_PID}" ] && { echo "[relay] stopping bridge (pid=${RELAY_BRIDGE_PID})"; kill "${RELAY_BRIDGE_PID}" 2>/dev/null; }
  [ -n "${RELAY_APP_STOP}" ] && { echo "[relay] stopping Modal app ${RELAY_APP_STOP}"; uv run modal app stop "${RELAY_APP_STOP}" </dev/null >/dev/null 2>&1 || true; }
  [ -n "${VLLM_PID}" ] && { echo "[eval] stopping vLLM (pid=${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null; }
}
trap cleanup EXIT

# --- self-serve the model with vLLM (if MHO_MODEL set) -----------------------
if [ -n "${MHO_MODEL:-}" ]; then
  require_sif_dir
  # vLLM serving runs from BASE_SIF itself (it ships vllm serve); no separate vllm-cuda.sif.
  VLLM_SIF="${BASE_SIF}"
  VLLM_PORT="${VLLM_PORT:-8000}"; SERVED_NAME="${SERVED_NAME:-$(basename "${MHO_MODEL}")}"
  [ -f "$VLLM_SIF" ] || { echo "ERROR: MHO_MODEL set but BASE_SIF missing: ${BASE_SIF}" >&2; exit 1; }
  MODEL_BIND=(); [ -d "${MHO_MODEL}" ] && MODEL_BIND=(--bind "${MHO_MODEL}:${MHO_MODEL}")
  mkdir -p "${MHO_OUT:-/tmp}"
  # mini-swe-agent sends tool_choice=auto; vLLM needs --enable-auto-tool-choice +
  # --tool-call-parser. Qwen3.5 uses the 'qwen3_coder' parser (same name in this vLLM as on the
  # sglang training path). Override via VLLM_TOOL_PARSER; set VLLM_REASONING_PARSER
  # (e.g. deepseek_r1 / qwen3) to strip <think>.
  VLLM_TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_PARSER:-qwen3_coder}" --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}")
  [ -n "${VLLM_REASONING_PARSER:-}" ] && VLLM_TOOL_ARGS+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  # Small model (fits on 1 GPU) + many concurrent agents -> DATA-parallel (N replicas, load
  # balanced) maximizes throughput; tensor-parallel would only add comm overhead. Default DP to
  # the number of visible GPUs so it uses the whole allocation; override with VLLM_DATA_PARALLEL
  # (=1 to disable) and VLLM_TENSOR_PARALLEL (for models too big for one GPU).
  _NGPU="$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr "," "\n" | grep -c . || true)"
  [ "${_NGPU:-0}" -ge 1 ] 2>/dev/null || _NGPU="$(nvidia-smi -L 2>/dev/null | grep -c . || echo 1)"
  VLLM_DATA_PARALLEL="${VLLM_DATA_PARALLEL:-${_NGPU:-1}}"
  VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
  [ "${VLLM_DATA_PARALLEL}" -gt 1 ] 2>/dev/null && VLLM_TOOL_ARGS+=(--data-parallel-size "${VLLM_DATA_PARALLEL}")
  [ "${VLLM_TENSOR_PARALLEL}" -gt 1 ] 2>/dev/null && VLLM_TOOL_ARGS+=(--tensor-parallel-size "${VLLM_TENSOR_PARALLEL}")
  echo "[eval] serving ${MHO_MODEL} as '${SERVED_NAME}' on :${VLLM_PORT} (sif=${VLLM_SIF}) tool-parser=${VLLM_TOOL_PARSER:-qwen3_coder} dp=${VLLM_DATA_PARALLEL} tp=${VLLM_TENSOR_PARALLEL}"
  singularity exec --nv \
    --bind "${HF_DIR:-$HOME/.cache/huggingface}:/root/.cache/huggingface" "${MODEL_BIND[@]}" \
    "$VLLM_SIF" vllm serve "${MHO_MODEL}" --port "${VLLM_PORT}" --served-model-name "${SERVED_NAME}" \
      "${VLLM_TOOL_ARGS[@]}" ${VLLM_EXTRA_ARGS:-} >"${MHO_OUT:-/tmp}/vllm_serve.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 120); do curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && break; sleep 5; done
  curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 \
    || { echo "[eval] vLLM not healthy; see ${MHO_OUT:-/tmp}/vllm_serve.log" >&2; exit 1; }
  MODEL="litellm_proxy/${SERVED_NAME}"; VLLM_LOCAL_URL="http://localhost:${VLLM_PORT}"
  if [ "${HARBOR_ENV}" = "modal" ]; then
    RELAY=1                                   # cloud sandboxes reach the vLLM via the relay
  else
    # local singularity: the sandbox shares the host net namespace -> hit vLLM directly.
    export LITELLM_PROXY_API_BASE="http://localhost:${VLLM_PORT}/v1"
    export LITELLM_PROXY_API_KEY="${LITELLM_PROXY_API_KEY:-EMPTY}"
  fi
fi
MODEL="${MODEL:-litellm_proxy/Qwen/Qwen3.8-27B}"

# --- validate agent import (before paying for any sandbox) -------------------
"${VENV_BIN}/python" - "${AGENT_IMPORT}" <<'PYV'
import importlib, sys
mod, _, cls = sys.argv[1].partition(":")
if not cls: sys.exit(f"agent import must be 'module:Class', got {sys.argv[1]!r}")
obj = getattr(importlib.import_module(mod), cls, None)
if not isinstance(obj, type): sys.exit(f"{sys.argv[1]} did not resolve to a class")
from harbor.agents.base import BaseAgent
if not issubclass(obj, BaseAgent): sys.exit(f"{sys.argv[1]} is not a harbor BaseAgent subclass")
print(f"[validate] agent OK: {sys.argv[1]}")
PYV

TASK_FLAGS=()
case "${TASK_SET}" in
  full)  ;;
  smoke) TASK_FLAGS+=(-i "${SMOKE_TASK:-hello-world}") ;;
  *) echo "Unknown TASK_SET '${TASK_SET}' (full|smoke)" >&2; exit 1 ;;
esac

# harbor run takes EITHER -p <local dir> OR -d <registry name>; if both are passed it
# silently ignores -d. Both default empty -> append only the one that is set, -p first.
DATA_FLAG=()
if [ -n "${HARBOR_PATH}" ]; then DATA_FLAG=(-p "${HARBOR_PATH}")
elif [ -n "${HARBOR_DATASET}" ]; then DATA_FLAG=(-d "${HARBOR_DATASET}")
else echo "ERROR: set MHO_DATA_PATH/HARBOR_PATH (-p) or MHO_DATA/HARBOR_DATASET (-d)" >&2; exit 2; fi
# Environment selection: local singularity (our harbor-singularity-hpc SingularityWritableEnvironment, run on the host
# -- no nesting, no external executor) or modal (cloud).
# Environment selection. `singularity` is a custom class from harbor-singularity-hpc, so it is
# picked by --environment-import-path (harbor's -e is a fixed enum). modal uses the built-in -e.
ENV_FLAGS=()
if [ "${HARBOR_ENV}" = "singularity" ]; then
  require_sif_dir
  [ -n "${SINGULARITY_CACHE_DIR}" ] && mkdir -p "${SINGULARITY_CACHE_DIR}" 2>/dev/null || true
  ENV_FLAGS=( --environment-import-path "${HARBOR_SING_IMPORT}" )
  # The .sif cache: a PERSISTENT dir for converted task images -- NOT the sif-in-use dir.
  [ -n "${SINGULARITY_CACHE_DIR}" ] && ENV_FLAGS+=( --ek "singularity_image_cache_dir=${SINGULARITY_CACHE_DIR}" )
  [ -n "${SINGULARITY_FORCE_PULL}" ] && ENV_FLAGS+=( --ek "singularity_force_pull=${SINGULARITY_FORCE_PULL}" )
  [ -n "${SINGULARITY_NO_MOUNT}" ] && ENV_FLAGS+=( --ek "singularity_no_mount=${SINGULARITY_NO_MOUNT}" )
  [ -n "${SINGULARITY_WRITABLE_SANDBOX}" ] && ENV_FLAGS+=( --ek "singularity_writable_sandbox=${SINGULARITY_WRITABLE_SANDBOX}" )
  [ -n "${SINGULARITY_SANDBOX_DIR}" ] && ENV_FLAGS+=( --ek "singularity_sandbox_dir=${SINGULARITY_SANDBOX_DIR}" )
  [ -n "${SINGULARITY_MEMORY_MB}" ] && ENV_FLAGS+=( --ek "override_memory_mb=${SINGULARITY_MEMORY_MB}" )
  [ -n "${SINGULARITY_MEMORY_ENFORCEMENT}" ] && ENV_FLAGS+=( --ek "memory_enforcement_policy=${SINGULARITY_MEMORY_ENFORCEMENT}" )
else
  ENV_FLAGS=( -e modal )
  [ -n "${MODAL_APP_NAME}" ] && ENV_FLAGS+=( --ek "app_name=${MODAL_APP_NAME}" )
  [ -n "${MODAL_SANDBOX_TIMEOUT_SEC}" ] && ENV_FLAGS+=( --ek "sandbox_timeout_secs=${MODAL_SANDBOX_TIMEOUT_SEC}" )
  [ -n "${MODAL_SANDBOX_IDLE_TIMEOUT_SEC}" ] && ENV_FLAGS+=( --ek "sandbox_idle_timeout_secs=${MODAL_SANDBOX_IDLE_TIMEOUT_SEC}" )
  [ -n "${ALLOW_HOST}" ] && ENV_FLAGS+=( --allow-agent-host "${ALLOW_HOST}" )
# Math verifier for standalone harbor evals (defaults to the repo custom grader,
# mirroring miles_agent_fn.py). Passed to `harbor run --verifier`.
MHO_VERIFIER="${MHO_VERIFIER:-harness.math_verifier:MathVerifier}"
fi
# Output location: write to a deterministic dir (MHO_OUT, else runs_output/eval) with a stable
# job name, so results are findable (harbor's default is a timestamped dir under ./jobs).
EVAL_JOBS_DIR="${MHO_OUT:-${REPO_DIR}/runs_output/eval}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${EVAL_JOBS_DIR}"
CMD=(
  "${VENV_BIN}/harbor" run --agent "${AGENT_IMPORT}" "${DATA_FLAG[@]}" -m "${MODEL}" "${ENV_FLAGS[@]}"
  --jobs-dir "${EVAL_JOBS_DIR}" --job-name "${EVAL_JOB_NAME}"
  -n "${N_CONCURRENT}" --n-attempts "${RUNS}" --ak "temperature=${AGENT_TEMPERATURE}"
  ${MHO_VERIFIER:+--verifier "${MHO_VERIFIER}"}
)
[ -n "${MINI_FORK_LOCAL:-}" ] && CMD+=( --ak "mini_fork_local=${MINI_FORK_LOCAL}" )

# --- Modal reverse relay (only for HARBOR_ENV=modal: expose local vLLM to cloud sandboxes) --
if [ "${RELAY:-0}" = "1" ]; then
  VLLM_LOCAL_URL="${VLLM_LOCAL_URL:-http://localhost:8000}"
  RELAY_APP_NAME="${RELAY_APP_NAME:-mho-vllm-relay-$$}"
  RELAY_SECRET="${RELAY_SECRET:-$(openssl rand -hex 16)}"
  echo "[relay] deploying ${RELAY_APP_NAME} (vllm=${VLLM_LOCAL_URL}) ..."
  _DEPLOY_OUT="$(RELAY_APP_NAME="${RELAY_APP_NAME}" RELAY_SECRET="${RELAY_SECRET}" \
      uv run modal deploy src/harness/relay/modal_relay.py 2>&1)" || { echo "${_DEPLOY_OUT}" >&2; exit 1; }
  _RELAY_HOST="$(printf '%s\n' "${_DEPLOY_OUT}" | grep -oE 'https://[a-z0-9.-]+\.modal\.run' | head -1)"
  [ -n "${_RELAY_HOST}" ] || { echo "ERROR: no relay URL in deploy output:" >&2; echo "${_DEPLOY_OUT}" >&2; exit 1; }
  RELAY_APP_STOP="${RELAY_APP_NAME}"
  echo "[relay] app URL: ${_RELAY_HOST}"
  RELAY_WS_URL="${_RELAY_HOST/https:/wss:}/bridge" RELAY_SECRET="${RELAY_SECRET}" VLLM_LOCAL_URL="${VLLM_LOCAL_URL}" \
    uv run --with websockets --with httpx python src/harness/relay/bridge.py &
  RELAY_BRIDGE_PID=$!
  echo "[relay] waiting for bridge to register ..."; _OK=0
  for _ in $(seq 1 24); do
    curl -sf --max-time 10 "${_RELAY_HOST}/health" 2>/dev/null | grep -q '"bridge_connected":true' && { _OK=1; break; }
    sleep 5
  done
  [ "${_OK}" = "1" ] || { echo "ERROR: bridge never registered with ${RELAY_APP_NAME}" >&2; exit 1; }
  echo "[relay] bridge connected."
  export LITELLM_PROXY_API_BASE="${_RELAY_HOST}/v1"; export LITELLM_PROXY_API_KEY="${RELAY_SECRET}"
fi

# litellm_proxy passthrough: inject proxy base/key into the sandbox agent env.
case "${MODEL}" in
  litellm_proxy/*)
    _PROXY_BASE="${LITELLM_PROXY_API_BASE:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
    _PROXY_KEY="${LITELLM_PROXY_API_KEY:-${OPENAI_API_KEY:-${MSWEA_API_KEY:-EMPTY}}}"
    [ -n "${_PROXY_BASE}" ] || { echo "ERROR: MODEL=litellm_proxy/... needs LITELLM_PROXY_API_BASE" >&2; exit 1; }
    # mini-swe-agent resolves its key ONLY from MSWEA_API_KEY (harbor mini_swe_agent.py:483) and
    # hard-requires it non-empty. litellm uses LITELLM_PROXY_API_KEY for the endpoint; local vLLM
    # ignores auth, so any non-empty MSWEA_API_KEY works. Inject both into the sandbox agent env.
    CMD+=( --ae "LITELLM_PROXY_API_BASE=${_PROXY_BASE}" --ae "LITELLM_PROXY_API_KEY=${_PROXY_KEY}"
           --ae "MSWEA_API_KEY=${_PROXY_KEY}" )
    echo "[model] litellm_proxy passthrough -> ${_PROXY_BASE}"
    ;;
esac
[ -n "${AGENT_TIMEOUT_MULTIPLIER}" ] && CMD+=( --agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER}" )
[ "${#TASK_FLAGS[@]}" -gt 0 ] && CMD+=("${TASK_FLAGS[@]}")

echo "=================================================================="
echo "agent: ${AGENT_IMPORT} | data: ${HARBOR_PATH:-${HARBOR_DATASET}} (${TASK_SET}) | model: ${MODEL}"
echo "env: ${HARBOR_ENV} | harness: ${MINI_FORK_LOCAL:-<base>} | n=${N_CONCURRENT} attempts=${RUNS}"
echo "output: ${EVAL_JOBS_DIR}/${EVAL_JOB_NAME}/"
echo "=================================================================="

RUN_CMD=("${CMD[@]}")
if command -v timeout >/dev/null 2>&1 && [ "${HARBOR_EXTERNAL_TIMEOUT:-0}" != "1" ]; then
  RUN_CMD=(timeout --signal=SIGINT "${HARBOR_TIMEOUT_SECONDS}" "${CMD[@]}")
fi
# Never exec here: a relay/vLLM child must be torn down by the EXIT trap.
"${RUN_CMD[@]}" & HARBOR_PID=$!; wait "${HARBOR_PID}"
