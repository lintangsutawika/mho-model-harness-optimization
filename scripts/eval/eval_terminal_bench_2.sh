#!/usr/bin/env bash
# Run a Terminal-Bench 2.0 evaluation with an MHO harness agent, on the Modal env.
#
# This is the CORE entrypoint. The PBS / Slurm wrappers (submit_pbs.sh /
# submit_slurm.sbatch) just set scheduler directives + cluster env and then call this.
#
# Usage:
#   scripts/eval/eval_terminal_bench_2.sh [AGENT_IMPORT] [TASK_SET] [RUNS] [N_CONCURRENT] [extra harbor flags...]
#     AGENT_IMPORT  harbor agent module:Class (default: harness.harbor_run:AgentHarness,
#                   which installs the micro snapshot named by MINI_FORK_LOCAL /
#                   MICRO_SCAFFOLD_DIR -- see harness/scaffold.py)
#     TASK_SET      full | smoke | <single task id via -i is also fine as an extra flag>
#     RUNS          --n-attempts per task (default 1)
#     N_CONCURRENT  parallel trials = parallel Modal sandboxes + in-flight model calls (default 16)
#
# Model: set MODEL (litellm id). For an API model (e.g. openai/gpt-5, anthropic/...),
# export that provider's key and Modal sandboxes reach it directly. For a self-hosted
# vLLM the agent runs INSIDE the Modal sandbox, so it needs a PUBLIC endpoint -- set
# RELAY=1 to stand up harness/relay (Modal reverse-relay to this node's vLLM).
#
# Env knobs: MODEL, N_ATTEMPTS, N_CONCURRENT, TASK_SET, HARBOR_TIMEOUT_SECONDS,
#   MODAL_APP_NAME, MODAL_SANDBOX_TIMEOUT_SEC, MODAL_SANDBOX_IDLE_TIMEOUT_SEC,
#   AGENT_TEMPERATURE, AGENT_TIMEOUT_MULTIPLIER, HARBOR_DATASET, MINI_FORK_LOCAL,
#   RELAY, VLLM_LOCAL_URL, RELAY_APP_NAME.

set -euo pipefail

AGENT_IMPORT="${1:-${AGENT_IMPORT:-harness.harbor_run:AgentHarness}}"
TASK_SET="${2:-${TASK_SET:-full}}"
RUNS="${3:-${N_ATTEMPTS:-1}}"
N_CONCURRENT="${4:-${N_CONCURRENT:-16}}"
if [ "$#" -gt 4 ]; then shift 4; else shift "$#"; fi
EXTRA_FLAGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Make `harness.*` importable by the harbor agent factory (--agent <import path>).
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Local .env overrides the shell environment (keys, base URLs, Modal tokens).
if [ -f "${REPO_DIR}/.env" ]; then set -a; . "${REPO_DIR}/.env"; set +a; fi

MODEL="${MODEL:-litellm_proxy/Qwen/Qwen3.8-27B}"
HARBOR_DATASET="${HARBOR_DATASET:-terminal-bench@2.0}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.7}"
# Scale EVERY task's per-task agent-execution timeout (each TB task sets its own
# timeout_sec in task.toml, e.g. 900s). e.g. AGENT_TIMEOUT_MULTIPLIER=2 -> 1800s per task.
# Empty = harbor default (1.0). Raise it for slow self-hosted inference through the relay.
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-}"
HARBOR_TIMEOUT_SECONDS="${HARBOR_TIMEOUT_SECONDS:-28800}"   # 8h wall, matches harbor default
# Modal cost/leak guards (harbor env-kwargs; harbor's defaults are 24h lifetime / no idle).
MODAL_APP_NAME="${MODAL_APP_NAME:-mho_tb2}"
MODAL_SANDBOX_TIMEOUT_SEC="${MODAL_SANDBOX_TIMEOUT_SEC:-7200}"
MODAL_SANDBOX_IDLE_TIMEOUT_SEC="${MODAL_SANDBOX_IDLE_TIMEOUT_SEC:-3600}"

# Fail fast if the agent import path is wrong (before paying for any sandbox).
uv run python - "${AGENT_IMPORT}" <<'PY'
import importlib, sys
mod, _, cls = sys.argv[1].partition(":")
if not cls:
    sys.exit(f"agent import must be 'module:Class', got {sys.argv[1]!r}")
obj = getattr(importlib.import_module(mod), cls, None)
if not isinstance(obj, type):
    sys.exit(f"{sys.argv[1]} did not resolve to a class")
from harbor.agents.base import BaseAgent
if not issubclass(obj, BaseAgent):
    sys.exit(f"{sys.argv[1]} is not a harbor BaseAgent subclass")
print(f"[validate] agent OK: {sys.argv[1]} -> {obj.__module__}.{obj.__qualname__}")
PY

# Task selection. `full` = whole dataset; `smoke` = one cheap task for bring-up.
TASK_FLAGS=()
case "${TASK_SET}" in
  full)  ;;
  smoke) TASK_FLAGS+=(-i "${SMOKE_TASK:-hello-world}") ;;
  *)     echo "Unknown TASK_SET '${TASK_SET}' (use: full | smoke). To pick tasks, pass '-i <id>' as an extra flag." >&2; exit 1 ;;
esac

CMD=(
  uv run harbor run
  --agent "${AGENT_IMPORT}"
  -d "${HARBOR_DATASET}"
  -m "${MODEL}"
  -e modal
  -n "${N_CONCURRENT}"
  --n-attempts "${RUNS}"
  --ak "temperature=${AGENT_TEMPERATURE}"
  --ek "app_name=${MODAL_APP_NAME}"
  --ek "sandbox_timeout_secs=${MODAL_SANDBOX_TIMEOUT_SEC}"
  --ek "sandbox_idle_timeout_secs=${MODAL_SANDBOX_IDLE_TIMEOUT_SEC}"
)

# Which candidate snapshot harness.harbor_run installs into the sandboxes. MINI_FORK_LOCAL
# is the snapshot dir from harness/scaffold.py:materialize (runs/run_<id>/candidate_<id>/
# micro-swe-agent). Passed as an agent kwarg; harbor_run also reads MICRO_SCAFFOLD_DIR from
# the env, so either works.
if [ -n "${MINI_FORK_LOCAL:-}" ]; then
  CMD+=( --ak "mini_fork_local=${MINI_FORK_LOCAL}" )
fi

# --- Modal reverse relay (RELAY=1) ------------------------------------------------
# micro runs INSIDE the Modal sandbox, so a self-hosted vLLM on this node is NOT
# reachable at localhost from there. RELAY=1 stands up harness/relay: deploy a public
# Modal app (HTTPS + a WebSocket the node dials out to) and run the node-side bridge that
# proxies to the local vLLM. Then LITELLM_PROXY_API_BASE/_KEY point at the relay, and the
# litellm_proxy wiring below injects them into the sandbox. Torn down on exit.
#   VLLM_LOCAL_URL   local vLLM base (default http://localhost:8000, no /v1)
#   RELAY_APP_NAME   Modal app name (default mho-vllm-relay-<pid>, per-run isolated)
RELAY_BRIDGE_PID=""
RELAY_APP_STOP=""
relay_cleanup() {
  [ -n "${RELAY_BRIDGE_PID}" ] && { echo "[relay] stopping bridge (pid=${RELAY_BRIDGE_PID})";
    kill "${RELAY_BRIDGE_PID}" 2>/dev/null; }
  [ -n "${RELAY_APP_STOP}" ] && { echo "[relay] stopping Modal app ${RELAY_APP_STOP}";
    uv run modal app stop "${RELAY_APP_STOP}" </dev/null >/dev/null 2>&1 || true; }
}
trap relay_cleanup EXIT

if [ "${RELAY:-0}" = "1" ]; then
  VLLM_LOCAL_URL="${VLLM_LOCAL_URL:-http://localhost:8000}"
  RELAY_APP_NAME="${RELAY_APP_NAME:-mho-vllm-relay-$$}"
  RELAY_SECRET="${RELAY_SECRET:-$(openssl rand -hex 16)}"
  echo "[relay] deploying ${RELAY_APP_NAME} (vllm=${VLLM_LOCAL_URL}) ..."
  _DEPLOY_OUT="$(RELAY_APP_NAME="${RELAY_APP_NAME}" RELAY_SECRET="${RELAY_SECRET}" \
      uv run modal deploy harness/relay/modal_relay.py 2>&1)" || { echo "${_DEPLOY_OUT}" >&2; exit 1; }
  _RELAY_HOST="$(printf '%s\n' "${_DEPLOY_OUT}" | grep -oE 'https://[a-z0-9.-]+\.modal\.run' | head -1)"
  [ -n "${_RELAY_HOST}" ] || { echo "ERROR: could not find relay URL in deploy output:" >&2; echo "${_DEPLOY_OUT}" >&2; exit 1; }
  RELAY_APP_STOP="${RELAY_APP_NAME}"
  echo "[relay] app URL: ${_RELAY_HOST}"

  # Node-side bridge: dials the relay WS, proxies to local vLLM. websockets+httpx via uv.
  RELAY_WS_URL="${_RELAY_HOST/https:/wss:}/bridge" RELAY_SECRET="${RELAY_SECRET}" \
    VLLM_LOCAL_URL="${VLLM_LOCAL_URL}" \
    uv run --with websockets --with httpx python harness/relay/bridge.py &
  RELAY_BRIDGE_PID=$!

  echo "[relay] waiting for bridge to register ..."
  _OK=0
  for _ in $(seq 1 24); do
    if curl -sf --max-time 10 "${_RELAY_HOST}/health" 2>/dev/null | grep -q '"bridge_connected":true'; then
      _OK=1; break
    fi
    sleep 5
  done
  [ "${_OK}" = "1" ] || { echo "ERROR: bridge never registered with relay ${RELAY_APP_NAME}" >&2; exit 1; }
  echo "[relay] bridge connected."

  # Point the deliberator at the relay; the litellm_proxy wiring below injects it.
  export LITELLM_PROXY_API_BASE="${_RELAY_HOST}/v1"
  export LITELLM_PROXY_API_KEY="${RELAY_SECRET}"
fi

# litellm_proxy/ passthrough vs openai/: litellm_proxy forwards the request as-is to an
# OpenAI-compatible server, bypassing openai/'s transforms (notably routing reasoning+tools
# to /v1/responses, which vLLM does not implement). But litellm resolves a litellm_proxy/
# endpoint from LITELLM_PROXY_API_BASE / LITELLM_PROXY_API_KEY -- NOT OPENAI_BASE_URL, which
# is all harbor's adapter injects. So when the model is litellm_proxy/, inject the proxy
# vars into the sandbox agent env (--ae), defaulting them from the OpenAI-style vars if the
# dedicated ones aren't set. (For an API/OpenAI model, leave MODEL=openai/... and skip this.)
case "${MODEL}" in
  litellm_proxy/*)
    _PROXY_BASE="${LITELLM_PROXY_API_BASE:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
    _PROXY_KEY="${LITELLM_PROXY_API_KEY:-${OPENAI_API_KEY:-${MSWEA_API_KEY:-EMPTY}}}"
    if [ -z "${_PROXY_BASE}" ]; then
      echo "ERROR: MODEL=litellm_proxy/... needs a base URL. Set LITELLM_PROXY_API_BASE" \
           "(or OPENAI_BASE_URL) to the vLLM/relay endpoint." >&2
      exit 1
    fi
    CMD+=( --ae "LITELLM_PROXY_API_BASE=${_PROXY_BASE}" )
    CMD+=( --ae "LITELLM_PROXY_API_KEY=${_PROXY_KEY}" )
    echo "[model] litellm_proxy passthrough -> ${_PROXY_BASE}"
    ;;
esac
# Scale per-task agent timeouts (slow inference needs more than the task's declared budget).
[ -n "${AGENT_TIMEOUT_MULTIPLIER}" ] && CMD+=( --agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER}" )
[ "${#TASK_FLAGS[@]}" -gt 0 ] && CMD+=("${TASK_FLAGS[@]}")
[ "${#EXTRA_FLAGS[@]}" -gt 0 ] && CMD+=("${EXTRA_FLAGS[@]}")

echo "=================================================================="
echo "agent:       ${AGENT_IMPORT}"
echo "dataset:     ${HARBOR_DATASET}   task_set=${TASK_SET}"
echo "model:       ${MODEL}   temp=${AGENT_TEMPERATURE}"
echo "env:         modal (app=${MODAL_APP_NAME}, sandbox<=${MODAL_SANDBOX_TIMEOUT_SEC}s, idle<=${MODAL_SANDBOX_IDLE_TIMEOUT_SEC}s)"
echo "concurrency: ${N_CONCURRENT}   n-attempts=${RUNS}   wall<=${HARBOR_TIMEOUT_SECONDS}s   agent-timeout-x=${AGENT_TIMEOUT_MULTIPLIER:-1.0}"
echo "=================================================================="

# Bound the whole run so a hung driver can't idle a Slurm/PBS allocation to its wall.
RUN_CMD=("${CMD[@]}")
if command -v timeout >/dev/null 2>&1 && [ "${HARBOR_EXTERNAL_TIMEOUT:-0}" != "1" ]; then
  RUN_CMD=(timeout --signal=SIGINT "${HARBOR_TIMEOUT_SECONDS}" "${CMD[@]}")
fi

# With a relay running we must NOT exec: run harbor as a child and wait, so the EXIT
# trap fires and tears the relay + bridge down. Without a relay, exec is fine (nothing
# to clean up).
if [ "${RELAY:-0}" = "1" ]; then
  "${RUN_CMD[@]}" &
  HARBOR_PID=$!
  wait "${HARBOR_PID}"
else
  exec "${RUN_CMD[@]}"
fi
