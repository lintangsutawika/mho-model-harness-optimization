#!/usr/bin/env bash
# One public OpenAI-compatible endpoint: SGLang (data parallel over this job's GPU slice) behind a
# per-model Modal reverse relay. Runs inside a PBS job (scripts/serve/pair.sh runs two of these
# on one rt_HF node, GPUs 0-3 and 4-7).
#
#   client --https /v1/*--> Modal app $RELAY_APP --ws--> bridge.py (this job) --> SGLang 127.0.0.1:$PORT
#
# Adapted from the MSL eval serve process (msl-multilingual-self-learning: scripts/rollout.sh with
# CAPTURE=0, launched per GPU half by scripts/serve/eval_pair.sh; relay src/relay/{public_api,
# node_forwarder}.py). Differences: no capture spool / weight watcher / dispatcher; our relay
# (src/harness/relay/modal_relay.py + bridge.py); harnessforge's SGLang flags; a STABLE relay key
# kept in a mode-600 key file (so the URL and key survive job restarts); and the relay app is NOT
# stopped on exit unless RELAY_STOP_ON_EXIT=1 (it then answers 503 until the next job connects).
#
# Env:
#   NAME            endpoint name (required), e.g. qwen35-9b
#   MODEL_PATH      local HF snapshot dir or checkpoint (required)
#   SERVED_NAME     model name clients send (default: NAME)
#   PORT            local SGLang port (default 30010)
#   CUDA_VISIBLE_DEVICES  this endpoint's GPUs; DP_SIZE defaults to their count, TP_SIZE to 1
#   CONTEXT_LENGTH  default 262144 (Qwen3.5 native)
#   SGLANG_EXTRA_ARGS  default: harnessforge pool flags (qwen3_coder tool parser, mem 0.85, metrics)
#   SERVE_DIR       key/endpoint/log root (default ~/hf-dev/serve)
#   RELAY_APP       Modal app name (default hf-serve-$NAME) -> https://<workspace>--$RELAY_APP-web.modal.run
#   KEY_FILE        relay key (default $SERVE_DIR/$NAME.key; created with mode 600 if missing)
#   RELAY_MAX_INPUTS / RELAY_REQUEST_TIMEOUT   relay concurrency / per-request timeout (128 / 1800)
#   RELAY_CPU       relay container CPUs (default 1.0; 4 waited a long time for Modal capacity)
#   MILES_SIF       SGLang image (default the shared miles_latest.sif)
set -euo pipefail
export PATH="${HOME}/.local/bin:/opt/pbs/bin:/usr/sbin:/usr/bin:/bin:${PATH}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
: "${NAME:?NAME is required}" "${MODEL_PATH:?MODEL_PATH is required}"
JOBTAG="${PBS_JOBID:-local-$$}"; JOBTAG="${JOBTAG%%.*}"
SERVED_NAME="${SERVED_NAME:-$NAME}"
PORT="${PORT:-30010}"
SERVE_DIR="${SERVE_DIR:-${HOME}/hf-dev/serve}"
LOG_DIR="${LOG_DIR:-${SERVE_DIR}/logs/${JOBTAG}-${NAME}}"
RELAY_APP="${RELAY_APP:-hf-serve-${NAME}}"
KEY_FILE="${KEY_FILE:-${SERVE_DIR}/${NAME}.key}"
MILES_SIF="${MILES_SIF:-/home/aci18914wh/mho-model-harness-optimization/miles_latest.sif}"
NGPU="$(printf '%s' "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | grep -c .)"
DP_SIZE="${DP_SIZE:-$(( NGPU / ${TP_SIZE:-1} ))}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---tool-call-parser qwen3_coder --mem-fraction-static 0.85 --enable-metrics}"
mkdir -p "$SERVE_DIR" "$LOG_DIR"
log() { echo "[endpoint ${NAME} $(date +%H:%M:%S)] $*"; }
log "job=${JOBTAG} model=${MODEL_PATH} served=${SERVED_NAME} gpus=${CUDA_VISIBLE_DEVICES:-} dp=${DP_SIZE} port=${PORT}"

# Stable relay key (owner-only). The value only ever travels through env vars, never argv/logs.
umask 077
[ -s "$KEY_FILE" ] || { openssl rand -hex 32 > "$KEY_FILE"; log "new relay key -> ${KEY_FILE}"; }
chmod 600 "$KEY_FILE"
RELAY_SECRET="$(cat "$KEY_FILE")"
API_KEY="$(openssl rand -hex 32)"     # per-launch: gates SGLang's /v1; only the bridge holds it
umask 022

PIDS=()
cleanup() {
  log "shutting down"
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  if [ "${RELAY_STOP_ON_EXIT:-0}" = 1 ]; then
    uv run -q --no-project --with modal modal app stop -y "$RELAY_APP" >/dev/null 2>&1 \
      || log "could not stop ${RELAY_APP}"
  elif [ -n "${RELAY_URL:-}" ]; then
    log "relay app ${RELAY_APP} left deployed (stop: modal app stop ${RELAY_APP})"
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# 1. SGLang, DP over this slice, bound to localhost only.
HF_CACHE_REAL="$(readlink -f "${HOME}/.cache/huggingface" 2>/dev/null || true)"
# miles_latest.sif keeps a root-only /root/Megatron-LM on sys.path; SGLang's scheduler dies importing
# it. Mask /root with an empty dir (as harnessforge's pool does).
mkdir -p "${SERVE_DIR}/emptyroot"
BINDS=(--bind /home:/home --bind "${SERVE_DIR}/emptyroot:/root")
[ -d /groups ] && BINDS+=(--bind /groups:/groups)
case "$HF_CACHE_REAL" in /home/*|/groups/*|"") ;; *) BINDS+=(--bind "${HF_CACHE_REAL}:${HF_CACHE_REAL}") ;; esac
# shellcheck disable=SC2086   # SGLANG_EXTRA_ARGS is deliberately word-split
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 singularity exec --nv --writable-tmpfs "${BINDS[@]}" "$MILES_SIF" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$PORT" \
  --tp-size "${TP_SIZE:-1}" --dp-size "$DP_SIZE" --context-length "${CONTEXT_LENGTH:-262144}" \
  --api-key "$API_KEY" $SGLANG_EXTRA_ARGS > "${LOG_DIR}/sglang.log" 2>&1 &
SGLANG_PID=$!; PIDS+=("$SGLANG_PID")
log "sglang pid=${SGLANG_PID}; waiting for health (log ${LOG_DIR}/sglang.log)"
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "$SGLANG_PID" 2>/dev/null || { log "sglang exited during startup"; tail -30 "${LOG_DIR}/sglang.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { log "sglang not healthy after 30 min"; exit 1; }
log "sglang healthy"

# 2. Modal relay (fixed app name = stable URL; the key is baked in from the env at deploy).
DEPLOY_OUT="$(RELAY_APP_NAME="$RELAY_APP" RELAY_SECRET="$RELAY_SECRET" \
  RELAY_MAX_INPUTS="${RELAY_MAX_INPUTS:-128}" RELAY_CPU="${RELAY_CPU:-1.0}" RELAY_REQUEST_TIMEOUT="${RELAY_REQUEST_TIMEOUT:-1800}" \
  uv run -q --no-project --with modal modal deploy src/harness/relay/modal_relay.py 2>&1)" \
  || { echo "$DEPLOY_OUT"; log "relay deploy failed"; exit 1; }
RELAY_URL="$(printf '%s' "$DEPLOY_OUT" | grep -oE 'https://[a-zA-Z0-9._-]+\.modal\.run' | head -1)"
[ -n "$RELAY_URL" ] || { echo "$DEPLOY_OUT"; log "could not parse relay URL"; exit 1; }
log "relay deployed: ${RELAY_URL}"

# 3. Bridge: dials out to the relay and proxies to SGLang.
start_bridge() {
  RELAY_WS_URL="wss://${RELAY_URL#https://}/bridge" RELAY_SECRET="$RELAY_SECRET" \
    VLLM_LOCAL_URL="http://127.0.0.1:${PORT}" BACKEND_API_KEY="$API_KEY" \
    BRIDGE_TIMEOUT="${RELAY_REQUEST_TIMEOUT:-1800}" \
    uv run -q --no-project --with websockets --with httpx python src/harness/relay/bridge.py \
    >> "${LOG_DIR}/bridge.log" 2>&1 &
  BRIDGE_PID=$!; PIDS+=("$BRIDGE_PID")
}
start_bridge

# Endpoint record for clients (no secret in it; the key stays in KEY_FILE).
cat > "${SERVE_DIR}/${NAME}.endpoint" <<EOF
RELAY_URL=${RELAY_URL}
OPENAI_BASE_URL=${RELAY_URL}/v1
SERVED_NAME=${SERVED_NAME}
KEY_FILE=${KEY_FILE}
JOB=${JOBTAG}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
EOF
log "endpoint -> ${SERVE_DIR}/${NAME}.endpoint"

# 4. Probe end to end through the relay (header via curl config on stdin: no key in argv).
sleep 15
if printf 'header = "Authorization: Bearer %s"\n' "$RELAY_SECRET" \
    | curl -sf -K - --max-time 60 "${RELAY_URL}/v1/models" > "${LOG_DIR}/probe.json" 2>&1; then
  log "probe ok: $(head -c 300 "${LOG_DIR}/probe.json")"
else
  log "PROBE FAILED (see ${LOG_DIR}/bridge.log)"
fi

# 5. Supervise: restart the bridge if it dies; end the job if SGLang dies.
log "running"
while true; do
  kill -0 "$SGLANG_PID" 2>/dev/null || { log "sglang exited"; tail -30 "${LOG_DIR}/sglang.log"; exit 1; }
  kill -0 "$BRIDGE_PID" 2>/dev/null || { log "bridge exited; restarting"; start_bridge; }
  sleep 30
done
