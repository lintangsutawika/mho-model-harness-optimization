#!/bin/bash
# Two public Modal-relayed SGLang endpoints on ONE 8-GPU node (rt_HF): model A on GPUs 0-3,
# model B on GPUs 4-7, each a full scripts/serve/endpoint.sh (SGLang DP4 + its own Modal relay
# app + bridge). ABCI has no 4-GPU resource type, so this is how two models get 4 GPUs each.
# Adapted from msl-multilingual-self-learning scripts/serve/eval_pair.sh.
#
#   qsub -N hf-serve -q R9920261300 -P gai51740 -v RTYPE=rt_HF -l select=1 \
#        -l walltime=24:00:00 -j oe -o ~/hf-dev/serve/logs/ scripts/serve/pair.sh
#
# Defaults: A = Qwen/Qwen3.5-9B (name qwen35-9b, port 30011), B = Qwen/Qwen3.5-4B (qwen35-4b, port 30012);
# served under the HF ids, so litellm clients use litellm_proxy/Qwen/Qwen3.5-9B.
# Override with NAME_A/MODEL_A/SERVED_A/PORT_A and the _B equivalents (qsub -v ...). MODEL_* may be
# a local dir or an HF repo id (downloaded into the HF cache first). Everything else passes through
# to endpoint.sh (CONTEXT_LENGTH, SGLANG_EXTRA_ARGS, RELAY_MAX_INPUTS, SERVE_DIR, ...).
set -euo pipefail
export PATH="${HOME}/.local/bin:/opt/pbs/bin:/usr/sbin:/usr/bin:/bin:${PATH}"
REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT"
JOBTAG="${PBS_JOBID:-local-$$}"; JOBTAG="${JOBTAG%%.*}"
SERVE_DIR="${SERVE_DIR:-${HOME}/hf-dev/serve}"; export SERVE_DIR
mkdir -p "${SERVE_DIR}/logs"
log() { echo "[serve-pair $(date +%H:%M:%S)] $*"; }

NAME_A="${NAME_A:-qwen35-9b}"; MODEL_A="${MODEL_A:-Qwen/Qwen3.5-9B}"; SERVED_A="${SERVED_A:-Qwen/Qwen3.5-9B}"; PORT_A="${PORT_A:-30011}"
NAME_B="${NAME_B:-qwen35-4b}"; MODEL_B="${MODEL_B:-Qwen/Qwen3.5-4B}"; SERVED_B="${SERVED_B:-Qwen/Qwen3.5-4B}"; PORT_B="${PORT_B:-30012}"

fetch() {  # repo id (or a local dir, used as is) -> local snapshot path
  if [ -d "$1" ]; then echo "$1"; return; fi
  case "$1" in /*) echo "checkpoint $1 does not exist" >&2; return 1;; esac
  uv run -q --no-project --with huggingface_hub python -c \
    "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$1" | tail -1
}
PATH_A="$(fetch "$MODEL_A")"; PATH_B="$(fetch "$MODEL_B")"
log "A ${NAME_A}: ${PATH_A}"; log "B ${NAME_B}: ${PATH_B}"

PIDS=()
trap 'for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done; wait' EXIT TERM INT
serve() {  # name served port gpus model_path
  env NAME="$1" SERVED_NAME="$2" PORT="$3" CUDA_VISIBLE_DEVICES="$4" MODEL_PATH="$5" REPO_ROOT="$REPO_ROOT" \
    bash scripts/serve/endpoint.sh > "${SERVE_DIR}/logs/${JOBTAG}-$1.out" 2>&1 &
  PIDS+=($!)
  log "$1 ($2) on GPUs $4, port $3 (log ${SERVE_DIR}/logs/${JOBTAG}-$1.out)"
}
serve "$NAME_A" "$SERVED_A" "$PORT_A" 0,1,2,3 "$PATH_A"
sleep 20   # stagger start-ups
serve "$NAME_B" "$SERVED_B" "$PORT_B" 4,5,6,7 "$PATH_B"
for p in "${PIDS[@]}"; do wait "$p" || log "endpoint pid $p exited with $?"; done
log "both endpoints finished"
