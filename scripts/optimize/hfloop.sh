#!/usr/bin/env bash
# hfloop.sh: one long CPU job that drives the harnessforge harness search (src/mho/hfloop.py): starts the ctxcap
# context-window proxy in front of the pool router, writes the proposer's endpoint file, then runs the loop.
#
# Env:
#   HFLOOP_CONFIG     loop config JSON (required; see mho.hfloop.Config)
#   POOL_STATE        the pool's state dir (router.json), e.g. ~/hf-dev/pool/state/n4hf72b (required)
#   TOKENIZER         tokenizer path for ctxcap (the served model's HF dir; required)
#   OPTIMIZER_SRC_ENV an env file with DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL / DEEPSEEK_JOB
#                     (mapped to the proposer's OPTIMIZER_* in a mode-600 node-local file; values are never printed)
#   HF_PYTHON         python with harnessforge's deps (default: python3)
#   CTX_WINDOW / CTX_MIN_COMPLETION / CTX_WORKERS   ctxcap settings (default 65536 / 2048 / 20)
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${HF_PYTHON:-python3}"
: "${HFLOOP_CONFIG:?}" "${POOL_STATE:?}" "${TOKENIZER:?}"
export PYTHONPATH="${REPO_DIR}/packages/harnessforge/src:${REPO_DIR}/src:${PYTHONPATH:-}"
export PLATOON_NODE_SCRATCH="${PBS_LOCALDIR:-${TMPDIR:-/tmp}}"
RUN_DIR=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["run_dir"])' "$HFLOOP_CONFIG")
AUX="$RUN_DIR/aux"; mkdir -p "$AUX"
echo "[hfloop.sh] $(date +%T) on $(hostname) job ${PBS_JOBID:-} run $RUN_DIR"

if [ -n "${OPTIMIZER_SRC_ENV:-}" ]; then
  ENVF="${PLATOON_NODE_SCRATCH}/optimizer.env"
  ( umask 077; set -a; . "$OPTIMIZER_SRC_ENV"; set +a
    printf 'export OPTIMIZER_MODEL=%q\nexport OPTIMIZER_LLM_API_KEY=%q\nexport OPTIMIZER_LLM_BASE_URL=%q\nexport OPTIMIZER_SERVER_JOB=%q\n' \
      "$DEEPSEEK_MODEL" "$DEEPSEEK_API_KEY" "$DEEPSEEK_BASE_URL" "${DEEPSEEK_JOB:-}" > "$ENVF" )
  export MHO_OPTIMIZER_ENV="$ENVF"
  echo "[hfloop.sh] proposer endpoint file written (names only: OPTIMIZER_MODEL/_LLM_API_KEY/_LLM_BASE_URL)"
fi

ROUTER=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]+"/router.json"))["url"])' "$POOL_STATE")
rm -f "$AUX/ctxcap.url"
"$PY" -m harnessforge.evals.ctxcap --upstream "$ROUTER" --tokenizer "$TOKENIZER" --window "${CTX_WINDOW:-65536}" \
  --min-completion "${CTX_MIN_COMPLETION:-2048}" --margin 32 --workers "${CTX_WORKERS:-20}" \
  --url-file "$AUX/ctxcap.url" --stats "$AUX/ctxcap-stats.json" --stats-every 60 > "$AUX/ctxcap.log" 2>&1 &
CTX=$!
trap 'kill $CTX 2>/dev/null' EXIT
for _ in $(seq 1 120); do [ -s "$AUX/ctxcap.url" ] && break; sleep 2; done
[ -s "$AUX/ctxcap.url" ] || { echo "[hfloop.sh] ctxcap did not start"; cat "$AUX/ctxcap.log"; exit 4; }
EP=$(cat "$AUX/ctxcap.url")
echo "[hfloop.sh] ctxcap up"
"$PY" -m mho.hfloop run --config "$HFLOOP_CONFIG" --endpoint "$EP"
RC=$?
echo "[hfloop.sh] $(date +%T) loop exit $RC"
exit $RC
