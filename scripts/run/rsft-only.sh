#!/usr/bin/env bash
# Mode entrypoint: the 'rsft-only' optimization loop. Thin wrapper around src/optimize_loop.py;
# extra args pass through, e.g.:  scripts/run/rsft-only.sh --iterations 10 --task math --run-name r1
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
. scripts/utils/load_env.sh && load_env .env
# rsft-only has no proposer (each candidate is an RSFT training phase); the propose job is
# wired anyway so the same env works for harness-rsft.
export MHO_PROPOSE_JOB="${MHO_PROPOSE_JOB:-${REPO_DIR}/scripts/optimize/propose.sh}"
exec "${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/src/optimize_loop.py" --mode rsft-only "$@"
