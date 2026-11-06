#!/usr/bin/env bash
# Mode entrypoint: the 'harness-rsft' optimization loop. Thin wrapper around src/optimize_loop.py;
# extra args pass through, e.g.:  scripts/run/harness-rsft.sh --iterations 10 --task math --run-name r1
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
. scripts/utils/load_env.sh && load_env .env
# harness-rsft alternates proposer-driven harness candidates with RSFT training phases, so it
# needs the proposer job wired.
export MHO_PROPOSE_JOB="${MHO_PROPOSE_JOB:-${REPO_DIR}/scripts/optimize/propose.sh}"
exec "${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/src/optimize_loop.py" --mode harness-rsft "$@"
