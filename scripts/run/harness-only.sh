#!/usr/bin/env bash
# Mode entrypoint: the 'harness-only' optimization loop. Thin wrapper around src/optimize_loop.py;
# extra args pass through, e.g.:  scripts/run/harness-only.sh --iterations 10 --task math --run-name r1
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
[ -f .env ] && { set -a; . .env; set +a; }
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export MHO_PROPOSE_JOB="${MHO_PROPOSE_JOB:-${REPO_DIR}/scripts/optimize/propose.sh}"
exec python3 "${REPO_DIR}/src/optimize_loop.py" --mode harness-only "$@"
