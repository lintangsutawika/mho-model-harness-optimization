#!/usr/bin/env bash
# Mode entrypoint: the 'model-only' optimization loop. Thin wrapper around src/optimize_loop.py;
# extra args pass through, e.g.:  scripts/run/model-only.sh --iterations 10 --task math --run-name r1
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
[ -f .env ] && { set -a; . .env; set +a; }
# model-only proposes a MODEL improvement (sigma), so it needs the proposer job wired (same as
# the harness modes). Without this the propose step is skipped and no sigma is produced.
export MHO_PROPOSE_JOB="${MHO_PROPOSE_JOB:-${REPO_DIR}/scripts/optimize/propose.sh}"
exec "${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/src/optimize_loop.py" --mode model-only "$@"
