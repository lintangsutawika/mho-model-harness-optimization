#!/usr/bin/env bash
# Mode entrypoint: the 'model-only' optimization loop. Thin wrapper around src/optimize_loop.py;
# extra args pass through, e.g.:  scripts/run/model-only.sh --iterations 10 --task math --run-name r1
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
[ -f .env ] && { set -a; . .env; set +a; }
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
exec python3 "${REPO_DIR}/src/optimize_loop.py" --mode model-only "$@"
