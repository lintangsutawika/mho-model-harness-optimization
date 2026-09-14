#!/usr/bin/env bash
# Optimizer propose phase: analyze a completed (model, harness) eval and emit a new harness.
# Env-only. Runs the OpenHands scaffold-evolution agent (mho.optimizer.meta_harness); the SDK
# + litellm are installed on demand via uv at job time.
#
# Env:
#   MHO_CANDIDATE_DIR   where to write the new scaffold snapshot (required)
#   MHO_TRIALS_DIR      a completed eval run (trials dir) to analyze (optional)
#   MHO_FRONTIER        frontier.json to consider (optional)
#   MHO_SUMMARY         evolution_summary.jsonl (optional)
#   MICRO_SCAFFOLD_BASE base scaffold to mutate from (default: sibling checkout)
#   OPTIMIZER_MODEL / OPTIMIZER_LLM_API_KEY / OPTIMIZER_LLM_BASE_URL   proposer LLM (litellm)
#   OPENHANDS_PKGS      uv --with packages (default: openhands-sdk openhands-tools litellm)
set -euo pipefail
REPO_DIR="${MHO_REPO_DIR:-${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}}"; cd "$REPO_DIR"
[ -f .env ] && { set -a; . .env; set +a; }
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
# Do not let cwd (the repo root) prepend to sys.path: the stale repo-root harness/ package
# would shadow src/harness/ (agent_harness/math_verifier live only in src). src stays on
# PYTHONPATH, so this makes src win. See also the sys.path fixes in service.py/miles_agent_fn.
export PYTHONSAFEPATH=1
[ -n "${MHO_CANDIDATE_DIR:-}" ] || { echo "MHO_CANDIDATE_DIR required" >&2; exit 2; }
OPENHANDS_PKGS="${OPENHANDS_PKGS:-openhands-sdk openhands-tools litellm}"
WITH_ARGS=(); for _p in $OPENHANDS_PKGS; do WITH_ARGS+=(--with "$_p"); done
echo "[propose] run=${MHO_TRIALS_DIR:-?} out=${MHO_CANDIDATE_DIR} model=${OPTIMIZER_MODEL:-<default>}"
exec uv run --no-project --python "${OPTIMIZER_PYTHON_VERSION:-3.12}" "${WITH_ARGS[@]}" \
  python -m mho.optimizer.meta_harness
