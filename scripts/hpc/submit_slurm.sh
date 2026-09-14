#!/usr/bin/env bash
# Submit a repo script as a SLURM batch job.
#
#   bash scripts/hpc/submit_slurm.sh <target-script> [args...]
#   e.g.  MHO_TRAINER=miles bash scripts/hpc/submit_slurm.sh scripts/optimize/train.sh
#
# Uses `sbatch --wrap` to cd to the repo and run <target> by relative path; --export=ALL carries
# the current env (set MHO_*/config vars before calling).
#
# Resources (env, overridable): SBATCH_GRES (default gpu:8 -- set to "" for a CPU-only eval),
#   SBATCH_TIME (24:00:00), SBATCH_PARTITION, SBATCH_ACCOUNT, JOB_NAME, SBATCH_OUT.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ "$#" -ge 1 ] || { echo "usage: bash $0 <target-script relative to repo> [args...]" >&2; exit 2; }
TARGET="$1"; shift
[ -f "$REPO_DIR/$TARGET" ] || echo "[submit_slurm] warning: $TARGET not found under repo" >&2

SB=(sbatch --parsable --export=ALL -J "${JOB_NAME:-mho-$(basename "$TARGET" .sh)}")
[ -n "${SBATCH_PARTITION:-}" ] && SB+=(-p "$SBATCH_PARTITION")
[ -n "${SBATCH_ACCOUNT:-}" ] && SB+=(-A "$SBATCH_ACCOUNT")
GRES="${SBATCH_GRES-gpu:8}"; [ -n "$GRES" ] && SB+=(--gres="$GRES")
SB+=(-t "${SBATCH_TIME:-24:00:00}")
[ -n "${SBATCH_OUT:-}" ] && SB+=(-o "$SBATCH_OUT")

echo "[submit_slurm] ${SB[*]}  --wrap 'cd repo && bash $TARGET $*'"
exec "${SB[@]}" --wrap="cd '$REPO_DIR' && exec bash $TARGET $*"
