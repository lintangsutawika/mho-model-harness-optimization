#!/usr/bin/env bash
# Submit a repo script as a PBS batch job.
#
#   bash scripts/hpc/submit_pbs.sh <target-script> [args...]
#   e.g.  MHO_TRAINER=miles bash scripts/hpc/submit_pbs.sh scripts/optimize/train.sh
#
# The submitted job cd's to the repo (PBS_O_WORKDIR) and runs <target> by RELATIVE path, so the
# target's own BASH_SOURCE-based repo-dir resolution stays correct despite PBS spooling. The
# current env is exported to the job (-V), so set MHO_*/config vars before calling this.
#
# Resources (env, overridable): PBS_SELECT (default select=1:ngpus=8 -- set to select=1 for a
#   CPU-only eval), PBS_WALLTIME (24:00:00), PBS_QUEUE, PBS_GROUP, JOB_NAME, PBS_OUT.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ "$#" -ge 1 ] || { echo "usage: bash $0 <target-script relative to repo> [args...]" >&2; exit 2; }
TARGET="$1"; shift
[ -f "$REPO_DIR/$TARGET" ] || echo "[submit_pbs] warning: $TARGET not found under repo" >&2

QSUB=(qsub -V -j oe -N "${JOB_NAME:-mho-$(basename "$TARGET" .sh)}")
[ -n "${PBS_QUEUE:-}" ] && QSUB+=(-q "$PBS_QUEUE")
[ -n "${PBS_GROUP:-}" ] && QSUB+=(-P "$PBS_GROUP")
SEL="${PBS_SELECT-select=1:ngpus=8}"; [ -n "$SEL" ] && QSUB+=(-l "$SEL")
QSUB+=(-l "walltime=${PBS_WALLTIME:-24:00:00}")
[ -n "${PBS_OUT:-}" ] && QSUB+=(-o "$PBS_OUT")

echo "[submit_pbs] ${QSUB[*]}  <job: cd repo && bash $TARGET $*>"
"${QSUB[@]}" <<PBSJOB
cd "\${PBS_O_WORKDIR:-$REPO_DIR}"
exec bash $TARGET $*
PBSJOB
