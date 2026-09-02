#!/usr/bin/env bash
#SBATCH --job-name=mho-tb2
#SBATCH --output=logs/%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# GPU/partition are NOT hardcoded: this driver only orchestrates harbor + Modal and
# calls an API (or a public relay) for the model, so it needs CPU + network, not a GPU.
# If you self-host the model on THIS node (vLLM), add --gres=gpu:N + your partition at
# submit time and wire the serve+relay before calling the eval (see README).
#
# Slurm wrapper around scripts/eval/eval_terminal_bench_2.sh.
# Configure the run with env vars (exported to the job), then submit from the repo root:
#
#   sbatch --export=ALL,AGENT_IMPORT=harness.harbor_run:AgentHarness,\
#TASK_SET=full,N_ATTEMPTS=1,N_CONCURRENT=16,MODEL=openai/gpt-5 \
#     scripts/hpc/submit_slurm.sbatch
#
# Or positionally (sbatch forwards $1..$4 to the script):
#   sbatch scripts/hpc/submit_slurm.sbatch <agent> <task_set> <runs> <n_concurrent>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=${SLURM_JOB_ID:-?} node=$(hostname) submit_dir=${SLURM_SUBMIT_DIR:-$(pwd)}"

# Forward positional args if given; otherwise the eval script reads the env vars
# (AGENT_IMPORT / TASK_SET / N_ATTEMPTS / N_CONCURRENT / MODEL / ...).
exec bash scripts/eval/eval_terminal_bench_2.sh "$@"
