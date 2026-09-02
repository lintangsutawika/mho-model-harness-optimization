#!/usr/bin/env bash
#PBS -N mho-tb2
#PBS -l select=1
#PBS -l walltime=08:00:00
#PBS -j oe
# PBS/Torque wrapper (ABCI / qsub) around scripts/eval/eval_terminal_bench_2.sh.
#
# qsub takes NO positional script args and does not export your shell env by default, so
# pass config via `-v` and request env export with `-V`:
#
#   qsub -V -P <group> -q <queue> -l select=1 \
#     -v AGENT_IMPORT=harness.harbor_run:AgentHarness,TASK_SET=full,N_ATTEMPTS=1,N_CONCURRENT=16,MODEL=openai/gpt-5 \
#     scripts/hpc/submit_pbs.sh

set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"
mkdir -p logs
if [ -n "${PBS_JOBID:-}" ]; then
    exec > >(tee -a "logs/${PBS_JOBID}.out") 2>&1
fi

echo "[pbs] job=${PBS_JOBID:-?} node=$(hostname) workdir=${PBS_O_WORKDIR:-$(pwd)}"

# qsub can't forward positional args, so the eval script is driven purely by env vars
# (AGENT_IMPORT / TASK_SET / N_ATTEMPTS / N_CONCURRENT / MODEL / ...), passed via `-v`.
exec bash scripts/eval/eval_terminal_bench_2.sh
