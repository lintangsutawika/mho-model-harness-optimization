#!/usr/bin/env bash
# Async checkpoint eval: run scripts/optimize/ckpt_eval_sidecar.py with .env loaded (scheduler,
# queue and eval resources), e.g. as a small CPU job beside a train job:
#   bash scripts/hpc/submit_pbs.sh scripts/run/ckpt-eval.sh --hf-dir <run>/candidates/<c>/train/checkpoints/hf \
#     --val-data <val> --base-model Qwen/Qwen3.5-4B --out-dir <run>/ckpt_evals --num-steps 10 \
#     --sif <miles.sif> --hf-cache ~/.cache/huggingface [--skip-final]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
. scripts/utils/load_env.sh && load_env .env
exec "${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/scripts/optimize/ckpt_eval_sidecar.py" "$@"
