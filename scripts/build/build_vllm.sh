#!/usr/bin/env bash
# Prepare a vLLM serving image (Apptainer/Singularity .sif) for SELF-HOSTING the model
# on a cluster node. Only needed for the self-hosted path -- if you eval against an API
# model (openai/anthropic/...), Modal reaches it directly and you don't build anything.
#
# Usage:
#   scripts/build/build_vllm.sh cuda     # NVIDIA (H100/H200/A100) -> vllm-cuda.sif
#   scripts/build/build_vllm.sh rocm     # AMD  (MI300/MI250)      -> vllm-rocm.sif
#
# Env:
#   VLLM_VERSION      CUDA base tag version (default 0.28.0 -> vllm/vllm-openai:v0.28.0)
#   VLLM_ROCM_TAG     ROCm base tag (default latest -> vllm/vllm-openai-rocm:latest)
#   OUT_DIR           where the .sif is written (default: repo root)
#   APPTAINER_BIN     override the container tool (auto: apptainer > singularity)
#   ADD_QWEN35_KERNELS=1  (CUDA only) also install causal-conv1d + fla for Qwen3.5
#                          Gated-DeltaNet kernels (harmless for other models)

set -euo pipefail
ACC="${1:-cuda}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_DIR}}"
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo apptainer)}"
mkdir -p "${OUT_DIR}"

case "${ACC}" in
  cuda)
    VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
    BASE_URI="docker://vllm/vllm-openai:v${VLLM_VERSION}"
    BASE_SIF="${OUT_DIR}/vllm-openai-cuda-${VLLM_VERSION}.sif"
    BUILT_SIF="${OUT_DIR}/vllm-cuda.sif"
    DEF_FILE="${SELF_DIR}/vllm-cuda.def"
    ;;
  rocm)
    VLLM_ROCM_TAG="${VLLM_ROCM_TAG:-latest}"
    BASE_URI="docker://vllm/vllm-openai-rocm:${VLLM_ROCM_TAG}"
    BUILT_SIF="${OUT_DIR}/vllm-rocm.sif"
    ;;
  *)
    echo "Unknown accelerator '${ACC}' (use: cuda | rocm)" >&2; exit 1 ;;
esac

echo "==> [${ACC}] container tool: ${APPTAINER_BIN}"

if [ "${ACC}" = "rocm" ]; then
  # ROCm vLLM ships a complete prebuilt serving image -- nothing to compile, just pull.
  if [ -f "${BUILT_SIF}" ]; then
    echo "==> ${BUILT_SIF} already exists (delete to re-pull)."
  else
    echo "==> Pulling ${BASE_URI} -> ${BUILT_SIF} ..."
    "${APPTAINER_BIN}" pull "${BUILT_SIF}" "${BASE_URI}"
  fi
  echo "==> Done: ${BUILT_SIF}"
  exit 0
fi

# CUDA: the base image already IS a complete vLLM install; we only layer small fixes.
if [ ! -f "${BASE_SIF}" ]; then
  echo "==> Pulling CUDA base ${BASE_URI} (one-time) ..."
  "${APPTAINER_BIN}" pull "${BASE_SIF}" "${BASE_URI}"
fi

{
  echo "Bootstrap: localimage"
  echo "From: ${BASE_SIF}"
  echo
  echo "%post"
  echo "    set -e"
  echo "    PY=\$(command -v python3)"
  # The base's prometheus-fastapi-instrumentator 8.0.0 500s every request under fastapi
  # 0.116+; 8.0.2 fixes it. Idempotent if the base already ships >=8.0.2.
  echo "    \$PY -m pip install -q 'prometheus-fastapi-instrumentator==8.0.2'"
  if [ "${ADD_QWEN35_KERNELS:-0}" = "1" ]; then
    echo "    \$PY -m pip install -q 'causal-conv1d>=1.5.0'"
    echo "    \$PY -m pip install -q --no-deps 'fla-core==0.5.1' 'flash-linear-attention==0.5.1'"
  fi
  echo "    \$PY -c \"import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)\""
  echo
  echo "%environment"
  echo "    :"
} > "${DEF_FILE}"

echo "==> Building ${BUILT_SIF} ..."
"${APPTAINER_BIN}" build --fakeroot "${BUILT_SIF}" "${DEF_FILE}"
echo "==> Done: ${BUILT_SIF}"
