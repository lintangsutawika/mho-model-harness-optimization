#!/usr/bin/env bash
# Build a SkyRL[megatron] training image (Apptainer/Singularity .sif).
#
# WHY: the base SIF (skyrl-train-ray-...-cu13.0-megatron.sif) is system-only -- CUDA 13
# + uv, but NO Python ML packages (verified: no torch/skyrl/megatron baked). So the
# current train script rebuilds the full ~385-pkg env at runtime via `uv run --isolated`,
# which is the only reason pyproject.toml has to mirror SkyRL's entire [tool.uv] megatron
# resolution config.
#
# This bakes that env INTO the image instead -- the same pattern this repo already uses
# for vLLM (build_vllm.sh: "the base image already IS a complete install; we only layer
# small fixes"). The megatron resolution is done ONCE here, from SkyRL's OWN pyproject
# (an unmodified clone -- NO fork), then our thin deps are layered on top. Afterwards the
# launch runs the baked venv directly and pyproject.toml no longer needs the mirror.
#
# Our own src/ is NOT baked -- it's bind-mounted at run time (PYTHONPATH), so code edits
# never require an image rebuild.
#
# RUN THIS on a build node with `apptainer build --fakeroot` + network. It needs the CUDA
# toolkit (already in the base SIF) but does NOT need a GPU.
#
# Usage:
#   scripts/build/build_train.sh
#
# Env:
#   BASE_SIF       base image to build FROM (default: repo's ...-megatron.sif)
#   OUT_SIF        output skyrl-megatron image           (default: <BASE_SIF dir>/skyrl-megatron.sif)
#   SKYRL_REV      SkyRL git rev to bake -- MUST match the base SIF's stack
#                  (default: c516f3a5..., the rev the base SIF was built from)
#   HARBOR_REV     harbor git rev (default: c178c20..., matches pyproject.toml)
#   APPTAINER_BIN  container tool (auto: apptainer > singularity)

set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SELF_DIR}/../.." && pwd)"

# The base SIF sits one level up from the repo (see .env: BASE_SIF=${USER_DATA}/mho-.../..sif
# with USER_DATA=/home/<user>), i.e. next to the repo dir, not inside it.
BASE_SIF="${BASE_SIF:-$(dirname "${REPO_DIR}")/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron.sif}"
OUT_SIF="${OUT_SIF:-$(dirname "${BASE_SIF}")/skyrl-megatron.sif}"
SKYRL_REV="${SKYRL_REV:-c516f3a5634701f2d157753cb47bc3f8271b0f11}"
HARBOR_REV="${HARBOR_REV:-c178c20710c362ef806c5d5d18852f95b21ca34b}"
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo apptainer)}"
DEF_FILE="${SELF_DIR}/train-megatron.def"

[ -f "${BASE_SIF}" ] || { echo "base SIF not found: ${BASE_SIF}" >&2; exit 1; }

echo "==> container tool: ${APPTAINER_BIN}"
echo "==> base:   ${BASE_SIF}"
echo "==> out:    ${OUT_SIF}"
echo "==> SkyRL:  ${SKYRL_REV}"

cat > "${DEF_FILE}" <<EOF
Bootstrap: localimage
From: ${BASE_SIF}

# SkyRL[megatron] training image. The megatron env is resolved & built at BUILD time
# from SkyRL@${SKYRL_REV} (SkyRL's own [tool.uv] does the work -- no fork of SkyRL). Our
# own src/ is bind-mounted at run time, not baked, so code edits don't need a rebuild.
%post
    set -eux
    export CUDA_HOME=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:/usr/local/bin:\$PATH"
    export UV_LINK_MODE=copy
    # nv-grouped-gemm is the one source build; no GPU at build time, so target the archs
    # we run on (H100=9.0, B200=10.0). Adjust if you run on other GPUs.
    export TORCH_CUDA_ARCH_LIST="\${TORCH_CUDA_ARCH_LIST:-9.0;10.0}"

    # 1) SkyRL[megatron] env, from SkyRL's OWN pyproject + committed uv.lock (--frozen,
    #    so it's the exact resolution SkyRL ships -- the config we no longer mirror).
    git clone https://github.com/NovaSky-AI/SkyRL.git /opt/SkyRL
    git -C /opt/SkyRL checkout ${SKYRL_REV}
    cd /opt/SkyRL
    # Prefer the committed lock (exact resolution the SIF expects); resolve fresh only
    # if this rev shipped without one.
    if [ -f uv.lock ]; then
        uv sync --python 3.12 --extra megatron --frozen
    else
        uv sync --python 3.12 --extra megatron
    fi

    VENV=/opt/SkyRL/.venv

    # 2) Layer our thin deps into the SAME venv. All plain PyPI/git -- no [tool.uv] needed.
    uv pip install --python "\$VENV/bin/python" \\
        "harbor[modal] @ git+https://github.com/harbor-framework/harbor.git@${HARBOR_REV}" \\
        "terminal-bench==0.2.18" litellm python-dotenv pyyaml sympy loguru

    # 2b) The base image's Python 3.12 lives under the 'ray' user's conda (mode 750
    #     root:root), and 'uv sync' pointed the venv at it. This container runs as an
    #     arbitrary cluster uid (not root/ray), so open read+exec on the interpreter tree
    #     and on our editable skyrl source, or runtime imports die 'Permission denied'.
    #     (Metadata-only chmod; doesn't grow the image. Path derived so it survives base
    #     changes / a relocated interpreter.)
    REAL_PY="\$(readlink -f "\$VENV/bin/python")"
    PY_PREFIX="\$(cd "\$(dirname "\$REAL_PY")/.." && pwd)"   # e.g. /home/ray/anaconda3
    chmod a+rx "\$(dirname "\$PY_PREFIX")"                    # traverse into owning dir (/home/ray)
    chmod -R a+rX "\$PY_PREFIX" /opt/SkyRL

    # 3) Smoke-test the baked stack (fails the build early if anything is wrong).
    #    NB: the importable package is 'skyrl' (skyrl.train), NOT 'skyrl_train'.
    "\$VENV/bin/python" - <<'PY'
import torch, vllm, transformer_engine
import megatron.core  # noqa: F401
from skyrl.train.entrypoints.main_base import BasePPOExp  # what src/train_entrypoint.py needs
print("OK torch", torch.__version__, "| vllm", vllm.__version__, "| skyrl.train + megatron.core + TE OK")
PY
    "\$VENV/bin/python" -c "import harbor; print('OK harbor')" \\
        || echo "WARN: 'import harbor' failed -- check the module name if runtime import breaks"

%environment
    export SKYRL_VENV=/opt/SkyRL/.venv
    export PATH="/opt/SkyRL/.venv/bin:\$PATH"
EOF

echo "==> Building ${OUT_SIF} ..."
"${APPTAINER_BIN}" build --fakeroot "${OUT_SIF}" "${DEF_FILE}"

cat <<MSG
==> Done: ${OUT_SIF}

Run training against the skyrl-megatron image (train script auto-detects the baked env):

  export BASE_SIF="${OUT_SIF}"
  bash scripts/train/train_math_dapo.sh

(Point BASE_SIF back at the base SIF for the old 'uv run --isolated' path -- the
train script auto-selects the interpreter from whichever image it finds.)
MSG
