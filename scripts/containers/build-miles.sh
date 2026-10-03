#!/usr/bin/env bash
# Build the Miles training image (Apptainer/Singularity .sif) by layering what the
# Harbor rollout path needs onto radixark/miles:latest.
#
# radixark/miles:latest bundles Megatron-LM at /root/Megatron-LM and Miles at
# /root/miles (importable via the /opt/sglang python), but:
#   * /root is mode 0700 (root) -> a non-root cluster uid can't read miles/Megatron-LM
#   * /usr/bin/python3 has NO pip; the working interpreter is /opt/sglang/bin/python
#     (Miles' own runtime, has pip) -- harbor + our thin deps must be layered into THAT.
# This layers:
#   1) perms fix  -- /root world-traversable + miles/Megatron-LM world-readable
#   2) miles refresh -- git pull + `pip install -e /root/miles --no-deps`
#   3) harbor + thin deps into /opt/sglang/bin/python (in-process Harbor rollout)
#
# Our own src/ is NOT baked -- it's bind-mounted at run time (PYTHONPATH).
#
# RUN THIS on a build node with `singularity build --fakeroot` + network. No GPU needed.
#
# Usage:
#   scripts/build/build-miles.sh
#
# Env:
#   MILES_BASE_SIF  base image (default: <parent dir>/miles_latest.sif)
#   OUT_SIF         output image          (default: <base dir>/miles.sif)
#   HARBOR_REV      harbor git rev (default: c178c20..., matches pyproject.toml)
#   APPTAINER_BIN   container tool (auto: apptainer > singularity)
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
MILES_BASE_SIF="${MILES_BASE_SIF:-$(dirname "${REPO_DIR}")/miles_latest.sif}"
OUT_SIF="${OUT_SIF:-$(dirname "${MILES_BASE_SIF}")/miles.sif}"
HARBOR_REV="${HARBOR_REV:-c178c20710c362ef806c5d5d18852f95b21ca34b}"
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo apptainer)}"
DEF_FILE="${SELF_DIR}/miles.def"
[ -f "${MILES_BASE_SIF}" ] || { echo "base SIF not found: ${MILES_BASE_SIF}" >&2; exit 1; }
echo "==> container tool: ${APPTAINER_BIN}"
echo "==> base:   ${MILES_BASE_SIF}"
echo "==> out:    ${OUT_SIF}"
echo "==> harbor: ${HARBOR_REV}"
cat > "${DEF_FILE}" <<EOF
Bootstrap: localimage
From: ${MILES_BASE_SIF}
# Miles training image: perms fix + miles refresh + harbor/thin-deps layered on
# radixark/miles:latest. src/ is bind-mounted at run time, not baked.
%post
    set -eux
    export PATH="/usr/local/bin:/usr/bin:\$PATH"
    # 1) Perms: /root is 0700 root -> non-root cluster uid can't see miles/Megatron-LM.
    chmod a+rx /root
    chmod -R a+rX /root/miles /root/Megatron-LM
    # 2) Miles' working interpreter: /opt/sglang/bin/python (has pip; /usr/bin/python3
    #    does not). Refresh miles to latest + editable (no deps).
    PY=/opt/sglang/bin/python
    if [ -d /root/miles/.git ]; then
        git -C /root/miles pull --ff-only || echo "WARN: miles git pull failed (network?)"
    fi
    \$PY -m pip install -e /root/miles --no-deps
    # 3) Layer harbor + thin deps into the SAME interpreter (in-process Harbor rollout).
    \$PY -m pip install \\
        "harbor[modal] @ git+https://github.com/harbor-framework/harbor.git@${HARBOR_REV}" \\
        "terminal-bench==0.2.18" litellm python-dotenv pyyaml sympy loguru
    # 4) Smoke-test the layered stack -- any failure here MUST abort the build.
    \$PY - <<'PY'
import sys
import harbor, litellm, loguru, yaml, sympy   # noqa: F401
import pathlib
assert pathlib.Path("/root/miles").is_dir(), "/root/miles missing"
assert pathlib.Path("/root/Megatron-LM").is_dir(), "/root/Megatron-LM missing"
print("OK harbor + thin deps import; miles+Megatron-LM present; python", sys.version)
PY
%environment
    export PYTHONPATH="/root/miles:/root/Megatron-LM:\${PYTHONPATH:-}"
EOF
echo "==> Building ${OUT_SIF} ..."
"${APPTAINER_BIN}" build --fakeroot "${OUT_SIF}" "${DEF_FILE}"
cat <<MSG
==> Done: ${OUT_SIF}
Run training against the miles image:
  export BASE_SIF="${OUT_SIF}"
  bash scripts/optimize/train.sh
MSG