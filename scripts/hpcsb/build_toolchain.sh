#!/usr/bin/env bash
# Build the read-only agent toolchain hpc-sandbox trials run micro-swe-agent from (no per-trial uv/PyPI).
#
#   scripts/hpcsb/build_toolchain.sh [TARGET]      # in a CPU job (needs uv and network access to PyPI)
#
# TARGET (default /groups/gai51740/aci18914wh/mho/toolchains/hf-micro-swe-1) becomes a uv-managed CPython 3.12
# (python-build-standalone: relocatable, needs glibc in the task image) with the hash-locked packages of
# packages/harnessforge/src/harnessforge/evals/assets/hf-micro-swe-1.txt -- the same lock (and the same steps) as
# platoon's `platoon toolchain build` of harnessforge's hf-micro-swe-1, i.e. micro-swe-agent's dependencies plus
# the `--with litellm --with mcp --with orjson --with fastapi` extras the per-trial install added. manifest.json
# lists the installed distributions; MicroSweAgent compares a candidate's pyproject dependencies against it and
# falls back to the per-trial `uv tool install` for a candidate that needs more (see micro_swe_agent.py).
#
# hpc_sandbox_env (HPCSB_TOOLCHAIN=TARGET) copies it to node-local scratch once per node and bind-mounts it
# read-only at /opt/mho/toolchain in every pod. An existing TARGET is moved aside (TARGET.old-<time>), never
# deleted.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${1:-/groups/gai51740/aci18914wh/mho/toolchains/hf-micro-swe-1}"
LOCK="${TOOLCHAIN_LOCK:-${REPO}/packages/harnessforge/src/harnessforge/evals/assets/hf-micro-swe-1.txt}"
PY="${TOOLCHAIN_PYTHON:-3.12}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
[ -x "$UV" ] || { echo "uv not found (set UV=...)" >&2; exit 2; }
[ -f "$LOCK" ] || { echo "lock $LOCK not found" >&2; exit 2; }

parent="$(dirname "$TARGET")"; mkdir -p "$parent"
scratch="$(mktemp -d "${parent}/.$(basename "$TARGET").build-XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_LOCALDIR:-$scratch}/uv-cache}"
echo "[toolchain] CPython ${PY} via $("$UV" --version)"
"$UV" python install "$PY" --install-dir "$scratch/python" --no-bin --compile-bytecode
found=(); for d in "$scratch"/python/cpython-*; do [ -d "$d" ] && [ ! -L "$d" ] && found+=( "$d" ); done  # skip uv's minor-version link
[ "${#found[@]}" -eq 1 ] && [ -x "${found[0]}/bin/python3" ] || { echo "unexpected uv layout: ${found[*]}" >&2; exit 1; }
built="$scratch/tc"; mv "${found[0]}" "$built"   # the interpreter finds its stdlib relative to itself
echo "[toolchain] installing $(basename "$LOCK")"
# Wheels for glibc 2.28+ (manylinux_2_28; tiktoken has no 2_17 wheel), not the build node's: task images are as old as Ubuntu 20.04 (glibc 2.31),
# where e.g. cryptography's manylinux_2_34 wheel does not load.
"$UV" pip install --python "$built/bin/python3" --break-system-packages --require-hashes --compile-bytecode \
  --link-mode copy --python-platform "${TOOLCHAIN_PLATFORM:-x86_64-manylinux_2_28}" -r "$LOCK"
# Import check with the same flags the agent's launcher uses (and no cost-map fetch).
LITELLM_LOCAL_MODEL_COST_MAP=True "$built/bin/python3" -I -c \
  "import litellm, mcp, orjson, fastapi, jinja2, pydantic, tenacity, yaml; print('imports ok')"
"$built/bin/python3" -I - "$built" "$LOCK" <<'PY'
import hashlib, importlib.metadata as m, json, platform, sys, time
target, lock = sys.argv[1], sys.argv[2]
manifest = {
    "name": target.rstrip("/").rsplit("/", 1)[-1],
    "python": platform.python_version(),
    "lock": lock,
    "lock_sha256": hashlib.sha256(open(lock, "rb").read()).hexdigest(),
    "packages": {d.metadata["Name"]: d.version for d in m.distributions()},
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
open(f"{target}/manifest.json", "w").write(json.dumps(manifest, indent=1, sort_keys=True))
print(f"[toolchain] {len(manifest['packages'])} packages, python {manifest['python']}")
PY
chmod -R a+rX,go-w "$built"   # readable by any user a task's agent runs as
if [ -e "$TARGET" ]; then mv "$TARGET" "${TARGET}.old-$(date +%Y%m%d%H%M%S)"; fi
mv "$built" "$TARGET"
echo "[toolchain] ready: $TARGET ($(du -sh "$TARGET" | cut -f1))"
