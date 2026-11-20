#!/bin/bash
# Build the training-only platoon venv: a copy of the evals venv with platoon-tito.patch applied
# (token-in/token-out prompts in platoon's model gateway). The evals venv is only read.
#
#   build_platoon_tito_venv.sh [SRC_VENV] [DEST_VENV]
#     SRC_VENV   default ~/hf-dev/evals/venv            (platoon 0.6.0)
#     DEST_VENV  default ~/hf-dev/train/platoon-tito-venv
#
# Run it in a CPU job (it copies ~6 GB), e.g.
#   qsub -q R9920261300 -P gai51740 -v RTYPE=rt_HC -l select=1 -l walltime=1:00:00 -j oe -o <log> -- \
#     /bin/bash -c "bash <this script>"
# Refuses an existing DEST (move it aside first). A run uses it through run.json's driver.platoon =
# DEST/bin/platoon; PLATOON_TITO=0 in driver.env turns the stitching off again in the same venv.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH=$HERE/platoon-tito.patch
SRC=$(realpath "${1:-$HOME/hf-dev/evals/venv}")
DEST=$(realpath -m "${2:-$HOME/hf-dev/train/platoon-tito-venv}")
PY=python3.12

[ -f "$PATCH" ] || { echo "no $PATCH" >&2; exit 1; }
[ -x "$SRC/bin/python" ] || { echo "no venv at $SRC" >&2; exit 1; }
[ -e "$DEST" ] && { echo "$DEST exists; move it aside first" >&2; exit 1; }
SP_REL=lib/$PY/site-packages
VERSION=$(sed -n 's/^Version: //p' "$SRC/$SP_REL"/platoon-*.dist-info/METADATA)
[ "$VERSION" = "0.6.0" ] || { echo "platoon $VERSION in $SRC; the patch is for 0.6.0" >&2; exit 1; }

TMP=$DEST.building.$$
echo "copying $SRC -> $TMP"
mkdir -p "$(dirname "$DEST")"
cp -a "$SRC" "$TMP"

# Entry points and activate scripts name the venv by absolute path (shebangs, VIRTUAL_ENV): point them at DEST.
# Regular files only: bin/python* are symlinks to the base interpreter and stay as they are.
find "$TMP/bin" -maxdepth 1 -type f -print0 | xargs -0 grep -lIZ -- "$SRC" | xargs -0 -r sed -i "s#$SRC#$DEST#g"
if grep -rlI -- "$SRC" "$TMP/bin" "$TMP/pyvenv.cfg" "$TMP/$SP_REL"/*.pth 2>/dev/null; then
    echo "still names $SRC (above)" >&2; exit 1
fi

echo "patching platoon in $TMP/$SP_REL"
(cd "$TMP/$SP_REL" && patch -p1 --dry-run -s < "$PATCH" && patch -p1 -s < "$PATCH")
mv "$TMP" "$DEST"

# Checks: the copy runs its own (patched) platoon, and its entry point uses its own interpreter.
head -1 "$DEST/bin/platoon" | grep -qx "#!$DEST/bin/python" || { echo "bad shebang in $DEST/bin/platoon" >&2; exit 1; }
"$DEST/bin/python" - "$DEST" <<'EOF'
import os, sys
import platoon, platoon.gateway.core as core, platoon.gateway.tito as tito
dest = sys.argv[1]
assert os.path.realpath(platoon.__file__).startswith(os.path.realpath(dest)), platoon.__file__
assert hasattr(core.ModelGateway, "render_call") and tito.tito_from_env() is True
print("platoon", platoon.__file__, "TITO default on")
EOF
"$DEST/bin/platoon" --help > /dev/null

cat > "$DEST/PLATOON_TITO.json" <<EOF
{"source_venv": "$SRC", "platoon": "$VERSION", "patch": "$(basename "$PATCH")",
 "patch_sha256": "$(sha256sum "$PATCH" | cut -d' ' -f1)", "built_at": "$(date -Is)", "host": "$(hostname)"}
EOF
echo "built $DEST"
cat "$DEST/PLATOON_TITO.json"
