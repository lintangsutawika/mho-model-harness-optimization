#!/bin/bash
# Build a training-only platoon venv: a copy of the evals venv with platoon patches applied. The evals venv is only
# read.
#
#   build_platoon_tito_venv.sh [SRC_VENV] [DEST_VENV] [PATCH ...]
#     SRC_VENV   default ~/hf-dev/evals/venv            (platoon 0.6.0)
#     DEST_VENV  default ~/hf-dev/train/platoon-tito-venv
#     PATCH      patch files in this directory, applied in order; default platoon-tito.patch. The second venv:
#                ... ~/hf-dev/train/platoon-tito2-venv platoon-tito.patch platoon-tito2.patch
#
# Run it in a CPU job (it copies ~6 GB), e.g.
#   qsub -q R9920261300 -P gai51740 -v RTYPE=rt_HC -l select=1 -l walltime=1:00:00 -j oe -o <log> -- \
#     /bin/bash -c "bash <this script> ..."
# Refuses an existing DEST (move it aside first). The chat templates in ../templates are copied to DEST/templates (the
# path train.renderer_kwargs.chat_template names). A run uses the venv through run.json's driver.platoon =
# DEST/bin/platoon; PLATOON_TITO=0 / PLATOON_MIN_COMPLETION_TOKENS=0 in driver.env turn the switches off again.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(realpath "${1:-$HOME/hf-dev/evals/venv}")
DEST=$(realpath -m "${2:-$HOME/hf-dev/train/platoon-tito-venv}")
shift $(( $# < 2 ? $# : 2 ))
PATCHES=("$@")
[ ${#PATCHES[@]} -gt 0 ] || PATCHES=(platoon-tito.patch)
PY=python3.12

for p in "${PATCHES[@]}"; do [ -f "$HERE/$p" ] || { echo "no $HERE/$p" >&2; exit 1; }; done
[ -x "$SRC/bin/python" ] || { echo "no venv at $SRC" >&2; exit 1; }
[ -e "$DEST" ] && { echo "$DEST exists; move it aside first" >&2; exit 1; }
SP_REL=lib/$PY/site-packages
VERSION=$(sed -n 's/^Version: //p' "$SRC/$SP_REL"/platoon-*.dist-info/METADATA)
[ "$VERSION" = "0.6.0" ] || { echo "platoon $VERSION in $SRC; the patches are for 0.6.0" >&2; exit 1; }

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

for p in "${PATCHES[@]}"; do
    echo "applying $p in $TMP/$SP_REL"
    (cd "$TMP/$SP_REL" && patch -p1 --dry-run -s < "$HERE/$p" && patch -p1 -s < "$HERE/$p")
done
if [ -d "$HERE/../templates" ]; then
    mkdir -p "$TMP/templates" && cp "$HERE"/../templates/*.jinja "$TMP/templates/"
fi
mv "$TMP" "$DEST"

# Checks: the copy runs its own (patched) platoon, and its entry point uses its own interpreter.
head -1 "$DEST/bin/platoon" | grep -qx "#!$DEST/bin/python" || { echo "bad shebang in $DEST/bin/platoon" >&2; exit 1; }
"$DEST/bin/python" - "$DEST" "${PATCHES[@]}" <<'EOF'
import os, sys
import platoon, platoon.gateway.core as core, platoon.gateway.tito as tito
dest, patches = sys.argv[1], sys.argv[2:]
assert os.path.realpath(platoon.__file__).startswith(os.path.realpath(dest)), platoon.__file__
assert hasattr(core.ModelGateway, "render_call") and tito.tito_from_env() is True
print("platoon", platoon.__file__, "TITO default on")
if "platoon-tito2.patch" in patches:
    from platoon.train.tinker.renderers import ChatTemplateRenderer
    assert core.min_completion_tokens_from_env() == core.DEFAULT_MIN_COMPLETION_TOKENS == 2048
    assert "chat_template" in ChatTemplateRenderer.__init__.__code__.co_varnames
    for name in os.listdir(os.path.join(dest, "templates")):
        print("template", os.path.join(dest, "templates", name))
    print("context limit default 2048, chat_template override available")
EOF
"$DEST/bin/platoon" --help > /dev/null

SHAS=""
for p in "${PATCHES[@]}"; do SHAS="$SHAS\"$p\": \"$(sha256sum "$HERE/$p" | cut -d' ' -f1)\", "; done
cat > "$DEST/PLATOON_TITO.json" <<EOF
{"source_venv": "$SRC", "platoon": "$VERSION", "patches": {${SHAS%, }},
 "built_at": "$(date -Is)", "host": "$(hostname)"}
EOF
echo "built $DEST"
cat "$DEST/PLATOON_TITO.json"
