#!/bin/bash
# Oracle solution: the pull request's own patch (tests excluded).
set -Eeuo pipefail
cd {workdir}
# No --3way: these instance images carry a git index the option cannot resolve against, and it
# fails with "does not match index" on patches that apply cleanly without it. tests/test.py
# handles the same thing by falling back.
git apply -v --recount --ignore-space-change --whitespace=nowarn \
    "$(cd "$(dirname "$0")" && pwd)/patch.diff"
