#!/bin/bash
set -Eeuo pipefail
echo "=== STARTING SWE-rebench-V2 TEST EXECUTION ==="
mkdir -p /logs/verifier

# The SWE-rebench instance images all carry a conda python, but the repo under test can be any
# of 20 languages, so look the interpreter up instead of assuming `python3` resolves.
PYTHON=""
for candidate in python3 python /opt/conda/bin/python3 /usr/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
done
if [ -z "$PYTHON" ]; then
    echo "X no python interpreter in the task image; cannot grade"
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

set +e
"$PYTHON" /tests/test.py
exit_code=$?
set -e
echo "=== TEST EXECUTION COMPLETED (exit ${exit_code}) ==="
if [ $exit_code -eq 0 ]; then
    echo "OK - issue resolved"; echo 1 > /logs/verifier/reward.txt
else
    echo "X - issue not resolved"; echo 0 > /logs/verifier/reward.txt
fi
exit $exit_code
