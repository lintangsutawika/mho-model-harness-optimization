#!/bin/bash
set -Eeuo pipefail
echo "=== STARTING HMMT MATH TEST EXECUTION ==="
mkdir -p /logs/verifier
# math_verify (MathArena's grader) handles LaTeX/expression equivalence. Install it
# best-effort; the verifier itself degrades to sympy then to normalized string compare.
( pip install --quiet math-verify >/dev/null 2>&1 \
  || pip install --quiet sympy >/dev/null 2>&1 || true )
set +e
python3 /tests/test.py
exit_code=$?
set -e
echo "=== TEST EXECUTION COMPLETED (exit ${exit_code}) ==="
if [ $exit_code -eq 0 ]; then
    echo "OK - answer correct"; echo 1 > /logs/verifier/reward.txt
else
    echo "X - answer incorrect"; echo 0 > /logs/verifier/reward.txt
fi
exit $exit_code