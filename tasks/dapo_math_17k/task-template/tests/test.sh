#!/bin/bash
set -Eeuo pipefail
echo "=== STARTING DAPO MATH TEST EXECUTION ==="
mkdir -p /logs/verifier
# Self-contained answer-string grading (final Answer: $A, normalized compare):
# read the agent's "Answer: $A" line and compare (normalized) to the expected truth.
# test.py exits 0 on correct, non-zero on incorrect; we gate reward.txt on it so
# harbor's built-in file verifier can grade WITHOUT any src/ import.
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