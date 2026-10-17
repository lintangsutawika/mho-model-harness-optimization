#!/bin/bash
set -Eeuo pipefail
echo "=== STARTING TEST EXECUTION ==="
mkdir -p /logs/verifier

# Sigma-aware reward: when a sigma_reward.json (sigma config) is injected next to the task,
# delegate the reward to the shared generic reward_rule.py; else fall back to the original
# verifier below. Apply this to any task to make it sigma-aware without changing its problem.
SIGMA_REWARD_MANIFEST="${SIGMA_REWARD_MANIFEST:-/tests/../sigma_reward.json}"
OUTPUT_ARTIFACT="${OUTPUT_ARTIFACT:-/workspace/output.txt}"
TRAJ_ARTIFACT="${TRAJ_ARTIFACT:-/workspace/trajectory.json}"

if [ -f "$SIGMA_REWARD_MANIFEST" ] && [ -f "/tests/reward_rule.py" ]; then
    echo "=== sigma reward rule active (sigma_reward.json present) ==="
    exec python3 /tests/reward_rule.py "$SIGMA_REWARD_MANIFEST" "$OUTPUT_ARTIFACT" "$TRAJ_ARTIFACT"
fi

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