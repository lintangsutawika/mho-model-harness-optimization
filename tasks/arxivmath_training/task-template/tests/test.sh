#!/bin/bash
set -Eeuo pipefail
echo "=== STARTING ArXivMath-training TEST EXECUTION ==="
mkdir -p /logs/verifier

# Sigma-shaped reward: when a realization.json (sigma config: expected answer + checks) is
# injected next to the task, delegate the reward to the shared generic reward_rule.py. Otherwise
# fall back to the original verifier (test.py). This makes the task sigma-aware without changing
# its problem: inject realization.json to switch the reward, or run it standalone for the base.
REALIZATION_MANIFEST="${REALIZATION_MANIFEST:-/tests/../realization.json}"
OUTPUT_ARTIFACT="${OUTPUT_ARTIFACT:-/workspace/output.txt}"
TRAJ_ARTIFACT="${TRAJ_ARTIFACT:-/workspace/trajectory.json}"

if [ -f "$REALIZATION_MANIFEST" ] && [ -f "/tests/reward_rule.py" ]; then
    echo "=== sigma reward rule active (realization.json present) ==="
    exec python3 /tests/reward_rule.py "$REALIZATION_MANIFEST" "$OUTPUT_ARTIFACT" "$TRAJ_ARTIFACT"
fi

# --- original verifier (no realization.json -> unchanged base behavior) ---
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