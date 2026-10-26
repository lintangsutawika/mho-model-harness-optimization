#!/bin/bash

# Universal test runner for datagen-flash tasks
# This script installs test dependencies and runs pytest

set -e

# Check if we're in a valid working directory
if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set. Please set a WORKDIR in your Dockerfile before running this script."
    exit 1
fi

# Install pytest and CTRF plugin into system Python
# This ensures tests can access all packages installed in the dockerfile
echo "Installing pytest..."
pip install --break-system-packages --quiet pytest pytest-json-ctrf

# Install additional test dependencies if specified
# Format: One package per line in /tests/test_requirements.txt
if [ -f /tests/test_requirements.txt ]; then
    echo "Installing test dependencies from /tests/test_requirements.txt..."
    while IFS= read -r package || [[ -n "$package" ]]; do
        # Skip empty lines and comments
        [[ -z "$package" || "$package" =~ ^# ]] && continue
        echo "  Installing: $package"
        pip install --break-system-packages "$package" || echo "Warning: Failed to install $package"
    done < /tests/test_requirements.txt
    echo "Test dependencies installed."
fi

# Create logs directory if it doesn't exist
mkdir -p /logs/verifier

# Run pytest with CTRF output format
echo "Running pytest..."

# Disable set -e for pytest to allow capturing exit code for reward logic
set +e
python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
PYTEST_EXIT=$?
set -e

# Write reward based on test result
if [ $PYTEST_EXIT -eq 0 ]; then
  echo "All tests passed!"
  echo 1 > /logs/verifier/reward.txt
else
  echo "Some tests failed."
  echo 0 > /logs/verifier/reward.txt
fi

exit $PYTEST_EXIT

