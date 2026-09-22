#!/bin/sh
# DAPO math grading is done by harness.math_verifier:MathVerifier (in-process,
# read from the agent trajectory); this stub only satisfies harbor's task
# test-file requirement, mirroring dapo_math_17k/prepare.py.
# The standalone tests/test.py below is a faithful local re-implementation of
# that "Answer: $A" parse + normalized compare, for environments that run it.
exit 0