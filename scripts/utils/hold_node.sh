#!/usr/bin/env bash
# Park this node: submitted by the optimization loop (mho/jobs.py: submit_hold) pinned to a node
# whose sandboxes fail to start, so the loop's retried eval lands on a different node.
echo "[hold] parking $(hostname) for ${MHO_HOLD_SECONDS:-7200}s (eval sandboxes failed to start here; report it to the cluster admins)"
sleep "${MHO_HOLD_SECONDS:-7200}"
