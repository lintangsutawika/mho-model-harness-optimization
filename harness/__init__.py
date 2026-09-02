"""Meta-harness glue for optimizing the micro-swe-agent scaffold on Terminal-Bench 2.

Two pieces:
  * scaffold.py   -- materialize an immutable per-candidate snapshot of the scaffold
                     (runs/run_<id>/candidate_<id>/micro-swe-agent/).
  * harbor_run.py -- the harbor agent (AgentHarness) that uploads a snapshot into each
                     sandbox and installs+runs it. Fixed glue; identical per candidate.

The scaffold being OPTIMIZED lives in the micro-swe-agent package (its prompts, tools,
and agent loop); the meta-loop mutates a snapshot of it, not anything in here.
"""
