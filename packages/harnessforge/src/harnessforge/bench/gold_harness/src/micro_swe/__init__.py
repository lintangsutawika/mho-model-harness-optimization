"""micro-swe-agent: a stripped-down, self-contained agent scaffold.

`Agent` (agents/) drives a plan/act/observe loop over a `Model` (models/) and an
`Environment` (environments/), using the `bash` tool (tools/). It reuses mini-swe-agent's
trajectory format but vendors its own config, protocols, and utilities so the *scaffolding*
is what you edit here. `__main__.py` exposes the `mini-swe-agent` console entrypoint that
a runner drives by shelling out to it.
"""

__version__ = "0.1.0"
