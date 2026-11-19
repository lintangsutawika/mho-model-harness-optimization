"""Evaluation backends (workstream C): platoon eval for micro-swe-agent candidates, and a fake for tests.

- `PlatoonBackend` / `PlatoonRun` (`backend`): one `platoon eval` process per candidate, TASK_MAJOR, records read from
  the rollout directories while the run goes (`reader`), per-candidate stop.
- `FakeBackend` (`fake`): scripted rewards, simulated time, the same EvalRun behaviour, no I/O.
- `eval_node_job` (`jobs`): the PBS job an evaluation runs in.
- `harness`: a candidate dir as a platoon harness spec (toolchain + read-only mount, nothing installed in sandboxes).
- `pods` (imports platoon): a sandbox provider that runs every task sandbox as a one-service hpc-sandbox pod.

Only `pods` needs platoon; everything else is stdlib, so other workstreams can import this package without it.
"""
from harnessforge.evals.backend import PlatoonBackend, PlatoonRun
from harnessforge.evals.fake import FakeBackend, FakeRun
from harnessforge.evals.jobs import eval_node_job, max_concurrency
from harnessforge.evals.reader import RolloutReader, classify, read_rollouts

__all__ = ["FakeBackend", "FakeRun", "PlatoonBackend", "PlatoonRun", "RolloutReader", "classify",
           "eval_node_job", "max_concurrency", "read_rollouts"]
