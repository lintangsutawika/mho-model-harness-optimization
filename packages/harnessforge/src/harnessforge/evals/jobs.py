"""The batch job an evaluation runs in: a CPU node (sandboxes need cores) with node-local scratch for images.

The backend runs where its caller runs, so the caller (a search loop, or `python -m harnessforge.evals run`) is
what the job executes. `eval_node_job` describes that job as a `JobSpec` for a Launcher (workstream B); on ABCI:

    spec = eval_node_job("hf-eval", ["python", "-m", "harnessforge.evals", "run", "request.json"],
                         log=Path("eval.log"), seed_image_cache="/groups/.../hpc-sandbox-images")

Sandboxes take about one core each (tmax tasks ask for `cpus = 1`), so a node runs about `cpus - 4` trials at once,
summed over the candidates racing on it. Images are cached on node-local disk (`$PBS_LOCALDIR`, wiped with the job):
the job first copies the images its tasks need from `seed_image_cache` when given (`stage-images`), else they are
pulled on first use.
"""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnessforge.contracts import JobSpec

#: ABCI resource types: CPU node (32 cores here), one GPU (16 cores), full node (8 GPUs, 192 cores).
ABCI_CPUS = {"rt_HC": 32, "rt_HG": 16, "rt_HF": 192}
NODE_LOCAL = "${PBS_LOCALDIR:-${TMPDIR:-/tmp}}"


def eval_node_job(name: str, command: Sequence[str], *, resource: str = "rt_HC", walltime: str = "8:00:00",
                  queue: str | None = "R9920261300", account: str | None = "gai51740", log: Path | None = None,
                  seed_image_cache: str | None = None, request: str | Path | None = None,
                  env: Mapping[str, str] | None = None, python: str = "python") -> JobSpec:
    """A JobSpec running `command` on one eval node, after pointing platoon's node scratch and the sandbox image
    cache at node-local disk (`HF_EVAL_IMAGE_CACHE`, which a request's provider_options.image_cache can name) and,
    with `seed_image_cache` and `request`, copying the images the request's tasks need from that shared cache."""
    lines = [
        "set -euo pipefail",
        f"export PLATOON_NODE_SCRATCH={NODE_LOCAL}",
        f"export HF_EVAL_IMAGE_CACHE={NODE_LOCAL}/images",
        'mkdir -p "$HF_EVAL_IMAGE_CACHE"',
    ]
    if seed_image_cache and request:
        lines.append(f"{shlex.quote(python)} -m harnessforge.evals stage-images {shlex.quote(str(request))} "
                     f"--from {shlex.quote(seed_image_cache)} --to \"$HF_EVAL_IMAGE_CACHE\"")
    lines.append("exec " + shlex.join(command))
    options: dict[str, Any] = {"resource_type": resource}
    if queue:
        options["queue"] = queue
    if account:
        options["account"] = account
    return JobSpec(name=name, command=["/bin/bash", "-c", "\n".join(lines)], nodes=1, gpus=0,
                   cpus=ABCI_CPUS.get(resource, 0), walltime=walltime, env=dict(env or {}), log=log,
                   options=options)


def max_concurrency(resource: str = "rt_HC", reserve: int = 4) -> int:
    """How many trials a node of `resource` runs at once (about one core per sandbox), summed over candidates."""
    return max(1, ABCI_CPUS.get(resource, 0) - reserve)
