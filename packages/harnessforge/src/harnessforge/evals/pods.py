"""A platoon sandbox provider that starts every task sandbox as a one-service hpc-sandbox pod (needs platoon[hpc]).

Why: platoon's singularity provider runs a single-container task in the node's own network namespace. Many tmax
tasks start a service on a fixed port (8080, 5000, ...), and task-major evaluation runs every trial of a task at
once, of every candidate racing on it, so trials would collide on those ports (and on any port another task or the
node uses). Our harbor path (`mho.backends.hpc_sandbox_env`, `hpc_network=pod`) ran each trial as a pod: a network
namespace of its own, internet egress through slirp4netns, and the model reached through a Unix socket that the pod's
holder relays. This provider does the same for platoon:

- a sandbox with a network becomes a pod with `network="egress"`; its `main` service runs the task image, idle, as
  root (`run_as="subids"` by default: root that can switch users, as in Docker; `pod_users="single"` maps your uid
  only), and its upstreams (platoon's model gateway socket) are relayed by the pod into ports of its loopback
  (18080, ...), which the sandbox's routes report, so the harness reaches the gateway through the route;
- a sandbox without a network (a no-network task) is platoon's own single sandbox (needs `allow_network_none`).

`keep_agent_processes: true` makes `kill_processes` a no-op. platoon's Harbor grading kills every process the agents
left running before the tests run (so nothing forges the reward), which fails every task whose tests check a service
the agent was asked to leave running (tmax has many: "start the API on 0.0.0.0:8080"). Harbor, and so our old path,
grades with those processes alive; keep them for parity, at the cost of that guard.

Two options keep per-trial setup cheap on a 32-core node running ~100 trials:

- `image_map`: a JSON file {image reference: SIF path} (`scripts/hpcsb/prepare_images.py` at the repo root writes
  it). A mapped task starts from that SIF: the image from the shared cache, or one derived from it with curl, git,
  gcc/make, python3 and pip3 baked in, so the `harbor_system_deps` step finds them and installs nothing. References
  match with or without `docker://`; an unmapped one starts as before.
- `max_starting` (default `$HPCSB_MAX_STARTING`, else 16; 0 = no cap): at most that many sandboxes of the batch job
  are being started at once (`start_gate`: flocks in a node-local directory, so the cap spans platoon's rollout
  worker processes and is shared with the harbor path's `mho.backends.start_gate`). A start beyond the cap waits for
  a slot before anything is started. platoon times no single sandbox start (hpc-sandbox's own `start_timeout_s`
  starts after the wait); the wait counts only against the whole-rollout deadlines, which add
  `sandbox_start_budget_s` (600 s) per start, and a slot is held for seconds, so with 96 trials and 16 slots the
  longest wait is about six starts.

Select it in a Harbor entry: `provider: harnessforge.evals.pods:PodSingularityProvider`, with the usual singularity
`provider_options` (image_cache, scratch_root, fakeroot, ...) plus `pod_users` and `keep_agent_processes`.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from platoon.sandbox import cleanup
from platoon.sandbox.hpc import HpcSandbox
from platoon.sandbox.routes import FIRST_PORT, parse_upstreams
from platoon.sandbox.singularity import SingularityProvider
from platoon.utils.stopping import let_finish

from harnessforge.evals.start_gate import DEFAULT_MAX_STARTING, StartGate


def load_image_map(path: str | os.PathLike | None) -> dict[str, str]:
    """{image reference: SIF path} from a prepared-images map; {} when unset or missing."""
    if not path or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"image map {path} is not a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def mapped_image(image_map: dict[str, str], ref: str) -> str | None:
    bare = ref.removeprefix("docker://")
    return image_map.get(ref) or image_map.get(bare) or image_map.get(f"docker://{bare}")


class _PodRoutes:
    """The pod's routes, relayed by its holder, which nothing in the sandbox can stop."""

    def __init__(self, routes: dict[str, str]) -> None:
        self._routes = dict(routes)

    @property
    def routes(self) -> dict[str, str]:
        return dict(self._routes)

    async def ensure(self, sandbox: Any) -> dict[str, str]:
        return self.routes


class _KeepingSandbox(HpcSandbox):
    """A sandbox whose `kill_processes` is a no-op when its provider keeps the agents' processes (harbor parity)."""

    async def kill_processes(self) -> None:
        if not getattr(self._provider, "keep_agent_processes", False):
            await super().kill_processes()


class PodSandbox(_KeepingSandbox):
    """The `main` service of a one-service pod; stopping it stops the pod."""

    def __init__(self, provider: "PodSingularityProvider", inner: Any, pod: Any, routes: _PodRoutes) -> None:
        super().__init__(provider, inner, routes=routes)
        self._pod = pod

    async def stop(self) -> None:
        try:
            await self._pod.stop()
        finally:
            cleanup.unregister(self._token)
            self._token = None


class PodSingularityProvider(SingularityProvider):
    def __init__(self, *, pod_users: str = "subids", keep_agent_processes: bool = False,
                 image_map: str | None = None, max_starting: int | None = None, **options: Any) -> None:
        if pod_users not in ("subids", "single"):
            raise ValueError(f"pod_users must be subids or single, not {pod_users!r}")
        super().__init__(**options)
        self.pod_users = pod_users
        self.keep_agent_processes = bool(keep_agent_processes)
        self.image_map = load_image_map(image_map)
        if max_starting is None:
            raw = os.environ.get("HPCSB_MAX_STARTING", "").strip()
            max_starting = int(raw) if raw else DEFAULT_MAX_STARTING
        self.start_gate = StartGate(max_starting)
        # what a reap rebuilds the provider from
        self.options = {**self.options, "pod_users": pod_users, "keep_agent_processes": self.keep_agent_processes,
                        "image_map": image_map, "max_starting": max_starting}
        # Its sandboxes do not share this host's network: the harness reaches the gateway through the pod's route.
        self.capabilities = frozenset(self.capabilities - {"host_loopback"})

    def sandbox_for(self, inner: Any, *, routes: Any = None) -> HpcSandbox:
        return _KeepingSandbox(self, inner, routes=routes)  # no-network tasks' single sandboxes

    async def start(self, spec: Any) -> HpcSandbox:
        if mapped := mapped_image(self.image_map, spec.image or ""):
            spec = dataclasses.replace(spec, image=mapped)
        async with self.start_gate.slot():
            if spec.network == "none":
                return await super().start(spec)
            return await self._start_pod(spec)

    async def _start_pod(self, spec: Any) -> HpcSandbox:
        hs = self.hpc_sandbox
        upstreams: dict[str, str] = {}
        for upstream in parse_upstreams(spec.upstreams):
            if upstream.socket is None:
                raise ValueError(f"a pod relays only unix:// upstreams, not {upstream.name}={upstream.url}")
            upstreams[upstream.name] = upstream.url
        ports = {name: FIRST_PORT + i for i, name in enumerate(sorted(upstreams))}
        labels = {**spec.labels, **cleanup.labels()}
        pod_spec = hs.PodSpec(network="egress", upstreams=upstreams, upstream_ports=ports, labels=labels,
                              users=self.pod_users, hosts=[])
        token = cleanup.register(provider=cleanup.provider_ref(self), options=self.options, labels=labels)
        try:
            pod = await self.inner.start_pod(pod_spec)
        except BaseException:
            cleanup.unregister(token)
            raise
        try:
            service_spec = hs.ServiceSpec(
                name="main",
                sandbox=self.inner_spec(dataclasses.replace(spec, upstreams={}), labels=labels),
                command=None,
                run_as="subids" if self.pod_users == "subids" else "root",
            )
            inner = await pod.start_service(service_spec)
            sandbox = PodSandbox(self, inner, pod, _PodRoutes({name: pod.routes[name] for name in upstreams}))
            sandbox._token = token
            await sandbox.record_own_processes()
        except BaseException:
            # The pod stops even if the caller gave up meanwhile (a cleanup of the rollout's own).
            await asyncio.shield(let_finish(asyncio.ensure_future(pod.stop())))
            cleanup.unregister(token)
            raise
        cleanup.attach(token, sandbox)
        return sandbox
