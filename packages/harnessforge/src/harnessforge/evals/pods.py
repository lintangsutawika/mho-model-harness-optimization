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

Select it in a Harbor entry: `provider: harnessforge.evals.pods:PodSingularityProvider`, with the usual singularity
`provider_options` (image_cache, scratch_root, fakeroot, ...) plus `pod_users`.
"""
from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from platoon.sandbox import cleanup
from platoon.sandbox.hpc import HpcSandbox
from platoon.sandbox.routes import FIRST_PORT, parse_upstreams
from platoon.sandbox.singularity import SingularityProvider
from platoon.utils.stopping import let_finish


class _PodRoutes:
    """The pod's routes, relayed by its holder, which nothing in the sandbox can stop."""

    def __init__(self, routes: dict[str, str]) -> None:
        self._routes = dict(routes)

    @property
    def routes(self) -> dict[str, str]:
        return dict(self._routes)

    async def ensure(self, sandbox: Any) -> dict[str, str]:
        return self.routes


class PodSandbox(HpcSandbox):
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
    def __init__(self, *, pod_users: str = "subids", **options: Any) -> None:
        if pod_users not in ("subids", "single"):
            raise ValueError(f"pod_users must be subids or single, not {pod_users!r}")
        super().__init__(**options)
        self.pod_users = pod_users
        self.options = {**self.options, "pod_users": pod_users}  # what a reap rebuilds the provider from
        # Its sandboxes do not share this host's network: the harness reaches the gateway through the pod's route.
        self.capabilities = frozenset(self.capabilities - {"host_loopback"})

    async def start(self, spec: Any) -> HpcSandbox:
        if spec.network == "none":
            return await super().start(spec)
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
