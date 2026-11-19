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

`collect_dir` (set by `PlatoonBackend` unless its option `collect` is false): before a sandbox stops, copy what platoon
does not keep out of it into `<collect_dir>/<rollout id>/` (the rollout id is the sandbox's `platoon.rollout_id` label,
which is also the id of the rollout's trajectory_collection.json): the tests' output (`/logs/verifier/test-stdout.txt`,
its last 4 MB), the agent's own trajectory (`/tmp/platoon-harness/<agent>/trajectory.json`, which our harness spec
writes) as `agent/<agent>.trajectory.json`, and `sandbox.json` (labels; when its start was asked for, when it was up,
when it was stopped). `harnessforge.evals.harbor_export` reads them. Collection is best effort: it is bounded by
`COLLECT_TIMEOUT_S`, never fails the stop, and logs what it could not read.

Select it in a Harbor entry: `provider: harnessforge.evals.pods:PodSingularityProvider`, with the usual singularity
`provider_options` (image_cache, scratch_root, fakeroot, ...) plus `pod_users` and `keep_agent_processes`.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import posixpath
import time
from pathlib import Path
from typing import Any

from platoon.sandbox import cleanup
from platoon.sandbox.hpc import HpcSandbox
from platoon.sandbox.routes import FIRST_PORT, parse_upstreams
from platoon.sandbox.singularity import SingularityProvider
from platoon.utils.stopping import let_finish

from harnessforge.evals.start_gate import DEFAULT_MAX_STARTING, StartGate


logger = logging.getLogger(__name__)

#: What the collect hook copies (see the module docstring).
TEST_STDOUT_PATH = "/logs/verifier/test-stdout.txt"
AGENT_STATE_ROOT = "/tmp/platoon-harness"  # platoon.harness.runner.SANDBOX_STATE_ROOT
AGENT_TRAJECTORY = "trajectory.json"       # harness.render_spec: --output={state_dir}/trajectory.json
COLLECT_MAX_BYTES = 4 * 1024 * 1024
COLLECT_TIMEOUT_S = 60.0
ROLLOUT_LABEL = "platoon.rollout_id"


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)[:200] or "_"


def _write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


async def collect(sandbox: Any) -> Path | None:
    """Copy the tests' output and the agent's trajectory out of `sandbox` into its provider's `collect_dir` (see the
    module docstring); returns the rollout's collect dir, or None when there is nothing to collect into. Never
    raises (but for cancellation)."""
    root = getattr(getattr(sandbox, "_provider", None), "collect_dir", None)
    labels = getattr(sandbox, "_hf_labels", None) or {}
    rollout = labels.get(ROLLOUT_LABEL)
    if not root or not rollout:
        return None
    dest = Path(root) / _safe_name(str(rollout))
    try:
        await asyncio.wait_for(_collect(sandbox, dest, labels), COLLECT_TIMEOUT_S)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # a dead sandbox, a timeout: the trial's other files still count
        logger.warning("collecting from sandbox %s (rollout %s) failed: %r", getattr(sandbox, "id", "?"), rollout, exc)
    return dest


async def _collect(sandbox: Any, dest: Path, labels: dict[str, str]) -> None:
    verifier = labels.get("platoon.role") == "verifier"
    times = dict(getattr(sandbox, "_hf_times", None) or {})
    times["stop_requested"] = time.time()
    _write_atomic(dest / ("sandbox-verifier.json" if verifier else "sandbox.json"),
                  json.dumps({"id": getattr(sandbox, "id", None), "labels": labels, **times}, indent=1))
    out = await sandbox.exec(["sh", "-c", f'[ -f "$1" ] && tail -c {COLLECT_MAX_BYTES} "$1"', "collect",
                              TEST_STDOUT_PATH], cwd="/", timeout_s=COLLECT_TIMEOUT_S)
    if out.exit_code == 0 and not out.timed_out:
        _write_atomic(dest / "test-stdout.txt", out.stdout)
    if verifier:
        return
    listing = await sandbox.exec(["sh", "-c", 'for f in "$1"/*/"$2"; do [ -f "$f" ] && echo "$f"; done; true',
                                  "collect", AGENT_STATE_ROOT, AGENT_TRAJECTORY], cwd="/", timeout_s=COLLECT_TIMEOUT_S)
    for path in (listing.stdout or "").splitlines():
        path = path.strip()
        if not path:
            continue
        got = await sandbox.exec(["cat", path], cwd="/", timeout_s=COLLECT_TIMEOUT_S)
        if got.exit_code == 0 and not got.timed_out and got.stdout:
            agent = _safe_name(posixpath.basename(posixpath.dirname(path)))
            _write_atomic(dest / "agent" / f"{agent}.{AGENT_TRAJECTORY}", got.stdout)


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

    async def stop(self) -> None:
        try:
            await collect(self)
        finally:
            await super().stop()


class PodSandbox(_KeepingSandbox):
    """The `main` service of a one-service pod; stopping it stops the pod."""

    def __init__(self, provider: "PodSingularityProvider", inner: Any, pod: Any, routes: _PodRoutes) -> None:
        super().__init__(provider, inner, routes=routes)
        self._pod = pod

    async def stop(self) -> None:
        try:
            await collect(self)
            await self._pod.stop()
        finally:
            cleanup.unregister(self._token)
            self._token = None


class PodSingularityProvider(SingularityProvider):
    def __init__(self, *, pod_users: str = "subids", keep_agent_processes: bool = False,
                 image_map: str | None = None, max_starting: int | None = None, collect_dir: str | None = None,
                 **options: Any) -> None:
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
        self.collect_dir = str(collect_dir) if collect_dir else None
        # what a reap rebuilds the provider from
        self.options = {**self.options, "pod_users": pod_users, "keep_agent_processes": self.keep_agent_processes,
                        "image_map": image_map, "max_starting": max_starting, "collect_dir": self.collect_dir}
        # Its sandboxes do not share this host's network: the harness reaches the gateway through the pod's route.
        self.capabilities = frozenset(self.capabilities - {"host_loopback"})

    def sandbox_for(self, inner: Any, *, routes: Any = None) -> HpcSandbox:
        return _KeepingSandbox(self, inner, routes=routes)  # no-network tasks' single sandboxes

    async def start(self, spec: Any) -> HpcSandbox:
        requested = time.time()
        if mapped := mapped_image(self.image_map, spec.image or ""):
            spec = dataclasses.replace(spec, image=mapped)
        async with self.start_gate.slot():
            if spec.network == "none":
                sandbox = await super().start(spec)
            else:
                sandbox = await self._start_pod(spec)
        sandbox._hf_labels = dict(spec.labels or {})  # what collect() files it under
        sandbox._hf_times = {"start_requested": requested, "started": time.time()}
        return sandbox

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
