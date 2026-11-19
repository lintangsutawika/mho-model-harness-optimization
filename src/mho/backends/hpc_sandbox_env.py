"""Harbor environment on hpc-sandbox (https://github.com/ApGa/hpc-sandbox).

Runs each Harbor trial in an hpc-sandbox sandbox on the compute node itself: Singularity/Apptainer
(default), enroot or gVisor, rootless, no Docker, no Modal relay.

Networking (hpc_network):
  "pod" (default) -- each trial is a one-service pod with its OWN network namespace, so tasks can
      bind any port (8000, 27017, ...) without colliding with each other or with the node's vLLM.
      Internet egress goes through slirp4netns (never to the host's loopback); network_mode
      no-network turns egress off. Host services such as the model server reach the trial as
      "upstreams": hpc_upstreams / $HPCSB_UPSTREAMS = "name=/abs/path.sock@port[;...]" makes
      http://name:port inside every trial relay to that Unix socket on the host (pair it with
      mho.backends.uds_forward to put a TCP server such as vLLM behind the socket).
  "host" -- the sandbox shares the node's network (localhost reaches vLLM; ports collide).
  "none" -- loopback only, no upstreams (Singularity needs hpc_allow_network_none).
Users (hpc_pod_users): "subids" (default) maps your subordinate ids so root in the trial can switch
users (su), as Docker tasks expect; "single" maps only your uid as root.

What hpc-sandbox adds over harbor's stock SingularityEnvironment: one image cache shared by every
trial in the process (and by nodes, if on shared storage), per-sandbox node-local writable roots,
command timeouts and bounded output, a PSS memory watchdog instead of cgroups, and janitors that
reap sandboxes of dead processes. One provider is shared by all trials in a process.

    harbor run ... --environment-import-path mho.backends.hpc_sandbox_env:HpcSandboxEnvironment \
        [--ek hpc_runtime=singularity] [--ek hpc_image_cache=/groups/.../hpc-sandbox-images]

Images: [environment].docker_image, else the Dockerfile's last FROM (flags such as
--platform=... skipped). There is no image builder: with hpc_dockerfile_steps="run" (default) the
final stage's build steps are replayed in the started sandbox, as root, in order -- RUN as a shell
command under the WORKDIR/ENV in effect at that line, plain COPY/ADD (build-context sources, no
--from, no URLs) as uploads -- and a step that cannot be replayed fails that trial at start.
"ignore" skips build steps, "warn" skips and logs them, "error" fails any task that has them.
docker-compose files are ignored (single-service tasks only).

Prepared images (hpc_image_map / $HPCSB_IMAGE_MAP): a JSON file {image reference: /abs/path.sif} written by
scripts/hpcsb/prepare_images.py. A mapped task starts from that SIF instead of pulling its reference: the image from
the shared cache, or one derived from it with the system packages harbor's agent setup would otherwise apt-install in
every trial (curl, bash, gcc/make, git, python3, pip3). The dataset is not changed. A mapped SIF is copied to node-local
scratch once per node before use ($HPCSB_IMAGE_STAGE=0 starts it from shared storage).

Agent toolchain (hpc_toolchain / $HPCSB_TOOLCHAIN): a prebuilt CPython 3.12 with the agent's packages
(scripts/hpcsb/build_toolchain.sh), copied once per node to local scratch and bind-mounted read-only at
/opt/mho/toolchain in every sandbox; harness.micro_swe_agent runs the candidate from it instead of `uv tool install`ing
it from PyPI in every trial ($HPCSB_TOOLCHAIN_STAGE=0 mounts it from where it is).

Start cap ($HPCSB_MAX_STARTING, default 16, 0 = none; mho.backends.start_gate): at most that many trials of the job
are between "start the sandbox" and "agent set up" at once. Importing this module wraps harbor's Trial._prepare
(environment start + healthcheck + agent setup) so the slot is taken BEFORE harbor's timed regions: the wait does not
count against the environment-start (task build_timeout_sec, 600 s) or agent-setup (360 s) timeouts, and is not part
of the trial's environment_setup timing (logged instead). Without that wrapper (another trial class), start() takes
the slot for the sandbox start only.

Docker Hub pulls are authenticated if SINGULARITY_DOCKER_USERNAME / SINGULARITY_DOCKER_PASSWORD
(or APPTAINER_*) are set in the environment, as singularity reads them itself.

Kwargs (--ek): hpc_runtime ("singularity" | "enroot" | "gvisor"), hpc_scratch_root,
hpc_image_cache, hpc_fakeroot (default True: root inside, as Docker tasks expect),
hpc_exec_timeout_s (default 86400: harbor passes no timeout for whole agent loops),
hpc_output_limit_mb (default 64), hpc_allow_network_none (default False),
hpc_dockerfile_steps ("run" | "error" | "warn" | "ignore", default "run"), hpc_start_timeout_s,
hpc_network ("pod" | "host" | "none"), hpc_pod_users ("subids" | "single"), hpc_upstreams, hpc_image_map,
hpc_toolchain.
Env: HPC_SANDBOX_* as hpc-sandbox documents (e.g. HPC_SANDBOX_IMAGE_CACHE); HPCSB_IMAGE_MAP, HPCSB_TOOLCHAIN,
HPCSB_TOOLCHAIN_STAGE, HPCSB_MAX_STARTING, HPCSB_START_SLOT_DIR.
"""
from __future__ import annotations

import asyncio
import fcntl
import functools
import hashlib
import json
import os
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, NamedTuple

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)

from mho.backends.start_gate import StartGate

_PROVIDERS: dict[tuple, Any] = {}
#: Where the agent toolchain appears in every sandbox (harness.micro_swe_agent runs mini-swe-agent from it).
TOOLCHAIN_MOUNT = "/opt/mho/toolchain"
_GATE: StartGate | None = None
_STAGED: dict[str, Path] = {}
_STAGE_LOCKS: dict[str, asyncio.Lock] = {}


def start_gate() -> StartGate:
    global _GATE
    if _GATE is None:
        _GATE = StartGate.from_env()
    return _GATE


@functools.lru_cache(maxsize=8)
def _read_image_map(path: str, mtime: float) -> dict[str, str]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"image map {path} is not a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def load_image_map(path: str | None) -> dict[str, str]:
    """{image reference: SIF path} from a prepared-images map; {} when unset or missing."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    return _read_image_map(str(p), p.stat().st_mtime)


def mapped_image(image_map: dict[str, str], ref: str) -> str | None:
    """The prepared SIF for `ref`, spelled with or without docker://."""
    bare = ref.removeprefix("docker://")
    return image_map.get(ref) or image_map.get(bare) or image_map.get(f"docker://{bare}")


def toolchain_packages(toolchain: str | Path) -> frozenset[str]:
    """Normalized distribution names the toolchain holds (its manifest.json), lower-case with '-'."""
    try:
        packages = json.loads((Path(toolchain) / "manifest.json").read_text()).get("packages", {})
    except (OSError, ValueError):
        return frozenset()
    return frozenset(_norm(name) for name in packages)


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def _stage_copy(src: Path, scratch: Path) -> Path:
    """Copy `src` (a file, or a directory such as the toolchain) into node-local `scratch` once per node: a lock and
    a done-marker make concurrent processes share one copy. The copy's name carries a key of the source (a
    directory's manifest.json hash, a file's size and mtime), so a rebuilt source is copied anew."""
    if src.is_dir():
        manifest = src / "manifest.json"
        key = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12] if manifest.is_file() else "nomanifest"
        dest = scratch / f"{src.name}-{key}"
    else:
        st = src.stat()
        dest = scratch / f"{st.st_size}-{int(st.st_mtime)}-{src.name}"
    done = scratch / f".{dest.name}.done"
    if done.is_file() and dest.exists():
        return dest
    scratch.mkdir(parents=True, exist_ok=True)
    fd = os.open(scratch / f".{dest.name}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if done.is_file() and dest.exists():
            return dest
        if dest.exists():
            dest.rename(scratch / f".{dest.name}.partial-{os.getpid()}-{int(time.time())}")
        tmp = scratch / f".{dest.name}.tmp-{os.getpid()}"
        if src.is_dir():
            shutil.copytree(src, tmp, symlinks=True)
        else:
            shutil.copyfile(src, tmp)
            tmp.chmod(0o444)
        tmp.rename(dest)
        done.touch()
        return dest
    finally:
        os.close(fd)


async def stage_copy(src: str | Path, scratch: str | Path) -> Path:
    """The node-local copy of `src` (copied in a thread on first use; once per process and per node)."""
    key = f"{Path(src).resolve()}|{scratch}"
    if key in _STAGED:
        return _STAGED[key]
    async with _STAGE_LOCKS.setdefault(key, asyncio.Lock()):  # one copy per source; different sources in parallel
        if key not in _STAGED:
            _STAGED[key] = await asyncio.to_thread(_stage_copy, Path(src).resolve(), Path(scratch))
    return _STAGED[key]


stage_toolchain = stage_copy


# Harbor agents write bash; images without bash (alpine) still get sh.
_BASH_OR_SH = 'if command -v bash >/dev/null 2>&1; then exec bash -c "$1"; else exec sh -c "$1"; fi'
# Dockerfile instructions that only describe the image (or are harmless to skip).
_DESCRIPTIVE = {"FROM", "WORKDIR", "ENV", "ARG", "LABEL", "MAINTAINER", "EXPOSE", "CMD",
                "ENTRYPOINT", "USER", "SHELL", "STOPSIGNAL", "HEALTHCHECK", "VOLUME", "ONBUILD"}


def _truthy(v: Any) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def _dockerfile_instructions(text: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, args) with line continuations joined and comments dropped."""
    out: list[tuple[str, str]] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1].rstrip() + " "
            continue
        buf += line
        parts = buf.strip().split(None, 1)
        if parts:
            out.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
        buf = ""
    return out


def parse_from(args: str) -> str:
    """Image of a FROM line: skip --flags (e.g. --platform=linux/amd64) and drop `AS <stage>`."""
    toks = [t for t in args.split() if not t.startswith("--")]
    if len(toks) >= 3 and toks[1].upper() == "AS":
        toks = toks[:1]
    return toks[0] if toks else ""


def parse_env(args: str) -> dict[str, str]:
    """ENV k=v [k2=v2 ...] or legacy ENV k v."""
    toks = shlex.split(args)
    if toks and "=" not in toks[0]:
        return {toks[0]: " ".join(toks[1:])}
    return dict(t.split("=", 1) for t in toks if "=" in t)


class Step(NamedTuple):
    ins: str
    args: str
    workdir: str | None  # WORKDIR in effect at this line
    env: dict[str, str]  # ENV in effect at this line


class DockerfileInfo:
    def __init__(self, path: Path):
        self.image: str | None = None
        self.workdir: str | None = None
        self.env: dict[str, str] = {}
        self.steps: list[Step] = []
        if not path.exists():
            return
        for ins, args in _dockerfile_instructions(path.read_text(errors="replace")):
            if ins == "FROM":  # last stage wins; reset per stage
                self.image, self.workdir, self.env, self.steps = parse_from(args), None, {}, []
            elif ins == "WORKDIR":
                wd = args.strip()
                self.workdir = wd if wd.startswith("/") else f"{(self.workdir or '/').rstrip('/')}/{wd}"
            elif ins == "ENV":
                self.env.update(parse_env(args))
            elif ins not in _DESCRIPTIVE:
                self.steps.append(Step(ins, args, self.workdir, dict(self.env)))

    @property
    def build_steps(self) -> list[str]:
        return [f"{s.ins} {s.args}"[:120] for s in self.steps]


class UnsupportedStep(RuntimeError):
    pass


def run_command(args: str) -> str:
    """Shell command of a RUN line: exec form ["a", "b"] -> shell-quoted; --mount/--network
    flags are unsupported (they change what the command sees)."""
    a = args.strip()
    if a.startswith("--"):
        raise UnsupportedStep(f"RUN with build flags is not replayable: RUN {a[:80]}")
    if a.startswith("["):
        try:
            return shlex.join(json.loads(a))
        except (ValueError, TypeError):
            pass
    return a


def copy_plan(args: str, context: Path, workdir: str | None) -> list[tuple[Path, str]]:
    """(host source, sandbox destination) pairs for a COPY/ADD line, with Docker's rules: a
    directory's contents go into dst; files go to dst, or into it when dst ends in / or there are
    several sources. --chown/--chmod are dropped (root owns everything here); --from, URLs and
    ADD's archive extraction are unsupported."""
    toks = shlex.split(args) if not args.strip().startswith("[") else json.loads(args)
    flags = [t for t in toks if t.startswith("--")]
    toks = [t for t in toks if not t.startswith("--")]
    if any(f.startswith(("--from", "--link", "--parents", "--exclude")) for f in flags):
        raise UnsupportedStep(f"COPY/ADD flag not replayable: {flags}")
    if len(toks) < 2:
        raise UnsupportedStep(f"COPY/ADD needs a source and a destination: {args[:80]}")
    *srcs, dst = toks
    if not dst.startswith("/"):
        dst = f"{(workdir or '/').rstrip('/')}/{dst}"
    sources: list[Path] = []
    for s in srcs:
        if "://" in s:
            raise UnsupportedStep(f"COPY/ADD from a URL is not replayable: {s}")
        matches = sorted(context.glob(s.lstrip("/"))) if any(c in s for c in "*?[") else [context / s.lstrip("/")]
        for m in matches:
            if not m.exists():
                raise UnsupportedStep(f"COPY/ADD source not in the build context: {s}")
            sources.append(m)
    into_dir = dst.endswith("/") or len(sources) > 1
    plan = []
    for src in sources:
        if src.is_dir():
            plan.append((src, dst))
        else:
            plan.append((src, f"{dst.rstrip('/')}/{src.name}" if into_dir else dst))
    return plan


def parse_upstreams(text: str) -> dict[str, tuple[str, int | None]]:
    """"llm=/abs/llm.sock@39317;judge=/abs/j.sock" -> {name: ("unix:///abs/...", port or None)}."""
    out: dict[str, tuple[str, int | None]] = {}
    for item in (text or "").replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        name, sep, rest = item.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"upstream {item!r} is not name=/abs/path.sock[@port]")
        path, _, port = rest.strip().rpartition("@") if "@" in rest else (rest.strip(), "", "")
        path = path.removeprefix("unix://")
        if not path.startswith("/"):
            raise ValueError(f"upstream {name!r}: socket path must be absolute, got {path!r}")
        out[name.strip()] = (f"unix://{path}", int(port) if port else None)
    return out


def _provider(runtime: str, **opts: Any):
    key = (runtime, tuple(sorted(opts.items())))
    if key not in _PROVIDERS:
        import hpc_sandbox as hs

        cls = {"singularity": hs.SingularityProvider, "enroot": hs.EnrootProvider,
               "gvisor": hs.GvisorProvider}[runtime]
        if runtime != "singularity":
            opts.pop("fakeroot", None)
        if runtime == "gvisor":
            opts.pop("allow_network_none", None)
        _PROVIDERS[key] = cls(**opts)
    return _PROVIDERS[key]


class HpcSandboxEnvironment(BaseEnvironment):
    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: Any,
        task_env_config: Any,
        logger: Any = None,
        hpc_runtime: str = "singularity",
        hpc_scratch_root: str | None = None,
        hpc_image_cache: str | None = None,
        hpc_fakeroot: bool | str = True,
        hpc_exec_timeout_s: float | str = 86400,
        hpc_start_timeout_s: float | str = 1800,
        hpc_output_limit_mb: int | str = 64,
        hpc_allow_network_none: bool | str = False,
        hpc_dockerfile_steps: str = "run",
        hpc_network: str = "pod",
        hpc_pod_users: str = "subids",
        hpc_upstreams: str | None = None,
        hpc_image_map: str | None = None,
        hpc_toolchain: str | None = None,
        **kwargs: Any,
    ):
        # capabilities/_validate_definition read these during super().__init__
        self._runtime = hpc_runtime
        self._network_mode = hpc_network
        self._pod_users = hpc_pod_users
        self._upstreams = parse_upstreams(
            hpc_upstreams if hpc_upstreams is not None else os.environ.get("HPCSB_UPSTREAMS", ""))
        self._allow_network_none = _truthy(hpc_allow_network_none)
        self._dockerfile_steps = hpc_dockerfile_steps
        self._dockerfile = DockerfileInfo(Path(environment_dir) / "Dockerfile")
        self._image_map_path = hpc_image_map if hpc_image_map is not None else os.environ.get("HPCSB_IMAGE_MAP")
        tc = hpc_toolchain if hpc_toolchain is not None else os.environ.get("HPCSB_TOOLCHAIN")
        self._toolchain = Path(tc) if tc and (Path(tc) / "bin" / "python3").exists() else None
        if tc and self._toolchain is None and logger is not None:
            logger.warning(f"hpc_toolchain {tc} has no bin/python3: agents install per trial")
        self._scratch_root = hpc_scratch_root
        self._slot_from_trial = False  # set by the Trial._prepare wrapper while it holds this trial's start slot
        super().__init__(environment_dir=environment_dir, environment_name=environment_name,
                         session_id=session_id, trial_paths=trial_paths,
                         task_env_config=task_env_config, logger=logger, **kwargs)
        self._exec_timeout_s = float(hpc_exec_timeout_s)
        opts: dict[str, Any] = dict(
            default_timeout_s=self._exec_timeout_s,
            start_timeout_s=float(hpc_start_timeout_s),
            output_limit_bytes=int(hpc_output_limit_mb) * 1024 * 1024,
            allow_network_none=self._allow_network_none,
            fakeroot=_truthy(hpc_fakeroot),
        )
        if hpc_scratch_root:
            opts["scratch_root"] = hpc_scratch_root
        if hpc_image_cache:
            opts["image_cache"] = hpc_image_cache
        self._provider_opts = opts
        self._sandbox: Any = None  # the sandbox, or the pod's "main" service
        self._pod: Any = None

    # --- harbor contract -----------------------------------------------------------------------

    @staticmethod
    def type() -> str:
        return "hpc-sandbox"

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        none_ok = (getattr(self, "_network_mode", "pod") == "pod"  # a pod without egress
                   or self._runtime in ("enroot", "gvisor")
                   or getattr(self, "_allow_network_none", False))
        return EnvironmentCapabilities(mounted=True, disable_internet=none_ok)

    @property
    def image(self) -> str | None:
        return self.task_env_config.docker_image or self._dockerfile.image

    def _validate_definition(self) -> None:
        # Runs in __init__, where an exception aborts the whole harbor job: only check what makes
        # the environment meaningless. Build-step problems fail the trial in start().
        if self._dockerfile_steps not in ("run", "error", "warn", "ignore"):
            raise ValueError(f"hpc_dockerfile_steps must be run|error|warn|ignore, "
                             f"not {self._dockerfile_steps!r}")
        if self._network_mode not in ("pod", "host", "none"):
            raise ValueError(f"hpc_network must be pod|host|none, not {self._network_mode!r}")
        if self._pod_users not in ("subids", "single"):
            raise ValueError(f"hpc_pod_users must be subids|single, not {self._pod_users!r}")
        if self._network_mode == "pod" and self._runtime == "gvisor":
            raise ValueError("hpc_network=pod needs the singularity or enroot runtime (gVisor has no pods)")
        if not self.image:
            raise ValueError(f"{self.environment_dir}: no [environment].docker_image and no "
                             "Dockerfile FROM: hpc-sandbox needs a pullable image")

    def _pending_steps(self) -> list[Step]:
        """Build steps to replay; a task with docker_image uses that image as-is (harbor's rule:
        the Dockerfile is then not built)."""
        if self.task_env_config.docker_image or not self._dockerfile.steps:
            return []
        mode = self._dockerfile_steps
        msg = (f"{self.environment_dir}/Dockerfile has build steps on top of {self.image}: "
               f"{self._dockerfile.build_steps[:3]}")
        if mode == "error":
            raise RuntimeError(msg + " (hpc_dockerfile_steps=error)")
        if mode == "warn":
            self.logger.warning(msg + " -- skipped")
        return self._dockerfile.steps if mode == "run" else []

    async def _replay_steps(self, steps: list[Step]) -> None:
        for s in steps:
            cwd = s.workdir or "/"
            if s.ins == "RUN":
                cmd = run_command(s.args)
                await self._root_exec(f"mkdir -p {shlex.quote(cwd)}", timeout_s=60)
                r = await self._root_exec(cmd, cwd=cwd, env=s.env or None)
                if r.exit_code != 0:
                    raise RuntimeError(f"Dockerfile RUN failed (exit {r.exit_code}): {cmd[:200]}\n"
                                       f"{(r.stderr or r.stdout or '')[-1000:]}")
            elif s.ins in ("COPY", "ADD"):
                for src, dst in copy_plan(s.args, self.environment_dir, s.workdir):
                    if s.ins == "ADD" and src.is_file() and src.name.endswith(
                            (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
                        raise UnsupportedStep(f"ADD of an archive (auto-extract) is not replayable: {src.name}")
                    if src.is_dir():
                        await self.upload_dir(src, dst)
                    else:
                        await self.upload_file(src, dst)
            else:
                raise UnsupportedStep(f"Dockerfile {s.ins} is not replayable: {s.args[:80]}")

    @property
    def _workdir(self) -> str:
        return self.task_env_config.workdir or self._dockerfile.workdir or "/app"

    @property
    def agent_toolchain(self) -> tuple[str, frozenset[str]] | None:
        """(where the agent toolchain is mounted in the sandbox, the packages it holds), or None without one."""
        if self._toolchain is None:
            return None
        return TOOLCHAIN_MOUNT, toolchain_packages(self._toolchain)

    def _image_ref(self) -> str:
        ref = self.image or ""
        if mapped := mapped_image(load_image_map(self._image_map_path), ref):
            return mapped
        if ref.endswith(".sif") or ref.endswith(".sqsh") or "://" in ref:
            return ref
        return f"docker://{ref}"

    async def start(self, force_build: bool) -> None:
        import hpc_sandbox as hs

        mounts = [hs.Mount(source=str(m["source"]), target=m["target"],
                           read_only=bool(m.get("read_only", False)))
                  for m in self._mounts if m.get("type", "bind") == "bind" and m.get("source")]
        for m in mounts:
            Path(m.source).mkdir(parents=True, exist_ok=True)
        if self._toolchain is not None:
            source = self._toolchain
            if os.environ.get("HPCSB_TOOLCHAIN_STAGE", "1") not in ("0", "false", "no"):
                try:
                    source = await stage_copy(self._toolchain, self._stage_scratch() / "mho-toolchains")
                except Exception as e:  # noqa: BLE001 - the shared copy still works, only slower
                    self.logger.warning(f"staging toolchain {self._toolchain} failed ({e}); mounting it in place")
            mounts.append(hs.Mount(source=str(source), target=TOOLCHAIN_MOUNT, read_only=True))
        labels = {"harbor": "1", "pbs_job": os.environ.get("PBS_JOBID", "")[:60]}
        no_net = self._network_disabled or self._network_mode == "none"
        spec = hs.SandboxSpec(
            image=await self._staged_image(self._image_ref()),
            workdir=self._workdir,
            env={**self._dockerfile.env, **self._startup_env()},
            mounts=mounts,
            network="none" if no_net else "host",  # ignored for a pod service
            memory_mb=self._effective_memory_mb,
            labels=labels,
        )
        steps = self._pending_steps()  # raises (fails this trial) before anything is started
        provider = _provider(self._runtime, **self._provider_opts)
        if force_build and hasattr(provider, "images"):
            try:
                await provider.images.pull(spec.image, refresh=True)
            except Exception as e:  # a refresh is best-effort; start() pulls if needed
                self.logger.warning(f"refresh pull of {spec.image} failed: {e}")
        try:
            if self._slot_from_trial:  # the Trial._prepare wrapper holds this trial's slot
                await self._start_sandbox(provider, spec, labels, no_net)
            else:
                async with start_gate().slot():
                    await self._start_sandbox(provider, spec, labels, no_net)
        except Exception as e:
            await self.stop(delete=True)
            raise RuntimeError(f"hpc-sandbox start of {spec.image} failed: {e}") from e
        try:
            await self._replay_steps(steps)
            await self._run_setup_files()
            await self._upload_environment_dir_after_start()
        except BaseException:
            await self.stop(delete=True)  # a half-set-up sandbox would otherwise outlive the trial
            raise

    def _stage_scratch(self) -> Path:
        return Path(self._scratch_root or os.environ.get("PBS_LOCALDIR") or f"/tmp/hpcsb-{os.getuid()}")

    async def _staged_image(self, ref: str) -> str:
        """A prepared SIF (an absolute path from the image map) copied to node-local scratch once per node: the
        prepared images live on shared storage, which hundreds of concurrent starts should not all read, and whose
        group-only directories a fakeroot runtime cannot always traverse."""
        if not ref.startswith("/") or not ref.endswith(".sif") or not Path(ref).is_file():
            return ref
        if os.environ.get("HPCSB_IMAGE_STAGE", "1") in ("0", "false", "no"):
            return ref
        try:
            return str(await stage_copy(ref, self._stage_scratch() / "mho-images"))
        except Exception as e:  # noqa: BLE001 - starting from the shared copy still works
            self.logger.warning(f"staging {ref} failed ({e}); starting from it in place")
            return ref

    async def _start_sandbox(self, provider: Any, spec: Any, labels: dict[str, str], no_net: bool) -> None:
        if self._network_mode == "pod":
            await self._start_pod(provider, spec, labels, egress=not no_net)
        else:
            self._sandbox = await provider.start(spec)

    async def _start_pod(self, provider: Any, spec: Any, labels: dict[str, str], *, egress: bool) -> None:
        """One-service pod: the trial gets its own network namespace (any port is free), egress
        through slirp4netns unless the task disables networking, and each upstream reachable as
        http://<name>:<port>."""
        import hpc_sandbox as hs

        missing = [p for p, _ in self._upstreams.values() if not Path(p.removeprefix("unix://")).exists()]
        if missing:
            raise RuntimeError(f"upstream socket(s) not found on the host: {missing}")
        pod_spec = hs.PodSpec(
            network="egress" if egress else "none",
            upstreams={n: p for n, (p, _) in self._upstreams.items()},
            upstream_ports={n: port for n, (_, port) in self._upstreams.items() if port},
            hosts=list(self._upstreams),
            labels=labels,
            users=self._pod_users,
        )
        self._pod = await provider.start_pod(pod_spec)
        run_as = "subids" if self._pod_users == "subids" else "root"
        self._sandbox = await self._pod.start_service(
            hs.ServiceSpec("main", spec, run_as=run_as), wait="started")

    async def _root_exec(self, command: str, *, cwd: str | None = None,
                         env: dict[str, str] | None = None, timeout_s: float | None = None):
        """Setup-time command as the container's root, ignoring harbor's default_user."""
        return await self._sandbox.exec(["/bin/sh", "-c", _BASH_OR_SH, "sh", command],
                                        timeout_s=timeout_s or self._exec_timeout_s, env=env, cwd=cwd)

    async def _run_setup_files(self) -> None:
        """Parity with harbor's Singularity bootstrap: environment/files at /staging/env_files,
        and environment/files/setup.sh sourced once."""
        files = self.environment_dir / "files"
        if not files.is_dir():
            return
        await self._root_exec("mkdir -p /staging/env_files", timeout_s=60)
        await self._sandbox.upload(files, "/staging/env_files")
        if (files / "setup.sh").exists():
            r = await self._root_exec("cd / && bash /staging/env_files/setup.sh")
            if r.exit_code != 0:
                self.logger.warning(f"setup.sh exited {r.exit_code}: {(r.stderr or '')[-500:]}")

    async def stop(self, delete: bool) -> None:
        sb, self._sandbox = self._sandbox, None
        pod, self._pod = self._pod, None
        if pod is not None:
            await pod.stop()  # stops its services too
        elif sb is not None:
            await sb.stop()

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        if self._sandbox is None:
            raise RuntimeError("hpc-sandbox environment not started")
        resolved = self._resolve_user(user)
        if resolved is not None and str(resolved) not in ("root", "0"):
            who = (f"$(getent passwd {resolved} | cut -d: -f1)" if isinstance(resolved, int)
                   else shlex.quote(str(resolved)))
            command = f"su {who} -s /bin/bash -c {shlex.quote(command)}"
        import hpc_sandbox as hs

        try:
            r = await self._sandbox.exec(
                ["/bin/sh", "-c", _BASH_OR_SH, "sh", command],
                timeout_s=float(timeout_sec) if timeout_sec else self._exec_timeout_s,
                env=self._merge_env(env) or None,
                cwd=cwd,
            )
        except hs.SandboxLostError as e:
            raise RuntimeError(f"hpc-sandbox sandbox lost: {e}") from e
        if r.timed_out:
            self.logger.debug(f"command timed out after {timeout_sec}s")
        return ExecResult(stdout=r.stdout, stderr=r.stderr, return_code=r.exit_code)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        parent = str(Path(target_path).parent)
        await self._root_exec(f"mkdir -p {shlex.quote(parent)}", timeout_s=60)
        await self._sandbox.upload(Path(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._root_exec(f"mkdir -p {shlex.quote(target_dir)}", timeout_s=60)
        await self._sandbox.upload(Path(source_dir), target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._sandbox.download(source_path, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Contents of source_dir merged into target_dir (which may already exist)."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".hpcsb-dl-") as tmp:
            staged = Path(tmp) / "d"
            await self._sandbox.download(source_dir, staged)
            for item in staged.iterdir():
                dest = target / item.name
                if dest.is_dir() and not dest.is_symlink():
                    shutil.rmtree(dest)
                elif dest.exists() or dest.is_symlink():
                    dest.unlink()
                shutil.move(str(item), str(dest))


# --------------------------------------------------------------------------------------------- #
# Start cap: take a slot before harbor's timed environment start and release it once the agent is
# set up (Trial._prepare = environment start + healthcheck + agent setup). See the module docstring.
# --------------------------------------------------------------------------------------------- #
def gated_prepare(original: Any) -> Any:
    """Trial._prepare wrapped: an HpcSandboxEnvironment trial takes a start slot first (untimed), keeps it through
    environment start, healthcheck and agent setup, and releases it before the agent runs."""

    @functools.wraps(original)
    async def _prepare(self: Any) -> None:
        env = getattr(self, "agent_environment", None)
        gate = start_gate()
        if not isinstance(env, HpcSandboxEnvironment) or not gate.enabled:
            return await original(self)
        slot, waited = await gate.acquire()
        env._slot_from_trial = True
        if waited >= 1.0:
            env.logger.info(f"hpc-sandbox: waited {waited:.1f}s for a start slot (HPCSB_MAX_STARTING={gate.limit})")
        try:
            return await original(self)
        finally:
            env._slot_from_trial = False
            gate.release(slot)

    _prepare._hpcsb_gated = True  # type: ignore[attr-defined]
    return _prepare


def _install_trial_gate() -> bool:
    try:
        from harbor.trial.trial import Trial
    except Exception:  # noqa: BLE001 - another harbor layout: start() still caps the sandbox start
        return False
    if not getattr(Trial._prepare, "_hpcsb_gated", False):
        Trial._prepare = gated_prepare(Trial._prepare)
    return True


_install_trial_gate()
