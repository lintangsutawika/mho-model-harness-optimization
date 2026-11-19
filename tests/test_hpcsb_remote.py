"""RemoteHpcSandboxEnvironment (trainer side) -> service.py /hpcsb routes (host side) -> a fake
HpcSandboxEnvironment, wired in-process with httpx.ASGITransport."""
import asyncio
import shutil
from pathlib import Path

import pytest

pytest.importorskip("harbor")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from harbor.environments.base import ExecResult  # noqa: E402
from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402


class FakeHostEnv:
    """Stands in for HpcSandboxEnvironment: a dict-backed filesystem."""

    instances: list = []

    def __init__(self, payload):
        self.payload, self.files, self.execs, self.started, self.stopped = payload, {}, [], False, False
        FakeHostEnv.instances.append(self)

    async def start(self, force_build):
        self.started = True

    async def stop(self, delete):
        self.stopped = True

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.execs.append(dict(command=command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user))
        return ExecResult(stdout="ok", stderr="", return_code=7)

    async def upload_file(self, src, dst):
        self.files[dst] = Path(src).read_text()

    async def upload_dir(self, src, dst):
        for f in Path(src).rglob("*"):
            if f.is_file():
                self.files[f"{dst}/{f.relative_to(src)}"] = f.read_text()

    async def download_file(self, src, dst):
        if src not in self.files:
            raise FileNotFoundError(src)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text(self.files[src])

    async def download_dir(self, src, dst):
        for k, v in self.files.items():
            if k.startswith(src.rstrip("/") + "/"):
                out = Path(dst) / k[len(src.rstrip("/")) + 1:]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(v)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("HB_STAGING_ROOT", str(tmp_path / "staging"))
    from mho.backends.local_singularity import service
    from mho.backends import hpcsb_remote

    monkeypatch.setattr(service.tempfile, "tempdir", str(tmp_path / "staging"))
    (tmp_path / "staging").mkdir(exist_ok=True)
    monkeypatch.setattr(service, "_build_hpcsb_env", lambda payload, trial_dir: FakeHostEnv(payload))
    monkeypatch.setattr(hpcsb_remote, "_TRANSPORT", httpx.ASGITransport(app=service.app))
    monkeypatch.setenv("EXECUTOR_URL", "http://executor")
    FakeHostEnv.instances.clear()
    envdir = tmp_path / "task" / "environment"
    envdir.mkdir(parents=True)
    env = hpcsb_remote.RemoteHpcSandboxEnvironment(
        environment_dir=envdir, environment_name="t", session_id="t__x__env",
        trial_paths=TrialPaths(tmp_path / "trial"), task_env_config=EnvironmentConfig(),
        persistent_env={"P": "1"}, singularity_image_cache_dir="/ignored", hpc_runtime="enroot")
    return env, service, tmp_path


def test_round_trip(wired):
    env, service, tmp = wired

    async def go():
        await env.start(force_build=False)
        host = FakeHostEnv.instances[0]
        assert host.started and host.payload["session_id"] == "t__x__env"
        assert host.payload["env_kwargs"] == {"hpc_runtime": "enroot"}  # singularity_* dropped
        staging = env._staging
        assert staging.parent == tmp / "staging" and staging.is_dir()

        r = await env.exec("ls", cwd="/w", env={"X": "2"}, timeout_sec=9, user="agent")
        assert (r.return_code, r.stdout) == (7, "ok")
        assert host.execs[-1] == dict(command="ls", cwd="/w", env={"P": "1", "X": "2"},
                                      timeout_sec=9, user="agent")

        src = tmp / "a.txt"
        src.write_text("hello")
        await env.upload_file(src, "/app/a.txt")
        d = tmp / "d"
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "b.txt").write_text("bee")
        await env.upload_dir(d, "/app/d")
        assert host.files == {"/app/a.txt": "hello", "/app/d/sub/b.txt": "bee"}
        assert list(staging.iterdir()) == []  # staged copies cleaned up

        out = tmp / "out" / "a.txt"
        await env.download_file("/app/a.txt", out)
        assert out.read_text() == "hello"
        await env.download_dir("/app/d", tmp / "outdir")
        assert (tmp / "outdir" / "sub" / "b.txt").read_text() == "bee"
        with pytest.raises(RuntimeError, match="FileNotFoundError"):
            await env.download_file("/missing", tmp / "x")

        await env.stop(delete=True)
        assert host.stopped and not staging.exists() and not service._HPCSB
    asyncio.run(go())


def test_service_refuses_paths_outside_staging(wired):
    env, service, tmp = wired

    async def go():
        await env.start(force_build=False)
        sid = env._sandbox_id
        outside = tmp / "secret.txt"
        outside.write_text("s")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),
                                     base_url="http://executor") as c:
            r = await c.post(f"/hpcsb/{sid}/upload", json={"src": str(outside), "dst": "/x", "dir": False})
            assert r.status_code == 500 and r.json()["type"] == "PermissionError"
            r = await c.post(f"/hpcsb/{sid}/download", json={"src": "/x", "dst": str(tmp / "y"), "dir": False})
            assert r.json()["type"] == "PermissionError"
            r = await c.post("/hpcsb/nope/exec", json={"command": "x"})
            assert r.status_code == 404
        await env.stop(delete=True)
    asyncio.run(go())


def test_backend_swap_selects_remote_env(monkeypatch):
    from harbor.environments import factory
    from harbor.models.environment_type import EnvironmentType
    pytest.importorskip("harbor_agent_function")  # only importable inside the Miles container
    import mho.backends.local_singularity.backend as backend

    saved = factory._ENVIRONMENT_REGISTRY[EnvironmentType.SINGULARITY]
    try:
        backend._install_hpcsb_swap()
        e = factory._ENVIRONMENT_REGISTRY[EnvironmentType.SINGULARITY]
        assert (e.module_path if hasattr(e, "module_path") else e[0]) == "mho.backends.hpcsb_remote"
    finally:
        factory._ENVIRONMENT_REGISTRY[EnvironmentType.SINGULARITY] = saved
