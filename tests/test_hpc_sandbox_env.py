import asyncio
import contextlib
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("harbor")
hs = pytest.importorskip("hpc_sandbox")

from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from mho.backends import hpc_sandbox_env as he  # noqa: E402


# --- Dockerfile parsing -------------------------------------------------------------------------


def test_from_skips_platform_flag_and_stage():
    assert he.parse_from("--platform=linux/amd64 hamishivi/x:task_1-abc") == "hamishivi/x:task_1-abc"
    assert he.parse_from("python:3.12 AS build") == "python:3.12"
    assert he.parse_from("--platform=$BUILDPLATFORM ubuntu:24.04 as base") == "ubuntu:24.04"


def test_dockerfile_info(tmp_path):
    (tmp_path / "Dockerfile").write_text(
        "# c\nFROM golang:1 AS b\nRUN go build\n"
        "FROM --platform=linux/amd64 img:t\nWORKDIR /home/user\nENV A=1 B=\"two words\"\n"
        "ENV LEGACY some value\n"
    )
    info = he.DockerfileInfo(tmp_path / "Dockerfile")
    assert info.image == "img:t" and info.workdir == "/home/user"
    assert info.env == {"A": "1", "B": "two words", "LEGACY": "some value"}
    assert info.build_steps == []  # the RUN belonged to the earlier stage


def test_build_steps_detected_with_continuations(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM a\nRUN apt-get update && \\\n    apt-get install -y x\nCOPY f /f\n")
    info = he.DockerfileInfo(tmp_path / "Dockerfile")
    assert [s.split()[0] for s in info.build_steps] == ["RUN", "COPY"]


# --- environment against a fake provider ----------------------------------------------------------


class FakeSandbox:
    def __init__(self, spec):
        self.spec, self.execs, self.uploads, self.stopped = spec, [], [], False

    async def exec(self, cmd, *, timeout_s=None, env=None, cwd=None, stdin=None):
        self.execs.append((cmd, timeout_s, env, cwd))
        return hs.ExecResult(exit_code=0, stdout="out", stderr="")

    async def upload(self, src, dst):
        self.uploads.append((Path(src), dst))

    async def download(self, src, dst):
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "f.txt").write_text(src)
        return hs.DownloadReport([], 0)

    async def stop(self):
        self.stopped = True


class FakePod:
    def __init__(self, spec, provider):
        self.spec, self.provider, self.services, self.stopped = spec, provider, [], False

    async def start_service(self, svc, wait="started"):
        self.service_spec = svc
        sb = await self.provider.start(svc.sandbox)
        self.services.append(sb)
        return sb

    async def stop(self):
        self.stopped = True
        for s in self.services:
            s.stopped = True


class FakeProvider:
    def __init__(self):
        self.started, self.pods = [], []

    async def start(self, spec):
        sb = FakeSandbox(spec)
        self.started.append(sb)
        return sb

    async def start_pod(self, spec):
        pod = FakePod(spec, self)
        self.pods.append(pod)
        return pod


def _env(tmp_path, monkeypatch, dockerfile="FROM --platform=linux/amd64 img:t\nWORKDIR /w\nENV K=v\n",
         **kw):
    envdir = tmp_path / "environment"
    envdir.mkdir(parents=True)
    (envdir / "Dockerfile").write_text(dockerfile)
    prov = FakeProvider()
    monkeypatch.setattr(he, "_provider", lambda runtime, **o: prov)
    e = he.HpcSandboxEnvironment(
        environment_dir=envdir, environment_name="t", session_id="t__x__env",
        trial_paths=TrialPaths(tmp_path / "trial"), task_env_config=EnvironmentConfig(),
        mounts=[{"type": "bind", "source": str(tmp_path / "logs"), "target": "/logs/agent"}], **kw)
    return e, prov


def test_start_spec_exec_and_stop(tmp_path, monkeypatch):
    e, prov = _env(tmp_path, monkeypatch)
    asyncio.run(e.start(force_build=False))
    sb = prov.started[0]
    assert sb.spec.image == "docker://img:t" and sb.spec.workdir == "/w"
    assert sb.spec.env["K"] == "v" and sb.spec.network == "host"
    assert sb.spec.mounts[0].target == "/logs/agent" and not sb.spec.mounts[0].read_only
    assert (tmp_path / "logs").is_dir()
    r = asyncio.run(e.exec("echo hi", timeout_sec=5, env={"X": "1"}))
    assert r.return_code == 0 and r.stdout == "out"
    cmd, timeout, env, _ = sb.execs[-1]
    assert cmd[-1] == "echo hi" and timeout == 5.0 and env["X"] == "1"
    asyncio.run(e.exec("whoami", user="agent"))
    assert sb.execs[-1][0][-1].startswith("su agent -s /bin/bash -c ")
    asyncio.run(e.exec("true"))
    assert sb.execs[-1][1] == 86400.0  # harbor passes no timeout for whole agent loops
    asyncio.run(e.stop(delete=True))
    assert sb.stopped


def test_download_dir_merges_into_existing(tmp_path, monkeypatch):
    e, prov = _env(tmp_path, monkeypatch)
    asyncio.run(e.start(force_build=False))
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("k")
    asyncio.run(e.download_dir("/logs/verifier", out))
    assert (out / "keep.txt").read_text() == "k" and (out / "f.txt").read_text() == "/logs/verifier"


def test_build_steps_replayed_in_order(tmp_path, monkeypatch):
    df = ("FROM a\nWORKDIR /w\nENV P=1\nRUN mkdir -p /x && \\\n    touch /x/y\n"
          "COPY files/ /opt/f/\nCOPY run.sh tool.py /bin/\nRUN [\"echo\", \"a b\"]\nWORKDIR sub\nRUN pwd\n")
    e, prov = _env(tmp_path, monkeypatch, dockerfile=df)
    envdir = tmp_path / "environment"
    (envdir / "files").mkdir()
    (envdir / "run.sh").write_text("x")
    (envdir / "tool.py").write_text("y")
    asyncio.run(e.start(force_build=False))
    sb = prov.started[0]
    runs = [(c[-1], cwd, env) for c, _, env, cwd in sb.execs if not re.fullmatch(r"mkdir -p \S+", c[-1])]
    assert runs[0] == ("mkdir -p /x && touch /x/y", "/w", {"P": "1"})
    assert runs[1][0] == "echo 'a b'"
    assert runs[2][:2] == ("pwd", "/w/sub")  # relative WORKDIR resolved
    ups = [(p.name, d) for p, d in sb.uploads]
    assert ("files", "/opt/f/") in ups
    assert ("run.sh", "/bin/run.sh") in ups and ("tool.py", "/bin/tool.py") in ups
    assert sb.spec.workdir == "/w/sub"


def test_failed_or_unsupported_step_fails_trial_and_stops_sandbox(tmp_path, monkeypatch):
    e, prov = _env(tmp_path, monkeypatch, dockerfile="FROM a\nCOPY --from=b /x /y\n")
    with pytest.raises(he.UnsupportedStep):
        asyncio.run(e.start(force_build=False))
    assert prov.started[0].stopped
    e2, prov2 = _env(tmp_path / "2", monkeypatch, dockerfile="FROM a\nRUN false\n")

    async def failing(cmd, **kw):
        return hs.ExecResult(exit_code=3, stdout="", stderr="boom") if cmd[-1] == "false" else hs.ExecResult(0, "", "")
    orig = FakeProvider.start

    async def start(self, spec):
        sb = await orig(self, spec)
        sb.exec = failing
        return sb
    monkeypatch.setattr(FakeProvider, "start", start)
    with pytest.raises(RuntimeError, match="RUN failed .exit 3."):
        asyncio.run(e2.start(force_build=False))
    assert prov2.started[0].stopped


def test_modes_error_ignore_and_docker_image_wins(tmp_path, monkeypatch):
    e, _ = _env(tmp_path, monkeypatch, dockerfile="FROM a\nRUN make\n", hpc_dockerfile_steps="error")
    with pytest.raises(RuntimeError, match="build steps"):  # at start, not construction
        asyncio.run(e.start(force_build=False))
    e2, prov2 = _env(tmp_path / "2", monkeypatch, dockerfile="FROM a\nRUN make\n", hpc_dockerfile_steps="ignore")
    asyncio.run(e2.start(force_build=False))
    assert not any(c[-1] == "make" for c, *_ in prov2.started[0].execs)
    e3, prov3 = _env(tmp_path / "3", monkeypatch, dockerfile="FROM a\nRUN make\n")
    e3.task_env_config.docker_image = "prebuilt:1"
    asyncio.run(e3.start(force_build=False))
    assert prov3.started[0].spec.image == "docker://prebuilt:1"
    assert not any(c[-1] == "make" for c, *_ in prov3.started[0].execs)


def test_no_network_needs_capability(tmp_path, monkeypatch):
    e, _ = _env(tmp_path, monkeypatch, hpc_network="host")
    assert not e.capabilities.disable_internet
    e2, _ = _env(tmp_path / "2", monkeypatch, hpc_runtime="enroot", hpc_network="host")
    assert e2.capabilities.disable_internet


# --- pods: per-trial network namespace + upstreams -------------------------------------------------


def test_parse_upstreams():
    assert he.parse_upstreams("llm=/l/llm.sock@39317; judge=unix:///j.sock") == {
        "llm": ("unix:///l/llm.sock", 39317), "judge": ("unix:///j.sock", None)}
    assert he.parse_upstreams("") == {}
    with pytest.raises(ValueError):
        he.parse_upstreams("llm=relative.sock@1")


def test_pod_mode_default_with_upstream_and_egress(tmp_path, monkeypatch):
    sock = tmp_path / "llm.sock"
    sock.write_text("")  # existence is all start() checks
    monkeypatch.setenv("HPCSB_UPSTREAMS", f"llm={sock}@39317")
    e, prov = _env(tmp_path, monkeypatch)
    asyncio.run(e.start(force_build=False))
    (pod,) = prov.pods
    assert pod.spec.network == "egress" and pod.spec.users == "subids"
    assert pod.spec.upstreams == {"llm": f"unix://{sock}"} and pod.spec.upstream_ports == {"llm": 39317}
    assert pod.spec.hosts == ["llm"]
    assert pod.service_spec.name == "main" and pod.service_spec.run_as == "subids"
    assert pod.service_spec.sandbox.image == "docker://img:t"
    asyncio.run(e.exec("true"))
    assert prov.started[0].execs  # exec goes to the pod's main service
    asyncio.run(e.stop(delete=True))
    assert pod.stopped and prov.started[0].stopped
    assert e.capabilities.disable_internet  # a pod can run without egress


def test_pod_no_network_policy_and_missing_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("HPCSB_UPSTREAMS", f"llm={tmp_path}/absent.sock@1")
    e, prov = _env(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="upstream socket"):
        asyncio.run(e.start(force_build=False))
    monkeypatch.setenv("HPCSB_UPSTREAMS", "")
    e2, prov2 = _env(tmp_path / "2", monkeypatch)
    monkeypatch.setattr(type(e2), "_network_disabled", property(lambda self: True))
    asyncio.run(e2.start(force_build=False))
    assert prov2.pods[0].spec.network == "none"


def test_host_mode_and_bad_values(tmp_path, monkeypatch):
    e, prov = _env(tmp_path, monkeypatch, hpc_network="host")
    asyncio.run(e.start(force_build=False))
    assert not prov.pods and prov.started[0].spec.network == "host"
    with pytest.raises(ValueError, match="hpc_network"):
        _env(tmp_path / "2", monkeypatch, hpc_network="bridge")
    with pytest.raises(ValueError, match="gVisor has no pods"):
        _env(tmp_path / "3", monkeypatch, hpc_runtime="gvisor")


def test_uds_forward_round_trip():
    import tempfile
    from mho.backends import uds_forward

    sockdir = Path(tempfile.mkdtemp(prefix="uf", dir="/tmp"))

    async def main():
        async def echo(r, w):
            data = await r.read(100)
            w.write(b"pong:" + data)
            await w.drain()
            w.close()
        tcp = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = tcp.sockets[0].getsockname()[1]
        path = str(sockdir / "f.sock")
        ready = asyncio.Event()
        fwd = asyncio.create_task(uds_forward.serve(path, "127.0.0.1", port, ready))
        await ready.wait()
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"
        r, w = await asyncio.open_unix_connection(path)
        w.write(b"ping")
        await w.drain()
        w.write_eof()
        got = await r.read()
        w.close()
        fwd.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fwd
        tcp.close()
        return got
    try:
        assert asyncio.run(asyncio.wait_for(main(), 10)) == b"pong:ping"
        assert not (sockdir / "f.sock").exists()
    finally:
        sockdir.rmdir()
