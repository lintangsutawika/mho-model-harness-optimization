import asyncio

import pytest

pytest.importorskip("sail")
from sail.errors import ApiError  # noqa: E402
from sail.harbor import SailboxEnvironment  # noqa: E402

from mho.backends import sail_retry  # noqa: E402


def _run(monkeypatch, errors):
    calls = []

    async def fake_start(self, force_build):
        calls.append(force_build)
        if errors:
            raise errors.pop(0)

    monkeypatch.setattr(SailboxEnvironment, "start", fake_start)
    monkeypatch.setenv("MHO_SAIL_START_BACKOFF_S", "0")
    env = object.__new__(sail_retry.RetryingSailboxEnvironment)
    return calls, asyncio.run(env.start(False))


def test_retries_upload_503_then_succeeds(monkeypatch):
    calls, _ = _run(monkeypatch, [ApiError("local file upload failed: HTTP 503 Service Unavailable"),
                                  ApiError("x", status_code=502)])
    assert calls == [False, False, False]


def test_non_transient_raises_at_once(monkeypatch):
    with pytest.raises(ApiError):
        _run(monkeypatch, [ApiError("bad request", status_code=400)])


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setenv("MHO_SAIL_START_RETRIES", "2")
    with pytest.raises(ApiError):
        _run(monkeypatch, [ApiError("HTTP 503")] * 3)


# --- Docker Hub login --------------------------------------------------------------------------


class _FakeDual:
    def __init__(self):
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append(("sync", a, k))
        return "def"

    async def aio(self, *a, **k):
        self.calls.append(("aio", a, k))
        return "def"


class _FakeImage:
    def __init__(self):
        self.from_dockerfile, self.from_compose, self.from_registry = _FakeDual(), _FakeDual(), _FakeDual()
        self.debian = "passthrough"


def _creds(monkeypatch, on=True):
    for k in ("MHO_DOCKER_USERNAME", "MHO_DOCKER_PASSWORD", "SINGULARITY_DOCKER_USERNAME",
              "SINGULARITY_DOCKER_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    if on:
        monkeypatch.setenv("SINGULARITY_DOCKER_USERNAME", "u")
        monkeypatch.setenv("SINGULARITY_DOCKER_PASSWORD", "p")


def test_harbor_image_is_patched():
    import sail
    import sail.harbor
    if not hasattr(sail, "RegistryLogin"):
        pytest.skip("sail < 0.18")
    assert isinstance(sail.harbor.Image, sail_retry._ImageWithDockerLogin)


def test_compose_and_dockerfile_get_docker_login(monkeypatch):
    import sail
    if not hasattr(sail, "RegistryLogin"):
        pytest.skip("sail < 0.18")
    _creds(monkeypatch)
    fake = _FakeImage()
    img = sail_retry._ImageWithDockerLogin(fake)
    asyncio.run(img.from_compose.aio("/x", files=[]))
    img.from_dockerfile("/x/Dockerfile", context_dir="/x")
    for dual in (fake.from_compose, fake.from_dockerfile):
        (login,) = dual.calls[0][2]["registry_logins"]
        assert login.prefix == "docker.io" and login.username == "u"
    assert img.debian == "passthrough"


def test_explicit_logins_kept_and_no_creds_is_anonymous(monkeypatch):
    import sail
    if not hasattr(sail, "RegistryLogin"):
        pytest.skip("sail < 0.18")
    fake = _FakeImage()
    img = sail_retry._ImageWithDockerLogin(fake)
    _creds(monkeypatch)
    img.from_compose("/x", registry_logins=["mine"])
    assert fake.from_compose.calls[-1][2]["registry_logins"] == ["mine"]
    _creds(monkeypatch, on=False)
    img.from_compose("/x")
    assert "registry_logins" not in fake.from_compose.calls[-1][2]


def test_from_registry_login_only_for_docker_hub(monkeypatch):
    import sail
    if not hasattr(sail, "RegistryLogin"):
        pytest.skip("sail < 0.18")
    _creds(monkeypatch)
    fake = _FakeImage()
    img = sail_retry._ImageWithDockerLogin(fake)
    img.from_registry("hamishivi/swerl-tmax-hardened:t")
    img.from_registry("ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest")
    assert fake.from_registry.calls[0][2]["username"] == "u"
    assert "username" not in fake.from_registry.calls[1][2]
    assert sail_retry._is_docker_hub("python:3.13") and sail_retry._is_docker_hub("docker.io/a/b")
    assert not sail_retry._is_docker_hub("localhost:5000/a")
