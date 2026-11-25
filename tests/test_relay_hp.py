import importlib
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "harness", "relay"))


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("RELAY_WS_URL", "wss://x/bridge")
    monkeypatch.setenv("RELAY_SECRET", "s")
    monkeypatch.setenv("VLLM_LOCAL_URL", "http://127.0.0.1:30000")
    import bridge as b
    return importlib.reload(b)


def test_plain_path_goes_to_local(bridge):
    assert bridge.target_url("/sessions/a/v1/chat/completions") == "http://127.0.0.1:30000/sessions/a/v1/chat/completions"


def test_hp_routes_to_instance(bridge):
    ip = next(h for h in bridge.ALLOWED_HOSTS if h[0].isdigit())
    assert bridge.target_url(f"/_hp/{ip}:30017/sessions/a/v1/chat/completions") == \
        f"http://{ip}:30017/sessions/a/v1/chat/completions"


@pytest.mark.parametrize("path", ["/_hp/8.8.8.8:30001/x", "/_hp/127.0.0.1:22/x", "/_hp/127.0.0.1:8265/api"])
def test_hp_rejects_foreign_targets(bridge, path):
    with pytest.raises(ValueError):
        bridge.target_url(path)

