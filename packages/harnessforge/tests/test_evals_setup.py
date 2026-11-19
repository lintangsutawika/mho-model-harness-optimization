"""Per-trial setup on hpc-sandbox: prepared images (image_map) and the cap on concurrent sandbox starts."""
import asyncio
import dataclasses
import json

import pytest

from harnessforge.evals.backend import PlatoonBackend
from harnessforge.evals.start_gate import StartGate


def test_start_gate_caps_concurrency_across_gates(tmp_path):
    gate = StartGate(3, tmp_path / "s", poll_s=(0.01, 0.02))
    active, peak = [0], [0]

    async def one():
        async with gate.slot():
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.03)
            active[0] -= 1

    async def main():
        await asyncio.gather(*(one() for _ in range(10)))

    asyncio.run(main())
    assert peak[0] == 3 and gate.held() == 0
    other = StartGate(3, tmp_path / "s")  # another worker process's gate: same files, the flocks conflict
    held = [asyncio.run(other.acquire())[0] for _ in range(3)]
    assert gate._try_take() is None and gate.held() == 3
    for fd in held:
        other.release(fd)
    assert gate.held() == 0
    assert asyncio.run(StartGate(0, tmp_path / "s").acquire()) == (None, 0.0)


def test_start_gate_default_dir_is_per_job(tmp_path, monkeypatch):
    monkeypatch.delenv("HPCSB_START_SLOT_DIR", raising=False)
    monkeypatch.setenv("PBS_LOCALDIR", str(tmp_path))
    assert StartGate(1).slot_dir == tmp_path / "hpcsb-start-slots"
    monkeypatch.setenv("HPCSB_START_SLOT_DIR", str(tmp_path / "x"))
    assert StartGate(1).slot_dir == tmp_path / "x"
    monkeypatch.setenv("HPCSB_MAX_STARTING", "5")
    assert StartGate.from_env().limit == 5


def test_backend_passes_image_map_and_cap_to_the_pod_provider(tmp_path, monkeypatch):
    from test_evals import platoon_request

    monkeypatch.setenv("HPCSB_IMAGE_MAP", str(tmp_path / "map.json"))
    request = platoon_request(tmp_path)
    request.options["max_starting"] = 8
    PlatoonBackend(launch=False).start(request)
    kw = json.loads((tmp_path / "out" / "r1" / "c0" / "eval.yaml").read_text())["environments"][0]["service_kwargs"]
    assert kw["provider_options"]["image_map"] == str(tmp_path / "map.json")
    assert kw["provider_options"]["max_starting"] == 8


# ------------------------------------------------------------------ the pod provider (needs platoon[hpc])
@dataclasses.dataclass
class Spec:
    image: str
    network: str = "egress"


@pytest.fixture
def provider_cls(monkeypatch, tmp_path):
    pods = pytest.importorskip("harnessforge.evals.pods")

    def fake_init(self, **options):
        self.options, self.capabilities = dict(options), frozenset({"host_loopback", "pods"})

    started, active, peak = [], [0], [0]

    async def fake_start(self, spec):  # platoon's own single sandbox (no-network tasks) and our pods alike
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.03)
        active[0] -= 1
        started.append(spec.image)
        return spec.image

    monkeypatch.setattr(pods.SingularityProvider, "__init__", fake_init)
    monkeypatch.setattr(pods.SingularityProvider, "start", fake_start)
    monkeypatch.setattr(pods.PodSingularityProvider, "_start_pod", fake_start)
    monkeypatch.setenv("HPCSB_START_SLOT_DIR", str(tmp_path / "slots"))
    return pods.PodSingularityProvider, started, peak


def test_pod_provider_maps_images_and_caps_starts(provider_cls, tmp_path):
    cls, started, peak = provider_cls
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"alexgshaw/a:1": "/prep/a.sif"}))
    provider = cls(image_map=str(m), max_starting=2)
    assert provider.options["image_map"] == str(m) and provider.options["max_starting"] == 2

    async def main():
        specs = [Spec("docker://alexgshaw/a:1"), Spec("alexgshaw/a:1", network="none"), Spec("b:2")] * 3
        return await asyncio.gather(*(provider.start(s) for s in specs))

    images = asyncio.run(main())
    assert images.count("/prep/a.sif") == 6 and images.count("b:2") == 3
    assert peak[0] == 2 and provider.start_gate.held() == 0


def test_pod_provider_cap_from_env(provider_cls, monkeypatch):
    cls, _, _ = provider_cls
    monkeypatch.setenv("HPCSB_MAX_STARTING", "0")
    assert not cls().start_gate.enabled and cls().image_map == {}


def test_pod_provider_sysdeps_extracts_inside_the_start_slot(tmp_path, monkeypatch):
    import asyncio
    pytest.importorskip("platoon")
    from platoon.sandbox.base import ExecResult, SandboxSpec
    from harnessforge.evals import pods

    events = []

    class FakeSandbox:
        def __init__(self, spec):
            self.spec = spec

        async def exec(self, cmd, timeout_s=None, **k):
            events.append(("exec", cmd[-1], self.spec.mounts))
            return ExecResult(exit_code=0, stdout="", stderr="")

        async def stop(self):
            events.append(("stop",))

    class Slot:
        async def __aenter__(self):
            events.append(("slot+",))

        async def __aexit__(self, *a):
            events.append(("slot-",))

    bundle = tmp_path / "jammy"
    bundle.mkdir()
    prov = pods.PodSingularityProvider(sysdeps=str(bundle), max_starting=0, image_cache=str(tmp_path))
    prov.start_gate = type("G", (), {"slot": lambda self: Slot()})()

    async def fake_start_pod(spec):
        events.append(("start",))
        return FakeSandbox(spec)
    monkeypatch.setattr(prov, "_start_pod", fake_start_pod)
    sb = asyncio.run(prov.start(SandboxSpec(image="/x.sif")))
    kinds = [e[0] for e in events]
    assert kinds == ["slot+", "start", "exec", "slot-"]                 # extraction inside the slot
    cmd, mounts = events[2][1], events[2][2]
    assert "--skip-old-files" in cmd and pods.SYSDEPS_MOUNT in cmd
    assert [(m.source, m.target, m.read_only) for m in mounts] == [(str(bundle), pods.SYSDEPS_MOUNT, True)]
    assert "sysdeps_s" in sb._hf_times and prov.options["sysdeps"] == str(bundle)

    # a failed extraction fails the start (no agent runs without its tools) and stops the sandbox
    async def bad_exec(self, cmd, timeout_s=None, **k):
        return ExecResult(exit_code=2, stdout="", stderr="tar: bundle.tar: Cannot open")
    monkeypatch.setattr(FakeSandbox, "exec", bad_exec)
    events.clear()
    with pytest.raises(RuntimeError, match="sysdeps"):
        asyncio.run(prov.start(SandboxSpec(image="/x.sif")))
    assert ("stop",) in events
