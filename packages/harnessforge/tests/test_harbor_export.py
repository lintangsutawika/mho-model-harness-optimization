"""harbor_export: platoon rollout dirs -> harbor trial dirs that the old harbor-era readers (utils/gate.py,
utils/seq.py, the proposer's staging, analysis scripts) read with the same answers as the TrialRecords."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from harnessforge.evals.__main__ import main as cli
from harnessforge.evals.harbor_export import (ENV_EXCEPTION, SETUP_EXCEPTION, CandidateExport, RunExport, phases,
                                              rebuild_trajectory, trial_name)
from harnessforge.evals.reader import read_rollouts, safe_task_id
from harnessforge.records import per_task

FIXTURES = Path(__file__).parent / "fixtures"
REPO_SRC = Path(__file__).resolve().parents[3] / "src"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    path = REPO_SRC / "utils" / "gate.py"
    if not path.is_file():
        pytest.skip("the repo's src/utils/gate.py is not next to this package")
    return _load("hf_test_gate", path)


@pytest.fixture(scope="module")
def seq(gate):
    sys.modules.setdefault("utils", type(sys)("utils"))
    sys.modules["utils.gate"] = gate
    return _load("hf_test_seq", REPO_SRC / "utils" / "seq.py")


def write_trial(root: Path, task: str, k: int, *, status: str | None = "scored", reward: float = 1.0,
                failure_kind: str | None = None, reason: str | None = None, error: str | None = None,
                harness: dict | None = None, outcome_meta: dict | None = None, wall: float = 100.0,
                created: str = "2026-11-18T10:00:00+00:00", steps: int = 2, collection_id: str | None = None) -> Path:
    d = root / "rollouts" / safe_task_id(task) / f"rollout_{k}"
    d.mkdir(parents=True, exist_ok=True)
    if status is not None:
        outcome = {"status": status, "rewards": {"reward": reward} if status == "scored" else {}, "reason": reason}
        if failure_kind:
            outcome["failure_kind"] = failure_kind
        if outcome_meta:
            outcome["metadata"] = outcome_meta
        step_list = [{"misc": {"action_misc": {"completion_id": f"c{i}", "usage": {"completion_tokens": 10}},
                               "message": {"role": "assistant", "content": f"thinking {i}",
                                           "tool_calls": [{"id": f"t{i}", "type": "function", "function": {
                                               "name": "bash", "arguments": json.dumps({"command": f"ls {i}"})}}]}}}
                     for i in range(steps)]
        traj = {"task": {"goal": f"solve {task}"}, "reward": reward if status == "scored" else 0.0,
                "steps": step_list,
                "misc": {"outcome": outcome, "harness": harness or {"exit_code": 0, "timed_out": False},
                         "model_call_spans": {f"c{i}": [1000.0 + 10 * i, 1005.0 + 10 * i] for i in range(steps)}}}
        (d / "trajectory_collection.json").write_text(json.dumps(
            {"id": collection_id or f"{safe_task_id(task)}-{k:012x}", "trajectories": {"root": traj}}))
    meta = {"task_id": task, "rollout_index": k, "wall_time_seconds": wall, "error": error, "outcome_status": status,
            "created_at": created, "status": "completed" if error is None else "errored",
            "usage": {"policy": {"calls": steps, "prompt_tokens": 1000, "completion_tokens": 120,
                                 "cached_prompt_tokens": 7},
                      "aux": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}}}
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


#: (write_trial kwargs, expected env_failed, expected exception_type)
CASES = {
    "ok": (dict(), False, None),
    "zero": (dict(reward=0.0), False, None),
    "timeout": (dict(reward=0.0, reason="agent_timeout", harness={"exit_code": None, "timed_out": True}),
                False, "AgentTimeoutError"),
    "crash": (dict(reward=1.0, reason="harness_crash", harness={"exit_code": 1, "timed_out": False}),
              False, "NonZeroAgentExitCodeError"),
    "ctx": (dict(reward=0.0, reason="context_overflow"), False, "NonZeroAgentExitCodeError"),
    "verifier_timeout": (dict(status="invalid", failure_kind="verifier_timeout"), False, "VerifierTimeoutError"),
    "no_reward": (dict(status="error", failure_kind="verifier_no_reward"), False, "RewardFileNotFoundError"),
    "sandbox": (dict(status="error", failure_kind="sandbox_start"), True, ENV_EXCEPTION),
    "setup": (dict(status="error", failure_kind="harness_setup"), True, SETUP_EXCEPTION),
    "gateway": (dict(status="error", failure_kind="harness_crash_gateway"), True, ENV_EXCEPTION),
    "backend": (dict(reward=0.0, error="backend_unavailable: 3 calls failed"), True, ENV_EXCEPTION),
    "cancelled": (dict(status="cancelled"), True, ENV_EXCEPTION),
    "raised": (dict(status=None, error="RuntimeError: pod never started\ntraceback..."), True, ENV_EXCEPTION),
}


def build(tmp_path: Path) -> tuple[Path, list[str]]:
    out = tmp_path / "platoon"
    tasks = []
    for i, (name, (kw, _, _)) in enumerate(CASES.items()):
        task = f"task_{name}"
        tasks.append(task)
        write_trial(out, task, 0, created=f"2026-11-18T10:{i:02d}:00+00:00", **kw)
        write_trial(out, task, 1, created=f"2026-11-18T10:{i:02d}:30+00:00")
    return out, tasks


def test_every_outcome_maps_consistently(tmp_path, gate):
    out, tasks = build(tmp_path)
    dest = tmp_path / "job"
    stats = CandidateExport(out, dest, "cand", model="M", tasks=tasks, trials=2).sync()
    assert stats == {"trials": 2 * len(CASES), "finished": 2 * len(CASES), "new": 2 * len(CASES)}
    records = {(r.task, r.index): r for r in read_rollouts(out, "cand", tasks=tasks)}
    for name, (_, env_failed, exc) in CASES.items():
        result = json.loads((dest / trial_name(f"task_{name}", 0) / "result.json").read_text())
        record = records[(f"task_{name}", 0)]
        assert record.env_failed is env_failed, name
        assert gate.env_failed(result) is env_failed, name
        assert (result["exception_info"] or {}).get("exception_type") == exc, name
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        assert float(reward or 0.0) == float(record.reward or 0.0), name
        assert result["task_name"] == f"task_{name}"
        assert result["harnessforge"]["error"] == record.error
        if not env_failed:
            assert result["agent_setup"] and result["agent_execution"], name
            assert result["agent_result"]["n_output_tokens"] == 120
            assert result["agent_result"]["n_cache_tokens"] == 7
    # the gate's per-task view of the export == the records' per-task view
    pt_gate = gate.per_task(dest)
    pt_rec = per_task(records.values(), "cand")
    assert set(pt_gate) == set(pt_rec)
    for task, row in pt_gate.items():
        assert row["reward"] == pytest.approx(pt_rec[task].reward) and row["n"] == pt_rec[task].n
        assert row["cost"] == pytest.approx(pt_rec[task].cost)
    assert gate.env_failures(dest) == (sum(r.env_failed for r in records.values()), len(records))
    job = json.loads((dest / "result.json").read_text())
    assert job["n_total_trials"] == 2 * len(CASES) and job["stats"]["n_completed_trials"] == 2 * len(CASES)


def test_real_platoon_fixtures_and_seq_order(tmp_path, gate, seq):
    root = FIXTURES / "platoon_rollouts"
    tasks = json.loads((root / "tasks.json").read_text())
    expected = json.loads((root / "expected.json").read_text())
    dest = tmp_path / "eval"
    CandidateExport(root, dest, "base", model="Qwen3.5-4B", tasks=tasks, trials=2).sync()
    for key, exp in expected.items():
        task, k = key.split("/")
        d = dest / trial_name(task, int(k))
        assert (d / "config.json").is_file()
        if not exp["finished"]:
            assert not (d / "result.json").exists()
            continue
        result = json.loads((d / "result.json").read_text())
        assert gate.env_failed(result) is exp["env_failed"]
        assert result["agent_result"]["n_output_tokens"] == exp["output_tokens"]
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        assert float(reward or 0) == float(exp["reward"] or 0)
        traj = json.loads((d / "agent" / "mini-swe-agent.trajectory.json").read_text())
        assert traj["trajectory_format"] == "mini-swe-agent-1.1" and traj["info"]["reconstructed"]
        assert (d / "agent" / "mini-swe-agent.txt").read_text()
    # seq.launched sees every launched trial (in-flight included) in launch order
    launched = seq.launched(dest)
    assert {t: len(v) for t, v in launched.items()} == {
        t: sum(1 for key in expected if key.startswith(t + "/")) for t in tasks if any(
            key.startswith(t + "/") for key in expected)}
    assert launched["task_009561_689cbf6c"][0]["result"] is None
    two = launched["task_000035_1d654148"]
    assert [x["name"] for x in two] == [trial_name("task_000035_1d654148", 0), trial_name("task_000035_1d654148", 1)]


def test_collected_files_and_patched_outcome(tmp_path):
    out = tmp_path / "platoon"
    collect = tmp_path / "collected"
    write_trial(out, "t1", 0, reward=0.0, collection_id="t1-aaaaaaaaaaaa",
                outcome_meta={"test_output": "TAIL: 1 failed", "verifier_exit_code": 1, "verifier_seconds": 3.0})
    write_trial(out, "t2", 0, reward=0.0, collection_id="t2-bbbbbbbbbbbb",
                outcome_meta={"test_output": "TAIL only", "verifier_seconds": 2.0})
    # t1: the collect hook's copies
    c = collect / "t1-aaaaaaaaaaaa"
    (c / "agent").mkdir(parents=True)
    (c / "test-stdout.txt").write_text("FULL OUTPUT\n...\n1 failed")
    agent = {"info": {"exit_status": "Submitted"}, "trajectory_format": "mini-swe-agent-1.1",
             "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
                          {"role": "assistant", "content": "a", "extra": {"timestamp": 1795000010.0}},
                          {"role": "tool", "content": "obs"},
                          {"role": "assistant", "content": "b", "extra": {"timestamp": 1795000020.0}},
                          {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}}]}
    (c / "agent" / "abc.trajectory.json").write_text(json.dumps(agent))
    (c / "sandbox.json").write_text(json.dumps({"start_requested": 0, "started": None}))
    dest = tmp_path / "job"
    CandidateExport(out, dest, "c", tasks=["t1", "t2"], collect_dir=collect).sync()
    d1, d2 = dest / trial_name("t1", 0), dest / trial_name("t2", 0)
    assert (d1 / "verifier" / "test-stdout.txt").read_text().startswith("FULL OUTPUT")
    assert json.loads((d1 / "agent" / "mini-swe-agent.trajectory.json").read_text()) == agent
    assert "obs" in (d1 / "agent" / "mini-swe-agent.txt").read_text()
    r1 = json.loads((d1 / "result.json").read_text())
    assert r1["harnessforge"]["agent_trajectory"] == "collected" and r1["harnessforge"]["test_output"] == "collected"
    assert r1["harnessforge"]["verifier_exit_code"] == 1
    assert (d2 / "verifier" / "test-stdout.txt").read_text() == "TAIL only"
    r2 = json.loads((d2 / "result.json").read_text())
    assert r2["harnessforge"]["test_output"] == "platoon_outcome_tail"
    assert r2["harnessforge"]["agent_trajectory"] == "rebuilt"
    assert (d2 / "verifier" / "reward.txt").read_text().strip() == "0"
    records = read_rollouts(out, "c", tasks=["t1"])
    assert records[0].extra["verifier_exit_code"] == 1


def test_launch_order_survives_bad_times(tmp_path, seq):
    out = tmp_path / "platoon"
    for k in range(12):  # rollout_10 sorts before rollout_2 as text; later indices "started" earlier
        write_trial(out, "t", k, created=f"2026-11-18T10:{59 - k:02d}:00+00:00", wall=10.0)
    CandidateExport(out, tmp_path / "job", "c", tasks=["t"]).sync()
    names = [x["name"] for x in seq.launched(tmp_path / "job")["t"]]
    assert names == [trial_name("t", k) for k in range(12)]


def test_incremental_sync_and_stop(tmp_path, gate):
    out = tmp_path / "platoon"
    (out / "rollouts" / "t1" / "rollout_0").mkdir(parents=True)   # launched, in flight
    exp = CandidateExport(out, tmp_path / "job", "c", tasks=["t1"], trials=2)
    assert exp.sync()["new"] == 0
    d = tmp_path / "job" / trial_name("t1", 0)
    assert (d / "config.json").is_file() and not (d / "result.json").exists()
    write_trial(out, "t1", 0, reward=1.0)
    (out / "rollouts" / "t1" / "rollout_1").mkdir()
    assert exp.sync()["new"] == 1
    assert json.loads((d / "result.json").read_text())["verifier_result"]["rewards"]["reward"] == 1.0
    assert exp.sync()["new"] == 0                                    # nothing new: nothing rewritten
    s = exp.sync(stopped=True, final=True)                            # rollout_1 never finishes
    assert s["new"] == 1
    r = json.loads((tmp_path / "job" / trial_name("t1", 1) / "result.json").read_text())
    assert gate.env_failed(r) and r["harnessforge"]["error"] == "stopped"
    assert r["exception_info"]["exception_type"] == ENV_EXCEPTION
    # a fresh exporter (restart) does not rewrite what an earlier pass exported
    before = (d / "result.json").stat().st_mtime_ns
    assert CandidateExport(out, tmp_path / "job", "c", tasks=["t1"]).sync()["new"] == 0
    assert (d / "result.json").stat().st_mtime_ns == before


def test_run_export_cli(tmp_path, gate):
    run = tmp_path / "runs" / "r1"
    for cand in ("base", "cand/x"):
        slug = cand.replace("/", "_")
        write_trial(run / slug / "platoon", "tA", 0, reward=1.0 if cand == "base" else 0.0)
    (run / "dataset" / "tA").mkdir(parents=True)
    (run / "dataset" / "tA" / "task.toml").write_text('[task]\nname = "tmax/tA-v1"\n')
    (run / "request.json").write_text(json.dumps({
        "run_id": "r1", "tasks": ["tA"], "dataset": "/elsewhere", "trials": 1, "model": "Qwen", "policy": "p",
        "candidates": [{"id": "base", "harness_dir": "/x"}, {"id": "cand/x", "harness_dir": "/y"}]}))
    (run / "status.json").write_text(json.dumps({"done": True}))
    assert cli(["export", str(run), str(tmp_path / "h")]) == 0
    for slug, reward in (("base", 1.0), ("cand_x", 0.0)):
        result = json.loads((tmp_path / "h" / slug / trial_name("tA", 0) / "result.json").read_text())
        assert result["task_name"] == "tmax/tA-v1"                    # as harbor names it (task.toml [task].name)
        assert result["config"]["task"]["path"].endswith("dataset/tA")
        assert gate.per_task(tmp_path / "h" / slug)["tmax/tA-v1"]["reward"] == reward
    assert set(RunExport(run, tmp_path / "h2", candidates=["base"]).exports) == {"base"}


def test_phases_estimates():
    p = phases(started=100.0, finished=200.0, env_failed=False, call_spans=[[10, 20], [30, 40]],
               sandbox={"started": 110.0}, harness_log_mtime=None, verifier_seconds=10.0)
    # calls anchored so the last ends at finished - verifier_seconds = 190 -> agent 160..190
    assert p["environment_setup"]["finished_at"].endswith(":01:50.000000Z")
    assert p["agent_execution"]["started_at"].endswith(":02:40.000000Z")
    assert p["agent_execution"]["finished_at"].endswith(":03:10.000000Z")
    assert p["verifier"]["finished_at"].endswith(":03:20.000000Z")
    assert phases(started=1.0, finished=2.0, env_failed=True)["agent_setup"] is None


def test_rebuild_trajectory_keeps_commands():
    traj = rebuild_trajectory({"task": {"goal": "g"}, "steps": [{"misc": {"action_misc": {"completion_id": "c"},
                              "message": {"role": "assistant", "content": "x", "tool_calls": [
                                  {"id": "1", "function": {"name": "bash", "arguments": '{"command": "pwd"}'}}]}}}]},
                              exit_status="Submitted")
    assert [m["role"] for m in traj["messages"]] == ["user", "assistant", "exit"]
    assert traj["messages"][1]["extra"]["actions"] == [{"command": "pwd", "tool_call_id": "1"}]


def test_backend_sets_collect_dir(tmp_path):
    from harnessforge.contracts import Candidate, EvalRequest
    from harnessforge.evals import PlatoonBackend
    cand = tmp_path / "c"
    (cand / "src" / "micro_swe").mkdir(parents=True)
    (cand / "src" / "micro_swe" / "__main__.py").write_text("")
    data = tmp_path / "data" / "t"
    (data / "environment").mkdir(parents=True)
    (data / "task.toml").write_text('[environment]\ndocker_image = "x/y:1"\n')
    req = EvalRequest(run_id="r", candidates=[Candidate("A", cand)], tasks=["t"], dataset=str(tmp_path / "data"),
                      trials=1, endpoint="http://e/v1", out_dir=tmp_path / "out", options={"model": "m"})
    PlatoonBackend(launch=False).start(req)
    cfg = json.loads((tmp_path / "out" / "r" / "A" / "eval.yaml").read_text())
    assert cfg["environments"][0]["service_kwargs"]["provider_options"]["collect_dir"] == str(
        tmp_path / "out" / "r" / "A" / "collected")
    req2 = EvalRequest(run_id="r2", candidates=[Candidate("A", cand)], tasks=["t"], dataset=str(tmp_path / "data"),
                       trials=1, endpoint="http://e/v1", out_dir=tmp_path / "out",
                       options={"model": "m", "collect": False})
    PlatoonBackend(launch=False).start(req2)
    cfg = json.loads((tmp_path / "out" / "r2" / "A" / "eval.yaml").read_text())
    assert "collect_dir" not in cfg["environments"][0]["service_kwargs"]["provider_options"]


def test_pod_collect_hook(tmp_path):
    pytest.importorskip("platoon.sandbox.hpc")
    import asyncio
    from types import SimpleNamespace
    from harnessforge.evals import pods

    files = {pods.TEST_STDOUT_PATH: "tests said no",
             f"{pods.AGENT_STATE_ROOT}/agent1/{pods.AGENT_TRAJECTORY}": '{"messages": [1]}'}

    class Box:
        id = "sb1"
        _provider = SimpleNamespace(collect_dir=str(tmp_path / "col"))
        _hf_labels = {pods.ROLLOUT_LABEL: "t1-abc", "platoon.task": "t1"}
        _hf_times = {"start_requested": 1.0, "started": 2.0}

        async def exec(self, argv, cwd=None, timeout_s=None):
            if argv[0] == "cat":
                return SimpleNamespace(exit_code=0, stdout=files[argv[1]], timed_out=False)
            if "tail" in argv[2]:
                return SimpleNamespace(exit_code=0, stdout=files[argv[4]], timed_out=False)
            listing = "\n".join(p for p in files if p.startswith(argv[4]) and p.endswith(argv[5]))
            return SimpleNamespace(exit_code=0, stdout=listing + "\n", timed_out=False)

    dest = asyncio.run(pods.collect(Box()))
    assert dest == tmp_path / "col" / "t1-abc"
    assert (dest / "test-stdout.txt").read_text() == "tests said no"
    assert (dest / "agent" / "agent1.trajectory.json").read_text() == '{"messages": [1]}'
    assert json.loads((dest / "sandbox.json").read_text())["started"] == 2.0

    class Broken(Box):
        async def exec(self, *a, **k):
            raise RuntimeError("sandbox gone")

    assert asyncio.run(pods.collect(Broken())) == tmp_path / "col" / "t1-abc"   # logged, never raised
    Box._provider = SimpleNamespace(collect_dir=None)
    assert asyncio.run(pods.collect(Box())) is None
