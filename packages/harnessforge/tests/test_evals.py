import json
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harnessforge.contracts import Candidate, EvalBackend, EvalRequest, EvalRun, JobSpec, Order
from harnessforge.evals import FakeBackend, PlatoonBackend, eval_node_job, read_rollouts
from harnessforge.evals.dataset import dockerfile_info, materialize, replay_command
from harnessforge.evals.harness import CandidateError, check_candidate, render_spec
from harnessforge.evals.reader import RolloutReader, classify, safe_task_id
from harnessforge.records import per_task

FIXTURES = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------------- helpers
def write_trial(root: Path, task: str, k: int, *, status: str | None = "scored", reward: float = 1.0,
                failure_kind: str | None = None, reason: str | None = None, error: str | None = None,
                collection: bool = True, wall: float = 12.5, usage: dict | None = None) -> Path:
    d = root / "rollouts" / safe_task_id(task) / f"rollout_{k}"
    d.mkdir(parents=True, exist_ok=True)
    if collection and status is not None:
        outcome = {"status": status, "rewards": {"reward": reward} if status == "scored" else {}, "reason": reason}
        if failure_kind:
            outcome["failure_kind"] = failure_kind
        traj = {"reward": reward if status == "scored" else 0.0, "steps": [{}, {}],
                "misc": {"outcome": outcome, "harness": {"exit_code": 0, "timed_out": False}}}
        (d / "trajectory_collection.json").write_text(json.dumps({"trajectories": {"root-id": traj}}))
    meta = {"task_id": task, "rollout_index": k, "wall_time_seconds": wall, "error": error, "outcome_status": status,
            "created_at": "2026-11-18T10:00:00+00:00", "status": "completed" if error is None else "errored",
            "usage": usage if usage is not None else {
                "policy": {"calls": 4, "prompt_tokens": 1000, "completion_tokens": 120},
                "aux": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}}}
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


def micro_checkout(root: Path, deps=("litellm", "jinja2", "pydantic", "tenacity", "pyyaml", "mcp")) -> Path:
    (root / "src" / "micro_swe").mkdir(parents=True)
    (root / "src" / "micro_swe" / "__main__.py").write_text("def main(argv=None):\n    pass\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "micro-swe-agent"\ndependencies = [' + ", ".join(f'"{d}"' for d in deps) + "]\n")
    return root


def harbor_tasks(root: Path, names) -> Path:
    for name in names:
        t = root / name
        (t / "environment").mkdir(parents=True)
        (t / "tests").mkdir()
        (t / "task.toml").write_text('[agent]\ntimeout_sec = 600.0\n\n[environment]\nworkdir = "/app"\ncpus = 1\n')
        (t / "environment" / "Dockerfile").write_text(
            f"FROM --platform=linux/amd64 org/img:{name}\nUSER root\n"
            "RUN mkdir -p /workspace /output && \\\n    ([ -d /app ] || ln -sf /workspace /app)\nWORKDIR /app\n")
        (t / "instruction.md").write_text("do it")
        (t / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt")
    return root


# ----------------------------------------------------------------------------- reader
def test_classify_outcomes():
    scored = {"outcome_status": "scored"}
    root = {"reward": 1.0, "misc": {"outcome": {"status": "scored", "rewards": {"reward": 1.0}}}}
    assert classify(scored, root) == (1.0, False, None)
    graded_timeout = {"reward": 0.0, "misc": {"outcome": {"status": "scored", "reason": "agent_timeout"}}}
    assert classify(scored, graded_timeout) == (0.0, False, "scored:agent_timeout")
    # sandbox never started: the rollout raised, no outcome
    assert classify({"outcome_status": None, "error": "EnvSetupError: sandbox_start ..."}, None).env_failed
    for kind in ("sandbox_start", "env_setup", "harness_setup", "harness_crash_gateway", "env_busy",
                 "unexpected:SandboxLostError"):
        v = classify({"outcome_status": "error"}, {"misc": {"outcome": {"status": "error", "failure_kind": kind}}})
        assert v.env_failed and v.reward is None and kind in v.error
    # after the agent ran: counts, as 0
    for kind in ("harness_crash", "verifier_timeout", "verifier_no_reward", "context_overflow"):
        v = classify({"outcome_status": "error"}, {"misc": {"outcome": {"status": "error", "failure_kind": kind}}})
        assert not v.env_failed and v.reward == 0.0
    assert classify({"outcome_status": "timeout"}, {"misc": {"outcome": {"status": "timeout"}}}).reward == 0.0
    assert classify({"outcome_status": "cancelled"}, {}).env_failed
    backend = classify({"outcome_status": "scored", "error": "backend_unavailable: 2 model call(s) failed"}, root)
    assert backend.env_failed and backend.error == "backend_unavailable"


def test_reader_synthetic(tmp_path):
    out = tmp_path / "platoon"
    write_trial(out, "a/t1", 0, reward=1.0)
    write_trial(out, "a/t1", 1, reward=0.0)
    write_trial(out, "t2", 0, status=None, error="sandbox start failed\ntraceback", collection=False)
    write_trial(out, "t2", 1, status="error", failure_kind="harness_crash")
    inflight = out / "rollouts" / "t3" / "rollout_0"
    inflight.mkdir(parents=True)
    (inflight / "harness").mkdir()
    recs = read_rollouts(out, "cand", policy="pol", tasks=["a/t1", "t2", "t3"])
    by = {(r.task, r.index): r for r in recs}
    assert set(by) == {("a/t1", 0), ("a/t1", 1), ("t2", 0), ("t2", 1), ("t3", 0)}
    r = by[("a/t1", 0)]
    assert r.finished and r.reward == 1.0 and not r.env_failed and r.policy == "pol" and r.candidate == "cand"
    assert r.output_tokens == 120 and r.input_tokens == 1000 and r.wall_seconds == 12.5
    expected_end = datetime(2026, 11, 18, 10, tzinfo=timezone.utc).timestamp()
    assert r.finished_at == expected_end and r.started_at == expected_end - 12.5
    assert by[("t2", 0)].env_failed and by[("t2", 0)].error == "sandbox start failed"
    assert by[("t2", 1)].scored and by[("t2", 1)].reward == 0.0 and by[("t2", 1)].error == "error:harness_crash"
    t3 = by[("t3", 0)]
    assert not t3.finished and t3.reward is None and t3.started_at is not None  # in flight, task id mapped back
    stats = per_task(recs, "cand")
    assert stats["a/t1"].reward == 0.5 and stats["t2"].n == 1 and "t3" not in stats
    # once the run is stopped, unfinished trials end as cancelled evidence-free records
    stopped = {(r.task, r.index): r for r in read_rollouts(out, "cand", tasks=["t3"], stopped=True)}
    assert stopped[("t3", 0)].finished and stopped[("t3", 0)].env_failed and stopped[("t3", 0)].error == "stopped"


def test_reader_caches_finished_and_sees_completion(tmp_path):
    out = tmp_path / "p"
    d = out / "rollouts" / "t" / "rollout_0"
    d.mkdir(parents=True)
    reader = RolloutReader(out, "c", tasks=["t"])
    assert [r.finished for r in reader.records()] == [False]
    write_trial(out, "t", 0, reward=1.0)
    first = reader.records()
    assert first[0].finished and first[0].reward == 1.0
    (d / "metadata.json").write_text("not json anymore")  # finished records are cached, never re-read
    assert reader.records() == first


def test_reader_real_platoon_fixtures():
    root = FIXTURES / "platoon_rollouts"
    if not root.exists():
        pytest.skip("no recorded platoon rollouts")
    recs = read_rollouts(root, "base", tasks=json.loads((root / "tasks.json").read_text()))
    expected = json.loads((root / "expected.json").read_text())
    got = {f"{r.task}/{r.index}": {"finished": r.finished, "reward": r.reward, "env_failed": r.env_failed,
                                   "error": r.error, "output_tokens": r.output_tokens} for r in recs}
    assert got == expected


# ----------------------------------------------------------------------------- dataset / harness
def test_dockerfile_and_materialize(tmp_path):
    src = harbor_tasks(tmp_path / "src", ["t1", "t2"])
    image, runs = dockerfile_info(src / "t1" / "environment" / "Dockerfile")
    assert image == "org/img:t1"
    assert runs == ["mkdir -p /workspace /output && ([ -d /app ] || ln -sf /workspace /app)"]
    replays = materialize(src, ["t2"], tmp_path / "dst")
    assert set(replays) == {"t2"} and not (tmp_path / "dst" / "t1").exists()
    env = tomllib.loads((tmp_path / "dst" / "t2" / "task.toml").read_text())["environment"]
    assert env == {"docker_image": "org/img:t2", "workdir": "/app", "cpus": 1}
    assert replay_command(replays["t2"]) == "( " + runs[0] + " )"
    assert replay_command(['["echo", "a b"]']) == "( echo 'a b' )"


def test_render_spec_parses(tmp_path):
    cand = Candidate("cand/7", micro_checkout(tmp_path / "c"))
    cfg = tmp_path / "sampling.yaml"
    cfg.write_text("model:\n  model_kwargs:\n    temperature: 0.6\n")
    spec = render_spec(cand, tmp_path / "h", agent_config=cfg, prepare=["set -e\necho 'x' && echo \"$HOME\""])
    data = tomllib.loads(spec.read_text())
    assert data["name"] == "hf-micro-swe-cand-7" and data["install"]["toolchain"] == "hf-micro-swe-1"
    assert data["install"]["commands"] == ["set -e\necho 'x' && echo \"$HOME\""]
    argv = data["launch"]["argv"]
    assert argv[:4] == ["{toolchain}/bin/python3", "-I", "{state_dir}/hf_micro_runner.py", "--keep-children"]
    assert argv[argv.index("-c") + 1] == "mini" and "{state_dir}/agent-config.yaml" in argv
    assert "--cost-limit" in argv and argv[-1] == "--exit-immediately"
    assert data["launch"]["env"]["LITELLM_PROXY_API_BASE"] == "{base_url}/v1"
    assert (tmp_path / "h" / "hf_micro_runner.py").exists() and (tmp_path / "h" / "agent-config.yaml").exists()
    assert "resources" not in data


@pytest.mark.parametrize("keep", [True, False])
def test_runner_unmarks_the_agents_commands(tmp_path, keep):
    """platoon kills every process carrying its launch marker when the harness exits; with --keep-children the
    commands the agent starts (a server it was asked to leave running) do not carry it, the runner itself does."""
    import subprocess
    from harnessforge.evals.harness import RUNNER

    cand = micro_checkout(tmp_path / "c")
    (cand / "src" / "micro_swe" / "__main__.py").write_text(
        "import subprocess\n"
        "def main(argv=None):\n"
        "    out = [a for a in argv if a.startswith('--output=')][0].split('=', 1)[1]\n"
        "    cmd = ['sh', '-c', 'echo ${PLATOON_HARNESS_AGENT:-unmarked}']\n"
        "    open(out, 'w').write(subprocess.run(cmd, capture_output=True, text=True).stdout)\n")
    (tmp_path / "task.md").write_text("do it")
    env = {"PATH": "/usr/bin:/bin", "PLATOON_HARNESS_AGENT": "agent.123"}
    argv = [sys.executable, "-I", str(RUNNER), *(["--keep-children"] if keep else []), "--candidate", str(cand),
            "--task-file", str(tmp_path / "task.md"), f"--output={tmp_path / 'out'}"]
    subprocess.run(argv, env=env, check=True, timeout=60)
    assert (tmp_path / "out").read_text().strip() == ("unmarked" if keep else "agent.123")


def test_check_candidate(tmp_path):
    check_candidate(micro_checkout(tmp_path / "ok"))
    with pytest.raises(CandidateError):
        check_candidate(tmp_path)
    with pytest.raises(CandidateError, match="numpy"):
        check_candidate(micro_checkout(tmp_path / "np", deps=("litellm", "numpy>=2")))


def test_bench_tasks_and_harnesses_fit_the_platoon_backend(tmp_path):
    """harnessforge.bench output runs through PlatoonBackend unchanged: its tasks name their image in task.toml (no
    Dockerfile replay), a .sif path is used as is, and every gold/start/ablation variant is a valid candidate."""
    from harnessforge.bench import builder
    from harnessforge.bench.tasks.generate import generate
    from harnessforge.evals.__main__ import task_image

    sif = "/images/hf-bench-0.1.sif"
    manifest = generate(tmp_path / "bench", per_family=1, image=sif)
    ids = [t["task_id"] for t in manifest]
    assert len(ids) == 10
    assert materialize(tmp_path / "bench", ids, tmp_path / "ds") == {}  # docker_image set: nothing to replay
    for tid in ids:
        cfg = tomllib.loads((tmp_path / "ds" / tid / "task.toml").read_text())
        assert cfg["environment"]["docker_image"] == sif and not cfg.get("steps")
        assert "user" not in cfg.get("agent", {}) and (tmp_path / "ds" / tid / "tests" / "test.sh").is_file()
        assert task_image(tmp_path / "ds" / tid) == sif  # stage-images skips it; hpc-sandbox runs the .sif
    harnesses = builder.build_all(tmp_path / "h", names=["gold", "start", "ablate_context_trim"])
    for path in harnesses.values():
        check_candidate(path)
    request = EvalRequest(run_id="b1", candidates=[Candidate(k, v) for k, v in harnesses.items()], tasks=ids,
                          dataset=str(tmp_path / "bench"), trials=1, endpoint="http://gpu:8000/v1", policy="p",
                          concurrency=2, out_dir=tmp_path / "out",
                          options={"model": "p", "harbor_system_deps": False})
    PlatoonBackend(launch=False).start(request)
    spec = tomllib.loads(Path(json.loads((tmp_path / "out" / "b1" / "gold" / "eval.yaml").read_text())
                              ["environments"][0]["harness"]).read_text())
    assert not spec["install"].get("commands")


# ----------------------------------------------------------------------------- PlatoonBackend
def platoon_request(tmp_path, n_candidates=2, trials=2, tasks=("t1", "t2", "t3")):
    data = harbor_tasks(tmp_path / "data", tasks)
    cands = [Candidate(f"c{i}", micro_checkout(tmp_path / f"cand{i}")) for i in range(n_candidates)]
    return EvalRequest(run_id="r1", candidates=cands, tasks=list(tasks), dataset=str(data), trials=trials,
                       endpoint="http://gpu:8000/v1", policy="Qwen3.5-4B", concurrency=2, out_dir=tmp_path / "out",
                       options={"model": "Qwen3.5-4B", "agent_timeout_multiplier": 3})


def test_platoon_backend_writes_configs(tmp_path):
    request = platoon_request(tmp_path)
    run = PlatoonBackend(launch=False).start(request)
    assert isinstance(run, EvalRun) and run.order is Order.TASK_MAJOR
    cfg = json.loads((tmp_path / "out" / "r1" / "c0" / "eval.yaml").read_text())
    entry = cfg["environments"][0]
    kw = entry["service_kwargs"]
    assert kw["splits"] == {"eval": ["t1", "t2", "t3"]}  # the look order
    assert kw["dataset_dir"] == str(tmp_path / "out" / "r1" / "dataset")
    assert kw["mounts"][0]["source"] == str((tmp_path / "cand0").resolve()) and kw["mounts"][0]["read_only"]
    assert kw["agent_timeout_multiplier"] == 3.0 and kw["agent_user"] == "ignore"
    assert kw["provider"] == "harnessforge.evals.pods:PodSingularityProvider"
    assert kw["provider_options"]["keep_agent_processes"] is True
    assert entry["on_harness_exit"] == {"timeout": "grade", "crash": "grade"}
    inf = cfg["inference"]
    assert inf["model_name"] == "Qwen3.5-4B" and inf["hosted"]["base_url"] == "http://gpu:8000/v1"
    assert inf["workflow"]["num_rollouts_per_task"] == 2 and inf["workflow"]["num_concurrent_workers"] == 2
    spec = tomllib.loads(Path(entry["harness"]).read_text())
    commands = spec["install"]["commands"]
    assert commands[0].startswith("( mkdir -p /workspace /output") and "apt-get install -y curl" in commands[1]
    assert run.records() == [] and run.done()  # nothing launched


def test_platoon_backend_runs_and_stops_one_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_FAKE_TRIAL_S", "0.3")
    request = platoon_request(tmp_path, tasks=("t1", "t2", "t3", "t4"))
    backend = PlatoonBackend(platoon=FIXTURES / "fake_platoon.py", stop_grace_s=5)
    assert isinstance(backend, EvalBackend)
    run = backend.start(request)
    deadline = time.time() + 20
    while time.time() < deadline:  # wait until c1 has something in flight, then stop it
        recs = run.records()
        if any(r.candidate == "c1" and not r.finished for r in recs) and any(r.finished for r in recs):
            break
        time.sleep(0.05)
    run.stop("c1")
    assert run.wait(timeout=30, poll_s=0.1)
    recs = run.records()
    c0 = [r for r in recs if r.candidate == "c0"]
    c1 = [r for r in recs if r.candidate == "c1"]
    assert len(c0) == 8 and all(r.finished and r.scored for r in c0)  # untouched by c1's stop
    assert [r.reward for r in sorted(c0, key=lambda r: (r.task, r.index))] == [1.0, 0.0] * 4
    assert 0 < len(c1) < 8 and all(r.finished for r in c1)
    assert any(r.error == "stopped" and r.env_failed for r in c1)
    status = run.status()
    assert status["c0"]["returncode"] == 0 and status["c1"]["returncode"] == 143 and status["c1"]["stopped"]
    # launched task-major: c1's trials are a prefix of (t1,0),(t1,1),(t2,0),...
    order = [(t, k) for t in request.tasks for k in range(2)]
    assert sorted((r.task, r.index) for r in c1) == order[:len(c1)]


@pytest.mark.parametrize("terms,expected", [("2", 143), ("0", -9)])
def test_platoon_stop_escalates(tmp_path, monkeypatch, terms, expected):
    """A platoon that ignores the first SIGTERM gets a second at grace/2; one that ignores both is killed, and then
    its sandboxes are reaped with `platoon sandbox reap`."""
    monkeypatch.setenv("HF_FAKE_TRIAL_S", "30")
    monkeypatch.setenv("HF_FAKE_TERMS_TO_EXIT", terms)
    request = platoon_request(tmp_path, n_candidates=1, tasks=("t1",))
    run = PlatoonBackend(platoon=FIXTURES / "fake_platoon.py", stop_grace_s=1.0).start(request)
    deadline = time.time() + 20
    while not any(not r.finished for r in run.records()) and time.time() < deadline:
        time.sleep(0.05)
    run.stop("c0")
    assert run.wait(timeout=20, poll_s=0.1)
    assert run.status()["c0"]["returncode"] == expected
    assert {r.error for r in run.records()} == {"stopped"}
    reaped = tmp_path / "out" / "r1" / "c0" / "platoon" / "reaped"
    deadline = time.time() + 10
    while expected == -9 and not reaped.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert reaped.exists() == (expected == -9)


# ----------------------------------------------------------------------------- FakeBackend
def fake_request(cands=("A", "B"), tasks=("t1", "t2", "t3"), trials=2, concurrency=2):
    return EvalRequest(run_id="f", candidates=[Candidate(c, Path(f"/h/{c}")) for c in cands], tasks=list(tasks),
                       dataset="d", trials=trials, endpoint="e", policy="p", concurrency=concurrency)


def test_fake_backend_task_major_and_progressive():
    backend = FakeBackend({"A": {"t1": [1, 0], "t2": 1.0}, "B": {"t1": 0.0}}, durations={"t2": 3}, auto_advance=False)
    assert isinstance(backend, EvalBackend)
    run = backend.start(fake_request())
    assert run.order is Order.TASK_MAJOR
    first = run.records()
    assert {(r.candidate, r.task, r.index, r.finished) for r in first} == {
        ("A", "t1", 0, False), ("A", "t1", 1, False), ("B", "t1", 0, False), ("B", "t1", 1, False)}
    run.step()
    snap = {(r.candidate, r.task, r.index): r for r in run.records()}
    assert snap[("A", "t1", 0)].reward == 1 and snap[("A", "t1", 1)].reward == 0 and snap[("B", "t1", 0)].reward == 0
    assert not snap[("A", "t2", 0)].finished
    assert run.wait()
    final = run.records()
    assert len(final) == 12 and all(r.finished and r.scored for r in final)
    a = per_task(final, "A")
    assert a["t1"].reward == 0.5 and a["t2"].reward == 1.0 and a["t3"].reward == 0.0
    # deterministic
    again = FakeBackend({"A": {"t1": 0.5}}, seed=3).start(fake_request(trials=8))
    again.wait()
    other = FakeBackend({"A": {"t1": 0.5}}, seed=3).start(fake_request(trials=8))
    other.wait()
    assert again.records() == other.records()


def test_fake_backend_stop_and_env_failures():
    backend = FakeBackend({"A": {}, "B": {}}, default=1.0, env_failures={("A", "t1", 1)}, auto_advance=False)
    run = backend.start(fake_request(concurrency=1))
    run.step(2)
    run.stop("B")
    assert run.wait()
    recs = run.records()
    a = [r for r in recs if r.candidate == "A"]
    b = [r for r in recs if r.candidate == "B"]
    assert len(a) == 6 and sum(not r.scored for r in a) == 1
    assert [(r.task, r.index) for r in b] == [("t1", 0), ("t1", 1), ("t2", 0)]
    assert b[-1].error == "stopped" and b[-1].env_failed and b[0].scored


def test_fake_backend_auto_advance_polling_loop():
    run = FakeBackend(default=1.0).start(fake_request(cands=("A",)))
    polls = 0
    while not run.done():
        run.records()
        polls += 1
    assert polls == 3 and len(run.records()) == 6


# ----------------------------------------------------------------------------- jobs
def test_eval_node_job():
    spec = eval_node_job("hf-eval", ["python", "-m", "harnessforge.evals", "run", "req.json"],
                         seed_image_cache="/groups/x/images", request="req.json")
    assert isinstance(spec, JobSpec) and spec.gpus == 0 and spec.cpus == 32
    assert spec.options == {"resource_type": "rt_HC", "queue": "R9920261300", "account": "gai51740"}
    script = spec.command[-1]
    assert "PLATOON_NODE_SCRATCH" in script and "stage-images req.json --from /groups/x/images" in script
    assert script.splitlines()[-1] == "exec python -m harnessforge.evals run req.json"


def test_platoon_backend_remote_sandboxes(tmp_path):
    request = platoon_request(tmp_path, n_candidates=1)
    request.options["sandboxes"] = {"mode": "remote", "hosts_spec": "/fleet", "local_slots": 4}
    PlatoonBackend(launch=False).start(request)
    kw = json.loads((tmp_path / "out" / "r1" / "c0" / "eval.yaml").read_text())["environments"][0]["service_kwargs"]
    assert kw["provider"] == "harnessforge.sandboxhost.provider:RemotePodProvider"
    po = kw["provider_options"]
    assert po["hosts_spec"] == "/fleet" and po["local_slots"] == 4 and po["keep_agent_processes"] is True
    assert po["collect_dir"].endswith("/collected")
