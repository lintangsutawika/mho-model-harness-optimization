"""Task generator: structure, determinism, and every family solvable by an oracle (and not by doing nothing).

The verifiers are run locally by pointing /app and /logs/verifier at a temp dir (LAB_ROOT for labctl).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

import pytest

from harnessforge.bench.tasks import FAMILIES, generate, labctl, read_manifest
from harnessforge.bench.tasks.generate import specs
from harnessforge.bench.tasks.generate import test_sh as render_test_sh

LABCTL = Path(labctl.__file__)


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def test_generate_structure_and_counts(tmp_path):
    manifest = generate(tmp_path / "b", per_family=20, seed=0)
    assert len(manifest) == 200
    counts = {}
    for t in manifest:
        counts[t["family"]] = counts.get(t["family"], 0) + 1
        d = tmp_path / "b" / t["task_id"]
        for rel in ("task.toml", "instruction.md", "environment/Dockerfile", "environment/labctl.py", "tests/test.sh"):
            assert (d / rel).is_file(), (t["task_id"], rel)
        meta = tomllib.loads((d / "task.toml").read_text())
        assert meta["metadata"]["family"] == t["family"] and meta["metadata"]["seed"] == t["seed"]
        assert meta["environment"]["docker_image"] == "harnessforge-bench:0.1"
        assert os.access(d / "tests" / "test.sh", os.X_OK)
        assert "labctl setup " in (d / "instruction.md").read_text()
        assert t["family"] not in (d / "instruction.md").read_text()      # setup ids are opaque
    assert counts == {f: 20 for f in FAMILIES}
    assert read_manifest(tmp_path / "b") == manifest
    assert (tmp_path / "b" / "image" / "Dockerfile").read_text().startswith("FROM python:3.12-slim")
    assert len({t["seed"] for t in manifest}) == 200


def test_generate_is_deterministic(tmp_path):
    generate(tmp_path / "a", per_family=3, seed=7)
    generate(tmp_path / "b", per_family=3, seed=7)
    generate(tmp_path / "c", per_family=3, seed=8)
    assert _tree_digest(tmp_path / "a") == _tree_digest(tmp_path / "b")
    assert _tree_digest(tmp_path / "a") != _tree_digest(tmp_path / "c")
    with pytest.raises(FileExistsError):
        generate(tmp_path / "a", per_family=1)


# ----------------------------------------------------------------------------- oracle runs
def _labctl(root, *args, env_extra=None, **kw):
    env = {**os.environ, "LAB_ROOT": str(root), **(env_extra or {})}
    return subprocess.run([sys.executable, str(LABCTL), *args], env=env, capture_output=True, text=True,
                          check=True, **kw).stdout


def _verify(spec, root: Path) -> int:
    script = render_test_sh(spec)
    body = script.split("<<'HF_BENCH_CHECK'\n", 1)[1].rsplit("HF_BENCH_CHECK\n", 1)[0]
    body = body.replace("/logs/verifier", str(root / "_logs"))
    body = re.sub(r"(['\"])/app/", lambda m: f"{m.group(1)}{root}/", body)
    subprocess.run([sys.executable, "-c", body], capture_output=True, text=True, timeout=60)
    return int((root / "_logs" / "reward.txt").read_text())


def _solve_verify_first(spec, root):
    fn = spec.expected["fn"]
    impl = {
        "median": "def median(xs):\n    s = sorted(xs); n = len(s)\n"
                  "    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n",
        "percent_change": "def percent_change(old, new):\n    return None if old == 0 else round((new - old) / old * 100, 1)\n",
        "chunk": "def chunk(xs, size):\n    if size < 1:\n        raise ValueError(size)\n"
                 "    return [xs[i:i + size] for i in range(0, len(xs), size)]\n",
        "clamp": "def clamp(x, lo, hi):\n    if lo > hi:\n        raise ValueError((lo, hi))\n    return min(max(x, lo), hi)\n",
    }[fn]
    (root / "calc" / "stats.py").write_text(impl)


def _solve(spec, root: Path):
    fam, seed = spec.family, spec.seed
    app = root
    if fam == "huge_log":
        lines = _labctl(root, "huge-log", str(seed)).splitlines()
        run_id = lines[0].split()[0].split("=")[1]
        final = [l for l in lines if l.startswith("final_checksum=")][0].split("=")[1]
        (app / "answer.txt").write_text(f"{run_id} {final}\n")
    elif fam == "background_service" and seed % 2 == 0:
        proc = subprocess.Popen([sys.executable, str(LABCTL), "serve"], env={**os.environ, "LAB_ROOT": str(root)},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if (app / "service" / "port").exists():
                break
            time.sleep(0.05)
        port = int((app / "service" / "port").read_text())
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read().decode()
        (app / "health.txt").write_text(body)
        return proc
    elif fam == "background_service":
        out = _labctl(root, "crunch", str(seed), env_extra={"LAB_CRUNCH_SECONDS": "0.05"})
        (app / "result.txt").write_text(out.strip().splitlines()[-1].split(": ")[1])
    elif fam == "deep_dirs":
        d = labctl.deep_dirs_data(seed)
        for m, v in zip(d["modules"], d["values"]):
            md = app / "workspace" / d["proj"] / "modules" / m
            (md / "params" / "value.txt").write_text(f"{v}\n")
            subprocess.run(["./configure"], cwd=md, check=True, capture_output=True)
            subprocess.run(["./build"], cwd=md, check=True, capture_output=True)
    elif fam == "bulk_write":
        (app / "manifest.txt").write_text("\n".join(spec.expected["lines"]) + "\n")
    elif fam == "stuck_poll":
        info = _labctl(root, "job", "info")
        job = info.split()[1]
        code = re.search(r"approval code: (\S+)", info).group(1)
        (app / "queue" / "approvals").mkdir(parents=True, exist_ok=True)
        (app / "queue" / "approvals" / job).write_text(code + "\n")
        assert "DONE" in _labctl(root, "job", "status")
        (app / "result.txt").write_text(_labctl(root, "job", "result"))
    elif fam == "json_escape":
        (app / "config.json").write_text(json.dumps(spec.expected["config"]))
    elif fam == "verify_first":
        _solve_verify_first(spec, root)
    elif fam == "many_files":
        d = labctl.many_files_data(seed)
        name, n = d["chain"][0], 1
        while True:
            text = (app / "notes" / name).read_text()
            if m := re.search(r"^final: (\S+)$", text, re.M):
                break
            name, n = re.search(r"^next: (\S+)$", text, re.M).group(1), n + 1
        (app / "answer.txt").write_text(f"{m.group(1)}\n{n}\n")
    elif fam == "exact_text":
        line = _labctl(root, "emit", str(seed)).rstrip("\n")
        (app / "answer.txt").write_text(f"{line.count(chr(92))} {line.count(chr(34))} {line.count('<')}\n")
    elif fam == "control":
        (app / "answer.txt").write_text(spec.expected["answer"] + "\n")
    else:
        raise AssertionError(fam)
    return None


ORACLE_CASES = [(tid, spec) for tid, spec in specs(per_family=2, seed=3)]


@pytest.mark.parametrize("tid,spec", ORACLE_CASES, ids=[t for t, _ in ORACLE_CASES])
def test_oracle_solves_and_noop_fails(tmp_path, tid, spec):
    code = spec.instruction.split("labctl setup ", 1)[1].split("`", 1)[0]
    noop, solved = tmp_path / "noop", tmp_path / "solved"
    for root in (noop, solved):
        root.mkdir()
        assert "workspace ready" in _labctl(root, "setup", code)
    assert _verify(spec, noop) == 0
    proc = _solve(spec, solved)
    try:
        assert _verify(spec, solved) == 1
    finally:
        if proc is not None:
            proc.kill()


def test_families_cover_every_feature():
    from harnessforge.bench import FEATURE_NAMES
    covered = {f for _, s in specs(per_family=1) for f in s.features}
    assert covered == set(FEATURE_NAMES)


def test_verify_first_bug_is_real_and_has_hidden_part(tmp_path):
    # the shipped code fails the check, and fixing only the reported symptom is not enough
    for _, spec in specs(per_family=4, seed=1):
        if spec.family != "verify_first":
            continue
        root = tmp_path / str(spec.seed)
        root.mkdir()
        _labctl(root, "setup", f"v-{spec.seed}")
        r = subprocess.run([sys.executable, str(root / "calc" / "check.py")], capture_output=True, text=True)
        assert r.returncode == 1 and "FAIL" in r.stdout
        assert _verify(spec, root) == 0
