"""Write the toy benchmark as Harbor task directories.

    python -m harnessforge.bench.tasks --out data/hf-bench --per-family 20 --seed 0

Layout:
    <out>/manifest.json                 [{task_id, family, seed, features}], in generation order
    <out>/image/{Dockerfile,lab.def,labctl.py}
                                        the ONE shared image (python:3.12-slim + labctl); build it once
                                        (docker build -t harnessforge-bench:0.1 image/  or
                                         singularity build bench.sif image/lab.def)
    <out>/<task_id>/task.toml           [environment].docker_image = the shared image ref
    <out>/<task_id>/instruction.md
    <out>/<task_id>/environment/{Dockerfile,labctl.py}   same recipe (for backends that build)
    <out>/<task_id>/tests/test.sh       stdlib Python check -> /logs/verifier/reward.txt

Everything is a pure function of (seed, per_family, image): regenerating gives identical bytes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path

from harnessforge.bench.tasks import labctl
from harnessforge.bench.tasks.families import FAMILIES, TaskSpec

BENCH_VERSION = "0.1"
DEFAULT_IMAGE = f"harnessforge-bench:{BENCH_VERSION}"
LABCTL_SRC = Path(labctl.__file__)

DOCKERFILE = """FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl procps \\
    && rm -rf /var/lib/apt/lists/*
COPY labctl.py /opt/lab/labctl.py
RUN printf '#!/bin/sh\\nexec python3 /opt/lab/labctl.py "$@"\\n' > /usr/local/bin/labctl \\
    && chmod +x /usr/local/bin/labctl && mkdir -p /app /logs/verifier
WORKDIR /app
"""

SINGULARITY_DEF = """Bootstrap: docker
From: python:3.12-slim

%files
    labctl.py /opt/lab/labctl.py

%post
    apt-get update && apt-get install -y --no-install-recommends curl procps && rm -rf /var/lib/apt/lists/*
    printf '#!/bin/sh\\nexec python3 /opt/lab/labctl.py "$@"\\n' > /usr/local/bin/labctl
    chmod +x /usr/local/bin/labctl
    mkdir -p /app /logs/verifier
"""

CHECK_PRELUDE = """import base64, hashlib, importlib.util, json, sys, time, urllib.request
from pathlib import Path

EXPECTED = json.loads(base64.b64decode("{expected}").decode())

"""

CHECK_EPILOGUE = """

try:
    ok = bool(check())
except Exception as e:  # missing file, bad JSON, crash: no reward
    print("check error:", type(e).__name__, e)
    ok = False
Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
Path("/logs/verifier/reward.txt").write_text("1" if ok else "0")
print("reward:", int(ok))
"""


def task_seed(seed: int, family: str, i: int) -> int:
    return int(hashlib.sha256(f"hfbench:{seed}:{family}:{i}".encode()).hexdigest(), 16) % 1_000_000


def task_id(family: str, i: int) -> str:
    return f"{family}-{i:03d}"


def specs(per_family: int = 20, seed: int = 0) -> list[tuple[str, TaskSpec]]:
    """[(task_id, TaskSpec)] for every family, `per_family` tasks each."""
    return [(task_id(fam, i), make(task_seed(seed, fam, i)))
            for fam, make in FAMILIES.items() for i in range(per_family)]


def test_sh(spec: TaskSpec) -> str:
    expected = base64.b64encode(json.dumps(spec.expected, sort_keys=True).encode()).decode()
    body = CHECK_PRELUDE.format(expected=expected) + spec.check + CHECK_EPILOGUE
    return f"#!/bin/bash\nmkdir -p /logs/verifier\npython3 - <<'HF_BENCH_CHECK'\n{body}HF_BENCH_CHECK\n"


def task_toml(tid: str, spec: TaskSpec, image: str) -> str:
    feats = ", ".join(json.dumps(f) for f in spec.features)
    return (f'schema_version = "1.4"\n\n[task]\nname = "harnessforge-bench/{tid}"\n\n'
            f'[environment]\ndocker_image = "{image}"\ncpus = 1\nmemory = "1G"\nbuild_timeout_sec = 600.0\n\n'
            f"[agent]\ntimeout_sec = {spec.agent_timeout_sec}\n\n[verifier]\ntimeout_sec = 120.0\n\n"
            f'[metadata]\nbenchmark = "harnessforge-bench-v{BENCH_VERSION}"\ntask_id = "{tid}"\n'
            f'family = "{spec.family}"\nseed = {spec.seed}\nfeatures = [{feats}]\n')


def write_image(out: Path) -> Path:
    img = out / "image"
    img.mkdir(parents=True, exist_ok=True)
    (img / "Dockerfile").write_text(DOCKERFILE)
    (img / "lab.def").write_text(SINGULARITY_DEF)
    shutil.copyfile(LABCTL_SRC, img / "labctl.py")
    return img


def generate(out: str | Path, *, per_family: int = 20, seed: int = 0, image: str = DEFAULT_IMAGE,
             overwrite: bool = False) -> list[dict]:
    """Write all tasks under `out`; returns the manifest."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (pass overwrite=True)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_image(out)
    manifest = []
    for tid, spec in specs(per_family, seed):
        d = out / tid
        (d / "environment").mkdir(parents=True)
        (d / "tests").mkdir()
        (d / "task.toml").write_text(task_toml(tid, spec, image))
        (d / "instruction.md").write_text(spec.instruction)
        (d / "environment" / "Dockerfile").write_text(DOCKERFILE)
        shutil.copyfile(LABCTL_SRC, d / "environment" / "labctl.py")
        (d / "tests" / "test.sh").write_text(test_sh(spec))
        (d / "tests" / "test.sh").chmod(0o755)
        manifest.append({"task_id": tid, "family": spec.family, "seed": spec.seed, "features": list(spec.features)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def read_manifest(root: str | Path) -> list[dict]:
    return json.loads((Path(root) / "manifest.json").read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessforge.bench.tasks", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="docker_image ref (e.g. a prebuilt .sif path)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    m = generate(args.out, per_family=args.per_family, seed=args.seed, image=args.image, overwrite=args.overwrite)
    counts: dict[str, int] = {}
    for t in m:
        counts[t["family"]] = counts.get(t["family"], 0) + 1
    print(f"wrote {len(m)} tasks to {args.out}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0
