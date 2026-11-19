"""Prepare the task images of Harbor datasets for hpc-sandbox trials, once, in a CPU job (no per-trial apt).

    python scripts/hpcsb/prepare_images.py --dataset DIR [--dataset DIR ...] \
        [--cache /groups/.../hpc-sandbox-images] [--out /groups/.../mho/prepared-images] \
        [--toolchain /groups/.../mho/toolchains/hf-micro-swe-1] [--jobs 8] [--no-build] [--tasks a,b]

For every distinct image a task names (`[environment] docker_image`, else its Dockerfile's FROM, whose build steps
are still replayed on top at start):

1. **pull** it into the shared image cache (`--cache`, hpc-sandbox's content-addressed store), so eval jobs stop
   pulling and converting it from Docker Hub into their node-local cache every time;
2. **probe** it: which of the commands harbor's installed-agent setup ensures (`ensure_system_dependencies(curl,
   bash, build_tools, git, python3, python_pip)` checks `curl bash gcc make git python3 pip3`) are missing, its
   package manager and libc, and whether the agent toolchain (`--toolchain`, mounted read-only at
   /opt/mho/toolchain) imports there;
3. **derive** an image for each one that misses any of them: `singularity build --fakeroot` from the cached SIF
   with exactly the packages harbor's per-trial install would add (apt: curl bash build-essential git python3
   python3-pip; dnf/yum/apk equivalents), written to `<out>/sysdeps/`, and probe it again.

`<out>/image-map.json` maps each image reference to the SIF to run instead (the derived one, else the cached one);
existing entries for other references are kept. hpc_sandbox_env (`HPCSB_IMAGE_MAP` / `--ek hpc_image_map=`) and
harnessforge's pod provider (`provider_options.image_map`) start a mapped task from that file; harbor's
system-dependency check then finds everything and installs nothing. The dataset is not modified.
`<out>/report-<time>.json` holds every probe; the summary goes to stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[2] / "src")]

from mho.backends.hpc_sandbox_env import DockerfileInfo  # noqa: E402

DEFAULT_CACHE = "/groups/gai51740/aci18914wh/mho/hpc-sandbox-images"
DEFAULT_OUT = "/groups/gai51740/aci18914wh/mho/prepared-images"
DEFAULT_TOOLCHAIN = "/groups/gai51740/aci18914wh/mho/toolchains/hf-micro-swe-1"
TOOLCHAIN_MOUNT = "/opt/mho/toolchain"
#: What harbor's MiniSweAgent.install checks (SYSTEM_PACKAGES[...].commands of its dependencies).
REQUIRED = ("curl", "bash", "gcc", "make", "git", "python3", "pip3")
BUILD_TIMEOUT_S = int(os.environ.get("PREPARE_BUILD_TIMEOUT_S", "3600"))
EXTRA = ("apt-get", "dnf", "yum", "apk", "uv", "tar", "sh")

PROBE = r"""
for c in curl bash gcc make git python3 pip3 apt-get dnf yum apk uv tar sh; do
  if command -v "$c" >/dev/null 2>&1; then echo "cmd $c 1"; else echo "cmd $c 0"; fi
done
if ls /lib/ld-musl* >/dev/null 2>&1; then echo "libc musl"; else echo "libc $( (ldd --version 2>&1 || true) | head -1)"; fi
if [ -x /opt/mho/toolchain/bin/python3 ]; then
  if LITELLM_LOCAL_MODEL_COST_MAP=True /opt/mho/toolchain/bin/python3 -I -c \
     "import litellm, mcp, orjson, fastapi, jinja2, pydantic, tenacity, yaml" >/tmp/.tc.err 2>&1; then
    echo "toolchain 1"; else echo "toolchain 0 $(tail -1 /tmp/.tc.err)"; fi
else echo "toolchain -"; fi
"""

#: The install harbor's ensure_system_dependencies runs when a check fails, per package manager.
SYSDEPS = r"""
set -e
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get -o APT::Sandbox::User=root update
  apt-get -o APT::Sandbox::User=root install -y curl bash build-essential git python3 python3-pip || {
    # A mirror that lost a recommended package (e.g. bullseye's python3.9-dev, 404): the same packages without
    # recommends. harbor's per-trial apt fails outright on such an image.
    echo "retrying without recommends" >&2
    apt-get -o APT::Sandbox::User=root install -y --no-install-recommends curl bash build-essential git python3 python3-pip; }
elif command -v dnf >/dev/null 2>&1; then dnf install -y curl bash gcc gcc-c++ make git python3 python3-pip
elif command -v yum >/dev/null 2>&1; then yum install -y curl bash gcc gcc-c++ make git python3 python3-pip
elif command -v apk >/dev/null 2>&1; then apk add --no-cache curl bash build-base git python3 py3-pip
else echo "no supported package manager" >&2; exit 3
fi
"""


def task_images(datasets: list[Path], only: set[str] | None) -> dict[str, list[str]]:
    """image reference -> the tasks that start from it (docker_image, else the Dockerfile's FROM)."""
    out: dict[str, list[str]] = {}
    for ds in datasets:
        for toml in sorted(ds.glob("*/task.toml")):
            task = toml.parent.name
            if only and task not in only:
                continue
            env = tomllib.loads(toml.read_text()).get("environment", {})
            ref = env.get("docker_image")
            if not ref:
                # The FROM image; its build steps are still replayed on top of the prepared image at start.
                ref = DockerfileInfo(toml.parent / "environment" / "Dockerfile").image
            if ref and not ref.endswith(".sif"):
                out.setdefault(ref, []).append(task)
    return out


def slug(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref)[:120]


async def run(argv: list[str], timeout: float) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "timeout"
    return proc.returncode or 0, out.decode(errors="replace")


async def probe(singularity: str, sif: Path, toolchain: Path | None) -> dict:
    argv = [singularity, "exec", "--containall", "--no-home", "--cleanenv"]
    if toolchain:
        argv += ["-B", f"{toolchain}:{TOOLCHAIN_MOUNT}:ro"]
    code, out = await run([*argv, str(sif), "/bin/sh", "-c", PROBE], timeout=300)
    res: dict = {"probe_exit": code, "cmds": {}, "libc": None, "toolchain": None}
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if parts[0] == "cmd" and len(parts) == 3:
            res["cmds"][parts[1]] = parts[2] == "1"
        elif parts[0] == "libc":
            res["libc"] = line[5:].strip()
        elif parts[0] == "toolchain":
            res["toolchain"] = {"1": True, "0": False}.get(parts[1] if len(parts) > 1 else "-")
            if res["toolchain"] is False:
                res["toolchain_error"] = line[12:].strip()[:300]
    if not res["cmds"]:
        res["probe_output"] = out[-1000:]
    res["missing"] = [c for c in REQUIRED if not res["cmds"].get(c)]
    return res


async def derive(singularity: str, base: Path, target: Path, log: Path) -> tuple[bool, str]:
    """Build in node-local scratch: a fakeroot build cannot traverse the group-only directories of /groups."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("PBS_LOCALDIR") or None) as td:
        local_base, built = Path(td) / "base.sif", Path(td) / "derived.sif"
        await asyncio.to_thread(shutil.copyfile, base, local_base)
        deffile = Path(td) / "sysdeps.def"
        deffile.write_text(f"Bootstrap: localimage\nFrom: {local_base}\n\n%post\n{SYSDEPS}\n")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb") as fh:  # streamed, so a build that hangs still shows how far it got
            proc = await asyncio.create_subprocess_exec(
                singularity, "build", "--fakeroot", "--force", str(built), str(deffile),
                stdout=fh, stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL)
            try:
                code = await asyncio.wait_for(proc.wait(), BUILD_TIMEOUT_S)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                code = 124
        if code != 0 or not built.is_file():
            tail = log.read_text(errors="replace")[-600:]
            return False, ("timeout after %ds: " % BUILD_TIMEOUT_S if code == 124 else "") + tail
        partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
        await asyncio.to_thread(shutil.copyfile, built, partial)
        partial.chmod(0o444)
        os.replace(partial, target)
    return True, ""


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--toolchain", default=DEFAULT_TOOLCHAIN)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-build", action="store_true", help="probe only; map nothing to derived images")
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: all)")
    a = ap.parse_args()

    import hpc_sandbox as hs

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get("PBS_LOCALDIR") or tempfile.mkdtemp()) / "prepare-images"
    provider = hs.SingularityProvider(image_cache=a.cache, scratch_root=scratch, stage_images=False)
    singularity = provider.runtime.path
    toolchain = Path(a.toolchain) if a.toolchain and Path(a.toolchain, "bin", "python3").exists() else None
    if toolchain is None:
        print(f"[prepare] no toolchain at {a.toolchain}: skipping the toolchain import probe", flush=True)
    images = task_images([Path(d) for d in a.dataset], set(filter(None, a.tasks.split(","))) or None)
    print(f"[prepare] {len(images)} images for {sum(map(len, images.values()))} tasks", flush=True)
    sem = asyncio.Semaphore(a.jobs)
    report: dict[str, dict] = {}

    async def one(ref: str) -> None:
        async with sem:
            rec: dict = {"tasks": images[ref]}
            report[ref] = rec
            t0 = time.time()
            try:
                img = await provider.images.pull(ref)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"pull: {e}"[:600]
                print(f"[prepare] PULL FAILED {ref}: {rec['error'][:200]}", flush=True)
                return
            rec["base"] = str(img.path)
            rec["pull_s"] = round(time.time() - t0, 1)
            rec["probe"] = await probe(singularity, img.path, toolchain)
            rec["sif"] = str(img.path)
            missing = rec["probe"]["missing"]
            if missing and not a.no_build:
                digest = (img.sha256 or hashlib.sha256(str(img.path).encode()).hexdigest())[:12]
                target = out / "sysdeps" / f"{slug(ref)}-{digest}.sif"
                if target.is_file():
                    ok, err = True, ""
                else:
                    t1 = time.time()
                    ok, err = await derive(singularity, img.path, target, out / "logs" / f"{slug(ref)}.build.log")
                    rec["build_s"] = round(time.time() - t1, 1)
                if ok:
                    rec["derived"] = str(target)
                    rec["derived_probe"] = await probe(singularity, target, toolchain)
                    if not rec["derived_probe"]["missing"]:
                        rec["sif"] = str(target)
                    else:
                        rec["error"] = f"derived image still misses {rec['derived_probe']['missing']}"
                else:
                    rec["error"] = f"build: {err}"
            print(f"[prepare] {ref}: missing={missing} libc={rec['probe']['libc']!r} "
                  f"toolchain={rec['probe']['toolchain']} -> {Path(rec['sif']).name}"
                  + (f" ERROR {rec['error'][:160]}" if rec.get("error") else ""), flush=True)

    (out / "logs").mkdir(exist_ok=True)
    await asyncio.gather(*(one(ref) for ref in sorted(images)))

    map_path = out / "image-map.json"
    mapping = json.loads(map_path.read_text()) if map_path.is_file() else {}
    mapping.update({ref: rec["sif"] for ref, rec in report.items() if rec.get("sif")})
    tmp = map_path.with_name(f".{map_path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(dict(sorted(mapping.items())), indent=1))
    os.replace(tmp, map_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"report-{stamp}.json").write_text(json.dumps(report, indent=1))

    probed = [r for r in report.values() if "probe" in r]
    print(f"\n[prepare] summary over {len(probed)} probed images ({len(report) - len(probed)} failed to pull):")
    for c in REQUIRED:
        print(f"  lacking {c:8s}: {sum(1 for r in probed if c in r['probe']['missing'])}")
    print(f"  lacking any      : {sum(1 for r in probed if r['probe']['missing'])}")
    print(f"  musl libc        : {sum(1 for r in probed if r['probe']['libc'] == 'musl')}")
    print(f"  toolchain ok/fail: {sum(1 for r in probed if r['probe']['toolchain'] is True)}/"
          f"{sum(1 for r in probed if r['probe']['toolchain'] is False)}")
    pm = {}
    for r in probed:
        m = next((p for p in ("apt-get", "dnf", "yum", "apk") if r["probe"]["cmds"].get(p)), "none")
        pm[m] = pm.get(m, 0) + 1
    print(f"  package managers : {pm}")
    print(f"  derived          : {sum(1 for r in report.values() if r.get('derived'))}, "
          f"errors: {sum(1 for r in report.values() if r.get('error'))}")
    print(f"[prepare] map: {map_path} ({len(mapping)} entries)")


if __name__ == "__main__":
    asyncio.run(main())
