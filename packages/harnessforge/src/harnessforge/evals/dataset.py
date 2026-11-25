"""Materialize the tasks of an evaluation as a Harbor dataset directory platoon runs as our harbor path did.

platoon runs a dataset directory's task from `[environment] docker_image` (else the service's `default_image`); it
builds `environment/Dockerfile` only through an image map. Our tmax tasks name their image only as the Dockerfile's
`FROM` (and in `[metadata]`), and our harbor environment (`mho.backends.hpc_sandbox_env`) ran that image and replayed
the Dockerfile's remaining steps in the started sandbox. `materialize` copies the requested tasks into a directory of
the run's own, giving each task without a `docker_image` its Dockerfile's last `FROM`, and returns the Dockerfile
steps (RUN lines) the caller must replay (the harness's prepare commands do: see `PlatoonBackend`).
"""
from __future__ import annotations

import re
import shlex
import shutil
import tomllib
from pathlib import Path
from typing import Sequence

_IGNORE = shutil.ignore_patterns(".git", "__pycache__")


def dockerfile_info(path: Path) -> tuple[str | None, list[str]]:
    """(image of the last FROM, RUN commands after it) of a Dockerfile; continuation lines joined."""
    image, runs, buf = None, [], ""
    if not path.is_file():
        return None, []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1].rstrip() + " "
            continue
        buf, line = "", (buf + line).strip()
        word, _, rest = line.partition(" ")
        word = word.upper()
        if word == "FROM":
            toks = [t for t in rest.split() if not t.startswith("--")]
            image, runs = (toks[0] if toks else None), []
        elif word == "RUN":
            runs.append(rest.strip())
    return image, runs


def _with_docker_image(text: str, image: str) -> str:
    line = f"docker_image = {_toml_str(image)}"
    match = re.search(r"(?m)^\[environment\][ \t]*$", text)
    if match is None:
        return text.rstrip("\n") + f"\n\n[environment]\n{line}\n"
    return text[: match.end()] + "\n" + line + text[match.end():]


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def materialize(source: str | Path, tasks: Sequence[str], dest: str | Path) -> dict[str, list[str]]:
    """Copy `tasks` (ids relative to `source`) into `dest`; give each a docker_image from its Dockerfile when it has
    none. Returns {task: RUN commands of its Dockerfile after the FROM} for tasks that got their image that way."""
    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    replays: dict[str, list[str]] = {}
    for task in tasks:
        src = source / task
        if not (src / "task.toml").is_file():
            raise FileNotFoundError(f"{src}: no task.toml")
        dst = dest / task
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=False, ignore=_IGNORE)
        toml_path = dst / "task.toml"
        text = toml_path.read_text()
        if tomllib.loads(text).get("environment", {}).get("docker_image"):
            continue
        image, runs = dockerfile_info(dst / "environment" / "Dockerfile")
        if image is None:
            continue
        toml_path.write_text(_with_docker_image(text, image))
        replays[task] = runs
    return replays


def replay_command(runs: Sequence[str]) -> str | None:
    """One shell command running a Dockerfile's RUN lines in order (shell form; exec form converted), or None."""
    parts = []
    for run in runs:
        if run.startswith("--"):
            raise ValueError(f"RUN with build flags cannot be replayed: RUN {run[:80]}")
        if run.startswith("["):
            import json

            run = shlex.join(json.loads(run))
        parts.append(f"( {run} )")
    return " && ".join(parts) if parts else None
