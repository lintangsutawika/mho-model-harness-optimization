"""Harbor agent that installs a micro-swe-agent SNAPSHOT into each sandbox and runs it.

This is the fixed *deploy* half of the meta-loop -- the counterpart to scaffold.py's
*produce* half:

    scaffold.py        -> makes runs/run_<id>/candidate_<id>/micro-swe-agent/
    harbor_run.py       -> installs that dir into each Modal sandbox, then runs it

It subclasses harbor's installed `MiniSweAgent` and overrides ONLY `install()`: harbor's
stock adapter does `uv tool install mini-swe-agent` (PyPI), which has no way to install a
LOCAL snapshot, so we redirect that one step to upload the candidate dir and install from
it. Everything else -- running `mini-swe-agent --yolo --model=... -c mini -c <custom> ...`
and parsing the trajectory into ATIF -- is inherited unchanged.

It defines NO prompts, tools, or loop logic: those live in the snapshot (micro's own
config/mini.yaml, tools/, agents/default.py), which the meta-loop mutates. This file is
pure plumbing and is identical for every candidate.

Run it:
    harbor run --agent src.harness.agent_harness:AgentHarness \
        --ak mini_fork_local=runs/run_<id>/candidate_<id>/micro-swe-agent -d terminal-bench@2.0 ...
    # or set MICRO_SCAFFOLD_DIR in the env instead of --ak mini_fork_local=...
"""

from __future__ import annotations

import os
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent

# Where the snapshot is uploaded inside the sandbox.
_SCAFFOLD_SANDBOX_DIR = "/tmp/micro-scaffold"
# Repo root, used to resolve a relative snapshot path (e.g. "runs/run_x/candidate_y/...").
_REPO_ROOT = Path(__file__).resolve().parent.parent


class AgentHarness(MiniSweAgent):
    """mini-swe-agent whose install step ships a local micro snapshot into the sandbox."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The candidate snapshot dir: `--ak mini_fork_local=<path>` wins, else the env.
        self._scaffold_dir = kwargs.pop("mini_fork_local", None) or os.environ.get("MICRO_SCAFFOLD_DIR")
        super().__init__(*args, **kwargs)

    async def install(self, environment) -> None:  # type: ignore[override]
        """Agent-setup hook (runs in the sandbox). Upload the snapshot + `uv tool install`
        it, giving the sandbox the `mini-swe-agent` binary from THIS candidate's code."""
        if not self._scaffold_dir:
            raise ValueError(
                "No scaffold snapshot set. Pass --ak mini_fork_local=<dir> or set "
                "MICRO_SCAFFOLD_DIR (see harness/scaffold.py:materialize)."
            )
        src = Path(self._scaffold_dir)
        if not src.is_absolute():
            src = _REPO_ROOT / src
        if not (src / "pyproject.toml").exists():
            raise FileNotFoundError(f"scaffold {src} is not an installable package (no pyproject.toml)")

        # Ship the candidate into the sandbox as a single tarball rather than
        # environment.upload_dir() (which stages via copytree + a two-`cp` merge and
        # races under concurrent trials). The sandbox's writable-tmpfs is fuse-overlayfs,
        # which is unreliable for filesystem MUTATIONS under concurrent load: a fresh
        # `mkdir` can be momentarily invisible ("tar: <dir>: Cannot open"), and `rm -rf`
        # spuriously fails ("rm: cannot remove '.../models': Is a directory"). So we do
        # NEITHER: extract to a FRESH, unique dir per install via a single `tar` that
        # creates the dir itself when extracting into the always-present /tmp. The
        # sandbox is ephemeral (per-trial, delete=True), so nothing needs cleaning up,
        # and `uv tool install` below points at this unique dir.
        _parent = str(Path(_SCAFFOLD_SANDBOX_DIR).parent)          # /tmp
        _uid = uuid.uuid4().hex
        _leaf = f"{Path(_SCAFFOLD_SANDBOX_DIR).name}-{_uid}"       # micro-scaffold-<uid>
        scaffold_dir = f"{_parent}/{_leaf}"                        # /tmp/micro-scaffold-<uid>
        with tempfile.TemporaryDirectory() as _td:
            tar_name = f"scaffold-{_uid}.tgz"
            tar_path = Path(_td) / tar_name
            with tarfile.open(tar_path, "w:gz") as _tf:
                _tf.add(src, arcname=_leaf)
            remote_tar = f"/tmp/{tar_name}"
            await environment.upload_file(tar_path, remote_tar)
        await self.exec_as_root(
            environment,
            command=f"set -eu; tar xzf {remote_tar} -C {_parent}",
        )

        # Mirror harbor's own mini install (system deps + uv bootstrap), but install from
        # the uploaded snapshot. Same console-script name (`mini-swe-agent`) + the same
        # `--with` extras, so the inherited run() invocation resolves unchanged.
        await self.ensure_system_dependencies(
            environment,
            ("curl", "bash", "build_tools", "git", "python3", "python_pip"),
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v uv >/dev/null 2>&1; then "
                "  curl -LsSf https://astral.sh/uv/install.sh | sh; fi && "
                'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi && '
                'export PATH="$HOME/.local/bin:$PATH" && '
                f"uv tool install {scaffold_dir} "
                "--with litellm --with orjson --with fastapi && "
                "mini-swe-agent --help"
            ),
        )
