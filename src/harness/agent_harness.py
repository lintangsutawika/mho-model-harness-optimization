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

import json
import os
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.models.agent.context import AgentContext

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

    async def _exec(  # type: ignore[override]
        self,
        environment,
        command: str,
        user=None,
        env=None,
        cwd=None,
        timeout_sec=None,
    ):
        """Raise harbor's per-exec cap for the (single, long-running) agent exec.

        ``mini_swe_agent.run()`` calls ``exec_as_agent`` WITHOUT a ``timeout_sec``, so the
        whole agent runs under harbor's hardcoded ``_DEFAULT_HTTP_TIMEOUT = 600`` in
        singularity.exec. Long reasoning trajectories (a single 16k-token turn ≈ 630s at the
        observed ~26 tok/s) exceed it and are killed as ``AgentTimeoutError`` — for BOTH eval
        and training rollouts (the generator runs the same trial). The outer agent-phase
        timeout (``override_timeout_sec``) is unset → None (unlimited), so the 600s exec cap is
        the only binding one, and it lives in the vendored harbor pkg. Inject a higher default
        here (tunable via ``AGENT_EXEC_TIMEOUT_SEC``, default 1800s) whenever a caller left it
        None; explicit values (e.g. a future per-call cap) pass through untouched. This runs in
        the ray worker (harbor client side), so it reads the worker env, not the sandbox env.
        """
        if timeout_sec is None:
            timeout_sec = int(os.environ.get("AGENT_EXEC_TIMEOUT_SEC", "1800"))
        return await super()._exec(
            environment, command, user=user, env=env, cwd=cwd, timeout_sec=timeout_sec
        )

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

    def populate_context_post_run(self, context: AgentContext) -> None:  # type: ignore[override]
        """Assemble per-turn RL rollout details on top of harbor's stock post-run.

        Harbor's ``MiniSweAgent.populate_context_post_run`` records only token *counts* and
        writes the ATIF trajectory; it never sets ``rollout_details``/``metadata``, so SkyRL's
        generator (which trains on per-turn ``completion_token_ids``) drops every trajectory.

        The micro-swe-agent fork stashes ``{prompt_token_ids, completion_token_ids, logprobs}``
        under each assistant message's ``extra.rollout`` (see micro_swe/models/litellm.py, gated
        by ``MICRO_RETURN_TOKEN_IDS``). We read the fork's own trajectory, collect those turns,
        and emit the single linear rollout segment the generator asserts on:
        ``rollout_details = [{prompt_token_ids: [per-turn], completion_token_ids: [per-turn],
        logprobs: [per-turn]}]`` with ``metadata["n_episodes"] = n_turns``. Only turns whose
        three arrays are present and length-consistent (``len(logprobs)==len(completion)``) are
        kept, matching the generator's assertions; an empty result leaves it a dropped trajectory.
        """
        super().populate_context_post_run(context)

        mini_trajectory_path = self.logs_dir / "mini-swe-agent.trajectory.json"
        if not mini_trajectory_path.exists():
            return
        try:
            mini_trajectory = json.loads(mini_trajectory_path.read_text())
        except Exception as e:
            self.logger.debug(f"rollout: failed to load fork trajectory: {e}")
            return

        prompt_ids_per_turn: list[list[int]] = []
        completion_ids_per_turn: list[list[int]] = []
        logprobs_per_turn: list[list[float]] = []
        for message in mini_trajectory.get("messages") or []:
            if message.get("role") != "assistant":
                continue
            rollout = (message.get("extra") or {}).get("rollout")
            if not rollout:
                continue
            p_ids = rollout.get("prompt_token_ids")
            c_ids = rollout.get("completion_token_ids")
            lps = rollout.get("logprobs")
            # Enforce the generator's per-turn invariants; skip any turn that can't satisfy them.
            if not (isinstance(p_ids, list) and isinstance(c_ids, list) and isinstance(lps, list)):
                continue
            if len(c_ids) == 0 or len(lps) != len(c_ids):
                continue
            prompt_ids_per_turn.append(p_ids)
            completion_ids_per_turn.append(c_ids)
            logprobs_per_turn.append(lps)

        n_turns = len(completion_ids_per_turn)
        if n_turns == 0:
            self.logger.debug("rollout: no token-id-bearing assistant turns found; trajectory will be dropped")
            return

        context.rollout_details = [
            {
                "prompt_token_ids": prompt_ids_per_turn,
                "completion_token_ids": completion_ids_per_turn,
                "logprobs": logprobs_per_turn,
            }
        ]
        context.metadata = {**(context.metadata or {}), "n_episodes": n_turns}
