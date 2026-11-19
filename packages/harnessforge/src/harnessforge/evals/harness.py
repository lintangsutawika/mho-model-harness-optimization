"""A micro-swe-agent candidate as a platoon harness spec that runs inside the task sandbox, installing nothing there.

Our harbor path (`harness.micro_swe_agent:MicroSweAgent`) uploads the candidate dir into each sandbox,
`uv tool install`s it with litellm/mcp/orjson/fastapi from the network, and runs

    mini-swe-agent --yolo --model=litellm_proxy/<served> --task=<instruction> --output=<traj> --cost-limit 0 \
        -c mini -c model.model_kwargs.timeout=3600 -c <sampling yaml> --exit-immediately

in the task's working directory, where the agent's LocalEnvironment runs each bash action as a subprocess. Here the
same command runs in the same place, from

- a **toolchain** (`TOOLCHAIN`): a prebuilt CPython 3.12 with micro-swe-agent's dependencies, hash-locked
  (`assets/hf-micro-swe-1.txt`), which platoon copies once per node and mounts read-only into every sandbox; and
- the **candidate dir**, mounted read-only at `CANDIDATE_MOUNT` (the Harbor service's `mounts`), whose `src/` the
  runner (`assets/hf_micro_runner.py`) puts first on `sys.path` before calling the candidate's own
  `micro_swe.__main__.main`.

So the agent loop, prompts, tools and parsing are the candidate's own code, run where it ran before, and the model is
reached through platoon's gateway (which forwards to the hosted endpoint) as `litellm_proxy/<model>`.
"""
from __future__ import annotations

import re
import shutil
import tomllib
from pathlib import Path
from typing import Sequence

from harnessforge.contracts import Candidate

ASSETS = Path(__file__).parent / "assets"
RUNNER = ASSETS / "hf_micro_runner.py"
#: The toolchain's name under the toolchain root; its recipe is `assets/hf-micro-swe-toolchain.toml`.
TOOLCHAIN = "hf-micro-swe-1"
TOOLCHAIN_SPEC = ASSETS / "hf-micro-swe-toolchain.toml"
#: Where the candidate dir appears in each sandbox.
CANDIDATE_MOUNT = "/opt/harnessforge/candidate"
#: What the toolchain holds (pip names, lower-case); a candidate that needs more is refused at start.
TOOLCHAIN_PACKAGES = frozenset({"litellm", "jinja2", "pydantic", "tenacity", "pyyaml", "mcp", "orjson", "fastapi"})


class CandidateError(ValueError):
    pass


def check_candidate(harness_dir: str | Path) -> None:
    """Refuse a candidate dir the toolchain cannot run: no `src/micro_swe/__main__.py`, or dependencies (pyproject
    `[project] dependencies`) beyond `TOOLCHAIN_PACKAGES`."""
    root = Path(harness_dir)
    if not (root / "src" / "micro_swe" / "__main__.py").is_file():
        raise CandidateError(f"{root}: no src/micro_swe/__main__.py (not a micro-swe-agent checkout)")
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        deps = tomllib.loads(pyproject.read_text()).get("project", {}).get("dependencies", [])
        names = {re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].strip().lower().replace("_", "-") for d in deps}
        if extra := sorted(names - TOOLCHAIN_PACKAGES):
            raise CandidateError(f"{root}: depends on {extra}, which the {TOOLCHAIN} toolchain does not hold")


def spec_name(candidate: Candidate) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", candidate.id.lower()).strip("-")[:48] or "candidate"
    return f"hf-micro-swe-{slug}"


def render_spec(candidate: Candidate, out_dir: str | Path, *, agent_config: str | Path | None = None,
                toolchain: str = TOOLCHAIN, prepare: Sequence[str] = (), timeout_s: int = 86400,
                memory_mb: int = 0, llm_timeout_s: int = 3600, keep_children: bool = True) -> Path:
    """Write the candidate's harness spec (and its assets) into `out_dir`; returns the spec's path.

    `agent_config` is the sampling YAML layered after `-c mini` (eval.sh's `configs/sampling/<model>.yaml`).
    `prepare` are shell commands run in the sandbox, as the task's user, before the agent starts (platoon's
    `[install] commands`): what our harbor path did to the environment before its agent ran (see `PlatoonBackend`).
    `timeout_s` is the spec's own limit; the task's `[agent] timeout_sec` x `agent_timeout_multiplier` normally binds
    first, as harbor's agent timeout did. `memory_mb` (default 0) is added to the task's memory limit for the agent's
    own processes; 0 counts them against the task's limit, as harbor did. `keep_children` (default, harbor parity)
    leaves what the agent's commands started running after the harness exits (the runner's `--keep-children`);
    platoon otherwise kills every process of the launch when it ends."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUNNER, out / RUNNER.name)
    argv = ["{toolchain}/bin/python3", "-I", "{state_dir}/" + RUNNER.name,
            *(["--keep-children"] if keep_children else []),
            "--candidate", CANDIDATE_MOUNT, "--task-file", "{state_dir}/instruction.md",
            "--yolo", "--model=litellm_proxy/{model}", "--output={state_dir}/trajectory.json", "--cost-limit", "0",
            "-c", "mini", "-c", f"model.model_kwargs.timeout={int(llm_timeout_s)}"]
    assets = {RUNNER.name: RUNNER.name}
    if agent_config is not None:
        shutil.copyfile(agent_config, out / "agent-config.yaml")
        assets["agent-config.yaml"] = "agent-config.yaml"
        argv += ["-c", "{state_dir}/agent-config.yaml"]
    argv.append("--exit-immediately")
    lines = [
        "# Generated by harnessforge.evals.harness: micro-swe-agent candidate " + _toml(candidate.id),
        f"name = {_toml(spec_name(candidate))}",
        'version = "1"',
        'dialect = "openai_chat"',
        f"timeout_s = {int(timeout_s)}",
        "",
        "[install]",
        f"toolchain = {_toml(toolchain)}",
        "commands = [" + ", ".join(_toml_multiline(c) for c in prepare) + "]",
        "",
        *(["[resources]", f"memory_mb = {int(memory_mb)}", ""] if memory_mb else []),
        "[launch]",
        "argv = [" + ", ".join(_toml(a) for a in argv) + "]",
        'files = { "instruction.md" = "{instruction}" }',
        "assets = { " + ", ".join(f"{_toml(k)} = {_toml(v)}" for k, v in assets.items()) + " }",
        "",
        "[launch.env]",
        'LITELLM_PROXY_API_BASE = "{base_url}/v1"',
        'LITELLM_PROXY_API_KEY = "{api_key}"',
        'MSWEA_API_KEY = "{api_key}"',
        'MSWEA_CONFIGURED = "true"',
        'MSWEA_COST_TRACKING = "ignore_errors"',
        'LITELLM_LOCAL_MODEL_COST_MAP = "True"',
        'PYTHONDONTWRITEBYTECODE = "1"',
        "",
        "[builtins]",
        'tools = ["bash"]',
        "",
    ]
    spec = out / "spec.toml"
    spec.write_text("\n".join(lines))
    return spec


def _toml_multiline(value: str) -> str:
    if "'''" in value:
        raise ValueError("a prepare command may not contain '''")
    return "'''\n" + value + "'''"


def _toml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
