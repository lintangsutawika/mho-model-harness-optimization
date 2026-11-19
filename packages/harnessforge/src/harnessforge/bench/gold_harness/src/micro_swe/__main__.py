"""`mini-swe-agent` console entrypoint.

A runner drives the agent by shelling out to the `mini-swe-agent` binary, e.g.:

    mini-swe-agent --yolo --model=<m> --task=<t> --output=<traj> \
        -c mini -c /path/to/custom.yaml -c model.model_kwargs.timeout=<N> ... \
        --exit-immediately

This entrypoint provides that binary: it parses the flag subset a runner uses, resolves
the `-c` config specs (get_config_from_spec + recursive_merge; bare `mini` -> builtin
config/mini.yaml), builds the Model + LocalEnvironment + Agent, runs the task, and writes
the trajectory to --output.

Flags this agent doesn't need (--yolo, --exit-immediately, ...) are tolerated and ignored
via argparse.parse_known_args, so a runner's exact command line just works.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from micro_swe.config import get_config_from_spec
from micro_swe.agents import Agent
from micro_swe.environments.local import LocalEnvironment
from micro_swe.models import Model
from micro_swe.utils.serialize import recursive_merge

# LocalEnvironmentConfig fields we forward from a config's `environment:` block.
_ENV_KEYS = ("cwd", "env", "timeout")


def _merge_configs(specs: list[str]) -> dict:
    """Merge `-c` specs left-to-right, exactly like mini's CLI. Empty -> builtin mini."""
    if not specs:
        specs = ["mini"]
    merged: dict = {}
    for spec in specs:
        merged = recursive_merge(merged, get_config_from_spec(spec))
    return merged


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mini-swe-agent", add_help=True)
    p.add_argument("-m", "--model", required=True, help="litellm model id (provider/name)")
    p.add_argument("-t", "--task", default="", help="task / problem statement")
    p.add_argument("-o", "--output", default=None, help="trajectory output path")
    p.add_argument("-c", "--config", action="append", default=[], help="config spec (repeatable)")
    p.add_argument("-l", "--cost-limit", type=float, default=None, help="cost limit (0 disables)")
    # Parse only what we know; ignore a runner's --yolo / --exit-immediately / etc.
    args, _ignored = p.parse_known_args(argv)

    cfg = _merge_configs(args.config)

    # model: --model wins over any config model_name/model_class.
    model_cfg = dict(cfg.get("model") or {})
    model_cfg.pop("model_name", None)
    model_cfg.pop("model_class", None)
    model = Model(model_name=args.model, **model_cfg)

    # environment: mini's LocalEnvironment (executes bash in the sandbox).
    env_cfg = {k: v for k, v in (cfg.get("environment") or {}).items() if k in _ENV_KEYS}
    env = LocalEnvironment(**env_cfg)

    # agent: micro's DefaultAgent. `agent:` block supplies the required system_template /
    # instance_template (from `-c mini`); pydantic ignores keys micro doesn't define.
    agent_cfg = dict(cfg.get("agent") or {})
    agent_cfg.pop("mode", None)  # mini-only knob; not a micro AgentConfig field
    if args.cost_limit is not None:
        agent_cfg["cost_limit"] = args.cost_limit
    output_path = Path(args.output) if args.output else None

    agent = Agent(model, env, output_path=output_path, **agent_cfg)
    agent.run(args.task)


if __name__ == "__main__":
    main()
