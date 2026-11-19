# micro-swe-agent

A stripped-down, self-contained terminal agent scaffold. `Agent` runs a
plan/act/observe loop over a litellm `Model` and a bash `Environment`, so the
**scaffolding** (loop, prompts, tools) is the part you edit.

It ships the `mini-swe-agent` console entrypoint, making it a drop-in replacement for
mini-swe-agent anywhere a runner shells out to that binary.

## Layout

```
src/micro_swe/
  agents/       # Agent — the loop (run/step/query/execute_actions)
  models/       # Model — litellm, native bash tool-calling (non-Claude focused)
  environments/ # LocalEnvironment — executes bash
  tools/        # bash tool: schema + action parse + observation format
  config/       # config resolver + mini.yaml (default prompts)
  utils/        # retry, serialize
  exceptions.py, protocols.py, __main__.py (the CLI)
```

## Install & run

```bash
uv pip install -e .          # standalone (no mini-swe-agent dependency)

mini-swe-agent \
  --model litellm_proxy/Qwen/Qwen3.8-27B \
  --task "Create primes.py that prints the first 20 primes, then run it." \
  --output traj.json \
  -c mini                    # -c specs merge like mini; bare `mini` = config/mini.yaml
```

The model endpoint is resolved by litellm from the environment (e.g.
`LITELLM_PROXY_API_BASE`/`_KEY`). Unknown flags (`--yolo`, `--exit-immediately`) are
tolerated, so a runner's command line works unchanged.

## Extend

Edit `agents/default.py` (`query`/`execute_actions` for control flow), `tools/bash.py`
(the action space), or `config/mini.yaml` (prompts).

## Test

```bash
uv pip install -e '.[dev]' && pytest
```
