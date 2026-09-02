# mho-model-harness-optimization

Optimizing the **agent scaffold** (workflow + tools) for a given model on
**Terminal-Bench 2.0**, run through **harbor** on the **Modal** environment.

The scaffold under study is **micro-swe-agent** (a sibling checkout) — a stripped-down,
self-contained agent whose loop / prompts / tools are meant to be edited. A meta-loop
continually mutates it and evaluates each candidate. To avoid git churn and cross-run
collisions, each candidate is an **immutable snapshot** of the scaffold; harbor installs
that snapshot into each sandbox and runs it.

## Layout

```
harness/
  scaffold.py                # materialize a per-candidate snapshot:
                             #   runs/run_<id>/candidate_<id>/micro-swe-agent/
  harbor_run.py              # AgentHarness: uploads a snapshot into each sandbox,
                             #   installs+runs it (fixed glue; same for every candidate)
scripts/eval/
  eval_terminal_bench_2.sh   # CORE: harbor run w/ harness.harbor_run on TB2, Modal env
scripts/hpc/
  submit_slurm.sbatch        # Slurm wrapper (babel/hpcfund) → calls the eval
  submit_pbs.sh              # PBS wrapper (ABCI/qsub)       → calls the eval
scripts/build/
  build_vllm.sh              # prepare vllm-cuda / vllm-rocm .sif (self-hosted model only)
../micro-swe-agent/          # the scaffold being optimized (its own package + tests)
src/                         # meta-harness reference checkout (read-only reference)
```

## Setup

```bash
uv sync                      # installs harbor[modal] + mini-swe-agent[full] + terminal-bench
modal setup                  # authenticate Modal (or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)
```

Dependency pins live in `pyproject.toml`; after the first `uv sync`, commit `uv.lock` and
treat a harbor/terminal-bench bump as a deliberate, all-cluster event (re-lock, re-sync,
fresh runs only — resuming across a harbor bump breaks the job lock).

## Run an eval

```bash
# 1) snapshot a candidate (base = ../micro-swe-agent; edit the snapshot to mutate it)
DIR=$(python -m harness.scaffold materialize --run-id demo --candidate-id 0)

# 2) evaluate that snapshot on TB2 (smoke = one cheap task)
MODEL=openai/gpt-5 MINI_FORK_LOCAL="$DIR" \
  bash scripts/eval/eval_terminal_bench_2.sh harness.harbor_run:AgentHarness smoke 1 4
```
`harbor_run` uploads the snapshot into each Modal sandbox and `uv tool install`s it, then
runs `harbor run -d terminal-bench@2.0 -m $MODEL -e modal …` with Modal sandbox caps
(2h lifetime / 1h idle) so a hung sandbox can't bill for harbor's 24h default. Knobs:
`MINI_FORK_LOCAL` (snapshot dir), `MODEL`, `AGENT_TEMPERATURE`, `N_CONCURRENT`,
`N_ATTEMPTS`, `TASK_SET`, `HARBOR_TIMEOUT_SECONDS`, `MODAL_APP_NAME`,
`MODAL_SANDBOX_TIMEOUT_SEC`, `MODAL_SANDBOX_IDLE_TIMEOUT_SEC`.

## Submit on a cluster

The wrappers set scheduler directives + cluster env, then call the eval. Both are driven
by the same env vars (incl. `MINI_FORK_LOCAL`), so the only difference is the submit command:

```bash
# Slurm
sbatch --export=ALL,MINI_FORK_LOCAL="$DIR",TASK_SET=full,N_CONCURRENT=16,MODEL=openai/gpt-5 \
  scripts/hpc/submit_slurm.sbatch

# PBS: qsub takes NO positional args -> pass everything via -v, export env with -V
qsub -V -P <group> -q <queue> -l select=1 \
  -v MINI_FORK_LOCAL="$DIR",TASK_SET=full,N_CONCURRENT=16,MODEL=openai/gpt-5 \
  scripts/hpc/submit_pbs.sh
```

## The meta-loop

Each iteration: `materialize(run_id, cand_id, mutate=...)` writes an immutable snapshot,
the `mutate` callback edits it (rewrite `src/micro_swe/config/mini.yaml`, `agents/`, or
`tools/`), then the eval scores it via `harbor_run`. Distinct `run_<id>` / `candidate_<id>`
trees keep concurrent runs collision-free, and each snapshot is a reproducible artifact —
no git commits per candidate. The scaffold logic lives in `../micro-swe-agent/`, never in
`harness/`.

## Model source: API vs self-hosted

- **API model** (default, simplest): set `MODEL=openai/gpt-5` (etc.) + the provider key.
  Modal sandboxes reach the API directly — nothing to serve, no GPU needed, so the cluster
  wrappers only need CPU + network.
- **Self-hosted vLLM**: the agent runs *inside* the Modal sandbox, so `localhost:8000`
  there is NOT your node's vLLM — it needs a **public** endpoint. Set `RELAY=1` and the eval
  stands up `harness/relay` (a Modal reverse-relay): it deploys a public Modal app + runs a
  node-side bridge that proxies to your local vLLM, points `LITELLM_PROXY_API_BASE` at it,
  and tears it down on exit. Requires `modal setup` + a running vLLM.
  ```bash
  # serve vLLM on the node first (e.g. via scripts/build/build_vllm.sh cuda|rocm), then:
  RELAY=1 VLLM_LOCAL_URL=http://localhost:8000 \
  MODEL=litellm_proxy/Qwen/Qwen3.8-27B MINI_FORK_LOCAL="$DIR" \
    bash scripts/eval/eval_terminal_bench_2.sh harness.harbor_run:AgentHarness full 1 16
  ```
