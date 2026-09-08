# notes — mho-model-harness-optimization (progress + how to resume)

Working notes so a new session can pick up. The README is the user-facing doc; this is the
"what state are we actually in / what's next" scratch file.

## Goal (one line)

Meta-loop that continually mutates an **agent scaffold** (`micro-swe-agent`) and scores each
candidate on **Terminal-Bench 2.0**, run through **harbor** on the **Modal** environment.

## Repo layout (this checkout)

Sibling checkouts live under `/data/user_data/lsutawik/mho-model-harness-optimization/`:
- `mho-model-harness-optimization/` — THIS repo (the meta-harness + eval + submit + relay)
- `micro-swe-agent/` — the scaffold being optimized (standalone installable package)
- `meta-harness/` — stanford-iris-lab reference (read-only; TB2 reference examples/prompts)
- `mini-swe-agent/` — upstream SWE-agent/mini-swe-agent checkout (reference)

This repo:
```
harness/
  scaffold.py        # materialize(run_id, cand_id, mutate=..) -> runs/run_<id>/candidate_<id>/micro-swe-agent/
  harbor_run.py      # AgentHarness(MiniSweAgent): overrides install() to upload+install the snapshot
  relay/             # Modal reverse-relay (modal_relay.py public app + bridge.py node-side dialer)
scripts/
  eval/eval_terminal_bench_2.sh   # CORE: harbor run on TB2, Modal env
  hpc/submit_slurm.sbatch         # babel/hpcfund
  hpc/submit_pbs.sh               # ABCI/qsub  (qsub takes NO positional args -> pass via -v, env via -V)
  build/build_vllm.sh             # vllm-cuda / vllm-rocm sif (self-hosted model path only)
src/                 # meta-harness reference checkout copied in (read-only reference)
pyproject.toml, uv.lock
```

## Key design decisions (why things are the way they are)

- **Snapshot, not git, for candidates.** `scaffold.materialize()` copies the base scaffold to
  an immutable `runs/run_<id>/candidate_<id>/micro-swe-agent/`, optional `mutate(dst)` edits it
  in place, harbor uploads+`uv tool install`s that dir. No commit churn, collision-free across runs.
  `MICRO_SCAFFOLD_BASE` env picks the base checkout (cluster layouts differ — don't assume sibling).
- **harbor_run.AgentHarness** subclasses harbor's `MiniSweAgent` and overrides ONLY `install()`:
  reads snapshot from `mini_fork_local` agent-kwarg or `MICRO_SCAFFOLD_DIR` env, `upload_dir` →
  `/tmp/micro-scaffold`, `uv tool install /tmp/micro-scaffold --with litellm orjson fastapi`.
  Everything else (run loop, ATIF parsing) inherited.
- **micro-swe-agent is fully self-contained** now: its own installable package providing the
  `mini-swe-agent` console entrypoint; NO harbor dep, NO mini-swe-agent dep. Vendored config/
  protocols/utils/tools/exceptions. Claude-specific code (thinking-block reorder, cache-control,
  multimodal) removed — it's a non-Claude litellm bash-tool-calling agent. 27 tests pass.
- **Model source: API vs self-hosted.** API model (`MODEL=openai/gpt-5` + key) is simplest —
  Modal sandboxes reach the API directly. Self-hosted vLLM needs `RELAY=1`: agent runs INSIDE
  the Modal sandbox so `localhost:8000` isn't the node — the relay deploys a public Modal app +
  node-side bridge that proxies to local vLLM, points `LITELLM_PROXY_API_BASE` at it, tears down on exit.
- **litellm_proxy prefix, not openai/.** `litellm_proxy/` is passthrough; `openai/` routes
  reasoning+tools to `/v1/responses` which vLLM doesn't implement. Reads `LITELLM_PROXY_API_BASE/_KEY`.
- **Modal sandbox cost caps.** harbor defaults to 24h sandbox lifetime + no idle kill. Eval sets
  `MODAL_SANDBOX_TIMEOUT_SEC=7200` (2h) + `MODAL_SANDBOX_IDLE_TIMEOUT_SEC=3600` (1h).

## Dependency pins (pyproject.toml) — DO NOT drift casually

- `harbor[modal]` pinned to git rev `c178c20710c362ef806c5d5d18852f95b21ca34b` (validated to expose
  the mini_swe_agent adapter, terminus_2, custom `--agent module:Class`, `--ek`). A harbor bump =
  deliberate all-cluster event: re-lock, re-sync, FRESH runs only (resume breaks the job lock).
- `terminal-bench==0.2.18` from **PyPI**. The git repo is NOT pip-installable anymore (no pyproject
  at main/v3/v4) — an earlier git-source pin (rev 1a6ffa9) broke `uv sync`; removed.
- `mini-swe-agent[full]` — `[full]` (not `[modal]`) pulls **swe-rex**, which mini's `swerex_modal`
  env path needs ("Unknown environment type: swerex_modal" if missing).

## TB2 run mechanics learned

- Dataset spec is `-d terminal-bench@2.0` (harbor resolves it; the pip package is authoring-side).
- Each task's `task.toml` `[agent] timeout_sec` is the CANONICAL per-task limit (e.g. 900s). The
  `AgentTimeoutError: ... 900 seconds` is that, not a harness default. `AGENT_TIMEOUT_MULTIPLIER`
  scales it, BUT multiplying makes results non-leaderboard-comparable — first check raw inference
  latency + trajectory (slow-but-progressing vs stuck) before bumping.

## Prompt landscape (for choosing candidate mutations)

Current `micro-swe-agent/src/micro_swe/config/mini.yaml` is stock mini: **SWE-bench framed**
("solve this issue / edit the source code"), one bash command per turn, no per-turn structure.
Other TB scaffolds seen:
- **Terminus 2** (harbor reference): "command-line tasks in a Linux env", XML `<response>` with
  `<analysis>/<plan>/<commands>` and tmux `<keystrokes duration=..>` batches.
- **KIRA** (meta-harness): Terminus framing + the TB-specific "verify **minimal state changes** /
  no side effects / no extra files" insight (TB verifiers check exact end-state).
- **terminal-bench-rl** (top Qwen3): mandatory 5-phase flow (Plan→Explore→Refine→Execute→Verify),
  YAML actions with bash/todo/file/search tools, verify-before-done.

**Cheapest high-value candidate_1 (prompt-only, no scaffold change):** reframe mini.yaml to
"complete this command-line task" + add KIRA minimal-state-change + rl plan/verify discipline.
Bigger swings (batched keystrokes, todo/file/search tools) are scaffold mutations for later.

## Status / where we are

- Repo scaffolding, eval script, submit wrappers, relay, and self-contained micro-swe-agent are
  all in place and validated at the unit/lock level. `uv lock` succeeds (203 pkgs).
- micro-swe-agent: 27 tests pass, standalone install validated, wheel ships config/mini.yaml.
- **Not yet done:** a first clean live TB2 eval run end-to-end (was hitting timeouts + relay setup).
- Nothing in this repo is committed yet beyond "Initial commit" (fbb6c75). `harness/`, `scripts/`,
  `src/`, `pyproject.toml`, `uv.lock` are untracked; `.gitignore`, `README.md` modified.

## TODO / next session

1. Run one live TB2 smoke end-to-end and debug (start API-model path to remove the relay variable):
   ```bash
   DIR=$(python -m harness.scaffold materialize --run-id demo --candidate-id 0)
   MODEL=openai/gpt-5 MINI_FORK_LOCAL="$DIR" \
     bash scripts/eval/eval_terminal_bench_2.sh harness.harbor_run:AgentHarness smoke 1 4
   ```
   Pick a real cheap TB2 task id for a `SMOKE_TASK` default.
2. Then test the self-hosted `RELAY=1` path against a served vLLM.
3. Write candidate_1 (terminal-framed mini.yaml) and A/B vs stock baseline.
4. Commit both repos (micro-swe-agent standalone package; this repo's harness+scripts+relay+pyproject).
5. Propagate to ABCI: pyproject.toml+uv.lock, harness/scaffold.py (MICRO_SCAFFOLD_BASE), relay,
   eval script; ensure micro-swe-agent checkout present there (rsync or commit+push).


RELAY=1 VLLM_LOCAL_URL=http://localhost:8000 \
MODEL=litellm_proxy/Qwen/Qwen3.8-27B MINI_FORK_LOCAL="$DIR" \
bash scripts/eval/eval_terminal_bench_2.sh harness.harbor_run:AgentHarness full 1 1 -i extract-elf
