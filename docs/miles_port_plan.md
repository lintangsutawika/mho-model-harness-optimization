# Miles Port Plan — DAPO-Math agentic RL

Handoff plan for porting the existing **SkyRL** DAPO-Math RL pipeline to **Miles**
(radixark's Megatron-LM RL trainer). Written for a fresh Opus agent with no prior
session context. Mirror the SkyRL setup, which **works end-to-end today** (rollouts,
rewards, a full training step).

---

## 0. Environment & how to work

- **Access:** `ssh abci` → ABCI login node (no GPU). Training runs on a GPU node via the
  user's **interactive PBS job** (H200, **~140 GiB/GPU**, 8 GPUs). You cannot SSH into the
  compute node from login4 — hand the user commands to run in their PBS session, or `qsub`.
- **Repo:** `/home/aci18914wh/mho-model-harness-optimization/mho-model-harness-optimization`
  (this is the git repo root; note it's nested one level under a same-named dir).
  Remote is git; branch `restore-executor-pipeline` holds the current work. **Commit only
  when the user asks.**
- **No Docker on ABCI** — singularity/apptainer only. `apptainer build --fakeroot` works on
  the build node (as `build_vllm.sh`/`build_train.sh` already assume).
- **Storage (two roots, from `.env`):**
  - `BASE_DIR` = ephemeral tmp/cache (`/scratch` or `$PBS_LOCALDIR`).
  - `USER_DATA` = persistent (`/data/user_data/lsutawik/…`). Run data/checkpoints/`torch_dist`.
  - Home (`/home/aci18914wh`) is quota-bound — keep big artifacts off it. The SkyRL SIF +
    `sif_cache` currently live at `/home/aci18914wh/mho-model-harness-optimization/` though.
- **Editing pattern:** edit files locally in scratch, `scp` to the repo; verify with
  `bash -n` (shell) / `python3 -c 'import ast; ast.parse(...)'`; verify imports by running
  inside the SIF with `singularity exec`. `$VAR` set locally does NOT cross into
  `ssh abci "…$VAR…"` — inline literal paths on the remote or set the var inside the quotes.
- **The user is often editing files concurrently.** Re-read before assuming state.

---

## 1. What exists today (SkyRL) — READ THESE, they are the templates

| File | Role |
|---|---|
| `scripts/train/train_math_dapo.sh` | The launch script. Mirror it for Miles. |
| `src/train_entrypoint.py` | `HarborExp(BasePPOExp)`, loads config, wires `HarborGenerator`. SkyRL-specific. |
| `src/mho/generator.py` | `HarborGenerator(GeneratorInterface)` — SkyRL rollout. Does NOT port directly (Miles has no such subclass). |
| `src/mho/dataset.py` | `HarborTaskDataset`. |
| `tasks/dapo_math_17k/prepare.py` | Data prep → harbor task dirs. **Reuse as-is.** |
| `tasks/dapo_math_17k/trial_config.yaml` | Harbor trial config (agent/verifier/env). **Reuse.** |
| `tasks/dapo_math_17k/agent_sandbox.def` | Sandbox image recipe (python3.11 + harbor server venv + math tools). **Reuse.** |
| `src/harness/agent_harness.py` | `AgentHarness(MiniSweAgent)` — the harbor agent. **Reuse** (import path `harness.agent_harness:AgentHarness`). |
| `src/harness/math_verifier.py` | `MathVerifier` — grades answers. **Reuse** (`harness.math_verifier:MathVerifier`). |
| `src/mho/backends/local_singularity/environment.py` | `WritableSingularityEnvironment` — the harbor singularity backend (extract-to-writable-dir + argv rewrite + /dev,resolv.conf binds + per-image extraction lock). **Reuse.** |
| `src/mho/backends/local_singularity/service.py` | Host FastAPI executor: runs harbor `Trial` on the host, returns `TrialResult` JSON. **Reuse.** |
| `src/mho/backends/local_singularity/client.py` | HTTP client (`executor_enabled`, `run_trial_async`). **Reuse.** |
| `scripts/build/build_train.sh` | Builds the fat SkyRL SIF. Template for `build-miles.sh`. |

### The external-executor architecture (critical — reuse wholesale)
On ABCI you **cannot** nest apptainer inside the training SIF, and there's no Docker/cloud.
So harbor trials run in a **separate host process** (`local_singularity/service.py`, a
FastAPI service on `127.0.0.1:8900`) that runs the full harbor `Trial` (agent + verifier +
sandbox) via `WritableSingularityEnvironment`. The trainer's rollout code POSTs trial
configs to it over HTTP (`client.run_trial_async`). This exists, is debugged, and is the
same on Miles — **only the trainer-side caller changes** (SkyRL generator → a Miles rollout
function).

Runtime pieces that already exist and are reused verbatim:
- Fat sandbox image `python_3.11-slim.sif` in `sif_cache` (built from `agent_sandbox.def`;
  has `/usr/bin/python3`, `/opt/harbor-server` venv w/ uvicorn+fastapi, math tools).
- Host executor venv `/home/aci18914wh/executor_env/.venv` (harbor + fastapi + uvicorn +
  httpx + **loguru + sympy** — the last two needed by the verifier).
- `.env`: `EXECUTOR_URL=http://127.0.0.1:8900`, `HARBOR_ENV_TYPE=singularity`, `BASE_SIF`,
  `SINGULARITY_DOCKER_*`, `WANDB_API_KEY`, storage roots.

### Hard-won config knobs already in `train_math_dapo.sh` (carry over)
- Executor **auto-start block** (starts `mho.backends.local_singularity.service` when
  `HARBOR_ENV_TYPE=singularity`; health-check; abort on failure). Reuse verbatim.
- `EVAL_BATCH_SIZE` = **number of eval instances** (val dir keyed `test_${EVAL_BATCH_SIZE}`).
- `EVAL_BEFORE_TRAIN` gate.
- `DUMP_DATA_BATCH=true` → dumps training rollouts.
- `MAX_MODEL_LEN=32768` — caps each step's (context+gen) sequence at 32 K (prevents the
  50 K-token runaway that OOM'd the policy backward). **Keep for Miles.**
- Non-colocated placement: 4 policy GPUs + 4 inference GPUs. `MEGATRON_TP` knob (SkyRL: 1,
  fits with the 32 K cap; raise to 2/4 if the policy backward OOMs → sequence-parallel).
- `--no-project --with datasets` for host-side data prep (avoids resolving the heavy
  pyproject on a mismatched-glibc host).

---

## 2. Miles facts (grounded in the docs, 2026-09)

- **Image:** `radixark/miles:latest` (fat; bundles Megatron-LM at `/root/Megatron-LM`, Miles
  at `/root/miles`). Refresh via `cd /root/miles && git pull && pip install -e . --no-deps`.
- **Model conversion REQUIRED** (unlike SkyRL, which uses Megatron-Bridge `AutoBridge` to
  load HF live). Miles uses Megatron-LM's offline converter:
  ```bash
  MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3-4B)"
  read -ra MODEL_ARGS <<< "$MODEL_ARGS_LINE"
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen3-4B --save /root/models/Qwen3-4B_torch_dist
  ```
  Output is a sharded `torch_dist` Megatron checkpoint you train from.
- **Environments = 3 nested rollout layers**, each a plug-in point (a connector occupies
  **exactly one**):
  - `--custom-agent-function-path` (**innermost**): manages the agent↔env loop; **token
    recording is done by the outer layers**, so this layer stays agnostic. **Harbor plugs in
    HERE.**
  - `--custom-generate-function-path` (middle): wraps the agent fn; owns trajectory + token
    recording (often via SGLang `/generate`); reward stays in Miles core.
  - `--rollout-function-path` (outermost): owns everything — loop, recording, rewards, data
    source, batch orchestration.
  - Reward enters via **`Sample.reward`** (+ RM hooks) into Miles' training pipeline **even
    when the environment self-grades**. Token in/out is lossless, handled by the outer layers.
- **Harbor connector = the agent-function layer.** CRITICAL (per environments doc): **"the
  in-process Harbor path passes `HARBOR_ENV_TYPE` straight to Harbor"**, so mechanically it
  reaches **ANY Harbor backend — including our `singularity` backend.** This means we do
  **NOT** need to write a custom rollout/generate function: Miles' native in-process Harbor
  agent-function already handles token recording (outer layers) and reward (`Sample.reward`).
  We only need to (a) point `HARBOR_ENV_TYPE`/`import_path` at our environment, and (b) solve
  the nested-apptainer problem (below).
- **Sandbox providers are an ORTHOGONAL axis, not a limit.** The provider table (AgentENV,
  Daytona, E2B, Modal) lists only pairings past *training runs* have used — they provision
  task containers *inside* a connector, they don't occupy a rollout layer. Singularity/
  Apptainer isn't listed, but the `HARBOR_ENV_TYPE` passthrough means our singularity backend
  is reachable regardless.
- **THE ONE REAL BLOCKER — nested apptainer.** "In-process" means the Harbor `Trial` (agent +
  verifier + sandbox) runs *inside* the Miles training SIF process. Running our
  `WritableSingularityEnvironment` there = **apptainer-inside-apptainer**, which ABCI forbids
  (the exact thing our external executor was built to avoid in SkyRL). Two ways out:
  1. **(preferred)** Register a thin Harbor **`import_path` environment that forwards the
     `Trial` over HTTP to our host executor** (`local_singularity/service.py` on
     `127.0.0.1:8900`), which runs the real `WritableSingularityEnvironment` on the host — no
     nesting. This slots cleanly into Miles' native connector via `import_path`; the executor,
     agent, verifier, tasks, and data are all reused unchanged. **No custom rollout function.**
     (In SkyRL the trainer-side generator did this HTTP hop; here the hop moves *into* a harbor
     environment so Miles' native connector stays in charge.)
  2. **(fallback)** If the native connector can't be pointed at a custom `import_path`, write a
     minimal `--custom-agent-function-path` that builds the trial config and calls
     `client.run_trial_async` directly (same executor), letting Miles' outer layers record
     tokens and take `Sample.reward`.
- **Launch:** `python scripts/run_qwen3_dense.py --model-name Qwen3-4B` (editable recipe).
- Docs: quick-start `https://miles.radixark.com/docs/getting-started/quick-start`;
  environments `https://miles.radixark.com/docs/user-guide/environments`; harbor
  `https://miles.radixark.com/docs/user-guide/harbor`. Example code (read inside the SIF at
  `/root/miles/examples/experimental/harbor` and `/root/miles/scripts/run_qwen3_dense.py`).

---

## 3. Deltas SkyRL → Miles

| Concern | SkyRL (exists) | Miles (build) |
|---|---|---|
| Base image | `build_train.sh`→`skyrl-megatron.sif` | **`build-miles.sh`**→`miles.sif` (+layer harbor/thin-deps) |
| Model load | Megatron-Bridge `AutoBridge`, live from HF | **offline convert → `torch_dist`** (fold into launch script, idempotent) |
| Rollout | `HarborGenerator(GeneratorInterface)` | **Miles native in-process Harbor agent-function** (`--custom-agent-function-path`) — no custom rollout fn; just `HARBOR_ENV_TYPE` + a forwarding `import_path` |
| Reward | SkyRL trajectory rewards | **`Sample.reward`** — Miles takes it natively from the harbor grade |
| Sandbox backend | `local_singularity` **whole-trial** executor | **refactor to Modal-style per-command**: `RemoteSingularityEnvironment` (in-SIF client) + host **sandbox-lifecycle** manager (reuses `WritableSingularityEnvironment` verbatim) |
| Agent/verifier | run in **host executor** venv | run **in-process in the Miles SIF** (deps layered into `miles.sif`) — enables native token recording |
| Data | `tasks/dapo_math_17k/prepare.py` + `trial_config.yaml` | **same** |
| Launch | `train_math_dapo.sh` | **`train_math_dapo_miles.sh`** |

---

## 4. Implementation steps (in order)

### Step 1 — `scripts/build/build-miles.sh`
Mirror `build_train.sh`. Pull the Miles image to a SIF and layer what the Harbor path needs.
```
BASE:  apptainer pull miles-base.sif docker://radixark/miles:latest   (creds in .env if needed)
DEF (localimage From base):  %post →
   cd /root/miles && git pull && pip install -e . --no-deps         # refresh Miles
   # layer harbor + our thin deps so the Harbor rollout + tasks import in-process:
   uv pip install (or pip) "harbor[modal] @ git+…@<HARBOR_REV>" terminal-bench==0.2.18 \
       litellm python-dotenv pyyaml sympy loguru
   # perms fix if the base hides its python under a non-world-readable home (see agent_sandbox/
   #   build_train.sh chmod trick) — verify with a non-root `singularity exec` import test.
OUT:   $(dirname BASE_SIF)/miles.sif
```
Verify: `singularity exec miles.sif python -c "import megatron; import <miles pkg>; print('ok')"`
and that `tools/convert_hf_to_torch_dist.py` + `miles/utils/external_utils/model_args_utils.py`
exist inside. **Open Q:** does the base image already contain `harbor`? If yes, skip that layer.

### Step 2 — Data prep (reuse)
No change. `tasks/dapo_math_17k/prepare.py` writes harbor task dirs; launch script calls it with
`uv run --no-project --with datasets …` (host side). Confirm Miles' Harbor connector consumes the
same task-dir layout (`instruction.md`, `task.toml`, `tests/`) — the docs say Harbor tasks are 4
files (`instruction.md`, `Dockerfile`, `test.sh`, `task.toml`); our tasks omit `Dockerfile` and use
`task.toml` + a stub `tests/test.sh` (grading is via `MathVerifier`, not `test.sh`). **Verify** the
Miles Harbor loader tolerates that (SkyRL's harbor did).

### Step 3 — Environment hookup (THE CRUX): refactor the backend to mirror Modal
> **STATUS: host-side IMPLEMENTED + VERIFIED (2026-09-10).** `remote_env.py`,
> `service.py` (`/sandbox` + `DELETE /sandbox/{id}` added; `/trial` kept for SkyRL),
> and `client.py` (`provision_sandbox`/`stop_sandbox`) are written and deployed. On the
> host: `POST /sandbox` launched a real container (writable extraction fired), `exec`
> over the returned port ran `python3` in-container (rc=0), `DELETE` tore it down. The
> only untested leg is the in-SIF `RemoteSingularityEnvironment` talking to the host
> manager from *inside* a SIF — but that is pure HTTP over the shared net namespace, and
> the container side is proven. **Key correctness finding:** the writable-sandbox
> extraction only fires when `docker_image` is a **docker ref** (e.g. `python:3.11-slim`),
> because harbor's `start()` skips `_convert_docker_to_sif` for a prebuilt `.sif` path
> (`_is_sif_image`). The task config already uses the docker ref (converter finds the
> pre-seeded `python_3.11-slim.sif` in the cache dir) — keep it that way. **Cross-boundary
> gotcha for Step 4:** bind-mount `source` paths + `singularity_image_cache_dir` in the
> config are resolved by the *host manager*, so they must be host-valid paths (the manager
> already overrides the cache dir from its own `SIF_IMAGE_CACHE_DIR`; the launch script
> must pass host paths, bound identically into the SIF, for any mount sources — the baked
> `/opt/harbor-server` venv means no `/opt` mount is needed at all).

Goal: use Miles' **native in-process Harbor connector** (the agent-function layer) so Miles does
token recording + `Sample.reward` for us. Because Miles passes `HARBOR_ENV_TYPE` straight through
to Harbor, the connector reaches any harbor environment, including ours. We do **NOT** write a
rollout/generate function. The one problem is nesting: the connector runs the harbor `Trial`
in-process (inside the Miles SIF), so a stock singularity env would spawn apptainer-in-apptainer.
Fix by making our backend a **Modal-style per-command sandbox provider**.

**Key finding (verified in harbor source):** harbor's stock `SingularityEnvironment`
(`harbor/environments/singularity/singularity.py`) is **already** built exactly like the Modal
env — a client/server split:
- `start()`/`_start_server()` launches a FastAPI server *inside* the container
  (`singularity exec … server.py --port`) and reserves a host port. **This is the ONLY line
  that spawns apptainer locally (singularity.py ~L401) — the sole nesting-prone step.**
- `exec()` sends each command over HTTP to `http://localhost:{_server_port}/exec`
  (singularity.py ~L749). `upload_file`/`download_file` use a bind-mounted staging dir + `exec`.
- The in-container server binds **`127.0.0.1:{port}`** (`server.py` ~L424), and singularity
  **shares the host network namespace** by default (no `--net`). ⇒ a container launched on the
  **host** is reachable at `localhost:{port}` from *inside* the Miles SIF. The exec HTTP hop
  crosses the SIF boundary for free — exactly how Modal's remote container is reached, but local.

So the refactor is small and surgical (provisioning moves to the host; interaction is inherited):

1. **`RemoteSingularityEnvironment(SingularityEnvironment)`** — new, runs **inside the Miles
   SIF** (so the agent runs in-process ⇒ Miles records tokens natively). File:
   `src/mho/backends/local_singularity/remote_env.py`.
   - Override **`start()`**: instead of spawning singularity locally, `POST /sandbox` to the host
     manager with `{sif_path, workdir, mounts, env_files_dir, no_mount, memory, session_id}`; it
     launches the container on the host and returns `{sandbox_id, server_port, staging_dir}`. Set
     `self._server_port`, `self._staging_dir` (the returned shared path), `self._http_client`,
     then call the inherited `_upload_environment_dir_after_start()`.
   - Override **`stop()`**: `DELETE /sandbox/{id}` → manager tears down on the host.
   - **Inherit `exec` / `upload_file` / `download_file` / `upload_dir` / `download_dir`
     UNCHANGED** — they already hit `localhost:{port}` + staging.
   - Select it via `HARBOR_ENV_TYPE=singularity` +
     `import_path=mho.backends.local_singularity.remote_env:RemoteSingularityEnvironment`
     (harbor's `EnvironmentFactory.create_environment_from_import_path` honors `import_path` and
     passes `environment.kwargs` through — confirmed in `factory.py`).
2. **Host manager** — refactor `service.py` from `/trial` (whole trial) to a **sandbox lifecycle**
   service: `POST /sandbox` (provision) + `DELETE /sandbox/{id}` (teardown) + `/health`.
   Internally it instantiates our **existing `WritableSingularityEnvironment` on the host**, calls
   `.start()`, keeps it in a registry keyed by `sandbox_id`, and returns
   `{sandbox_id, _server_port, _staging_dir}`. **All our writable-sandbox logic is reused
   verbatim** (extract-to-dir, `--writable` argv rewrite, `/dev`+resolv.conf binds, per-sif
   extraction lock) — it just changes role from "runs the trial" to "provisions the sandbox."
   `client.py` gains `provision_sandbox()` / `stop_sandbox()` (replacing `run_trial_async`).
3. **Staging** must be a host path both sides can see: create it under `BASE_DIR` (already bound
   into the Miles SIF *and* bind-mounted into the sandbox container at `/staging`). The in-SIF
   client writes upload files there; the container reads them. Verify the bind is present on both.
4. `PYTHONPATH` includes `src`, **`src` forced to the front of `sys.path`** to beat the repo-root
   `harness/` shadow (reuse the exact trick in `service.py`) — needed **in the Miles process now**,
   since the agent + verifier run there.

**Tradeoff (accepted):** the agent (`harness.agent_harness:AgentHarness`) and verifier
(`harness.math_verifier:MathVerifier`) now run **in-process in the Miles SIF**, not the host venv
— so their deps (mini-swe-agent/litellm, loguru, sympy, pyyaml) must be layered into `miles.sif`
(Step 1). In exchange we get Miles' native lossless token recording (the whole point of the native
connector) and drop the per-trial reward-reconstruction that SkyRL needed.

**De-risking:** `RemoteSingularityEnvironment` + the sandbox-lifecycle manager are testable under
the **existing SkyRL pipeline** before miles.sif exists — swap the trial `import_path` to
`RemoteSingularityEnvironment` and run the agent in the generator process instead of shipping the
whole trial to the executor. Validate exec/upload/reward there first, then reuse unchanged in Miles.

**Fallback** (only if Miles' native connector turns out NOT to honor a harbor `import_path`):
write a minimal `--custom-agent-function-path` that drives `RemoteSingularityEnvironment` directly.
Still no whole-trial executor.

### Step 4 — `scripts/train/train_math_dapo_miles.sh`
Start from `train_math_dapo.sh`. Keep: `source .env`, storage split, host data prep, the
**sandbox-manager auto-start block** (was the executor auto-start; now launches the
`/sandbox`-lifecycle service — same pkill/health-check/abort pattern),
`EVAL_BATCH_SIZE`/`EVAL_BEFORE_TRAIN`, `MAX_MODEL_LEN=32768`, non-colocated placement, wandb.
Change:
- **Add idempotent model conversion** (per the user's decision, inside this script):
  ```bash
  HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-4B}"     # ensure present in HF cache first
  MCORE_CKPT="${MCORE_CKPT:-${USER_DATA}/mho-.../models/Qwen3.5-4B_torch_dist}"
  MILES_MODEL_KEY="${MILES_MODEL_KEY:-qwen3-4B}"   # key for model_args_utils.py — VERIFY it supports Qwen3.5
  if [ ! -d "$MCORE_CKPT" ]; then
    singularity exec --nv --bind ... "$MILES_SIF" bash -lc '
      MA="$(python3 /root/miles/miles/utils/external_utils/model_args_utils.py '"$MILES_MODEL_KEY"')"
      read -ra MA <<< "$MA"
      PYTHONPATH=/root/Megatron-LM python /root/miles/tools/convert_hf_to_torch_dist.py \
        "${MA[@]}" --hf-checkpoint '"$HF_HF_PATH"' --save '"$MCORE_CKPT"' '
  fi
  ```
  (Resolve exact in-SIF paths for the two tools by inspecting the SIF; download the HF model to
  `HF_DIR` first, or point `--hf-checkpoint` at an existing local HF dir.)
- **Launch** via Miles (adapt `run_qwen3_dense.py`), passing: the `torch_dist` model, TP/PP,
  the rollout-function-path flag (step 3), the harbor task dir(s), `MAX_MODEL_LEN`, eval knobs,
  export/dump paths, wandb. Run it inside `singularity exec "$MILES_SIF"` with `--nv` and the same
  binds as SkyRL (`BASE_DIR`, `USER_DATA`, HF cache, tmp, `PYTHONPATH=src`).

---

## 5. Verification (mirror how SkyRL was validated)
1. `build-miles.sh` → `miles.sif`; `singularity exec` import check (megatron, miles, harbor, and
   `import mho.backends.local_singularity.remote_env` — or `miles_agent_fn` for the fallback).
2. Conversion produces `torch_dist/` (non-empty, has `*.distcp`/metadata).
3. Executor auto-starts; `curl 127.0.0.1:8900/health` ok; startup self-check logs
   `harness.agent_harness importable`.
4. One rollout completes with a real reward (not `stop_reason=error`); check the executor log
   (`/home/aci18914wh/executor_svc.log`) and a `trials/<name>/result.json`.
5. Eval-before-train prints a score; **first training step** completes (watch for CUDA OOM in the
   policy backward → raise `MEGATRON_TP` (2/4) or lower `MAX_MODEL_LEN`; the 32 K cap + TP=1 fit on
   140 GB in SkyRL).

---

## 6. Open questions / risks (resolve during implementation)
- **Qwen3.5 support in Miles `model_args_utils.py`.** The doc example is `qwen3-4B`. Qwen3.5-4B is a
  **VL** model; SkyRL used `language_model_only=true` to load only the LLM backbone. Verify Miles has
  a matching model-args key and a text-only path; may need to add a model-args entry or use the base
  Qwen3-4B. **This could be the biggest blocker.**
- **Native connector `import_path` support** — the whole "preferred" path (2a) hinges on Miles'
  in-process Harbor connector honoring a harbor `import_path` (so we can point it at our
  forwarding env). If it only accepts the built-in cloud providers, fall back to the custom
  agent-function (2b). Confirm by reading `examples/experimental/harbor` + grepping the source.
- **Where `HARBOR_ENV_TYPE` is read + where the grade → `Sample.reward`** — pin these exact call
  sites so the forwarding env returns the shape Miles expects.
- **Harbor task format** — Miles expects `instruction.md/Dockerfile/test.sh/task.toml`; ours has no
  Dockerfile and uses a stub test. Confirm the loader path we use (custom rollout → our executor →
  our `WritableSingularityEnvironment`) bypasses Miles' own task-format assumptions (it should, since
  we run harbor `Trial` ourselves in `service.py`).
- **Weight sync (Megatron→inference).** Miles has its own policy→inference sync; confirm which
  inference engine Miles uses (vLLM/SGLang) and that non-colocated placement is supported, and that
  `api_base` is exposed to the rollout the way SkyRL did.
- **Image contents** — does `radixark/miles:latest` include `harbor`, `vllm`, and a world-readable
  python (non-root exec on ABCI)? Apply the perms chmod from `build_train.sh` if not.
- **`torch_dist` TP/PP layout** — baked at conversion. If you change `MEGATRON_TP`, you may need to
  re-convert (or rely on torch_dist reshard-on-load). Verify.

---

## 7. Reference commands (SkyRL, to mirror)
```bash
# build fat SIF (template):        bash scripts/build/build_train.sh
# executor (auto-started by the launch script now); manual form:
PYTHONPATH="$PWD/src" EXECUTOR_PORT=8900 nohup \
  /home/aci18914wh/executor_env/.venv/bin/python -m mho.backends.local_singularity.service \
  > /home/aci18914wh/executor_svc.log 2>&1 &
# run training:                    export BASE_SIF=…/skyrl-megatron.sif; bash scripts/train/train_math_dapo.sh
# fast smoke test:                 EVAL_BEFORE_TRAIN=false EVAL_INTERVAL=9999 bash scripts/train/train_math_dapo.sh
```
Read `train_math_dapo.sh` end-to-end before writing the Miles launch script — it encodes every
lesson from the SkyRL bring-up (binds, DNS/dev, MSWEA_API_KEY, storage, OOM cap).
