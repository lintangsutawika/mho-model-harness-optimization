# Training on platoon through miles-tinker (`harnessforge.train`)

Workstream E. GRPO on Harbor tasks (tmax) with our micro-swe-agent harness. **platoon** is the client: it samples
tasks, runs the harness in hpc-sandbox pods, scores the rollouts, builds the datums and calls the Tinker API.
**miles-tinker** is the Tinker API server: it trains (Megatron, full fine-tuning) and samples on SGLang engines
whose weights it keeps current.

```
 gateway job (PBS, 1 + R rt_HF nodes, one Ray cluster)            driver (rt_HC job, or the gateway's head node)
 +---------------- head node -----------------+  +- node 1..R -+   platoon train  --- Tinker HTTP --->  :<port>
 | serve_tinker.py :<port>  (Tinker API)      |  | 8 SGLang    |     gateway per rollout (renders Qwen3.5,
 | Megatron actor, 8 GPUs (TP2 x DP4)         |  | engines TP1 |     parses qwen3_coder tool calls)
 | miles router -> engines                    |  | (miles-     |     harbor env: micro-swe-agent in
 +--------------------------------------------+  |  managed)   |     hpc-sandbox pods on THIS node
        | disk-delta: weight_vNNNNNN/ (zstd XOR)  +-------------+   HF exporter (replays the deltas)
        v                                                                  |
   <run>/weights/delta  ------------------------------------------------>  <run>/hf/step_<N>, latest.json
```

Run directory (`<runs_root>/<run_id>`, on `/groups`): `run.json` (the resolved config every job reads),
`gateway.json`, `driver.json`, `jobs.json`, `logs/`, `tinker/` (Tinker checkpoints), `weights/delta/`
(disk-delta stream), `hf/` (HF exports), `platoon/` (platoon's run: metrics, rollouts, checkpoints.jsonl),
`dataset/` (the materialized tasks), `harness/` (the rendered harness spec), `STOP` (ask everything to stop).

## Commands

```bash
cd ~/hf-dev/train/harnessforge        # this package on ABCI; host python 3.12, stdlib only
export PYTHONPATH=$PWD/src PY=~/.local/bin/python3.12
$PY -m harnessforge.train render --config configs/train/smoke.json     # resolved config, job specs, serve flags
$PY -m harnessforge.train launch --config configs/train/smoke.json     # submit gateway (+ driver) jobs
$PY -m harnessforge.train status --run-dir /groups/.../runs/<run_id>   # jobs, gateway/driver state, versions, exports
$PY -m harnessforge.train stop   --run-dir /groups/.../runs/<run_id>   # STOP, wait, then qdel what is left
$PY -m harnessforge.train export --run-dir ... [--follow]             # (in the miles SIF) HF from the deltas
$PY -m harnessforge.train unlend --run-dir /groups/.../runs/<run_id>   # external mode: return lent pool nodes
$PY -m harnessforge.train mapping                                       # the old-run mapping table below
```

`launch` only writes files and calls `qsub`, so it may run on the login node. Configs (`configs/train/`, JSON with
`extends`): `abci-base.json` (our paths and the old run's hyperparameters), `smoke.json`, `naive.json` (the full
comparison run), `pass_rate.json` (the Thompson-sampling variant; not to be run yet).

## One-time setup on ABCI

All of this runs in CPU jobs (no compute on the login node), and nothing is installed into the image.

1. **miles-tinker source.** `git archive` of neulab/miles-tinker at 25b6a76, unpacked to
   `~/hf-dev/train/miles-tinker` and bound over `/root/miles` in `miles_latest.sif`. Then apply the two patches
   in `src/harnessforge/train/patches/` (`patch -p1` in the tree; both are to upstream):
   - `miles-tinker-disk-delta-cast.patch`: see below.
   - `miles-tinker-max-tokens-per-datum.patch`: adds `--tinker-max-tokens-per-datum`. `GatewayConfig`
     hard-codes 32768, and the first full-size `forward_backward` refused a 43k-token datum. The gateway passes
     `train.context_window_length` (65536).
   For external rollout mode, a second tree, `~/hf-dev/train/miles-tinker-ext-run`: `git archive` of
   `external-engines-fixes` (4ce94830) with the same two patches applied. `COMMIT` in the tree records this.
   The managed tree is left as is.
2. **Tinker SDK for the server.** The SDK goes into `~/hf-dev/train/extra-site`, which is on the container's
   `PYTHONPATH`:
   ```bash
   singularity exec miles_latest.sif python -m pip install --no-deps --target ~/hf-dev/train/extra-site \
       "tinker==0.27.1" "pyqwest>=0.4.1"
   ```
   Install with `--no-deps`: with its dependencies, the SDK's opentelemetry shadows the image's and `ray start`
   fails. The image has everything else the SDK needs.
3. **Base model.** The HF snapshot is in `~/.cache/huggingface`. `--ref-load` is the old run's torch_dist
   conversion of the same snapshot, used read-only.
4. **platoon.** The evals venv (`~/hf-dev/evals/venv`: platoon 0.6.0, tinker 0.16.1), with its site profile and the
   `hf-micro-swe-1` toolchain. For training, optionally its patched copy `~/hf-dev/train/platoon-tito-venv`
   (token-in/token-out; see "Token-in/token-out prompts" below).

**The disk-delta patch.**
- *Problem.* Qwen3.5 stores 24 tensors in F32 in the HF checkpoint (`linear_attn.norm.weight`, `A_log`, ...).
  Megatron emits them as BF16. Disk-delta diffs raw bytes against the checkpoint files, so its baseline check
  refuses the run (`dtype=F32 ... trainer emitted dtype=BF16`).
- *Fix.* The patch casts those tensors to the checkpoint's dtype before diffing. The cast is a lossless upcast.
  Engines load F32 there anyway, exactly as from the base checkpoint.
- *Side effect.* Because of this, the first push (platoon's launch checkpoint) is not empty: it carries the BF16
  rounding of those 24 norms.
- *Status.* To upstream to miles-tinker.

## Rollout modes: miles-managed engines or lent pool engines

`rollout.mode` picks where the rollout engines come from.

- **managed** (default; `smoke`, `naive`, `fullcheck`): the gateway job has 1 + R rt_HF nodes. miles launches 8*R
  TP1 SGLang engines and its router on the R rollout nodes. This works with stock miles-tinker 25b6a76 plus our
  two patches.
- **external** (`naive_ext.json`): the gateway job is **one** rt_HF node (the trainer). The engines are borrowed
  from a running harnessforge pool and attached with `--rollout-external --rollout-external-engine-addrs`.
  - This needs miles-tinker branch `external-engines-fixes` (4ce94830 = 09995e57 attach + one fix,
    draft neulab/miles-tinker#21) plus our two patches, deployed as `~/hf-dev/train/miles-tinker-ext-run`.
  - Stock 25b6a76 raises `NotImplementedError` for `--rollout-external`.

### External mode (`harnessforge.train.external`)

```json
"rollout": {"mode": "external",
            "pool_spec": "/home/aci18914wh/hf-dev/pool/scripts/pool-n4hf72.json",
            "lend_nodes": [1, 2, 3],
            "engine_local_dir": "/dev/shm/{user}-hftrain-{run_id}/engine-local",
            "drain_timeout": 1800}
```

`train launch` with this config does the following:

1. **Checks the pool.**
   - The pool's policy path must equal `model.hf_checkpoint`, as a local path. miles refuses hub ids under
     disk-delta, and disk-delta seeds each engine host's local checkpoint from the engine's `--model-path`.
   - None of `lend_nodes` may be lent to another run.
   - Every engine of those nodes must answer `/health` now. pool.json's `status` is not trusted for this: it stays
     `starting` after `pool start --no-wait`.
2. **Lends the nodes.** It calls `SGLangPool.lend(nodes, to=<run_id>, on_restart="latest", policy_file=<run>/engine_restart_policy.json)`.
   - The policy file names the **base** checkpoint (`model.hf_checkpoint`).
   - A lent engine that dies is restarted by its node agent on the base. miles treats the comeback as a new
     generation and replays every published delta from the base.
   - It must never restart on a later HF export: the XOR deltas would be applied twice.
3. **Checks each lent engine** via `/server_info` and `/model_info`.
   - The local `model_path` must equal `model.hf_checkpoint`.
   - `tp_size` must be 1.
   - `weight_version` must be the pool policy's id (or unset), not a previous trainer's weights. With disk-delta
     the first sync only records the baseline, so a stale engine would serve stale weights.
   - These are miles' own attach checks; an engine that fails them never becomes ready, and the trainer would
     wait forever.
   - On any problem the nodes are **returned** (unlend), `rollout.json` says `refused` with the reasons, and
     launch fails.
4. **Writes `<run>/rollout.json`:** `{status: lent, pool_id, nodes, engines: [host:port...], engine_local_dir}`.
5. **Submits one gateway job (1 rt_HF).**
   - Flags: `--rollout-external --rollout-external-engine-addrs <engines>`, disk-delta,
     `--update-weight-local-checkpoint-dir <engine_local_dir>`. No `--rollout-num-gpus`: miles derives it from
     the engine count.
   - If the submission fails, the nodes are returned.

**Getting the nodes back.**
- The gateway job's teardown (`STOP`, `train stop`, the driver ending, or SIGTERM) first stops the server, then
  calls `unlend`. That reloads the pool policy into every live lent engine and verifies `weight_version`.
- If those checks fail, the nodes **stay lent** and `rollout.json` records `unlend_error`.
- After a hard kill, `train stop` (or `python -m harnessforge.train unlend --run-dir <run>`) returns the nodes once
  the gateway job has ended. It refuses while the job still runs.
- `stop --driver-only` never returns them.

**`engine_local_dir`.**
- It must be host-local, writable, and run-unique on every engine host. An earlier run's applied-version marker
  makes the first delta of a new run fail as out of order.
- It needs room for the checkpoint (9.3 GB for Qwen3.5-4B). `/tmp` on ABCI compute nodes holds about 8 GB: the
  first smoke attempts failed there with `ENOSPC`.
- The pool jobs' `/local/<jobid>` differs per node, and the flag takes one path for all hosts. So the default is
  `/dev/shm/<user>-hftrain-<run_id>/engine-local`.
- That is RAM-backed: 9.3 GB per lent host until the pool job ends. Nothing removes it remotely.

**Pool tooling.** lend/unlend only write `pool.json` and make HTTP calls. They run from this package
(`harnessforge_src`) and submit no pool jobs, so the pool tooling's path (`~/hf-dev/pool-lend`, whose agents run
n4hf72) does not matter here. A node restart would; use that checkout for `pool restart-node`.

**Not yet verified on the cluster by this PR.** The miles-side attach (disk-delta to n4hf72 engines) is being
cluster-tested separately. This PR's lend/check/unlend path is tested against fake engines and a real
`SGLangPool`.

## Lending pool nodes to a trainer (`SGLangPool.lend` / `unlend`)

Used by the external rollout mode above, which calls these through `harnessforge.train.external`.

```bash
python -m harnessforge.pool lend   --spec pool.json --node 3 --to <run_id> [--policy-file <run>/hf/latest.json --on-restart latest]
python -m harnessforge.pool unlend --spec pool.json --node 3            # reload the pool policy, back into the router
```

- `pool.json` gets `lent: {"3": {to, since, policy_file, on_restart}}`. The node agents act on it.
- **Router.** Node 0's agent drops the lent engines from the router. No evaluation reaches them.
  - `lend()` returns their URLs once the router no longer lists them and their in-flight requests have drained.
- **load() and wait_ready().** Both skip lent nodes; `load` records them as `skipped_lent_nodes`.
- **Restarts.** A lent engine that dies is never restarted on the pool's policy, which would silently put the
  base model into training.
  - `on_restart=down` (default): it stays down. The node file shows `held: <reason>` and `status` shows `HELD`.
  - `on_restart=latest`: it restarts on the trainer's latest published HF export (`policy_file` →
    `{id, path}`). If there is none, it stays down.
- **unlend.** It first reloads the pool's active policy into every live lent engine, including ones still
  starting, and checks `weight_version` on each.
  - If that fails, the node stays lent and `unlend` raises.
  - Held engines start on the active policy once the node is returned.
- **Agent version.** `lend` refuses nodes whose agent predates the feature (no `lend` in `features`): node 0 and the
  lent nodes. Restart them with the new code first (`restart-node`). The running `n4hf` pool's agents predate it.

## HF export (`harnessforge.train.export`)

miles' disk-delta writes each weight push as `weights/delta/weight_v<NNNNNN>/`: zstd-compressed XOR (or
overwrite) deltas of the changed tensors against the previous version, plus an index with the version, base
version and checksums. Version 0 is `--hf-checkpoint`, and nothing is ever pruned.

The exporter replays the stream on CPU into a working full HF copy:

- it starts from a copy of the base, so config, tokenizer, vision tower and MTP head are kept;
- it applies each version in place and verifies each tensor's xxh3 checksum;
- every `export.every` versions it copies the result to `hf/step_<N>/` (atomic rename, with `HF_EXPORT.json`) and
  points `hf/latest.json` at it. That pointer is what `pool load` takes, and the lent-node restart target.

**Version → step.** platoon pushes:
- once at launch (its launch checkpoint) → v1;
- once per optimizer step → step k = v(k+1);
- once for its final checkpoint, with the same weights.

Exports are therefore `hf/step_<k>` from v(k+1). The final export is named after the last version that changed
weights. `HF_EXPORT.json` records both numbers.

The driver runs the exporter beside platoon, inside the miles SIF (numpy, zstandard, xxhash), and takes a final
snapshot at the end. An apply interrupted midway rebuilds the working copy from the base, because XOR deltas are not
idempotent.

This replaces the old path's `--save-hf` + `complete_hf_export.py`: no Megatron→HF conversion, and nothing missing
from the export. Tinker `save_state` checkpoints (`checkpoint.every`, torch_dist + Adam, ~64 GB each for 4B) are
only for resuming.

## The old naive run → platoon + miles-tinker

Reference: `/groups/.../mho/runs/grpo200-tmax-naive-4node-64k--Qwen3.5-4B`. That was miles `train_async.py` with
in-process harbor trials and hpc-sandbox executors on 4 rt_HF nodes (3 rollout + 1 trainer). Its flags come from
its log (`logs/train-*.out`, the `ray job submit` line and the env).

| old setting | old value | platoon / miles-tinker | status |
|---|---|---|---|
| `--hf-checkpoint` | Qwen/Qwen3.5-4B | `model.name`, `model.hf_checkpoint` (local snapshot) | same |
| `--prompt-data` | tmax-v11-train-naive (6725 tasks) | `data.dataset` | same |
| `--rollout-shuffle` | epoch shuffle | `train.task_sampler: uniform` | mapped |
| `--num-rollout` | 200 | `train.max_training_steps: 200` | same |
| `--rollout-batch-size` × `--n-samples-per-prompt` | 8 × 32 | `train.batch_size: 8`, `workflow_config.group_size: 32` | same |
| `--global-batch-size` | 256 = one optimizer step per rollout | `train.num_minibatches: 1` | same |
| `--lr`, decay | 1e-6 constant | `train.optimizer.learning_rate: 1e-6` (sent with every optim_step) | same |
| Adam β1/β2, weight decay | 0.9 / 0.98 / 0.1 | `train.optimizer` | same |
| `--clip-grad` (Megatron default), Adam eps | 1.0, 1e-8 | `grad_clip_norm: 1.0`, `eps: 1e-8` (miles-tinker's defaults are 0 and 1e-12) | same |
| `--advantage-estimator grpo --disable-grpo-std-normalization` | r − group mean | platoon group advantage (r − mean, no std) | same |
| Dr.GRPO loss reducer | constant divisor | sum loss × 1/(action tokens per microbatch) | **differs**: per-token mean, a scale difference |
| `--eps-clip/--eps-clip-high` | 0.2 / 0.28 | `loss_fn: ppo`, `clip_low_threshold: 0.8`, `clip_high_threshold: 1.28` | mapped |
| `--use-tis` (fully async) | truncated IS vs rollout logprobs | PPO ratio taken against the sampler's logprobs | **differs**: no separate TIS weight |
| `--use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl` | KL to the frozen base | none | **differs**, see below |
| `--entropy-coef` | 0 | none | same |
| zero-std group filter + resampling | drop uniform-reward groups, resample until 8 informative | `group_admission: informative`, `refill.max_groups_per_step: 8` | mapped (capped at 64 groups per step) |
| `--fully-async --max-weight-staleness 4` | async, staleness ≤ 4 | `train.max_staleness: 4` | mapped |
| `--async-max-concurrent-samples` | 256 | `train.max_concurrent_rollouts: 160` | **differs**: limited by the client node's cores (remote sandbox hosts lift it: `sandboxes`, docs/sandboxhost.md) |
| `--rollout-temperature` | 1.0 | `inference_params.temperature: 1.0` (miles-tinker requires 1.0) | same |
| `AGENT_MAX_OUTPUT_TOKENS` | 16384 per call | `max_completion_tokens: 16384`, `sampling_precedence: config` | same |
| context: `AGENT_MAX_INPUT_TOKENS`, `--max-seq-len` | rollouts **uncapped**: SGLang served 262144, the agent kept going past 64k, and only training truncated each sample to 65536 | `naive_ext_64kcap`: `train.context_window_length: 65536` **at rollout**. platoon's gateway refuses an over-window prompt and the trajectory scores 0 (`on_context_overflow: grade`); training datums ≤ 65536 | **differs** (user decision). In the old run, over-64k trajectories were ~45-60 % of rollout time but only 5-10 % of successes (context study, `~/hf-dev/ctxstudy/out/`) |
| agent steps / tool calls | uncapped (micro-swe-agent `step_limit: 0`; harbor's agent timeout only) | **from step 6 of `naive_ext_64kcap`:** `harness.agent_config` = `agent.step_limit: 64`. micro-swe-agent counts model calls, ≈ tool calls (several calls in one response: 0.11 per trajectory). At the limit the agent ends with `LimitsExceeded`, a normal harness exit, and the verifier grades the sandbox | **differs** (user decision, mid-run): matches TMax 4B's "64 maximum tool calls" (allenai/tmax-4b, arXiv 2606.23321 §4.1) |
| prompt rendering | miles TITO session server: the next prompt is the exact previous token stream plus new tokens | **from step 6:** platoon with TITO stitching (`platoon-tito-venv`, PR #65); steps 1-5 re-rendered the history (3-4 datums per trajectory) | mid-run change; replay of steps 1-2: 1.2-1.4 datums per trajectory, about 2.4× fewer training tokens |
| sandbox placement | executors on the 4 GPU nodes, 256 concurrent | **from step 6:** remote sandbox hosts (#63): 6 rt_HC × 96 + 112 local slots on the trainer head node, `max_concurrent_rollouts` 640 | mid-run change (steps 1-5: local only, 160) |
| agent timeout | 1800 s override × 2.0 = 3600 s | `agent_timeout_multiplier: 6` (each task's 600 s × 6) | same |
| `AGENT_TRIAL_TIMEOUT` | 7200 s, scores 0 | `rollout_config.timeout: 7200` | mapped |
| harness | micro-swe-agent base scaffold via harbor's MiniSweAgent | the same dir, through the evals backend's spec (runner and toolchain) | same code |
| chat template / parsing | SGLang `qwen3_coder` + `qwen3` reasoning parser, TITO `qwen35` | platoon `chat_template` renderer, `tool_call_parser: qwen3_coder` | mapped (token in/out kept) |
| sandboxes | hpc-sandbox pods, executors on all 4 nodes | `harnessforge.evals.pods` on the client node, harbor parity shims | mapped |
| rollout engines | 24 GPUs, miles SGLang | `gateway.rollout_nodes: 3` (miles-managed) | same |
| trainer | 1 node, TP2, 65536 tokens/GPU, log-prob chunk 1024, CPU-offloaded precision-aware Adam | same TP, tokens and chunk; Adam on GPU (full mode forbids the precision-aware optimizer) | mapped |
| `--save-interval 10 --save-hf` | HF every 10 steps | `export.every: 10` (from the disk-delta stream) | same |

**Mid-run changes in `platoon-grpo200-tmax-naive-uniform-noKL-64kcap-ext--Qwen3.5-4B`.** One driver-only restart after
step 5 switched on TITO stitching, the 64 tool-call cap and remote sandboxes (W&B tags `tito`, `toolcap64`,
`remote-sbx640`; `labels.toolcap` and `LABEL.txt` in the run dir). platoon keeps groups that had settled before a
restart, so step 6 may still train a few pre-switch rollouts; from step 7 every rollout was sampled under the new
settings. A second driver-only restart (`platoon-tito2-venv`: Qwen3.5 template and end-at-limit changes) is
planned.

### What is not matched, and what it would take

| gap | effect | cost to match | recommendation |
|---|---|---|---|
| **KL to the reference** (0.01, low_var_kl) | Small at lr 1e-6 over 200 steps, but a confound | Server-side in miles-tinker full mode (details below). ~1-2 days + 1 smoke; +20-30% trainer time per step (old run: ref log-probs 36 s of a 190 s train step, against 5-10 min rollout phases, so ~5% wall time) | Run without it now, labelled `no-KL`; add it before any run whose purpose is to reproduce exactly |
| Dr.GRPO divisor vs per-token mean | Changes the effective learning rate, not the gradient direction | Reshape platoon's datum scaling (a platoon objective option) or a miles-tinker `loss_fn_config` divisor; ~0.5 day | Accept; record it |
| TIS | At staleness ≤ 4 the PPO ratio against the sampler's logprobs already corrects off-policy samples; TIS also corrects train/infer mismatch | Add a truncated-IS factor to miles-tinker's `ppo` loss (`loss_fn_config.tis_cap`); ~0.5 day | Accept for now |
| Concurrency 160 vs 256 | Rollout phase up to ~1.6× longer at the start (shorter later as groups refill) | Sandboxes on more nodes: `sandboxes: {mode: remote, ...}` runs pods on sandbox host jobs too, relaying the gateway's unix socket over TCP (docs/sandboxhost.md) | Accept; measure step time |

**KL in miles-tinker full mode, scoped.** The pieces exist:

- The Megatron actor keeps named weight copies in CPU memory (`TensorBackuper`); `load_other_checkpoint("ref",
  args.ref_load)` is what `train.py` uses for `--use-kl-loss`.
- miles' `low_var_kl` is in the loss hub.

What is missing:

1. Full mode does not request the ref copy. Add `with_ref` when a new `--tinker-kl-coef > 0`.
2. In `full/executor.py` `forward_backward`:
   - switch to the ref weights, run a forward-only log-prob pass over the batch's datums, switch back;
   - put the result into the batch as `ref_log_probs`.
   - Gradients accumulate across `forward_backward` calls, so this runs once per call, about one extra forward.
3. In `tinker_losses.py`, add `coef * sum(low_var_kl(lp, ref_lp) * mask)` to each loss, and report `kl` in the
   metrics.

Server-side and fixed at startup is the right shape:

- `loss_fn_inputs` cannot carry reference log-probs: `encoding.INPUT_ROW_KEYS` keeps only `weights`, `advantages`
  and `logprobs`.
- The client has no base-model sampler to compute them anyway: sampling is latest-only.

## Placement and concurrency

platoon runs Harbor sandboxes on the client's node, unless the run has a `sandboxes` section with `mode: remote`
(below). So by default the "driver" is the sandbox host too:

- `driver.where: hc`: an `rt_HC` job (32 cores, ~28 concurrent rollouts). Used for the smoke run.
- `driver.where: gateway`: on the gateway's head node (192 cores; the Megatron trainer uses little CPU). The
  gateway job starts it once the server is healthy. Used for `naive.json`, at 160 concurrent rollouts.

Remote sandboxes (`harnessforge.sandboxhost`, docs/sandboxhost.md): start a fleet of rt_HC sandbox host jobs
(`python -m harnessforge.sandboxhost start --spec configs/sandboxhost/grpo.json`) and add to the run

```json
"sandboxes": {"mode": "remote", "hosts_spec": "/abs/path/configs/sandboxhost/grpo.json", "local_slots": 160}
```

The driver then keeps up to `local_slots` pods on its own node and places the rest on the hosts (96 each), so
`train.max_concurrent_rollouts` can be `local_slots + hosts x 96` (160 + 5 x 96 = 640). A host that dies fails its
rollouts as infrastructure errors (untrainable), within a second, and is avoided; `start` again replaces it.

Images:

- `driver.seed_image_cache` stages the tasks' SIFs from the shared hpc-sandbox cache to node-local disk (fine for a
  few tasks).
- `driver.image_cache` points the pods at the shared cache directly (6725 tasks).

## Smoke test (2026-11-19, `configs/train/smoke.json`)

**Setup.**
- Qwen3.5-4B, full fine-tuning.
- Gateway: 1 trainer + 1 rollout rt_HF node (8 miles-managed SGLang engines), disk-delta weight sync.
- Driver: an rt_HC job.
- Training: platoon `train`, uniform sampler, 6 tmax train tasks with mixed 4B pass rates, 4 steps × 2 tasks ×
  group 4, 4096 tokens/call, 30 agent steps, PPO loss, lr 1e-6.
- Run dir: `/groups/gai51740/aci18914wh/mho/runs/platoon-grpo-smoke1`.

**Timings.**

| step | batch time | mean rollout time | trainer waited on rollouts | `actor_train_time` | weight sync (`update_weights`) | delta (changed bytes / on disk) |
|---|---|---|---|---|---|---|
| 1 | 139.7 s | 75.5 s | 76.8 s | 53.6 s (first step, warm-up) | 6.8 s | 1.71 % / 0.34 GB |
| 2 | 81.3 s | 95.0 s | 40.1 s | 21.1 s | 7.0 s | 1.34 % / 0.28 GB |
| 3 | 111.8 s | 91.5 s | 43.4 s | 22.9 s | 6.7 s | 1.22 % / 0.25 GB |
| 4 | 49.7 s | 99.9 s | 14.5 s | 4.5 s | 6.5 s | 1.00 % / 0.21 GB |

Mean rollout time is per rollout. "Waited on rollouts" is `trainer_queue_wait_seconds_total`. Weight sync covers
writing the delta, the engines pulling it, and the reload.

**Checks.**
- **Steps complete.** All 4 steps trained (`skipped_optimizer` 0, grad norm 0.39-0.85). Batch reward was
  0.50 / 0.50 / 0.75 / 0.75; at this size that is not evidence of learning.
- **Weight versions advance.** All 8 engines logged `Weight version changed` 0 → 1 → … → 5 in step with the
  pushes. Startup took 310 s to a healthy gateway: Ray, 8 engines, Megatron load, baseline snapshot.
- **HF export loads.** The exporter replayed the stream into `hf/step_*`, checking xxh3 checksums.
  `transformers` loads it as `Qwen3_5ForCausalLM`, all 426 language tensors. Against the base, 2.3 % of bytes
  differ (360/738 tensors), last-token logits differ by at most 0.17 (mean 0.024), and the greedy answer is the
  same.
- **Cleanup.** The driver's exit writes `STOP`, and the gateway tore down and its job ended. Nothing was left
  running.
- **Sandbox setup.** With `harbor_system_deps` on, the per-trial `apt-get install` was still running after
  10 minutes on the rt_HC node, with 8 sandboxes pulling about 95 MB each through slirp4netns. The smoke turned it
  off. `pp/sandbox-setup` removes it for good.

**Weight-sync cost.** At lr 1e-6, one step changes only 1-2 % of the BF16 bytes: most updates are below BF16
resolution, as in the old run. A delta is about 0.3 GB and a sync about 7 s. 200 steps come to roughly 60 GB of
deltas.

## Full-topology check (`configs/train/fullcheck.json`)

This is `naive.json` on its real topology, with only 2 small steps (4 tasks × group 8). The setup:
- 1 trainer + 3 rollout rt_HF nodes (24 engines).
- platoon and the sandboxes run on the head node.
- All 6726 tmax-v11-train-naive tasks, the shared image cache, harbor system deps **on**, 64k context,
  `group_admission: informative`.

**What happened.**
- The gateway was healthy after 360-370 s.
- Driver start-up took about 3.5 min, mostly copying the 6726 tasks.
- On the head node's network the apt system-deps step took about 3 min per sandbox. On the rt_HC node it had not
  finished after 10 min.

**First attempt.** It failed at the first `forward_backward`: `datum 0: 43456 tokens exceeds 32768`. That is
miles-tinker's hard-coded `max_tokens_per_datum`, fixed by `miles-tinker-max-tokens-per-datum.patch`.

**Second attempt, step 1.**

| | value |
|---|---|
| batch time | 951 s |
| trainer waited on rollouts | 779 s |
| groups admitted | 4 of 7 (3 uniform groups rejected: 2 all-fail, 1 all-pass) |
| datums / tokens | 47 / 872k |
| `forward_backward` | 163 s, of which `actor_train_time` 143 s (TP2, 65536 tokens/GPU) |
| weight sync to 24 engines | 7.9 s |
| delta | 1.66 % / 0.33 GB |
| trainer memory | peak ~115 of 140 GB on rank 0 |

**Step 2 and stopping.** Step 2 was still waiting on straggler rollouts (3600 s agent limit) after 19 min, so the
check was stopped with `python -m harnessforge.train stop`.
- `STOP` reached platoon, which began its interrupted checkpoint. The gateway tore down, and both jobs ended
  66 s after the command.
- The driver forwarded SIGTERM twice (on seeing `STOP`, and again when the gateway signalled it). platoon takes a
  second SIGTERM as "force exit" and skipped the checkpoint. Fixed: the driver forwards it once.

**Implication for the full run.** The old run took ~9-10 min per step (rollout-bound) with 256 concurrent
sandboxes. Here 160 concurrent sandboxes is the main limit, and steps will likely be ~15 min: 200 steps ≈ 2 days
(`gateway.walltime` is 72 h).

## Restarting only the driver

`python -m harnessforge.train stop --driver-only --run-dir <run>` stops platoon and keeps the gateway: trainer, Tinker
state and any lent engines stay as they are.

- **Driver in its own rt_HC job** (`driver.where: hc`). The job is cancelled. `train driver --run-dir <run>
  [--set key=json ...]` submits a new one.
- **Driver on the gateway's head node** (`driver.where: gateway`, as in `naive*.json`).
  - The command writes `KEEP_GATEWAY` and `DRIVER_STOP`. The driver stops platoon (one SIGTERM: platoon writes
    its interrupted checkpoint and reaps its sandboxes) and exits.
  - The gateway consumes `KEEP_GATEWAY` and starts a **fresh** driver process, which reads the current
    `run.json` and this package's code from disk.
  - `train driver --run-dir <run> --set stats.wandb.mode='"online"'` edits `run.json` and does the same.
  - The command waits until `driver.json` shows the new driver.
  - Without `KEEP_GATEWAY`, a driver exit still ends the run.
- **What platoon does on restart.** It resumes from its newest checkpoint in the same `log_path` (the launch
  checkpoint at least). A new `create_model` on the single-tenant gateway resets the trainer to that state.

## Stopping, resuming, and handing over to a new pool

**What a resume reuses.**
- **platoon** resumes when it is run again with the same `log_path`. It takes the newest entry of
  `checkpoints.jsonl`, which records the batch, policy version, optimizer steps, sampler/dispatch state files and
  `wandb_run_id`. It then calls `create_training_client_from_state_with_optimizer`.
- **miles-tinker** resolves `tinker://<model>/weights/<name>` through the `tinker_meta.json` sidecar in the run's
  persistent `tinker/` checkpoint root. That works on a new gateway job too, with the same tenant (`tml-local`).
  Weights and Adam state come back; the parallel layout must match the save (TP4).
- **A graceful stop** (`train stop`) gives platoon a single SIGTERM. platoon writes a permanent
  `NNNNNN_interrupted` checkpoint, so stopping right after any step loses nothing but in-flight rollouts.
- **Disk-delta and HF export.**
  - A new gateway starts a new stream from the base. The gateway first moves the previous `weights/delta` aside
    (`delta.<ts>`), because miles would wipe it.
  - The driver records `weights/stream_base.json` (`{stream, batch}`) so the exporter continues the step numbers:
    v1 of the new stream is platoon's launch push of the checkpoint's weights = `step_<batch>`.

**Resume test (2026-11-20, 131k run, gateway 2999542 → 3001153).**

*First stop.*
- STOP reached platoon during step 2's `forward_backward`, which takes ~30 min at 131k.
- The gateway killed the driver after its fixed 300 s, so platoon never wrote its interrupted checkpoint, and step 1
  was lost.
- Fixed: `driver.stop_timeout` (3600 s in `naive_ext_131k.json`); `train stop`'s default grace follows it.

*Relaunch, same run id.*
- platoon resumed from the newest checkpoint (`recovery-000000`, batch 0).
- miles-tinker's fresh server loaded it through the sidecar (`load_slot` ok, 16.8 s).
- The same W&B run (`5zfx8md6`) was resumed rather than a new one.
- The previous delta stream was moved to `weights/delta.<ts>`, and `stream_base.json` was written.
- The new stream's launch push reached the engines.

*Still to verify:* resuming from a mid-run checkpoint (batch ≥ 1, Adam state, sampler position), via a stop at a
step boundary with the longer timeout.

**Throughput at 131k (TP4, 3 lent nodes = 24 engines).** Step 1 took 4105 s:
- rollouts: the trainer waited 2280 s;
- `forward_backward`: 1800 s (20.2M tokens in 760 datums, about 3 per trajectory; `actor_train_time` 1311 s,
  ~15k tok/s);
- weight sync: 6.9 s.

At ~1 h per step, 200 steps take ~200 h, about 3 pool handovers at 72 h each.

**Checkpoint cadence and disk** (131k, `naive_ext_131k.json`).
- Recovery checkpoints every 10 steps plus permanent ones every 50; nothing is pruned automatically
  (`keep_recovery` unset).
- Worst case, a crash or forced handover loses 10 steps (~3-5 h at 131k). A graceful stop loses none.
- 200 steps on /groups:

  | item | size |
  |---|---|
  | Tinker states (20 recovery + 4 permanent + launch/final, ~47 GB each) | ~1.2 TB |
  | disk-delta stream (~0.3 GB/step, maybe more at 131k) | ~60-100 GB |
  | HF exports (20 × 9 GB) | ~180 GB |
  | **total** | **~1.4 TB** |

- Pruning older recovery checkpoints is a user decision. platoon can do it (`keep_recovery`, through the Tinker
  `delete_checkpoint` route).

**Handover from n4hf72 to a new pool.** n4hf72's jobs started 00:55 JST Nov 20 with 72 h walltimes, so they end
around 00:55 JST Nov 23. Node budget: 8 rt_HF.
- Today: n4hf72 4 + trainer 1 + hf-serve 1 (ends ~02:40 JST Nov 21).
- After hf-serve ends: 4 + 1 = 5, leaving room for a 3-node new pool to overlap.
- qsub into R9920261300 accepted walltimes up to 336 h (held test jobs, deleted at once). The binding limit is the
  reservation's end time, which `pbs_rstat` does not let us read; ask before relying on more than 72 h.

1. After hf-serve ends (or on the user's go-ahead), start the new pool on 3 nodes. Use the same image, flags and
   local-snapshot policy as n4hf72 (`~/hf-dev/pool/scripts/pool-n4hf72.json` with `pool_id: n3hfNEW`,
   `nodes: 3`, `walltime: <longest allowed>`), from the pool tooling checkout whose agents support lending:
   ```bash
   cd ~/hf-dev/pool-lend/packages/harnessforge && PYTHONPATH=src ~/.local/bin/python3.12 -m harnessforge.pool start --spec ~/hf-dev/pool/scripts/pool-n3hfNEW.json --no-wait
   PYTHONPATH=src ~/.local/bin/python3.12 -m harnessforge.pool wait --spec ~/hf-dev/pool/scripts/pool-n3hfNEW.json
   ```
2. Stop the run at a step boundary, which writes the interrupted checkpoint and returns n4hf72 nodes 1-3:
   ```bash
   cd ~/hf-dev/train/harnessforge-ext && PYTHONPATH=src ~/.local/bin/python3.12 -m harnessforge.train stop --run-dir <run> --grace 1800
   ```
3. Point the run at the new pool and resume. Same run id and run dir: platoon resumes from its newest checkpoint and
   W&B run.
   - Set `rollout.pool_spec` to the new spec and `rollout.lend_nodes: [0, 1, 2]` in `naive_ext_131k.json`. Don't
     lend node 0 if evaluations should keep a router: lend `[1, 2]` and accept 16 engines, or start a 4-node pool.
   - Set `driver.hold: false`.
   ```bash
   PYTHONPATH=src ~/.local/bin/python3.12 -m harnessforge.train launch --force --config configs/train/naive_ext_131k.json
   ```
4. Check that platoon logged `Resuming from ... batch N`, the W&B run id is unchanged, and `hf/step_<N+...>`
   exports continue. Then stop n4hf72 if nothing else uses it (a user call).

## Sandbox tools without per-trial apt (`provider_options.sysdeps`)

**What the images lack.** harbor's installed-agent setup ran
`apt-get update && apt-get install -y curl bash build-essential git python3 python3-pip` in every trial, and the
platoon path kept it for parity (`harbor_system_deps`). Probing 50 random tmax-v11-train-naive images
(`scripts/hpcsb/prepare_images.py --no-build`; 6684 distinct images, one per task):
- all are Ubuntu 22.04 (glibc 2.35) with apt, and the agent toolchain imports in all of them;
- curl is missing in **50/50**, git in 48/50, make in 43/50, gcc in 30/50.

So the step cannot simply be dropped. Deriving 6684 images was ruled out.

**The bundle: `/groups/gai51740/aci18914wh/mho/sysdeps/jammy-harbor-deps-2/bundle.tar`.**
- *Contents:* 299 MB, 6514 files, root-owned. It holds exactly the files that apt command adds to a tmax image
  that lacks all four tools, minus dpkg/apt state, caches and logs. Built in a fakeroot sandbox of
  `sha256-81f38290….sif`.
- *Records:* `manifest.json`, plus `packages-base.txt`, `packages-after.txt` and `packages-added-or-changed.txt`
  (121 packages).
- *How it is used:* the pod provider (`provider_options.sysdeps`) mounts the directory read-only at
  `/opt/mho/sysdeps`. **Inside the start-gate slot**, right after the pod starts, it runs
  `tar -xf bundle.tar --skip-old-files -C / && ldconfig`. A failed extraction fails the sandbox start.
- *Config:* use it with `harbor_system_deps: false` (`naive_ext_64kcap.json`).

**Validation** (rt_HC, fakeroot sandboxes of 49 other sample images):
- In every image: curl, git, make, gcc and g++ present; a C and a C++ hello world compile, link and run against
  the image's own libc/libstdc++; `git clone` of a local repo works; an agent's later
  `apt-get update && apt-get install jq` succeeds; tar exits 0. 49/49 passed.
- Extraction takes 0.1-0.4 s alone and 0.23-0.63 s each with 16 extracting at once (0.95 s wall). It adds 117-368 MB
  per sandbox. By comparison the apt step took ~1 min per trial on the trainer node and >10 min on an rt_HC node
  with 8 at once.
- Ownership: the first bundle (`-1`, built unprivileged) made tar exit 2 on `/usr/bin/ssh-agent`'s setgid group;
  `-2` is the same files owned by root.

**Parity with the old run's per-trial apt.** Same command, same package set and versions, among them:
`binutils` 2.38-4ubuntu2.12, `binutils-common` 2.38-4ubuntu2.12, `binutils-x86-64-linux-gnu` 2.38-4ubuntu2.12, `build-essential` 12.9ubuntu3, `bzip2` 1.0.8-5ubuntu0.1, `cpp` 4:11.2.0-1ubuntu1, `cpp-11` 11.4.0-1ubuntu1~22.04.3, `curl` 7.81.0-1ubuntu1.30, `dirmngr` 2.2.27-3ubuntu2.5, `dpkg-dev` 1.21.1ubuntu2.6, `fakeroot` 1.28-1ubuntu1, `fontconfig-config` 2.13.1-4.2ubuntu5, `fonts-dejavu-core` 2.37-2build1, `g++` 4:11.2.0-1ubuntu1, `g++-11` 11.4.0-1ubuntu1~22.04.3, `gcc` 4:11.2.0-1ubuntu1, `gcc-11` 11.4.0-1ubuntu1~22.04.3, `gcc-11-base` 11.4.0-1ubuntu1~22.04.3, `git` 1:2.34.1-1ubuntu1.18, `git-man` 1:2.34.1-1ubuntu1.18, `gnupg` 2.2.27-3ubuntu2.5, `gnupg-l10n` 2.2.27-3ubuntu2.5, `gnupg-utils` 2.2.27-3ubuntu2.5, `gpg` 2.2.27-3ubuntu2.5, `gpg-agent` 2.2.27-3ubuntu2.5, `gpg-wks-client` 2.2.27-3ubuntu2.5, `gpg-wks-server` 2.2.27-3ubuntu2.5, `gpgconf` 2.2.27-3ubuntu2.5, `gpgsm` 2.2.27-3ubuntu2.5, `less` 590-1ubuntu0.22.04.3, `libalgorithm-diff-perl` 1.201-1, `libalgorithm-diff-xs-perl` 0.04-6build3, `libalgorithm-merge-perl` 0.08-3, `libasan6` 11.4.0-1ubuntu1~22.04.3, `libassuan0` 2.5.5-1build1, `libatomic1` 12.3.0-1ubuntu1~22.04.3, `libbinutils` 2.38-4ubuntu2.12, `libbrotli1` 1.0.9-2ubuntu0.1, `libbsd0` 0.11.5-1, `libbz2-1.0` 1.0.8-5ubuntu0.1, `libc-dev-bin` 2.35-0ubuntu3.18, `libc-devtools` 2.35-0ubuntu3.18, `libc6` 2.35-0ubuntu3.18, `libc6-dev` 2.35-0ubuntu3.18, `libcbor0.8` 0.8.0-2ubuntu1, `libcc1-0` 12.3.0-1ubuntu1~22.04.3, `libcrypt-dev` 1:4.4.27-1, `libctf-nobfd0` 2.38-4ubuntu2.12, `libctf0` 2.38-4ubuntu2.12, `libcurl3-gnutls` 7.81.0-1ubuntu1.30, `libcurl4` 7.81.0-1ubuntu1.30, `libdeflate0` 1.10-2, `libdpkg-perl` 1.21.1ubuntu2.6, `libedit2` 3.1-20210910-1build1, `liberror-perl` 0.17029-1, `libfakeroot` 1.28-1ubuntu1, `libfido2-1` 1.10.0-1, `libfile-fcntllock-perl` 0.22-3build7, `libfontconfig1` 2.13.1-4.2ubuntu5, `libfreetype6` 2.11.1+dfsg-1ubuntu0.4, `libgcc-11-dev` 11.4.0-1ubuntu1~22.04.3, `libgd3` 2.3.0-2ubuntu2.3, `libgdbm-compat4` 1.23-1, `libgdbm6` 1.23-1, `libgomp1` 12.3.0-1ubuntu1~22.04.3, `libisl23` 0.24-2build1, `libitm1` 12.3.0-1ubuntu1~22.04.3, `libjbig0` 2.1-3.1ubuntu0.22.04.1, `libjpeg-turbo8` 2.1.2-0ubuntu1, `libjpeg8` 8c-2ubuntu10, `libksba8` 1.6.0-2ubuntu0.2, `libldap-2.5-0` 2.5.20+dfsg-0ubuntu0.22.04.1, `libldap-common` 2.5.20+dfsg-0ubuntu0.22.04.1, `liblocale-gettext-perl` 1.07-4build3, `liblsan0` 12.3.0-1ubuntu1~22.04.3, `libmd0` 1.0.4-1build1, `libmpc3` 1.2.1-2build1, `libmpfr6` 4.1.0-3build3, `libnghttp2-14` 1.43.0-1ubuntu0.4, `libnpth0` 1.6-3build2, `libnsl-dev` 1.3.0-2build2, `libperl5.34` 5.34.0-3ubuntu1.10, `libpng16-16` 1.6.37-3ubuntu0.7, `libpsl5` 0.21.0-1.2build2, `libquadmath0` 12.3.0-1ubuntu1~22.04.3, `librtmp1` 2.4+20151223.gitfa8646d.1-2build4, `libsasl2-2` 2.1.27+dfsg2-3ubuntu1.2, `libsasl2-modules` 2.1.27+dfsg2-3ubuntu1.2, `libsasl2-modules-db` 2.1.27+dfsg2-3ubuntu1.2, `libssh-4` 0.9.6-2ubuntu0.22.04.8, `libstdc++-11-dev` 11.4.0-1ubuntu1~22.04.3, `libtiff5` 4.3.0-6ubuntu0.13, `libtirpc-dev` 1.3.2-2ubuntu0.1, `libtsan0` 11.4.0-1ubuntu1~22.04.3, `libubsan1` 12.3.0-1ubuntu1~22.04.3, `libwebp7` 1.2.2-2ubuntu0.22.04.2, `libx11-6` 2:1.7.5-1ubuntu0.4, `libx11-data` 2:1.7.5-1ubuntu0.4, `libxau6` 1:1.0.9-1build5, `libxcb1` 1.14-3ubuntu3, `libxdmcp6` 1:1.1.3-0ubuntu5, `libxext6` 2:1.3.4-1build1, `libxmuu1` 2:1.1.3-3, `libxpm4` 1:3.5.12-1ubuntu0.22.04.4, `linux-libc-dev` 5.15.0-204.214, `lto-disabled-list` 24, `make` 4.3-4.1build1, `manpages` 5.10-1ubuntu1, `manpages-dev` 5.10-1ubuntu1, `netbase` 6.3, `openssh-client` 1:8.9p1-3ubuntu0.18, `patch` 2.7.6-7build2, `perl` 5.34.0-3ubuntu1.10, `perl-base` 5.34.0-3ubuntu1.10, `perl-modules-5.34` 5.34.0-3ubuntu1.10, `pinentry-curses` 1.1.1-1build2, `publicsuffix` 20211207.1025-1, `rpcsvc-proto` 1.4.2-0ubuntu6, `ucf` 3.0043, `xauth` 1:1.1-1build2, `xz-utils` 5.2.5-2ubuntu1.2.

One difference is deliberate. apt also **upgrades** packages already in the image that the new ones pin; notably
`libc6` 2.35-0ubuntu3.13 → 3.18, pulled in by `libc6-dev`. The no-clobber extract keeps the image's own files, so
the sandbox keeps the image's libc 3.13 runtime and libstdc++, next to the bundle's 3.18 headers and crt objects.
Both are glibc 2.35 patch releases, and linking works in every sampled image. The dpkg database does not know
about the copied files; an agent's apt still works and may reinstall over them.

## W&B logging

platoon logs the run to W&B. `abci-base.json` sets `stats.wandb` to mode `online`, project `mho-harness`, and run
name = run id; tags come from the config. miles' own W&B stays disabled.

**Credentials.**
- The driver reads only `WANDB_API_KEY` and `WANDB_ENTITY`, from its process environment or else from
  `driver.wandb_env_file`. The default is `~/mho-model-harness-optimization/mho-platoon-port/.env`, the worktrees'
  shared `.env` symlink.
- It passes them only into the `platoon train` process environment.
- They are not written to `run.json`, the rendered platoon config or any log. The driver logs only which variable
  came from which source.
- Never put the key in a config's `driver.env`: that is written to `run.json`.

**No key.** If no key is found, the driver logs a warning and falls back to `offline`. The offline run is under
`<run>/platoon/<run_id>/train/wandb/`; upload it later with `wandb sync`.

**Leak check.** `python -m harnessforge.train scan-secrets --run-dir <run> [--also <path>]` compares every file
under the run dir (and the extra paths) against the key's value. It prints counts and file paths, never the value,
and exits 1 on a hit.

## Token-in/token-out prompts (`patches/platoon-tito.patch`)

**Problem.** platoon's gateway renders every model call's messages from scratch with the chat template, so the
model's previous turn comes back as the template's rendering of the message parsed from its sampled tokens. That
differs from the sampled tokens whenever the model wrote whitespace the template normalizes (`\n\n</think>`, one `\n`
before a `<tool_call>`; `\n` and `\n\n` are each one Qwen token, so the token count often does not change), a tool
call did not parse, the reply had several tool calls, or the generation was cut. The next prompt then does not
extend the previous prompt + completion, and `trajectory_to_data` starts a new datum that repeats the shared prefix.
In the 64k-cap run, step 1 trained 759 datums / 21.25M tokens for 256 trajectories and step 2 991 / 29.28M.

**The patch** (platoon 0.6.0; `platoon/gateway/tito.py` new; `core.py`, `server.py`, `hosted.py`):
- Per agent, the gateway keeps the exact stream of its last call: prompt tokens + sampled tokens, as sampled, with the
  request messages.
- A request whose messages are the previous request's + the assistant message the gateway returned (role, trimmed
  content and reasoning, tool-call names and arguments) + new messages is sampled on
  `stream [+ <|im_end|> if the completion was cut] + suffix`. The suffix is the template's rendering of the new
  messages and the generation prompt: the text after the last assistant turn's `<|im_end|>` in a render of the whole
  conversation. Earlier tokens are never re-rendered.
- Anything else is rendered from scratch exactly as before and starts a new stream (a new datum): history dropped or
  rewritten, tools or chat-template kwargs changed, or the template renders earlier turns differently once the new
  messages follow.
- Datums: unchanged code. `trajectory_to_data` merges every call whose prompt extends the accumulated sequence, which
  is now every stitched call. Loss mask = the sampled tokens only. Inserted tokens (new messages, template glue, a
  repaired `<|im_end|>`) are prompt tokens: mask 0, logprob 0. Sampler logprobs stay aligned with their tokens.
- What the model sees: its earlier turns are its own sampled tokens rather than a re-rendering. Everything else is the
  template's rendering. With canonical samples the stitched prompt equals the from-scratch prompt token for token.
- Switches: `PLATOON_TITO=0` (in `driver.env`) turns it off in the patched venv; `PLATOON_TITO_VERIFY=1` also
  checks every stitched prompt against the from-scratch render as text.
- Each trajectory's misc gets `tito: {stitched, fresh, reasons: {...}}`. The rollout process logs `TITO: N of M calls
  extended ...` every 2000 calls.

**micro-swe-agent and Qwen3.5.** When a reply has no valid bash tool call (none, unparsed, cut at max_tokens),
micro-swe-agent raises `FormatError`. It **drops the reply** and appends a user message with the error. The Qwen3.5
template then also stops rendering the thinking of every earlier turn, because a new user query moved
`last_query_index`. That is a real change of what the model sees, so the call is rendered from scratch and starts a
new datum. These are the only breaks left with this harness: 54 of 13816 calls in step 1's trained trajectories and 95 of
14538 in step 2's.
The case "cut / unparsed reply kept by the harness" (the stream keeps the raw sampled tokens, plus a masked
`<|im_end|>` for a cut) is covered by tests, but micro-swe-agent never takes that path.

**The training venv.** The evals venv is never modified:
```bash
# in an rt_HC job: copies ~/hf-dev/evals/venv, rewrites its entry points' paths, applies the patch, checks it
bash src/harnessforge/train/patches/build_platoon_tito_venv.sh ~/hf-dev/evals/venv ~/hf-dev/train/platoon-tito-venv
```
The script:
- refuses an existing destination;
- checks that platoon is 0.6.0;
- checks that `bin/platoon` runs the copy's own interpreter and imports the patched platoon;
- writes `PLATOON_TITO.json` (source, patch sha256, time).
platoon workers re-exec `sys.executable`, which is the copy's interpreter, so every process runs the patched code.

**Validation** (`tests/test_platoon_tito.py`, `harnessforge.train.tito_replay`):
- Unit tests with the real Qwen3.5 tokenizer and template and a scripted sampler: whitespace variants, the
  canonical token-for-token equality, the mask/logprob check, multiple / unparsable / cut replies, dropped replies,
  truncated history, changed tools, the `PLATOON_TITO` switch, the HTTP server path.
- End to end: the real micro-swe-agent (litellm 1.110.0, as in the `hf-micro-swe-1` toolchain) against the patched
  `GatewayServer` with a scripted sampler. litellm sends replies back with their reasoning and tool calls intact, so
  every kept reply stitches, and only the two format errors start datums:
  - TITO: 3 datums, 4388 tokens;
  - without TITO: 5 datums, 6410 tokens;
  - mask = the 595 sampled tokens in both.
- Replay of the live run's recorded steps (the 256 trained trajectories of each step). The recordings hold each
  call's returned message, usage and generation record, not the observations or the raw sampled text, so:
  - the replay rebuilds the conversation as micro-swe-agent would: filler tool results sized from the recorded
    prompt growth, steered to the recorded prompt sizes;
  - it drives the patched gateway with TITO off and on;
  - `--perturb p` samples a fraction p of the calls with whitespace the template does not render, since rebuilt
    samples are otherwise canonical and never break the TITO-off path.

| trained trajectories (256 per step) | measured in the run (no TITO) | replay, TITO off (perturb 0 / 0.01 / 0.02) | replay, TITO on | TITO from recorded usage |
|---|---|---|---|---|
| step 1 datums/trajectory | 2.96 (759) | 1.22 / 1.82 / 2.31 | **1.211** (310) | 1.211 |
| step 1 training tokens | 21.25M | 9.97M / 13.28M / 16.00M | **9.90M** (2.15× fewer than measured) | 9.89M |
| step 2 datums/trajectory | 3.87 (991) | 1.38 / 1.88 / 2.36 | **1.371** (351) | 1.371 |
| step 2 training tokens | 29.28M | 12.18M / 15.22M / 18.02M | **12.08M** (2.42× fewer) | 12.07M |

Notes on the replay:
- Rebuilt prompts are within 12 tokens per call of the recorded ones (mean |diff|).
- With TITO on, every call whose history micro-swe-agent kept was stitched: 13506 and 14187 calls. The only fresh
  renders were the format errors (54 and 95) and each trajectory's first call.
- With canonical samples (perturb 0), all 27693 stitched prompts decode to exactly the from-scratch render.
- TITO-on numbers do not change with perturbation, while TITO-off grows towards the measured 3–4 datums per trajectory.

**What is left: the 64k wall.** Of the remaining breaks, 46 (step 1) and 69 (step 2) are calls cut at the context
wall: prompt ≈ 65.3k, so `max_tokens` is clamped to a few hundred. micro-swe-agent drops the reply and adds a user
message, Qwen3.5 then drops all past thinking (~10k tokens), and the trajectory continues on a new ~55k-token datum.
These cost 2.70M (step 1) and 4.23M (step 2) of the TITO-on tokens; without them a step would be about 7.2–7.8M. TITO
leaves this alone because what the model sees really changes. Removing it is a harness/run decision, e.g. ending a
trajectory at the wall instead of letting it continue.

**Switching the running GRPO run (driver-only restart).** Coordinate with the run's owner first:
1. Build the venv (above); check `~/hf-dev/train/platoon-tito-venv/PLATOON_TITO.json`.
2. From the run's harnessforge checkout (the one its gateway job runs), at a step boundary, run
   `train driver` with `--set driver.platoon=...` as shown below.
   - `run.json`'s `driver.platoon` is the binary the driver starts.
   - With `driver.where: gateway`, this writes `KEEP_GATEWAY` + `DRIVER_STOP`. platoon gets one SIGTERM, writes its
     interrupted checkpoint and reaps its sandboxes. The gateway then starts a fresh driver with the edited
     `run.json`, and platoon resumes from that checkpoint against the same trainer and engines.
   - In-flight rollouts are lost, as with any driver restart.
   ```bash
   PYTHONPATH=src ~/.local/bin/python3.12 -m harnessforge.train driver --run-dir <run> --grace 5400 \
       --set 'driver.platoon="/home/aci18914wh/hf-dev/train/platoon-tito-venv/bin/platoon"'
   ```
3. Check:
   - `logs/platoon.log` resumes from the checkpoint;
   - new rollouts' `trajectory_finished` events carry `misc.tito`, mostly `stitched`;
   - the next steps log `optim/num_datums` near the number of trained trajectories (256 + format-error breaks) and
     `optim/num_tokens` at the replay's TITO-on level.
4. To undo, use the same command with `driver.platoon` set back to `/home/aci18914wh/hf-dev/evals/venv/bin/platoon`,
   or keep the venv and add `--set 'driver.env.PLATOON_TITO="0"'`.

**Risks.**
- Prefix-cache and datum shape. One datum per trajectory is up to 65536 tokens (prompt + completion of the last
  call, by the gateway's window), within the trainer's per-datum limit (65536) and TP2 tokens per GPU. There are fewer
  datums of the same maximum length, so the batch packs differently. The loss is a per-token mean, so its scale does
  not change.
- Sampled-token contexts. A trajectory's later calls now see the model's own raw earlier turns (as the old miles TITO
  run did) instead of a normalized re-rendering. This only matters for whitespace-level differences. A run switched
  mid-way mixes both in its history.
- Harness echo. The comparison trusts the harness to send replies back unchanged (role, trimmed content and
  reasoning, tool calls). A harness that edits them falls back to rendering from scratch, which is safe but gives no
  saving; the `tito.reasons` counts show it.
- Two template renders per call (history and conversation) instead of one plus a full encode. This was not measured
  at 160 concurrent rollouts, but encoding the full 60k-token prompt was the larger cost.
- The patch targets platoon 0.6.0 exactly. The build script refuses other versions. A platoon upgrade needs the
  patch rebased (or upstreamed; draft in `docs/platoon-tito-upstream.md`).

## Keep-thinking template and the context limit (`patches/platoon-tito2.patch`)

Two mid-run changes on top of TITO, in a second training venv, `~/hf-dev/train/platoon-tito2-venv`. It was built with
`build_platoon_tito_venv.sh ~/hf-dev/evals/venv ~/hf-dev/train/platoon-tito2-venv platoon-tito.patch
platoon-tito2.patch`. `platoon-tito-venv` and the evals venv are left as they are.

**(a) Always keep thinking (TMax-style: "We edit chat templates such that thinking is preserved on intermediate
turns").**
- *Stock behaviour.* Qwen3.5's stock template renders an assistant turn's `reasoning_content` only after the last real
  user query. A user message (micro-swe-agent's format error) therefore drops the thinking of every earlier turn: about
  10k tokens near the 64k wall.
- *The template.* `src/harnessforge/train/templates/qwen3.5-keep-thinking-v1.jinja` is the snapshot's template with
  that one condition changed to always render the thinking. A test checks that it equals the stock template but for
  that line.
  - Conversations without a user query after an assistant turn render token for token as before (so does every
    rollout's normal tool loop).
  - The build script copies it to `<venv>/templates/`.
- *The hook.* platoon 0.6.0 has no template override: `renderer_kwargs` are set as renderer attributes, and the
  `preserve_thinking` kwarg is ignored by Qwen3.5's template. So the patch adds a `chat_template` attribute to the
  `chat_template` renderer (a `.jinja` path or the text). It is passed to `apply_chat_template` and enters the
  tool-call parser's cache key. Set it with
  `train.renderer_kwargs.chat_template = "<venv>/templates/qwen3.5-keep-thinking-v1.jinja"`. harnessforge deep-merges
  this with the default `tool_call_parser: qwen3_coder`.
- *Serving.* The engines (miles-tinker, SGLang) sample on the tokens platoon renders and never render a chat
  template, so nothing changes there.
- *Evals.* Evaluations differ: the hosted gateway forwards `/v1/chat/completions` to the pool's SGLang, which renders
  with the model's stock template. Evaluated policies would see dropped thinking after a format error, where training
  keeps it. To align them, start the pool engines with `engine_args: {"chat-template": "<path>/qwen3.5-keep-thinking-v1.jinja"}`
  (SGLang `--chat-template`). That is a pool restart, which this PR does not do. The case is rare (micro-swe-agent
  format errors), but a policy trained with (a) relies on its old thinking still being there.
- *Effect.* (a) alone saves no tokens. micro-swe-agent still drops the failed reply, so the call after a format error
  still starts a new datum. Prompts no longer shrink there, so more trajectories reach the 64k window and end by
  overflow (graded as before). Use it with (b).

**(b) End a trajectory near the window (the context limit).**
- *What changes.* platoon used to clamp `max_tokens` to the remaining room: "Clamping max_tokens 16384 -> 149". The
  model then got cut after a few hundred tokens, micro-swe-agent dropped the reply, and the stock template dropped the
  past thinking. The trajectory went on with a new ~55k-token datum, often several times.
- *The limit.* Now, in `ModelGateway.fit_max_tokens`, a prompt that leaves less than
  `min(min_completion_tokens, max_tokens)` under the window raises `ContextLimitError`. That is a subclass of
  `ContextWindowExceededError`, so harnesses get the usual context-length error. micro-swe-agent aborts on it.
  - `min_completion_tokens` comes from `PLATOON_MIN_COMPLETION_TOKENS` (in `driver.env`); the default in this venv is
    2048, and `0` restores the clamp.
- *Grading.* The agent counts the limit (`AgentSession.context_limits`) and records
  `misc.context_limit = {prompt_tokens, context_window, min_completion_tokens, calls}`. `platoon.harness.rollout` (and
  `inline.failure_outcome` for function harnesses) then grades the session with the verifier, with reason
  `context_limit`. The outcome is `scored`, not env_failed. With `on_context_overflow: exclude` it is excluded from
  training instead, like an overflow.
- *Visibility.*
  - Events: the trajectory's `misc.outcome.reason = "context_limit"` (the grader's own reason under
    `grade_reason`), plus `misc.context_limit`.
  - W&B: a new per-trajectory stat `context_limit` (0/1) next to `num_steps`, logged as `rollout*/context_limit/avg`,
    i.e. the share ended this way. `outcome/scored` is unchanged: these are scored.

**Replay (256 trained trajectories per step, through the patched gateway at the real 65536 window).**

| | measured (no TITO) | TITO | TITO + (a) | TITO + (b) | TITO + (a) + (b) |
|---|---|---|---|---|---|
| step 1 datums/trajectory | 2.96 | 1.188 | 1.180 | **1.031** | 1.031 |
| step 1 training tokens | 21.25M | 9.50M | 9.43M | **6.87M** (3.1× fewer) | 6.89M |
| step 1 ended at the window | – | 3 overflow | 19 overflow | 31 at the limit (12.1%) | 32 at the limit (12.5%) |
| step 2 datums/trajectory | 3.87 | 1.324 | 1.371 | **1.102** | 1.102 |
| step 2 training tokens | 29.28M | 11.29M | 12.18M | **7.51M** (3.9× fewer) | 7.58M |
| step 2 ended at the window | – | 5 overflow | 30 overflow | 41 at the limit (16.0%) | 43 at the limit (16.8%) |

How to read the table:
- *Earlier TITO numbers.* At the real window the TITO column is slightly below the earlier replay (9.90M / 12.08M,
  run at a 2× window). The rebuilt prompts are within about 12 tokens of the recorded ones, so 3–5 trajectories at the
  wall end by overflow in the replay where the run clamped.
- *Calls after (b).* The remaining datum breaks under (b) are micro-swe-agent format errors away from the wall: 8 in
  step 1, 26 in step 2.
- *Reward of the ended trajectories.* Their final reward in the real run, after continuing past the wall:
  - step 1 (TITO + a + b): mean 0.28, and 9 of 32 passed;
  - step 2: 0.07, and 3 of 43 passed.
  The verifier grades the state at the limit, so some of these may still pass. At most about 9/256 (step 1) and 3/256
  (step 2) rollouts could lose a pass. That is the cost of (b).

**Switching the run (driver-only restart; coordinate with the run's owner).** Both changes are mid-run changes. They
apply to rollouts started after the restart, and trajectories already in history keep what they saw. From the run's
harnessforge checkout:
```bash
PYTHONPATH=src ~/.local/bin/python3.12 -m harnessforge.train driver --run-dir <run> --grace 5400 \
    --set 'driver.platoon="/home/aci18914wh/hf-dev/train/platoon-tito2-venv/bin/platoon"' \
    --set 'train.renderer_kwargs.chat_template="/home/aci18914wh/hf-dev/train/platoon-tito2-venv/templates/qwen3.5-keep-thinking-v1.jinja"' \
    --set 'driver.env.PLATOON_MIN_COMPLETION_TOKENS="2048"'
```
- (b) alone: drop the `chat_template` line.
- (a) alone: set `PLATOON_MIN_COMPLETION_TOKENS="0"`.
- To check after the restart:
  - `<run>/platoon-train.json` has `train.renderer_kwargs.chat_template`;
  - events show `misc.outcome.reason == "context_limit"` on some trajectories and no `Clamping max_tokens ... -> <2048`
    lines in `logs/platoon.log`;
  - W&B shows `context_limit/avg` around 0.1–0.17, and `optim/num_tokens` around 7–8M per step.

**Risks.**
- The reward of rollouts that used to get past the wall after a shrink, above.
- A policy shift towards shorter trajectories, since the wall now ends a rollout.
- (a) makes training and evaluation prompts differ after a format error unless the pool is given the same template.
- The template is pinned to the Qwen3.5-4B snapshot `851bf6e8`. Another snapshot needs its own versioned file.
- `PLATOON_MIN_COMPLETION_TOKENS` is a process variable, not a platoon config key. It applies to every gateway in the
  driver's platoon process, its evaluations included.

## Known limits and risks

- **Qwen3.5 has never run through the Tinker gateway** (only Qwen3-4B was tested upstream). The smoke run is the
  first.
- Disk-delta's baseline check needs the trainer's emitted HF tensors to match the checkpoint's names, dtypes and
  layout byte for byte (`A_log` fp32, `model.language_model.*`).
- miles-tinker's `--ref-load` here is the old run's torch_dist conversion of the base model (read only).
- **Disk use.** Deltas are never pruned: about 0.3 GB per step (smoke), ~60 GB over 200 steps. Add 20 HF exports
  of 9 GB each.
- **Tinker states.** Each `save_state` (torch_dist + Adam) is ~47 GB. platoon saves one at launch, one at the end
  (or on interrupt) and every `checkpoint.every` (50 in `naive.json`): ~280 GB for the full run. The smoke and
  check runs hold 94 GB each under `<run>/tinker`.
- **No per-step weight-sync or rollout timings from the server.** miles-tinker logs `tinker perf` (train time, MFU)
  per `optim_step` into `<run>/tinker/perf.jsonl`. Disk-delta prints density/wire per version in the head log.
  platoon logs `timing/batch_total` and the rollout-side waits.
- **The gateway is single-tenant.** A second `create_model` resets to the base weights, and a resume needs a
  Tinker `save_state` (`checkpoint.every`).
- **platoon `TINKER_API_KEY`.** It must start with `tml-`; we use `tml-local`. miles-tinker has no auth, so the
  key only names the tenant.
