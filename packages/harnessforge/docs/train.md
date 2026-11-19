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
$PY -m harnessforge.train mapping                                       # the old-run mapping table below
```

`launch` only writes files and calls `qsub`, so it may run on the login node. Configs (`configs/train/`, JSON with
`extends`): `abci-base.json` (our paths and the old run's hyperparameters), `smoke.json`, `naive.json` (the full
comparison run), `pass_rate.json` (the Thompson-sampling variant; not to be run yet).

## One-time setup on ABCI

All of this runs in CPU jobs (no compute on the login node), and nothing is installed into the image.

1. **miles-tinker source.** `git archive` of neulab/miles-tinker at 25b6a76, unpacked to
   `~/hf-dev/train/miles-tinker` and bound over `/root/miles` in `miles_latest.sif`. Then apply
   `src/harnessforge/train/patches/miles-tinker-disk-delta-cast.patch` (see below).
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
   `hf-micro-swe-1` toolchain.

**The disk-delta patch.**
- *Problem.* Qwen3.5 stores 24 tensors in F32 in the HF checkpoint (`linear_attn.norm.weight`, `A_log`, ...).
  Megatron emits them as BF16. Disk-delta diffs raw bytes against the checkpoint files, so its baseline check
  refuses the run (`dtype=F32 ... trainer emitted dtype=BF16`).
- *Fix.* The patch casts those tensors to the checkpoint's dtype before diffing. The cast is a lossless upcast.
  Engines load F32 there anyway, exactly as from the base checkpoint.
- *Side effect.* Because of this, the first push (platoon's launch checkpoint) is not empty: it carries the BF16
  rounding of those 24 norms.
- *Status.* To upstream to miles-tinker.

## Why miles-managed engines, not the pool's

miles-tinker 25b6a76 cannot attach SGLang engines it did not start: `ServerCell.init()` raises
`NotImplementedError("external rollout address allocation was removed ...")` for `--rollout-external`
(`miles/ray/rollout/server_cell.py:151`, asserted by a fast test), `--rollout-external-engine-addrs` is parsed and
never read, and an external router is refused (`rollout_server.py:26`). `docs/advanced/disaggregated-rollout.md` is
stale. The working deploy (`deploy/abci/tinker_gateway.pbs`) is one Ray cluster over the trainer node and the rollout
nodes, with miles launching its own engines and router there. `harnessforge.train.gateway` does the same from Python
for 1 + R nodes (R = 1 for the smoke, 3 for the comparison run).

Making the pool's engines miles-tinker's rollout engines (one fleet, the PLAN's open question) is a follow-up in
miles-tinker:

- `miles/ray/rollout/server_cell.py`: an `ExternalServerCell` that wraps an address, with no Ray actor and no
  launch. Health, pause/continue and `update_weights_*` go through `SGLangApiClient` as they do now.
- `miles/ray/rollout/rollout_server.py`: build cells from `--rollout-external-engine-addrs`, and allow
  `--sglang-router-ip/port`, or register the external engines with miles' own router.
- `miles/ray/placement_group.py`: no rollout bundles in external mode. This part is already done.
- Weight sync:
  - disk-delta works as is, provided each engine host has the base checkpoint at the engine's `--model-path`.
  - broadcast needs `init_weights_update_group` over the network to engines outside Ray, so their hosts must
    reach the trainer's NCCL rendezvous.
- Tests: `tests/test_external_rollout.py`, revived.

Estimate: 2-4 days including a 2-node test. On our side the pool already has the matching half: lending
(below).

## Lending pool nodes to a trainer (`SGLangPool.lend` / `unlend`)

For when a trainer does attach pool engines. It is implemented and tested now; with miles-managed engines it is not
used.

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
| `--async-max-concurrent-samples` | 256 | `train.max_concurrent_rollouts: 160` | **differs**: limited by the client node's cores |
| `--rollout-temperature` | 1.0 | `inference_params.temperature: 1.0` (miles-tinker requires 1.0) | same |
| `AGENT_MAX_OUTPUT_TOKENS` | 16384 per call | `max_completion_tokens: 16384`, `sampling_precedence: config` | same |
| `AGENT_MAX_INPUT_TOKENS`, `--max-seq-len` | 65536 | `train.context_window_length: 65536`; SGLang context 65536 | same |
| agent timeout | 1800 s override × 2.0 = 3600 s | `agent_timeout_multiplier: 6` (each task's 600 s × 6) | same |
| `AGENT_TRIAL_TIMEOUT` | 7200 s, scores 0 | `rollout_config.timeout: 7200` | mapped |
| harness | micro-swe-agent base scaffold via harbor's MiniSweAgent | the same dir, through the evals backend's spec (runner and toolchain) | same code |
| chat template / parsing | SGLang `qwen3_coder` + `qwen3` reasoning parser, TITO `qwen35` | platoon `chat_template` renderer, `tool_call_parser: qwen3_coder` | mapped (token in/out kept) |
| sandboxes | hpc-sandbox pods, executors on all 4 nodes | `harnessforge.evals.pods` on the client node, harbor parity shims | mapped |
| rollout engines | 24 GPUs, miles SGLang | `gateway.rollout_nodes: 3` (miles-managed) | same |
| trainer | 1 node, TP2, 65536 tokens/GPU, log-prob chunk 1024, CPU-offloaded precision-aware Adam | same TP, tokens and chunk; Adam on GPU (full mode forbids the precision-aware optimizer) | mapped |
| `--save-interval 10 --save-hf` | HF every 10 steps | `export.every: 10` (from the disk-delta stream) | same |

### What is not matched, and what it would take

| gap | effect | cost to match | recommendation |
|---|---|---|---|
| **KL to the reference** (0.01, low_var_kl) | Small at lr 1e-6 over 200 steps, but a confound | Server-side in miles-tinker full mode (details below). ~1-2 days + 1 smoke; +20-30% trainer time per step (old run: ref log-probs 36 s of a 190 s train step, against 5-10 min rollout phases, so ~5% wall time) | Run without it now, labelled `no-KL`; add it before any run whose purpose is to reproduce exactly |
| Dr.GRPO divisor vs per-token mean | Changes the effective learning rate, not the gradient direction | Reshape platoon's datum scaling (a platoon objective option) or a miles-tinker `loss_fn_config` divisor; ~0.5 day | Accept; record it |
| TIS | At staleness ≤ 4 the PPO ratio against the sampler's logprobs already corrects off-policy samples; TIS also corrects train/infer mismatch | Add a truncated-IS factor to miles-tinker's `ppo` loss (`loss_fn_config.tis_cap`); ~0.5 day | Accept for now |
| Concurrency 160 vs 256 | Rollout phase up to ~1.6× longer at the start (shorter later as groups refill) | Sandboxes on more nodes. platoon runs Harbor sandboxes on the client's node only: its gateway reaches a sandbox over a node-local unix socket, so remote sandbox jobs would need a new provider plus a relay | Accept; measure step time |

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

platoon runs Harbor sandboxes on the client's node. So the "driver" is the sandbox host too:

- `driver.where: hc`: an `rt_HC` job (32 cores, ~28 concurrent rollouts). Used for the smoke run.
- `driver.where: gateway`: on the gateway's head node (192 cores; the Megatron trainer uses little CPU). The
  gateway job starts it once the server is healthy. Used for `naive.json`, at 160 concurrent rollouts.

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

## Known limits and risks

- **Qwen3.5 has never run through the Tinker gateway** (only Qwen3-4B was tested upstream). The smoke run is the
  first.
- Disk-delta's baseline check needs the trainer's emitted HF tensors to match the checkpoint's names, dtypes and
  layout byte for byte (`A_log` fp32, `model.language_model.*`).
- miles-tinker's `--ref-load` here is the old run's torch_dist conversion of the base model (read only).
- **Disk use.** Deltas are never pruned: about 0.3 GB per step (smoke), ~60 GB over 200 steps. Add 20 HF exports
  of 9 GB each, and platoon's launch and `checkpoint.every` Tinker states (torch_dist + Adam, tens of GB each).
- **No per-step weight-sync or rollout timings from the server.** miles-tinker logs `tinker perf` (train time, MFU)
  per `optim_step` into `<run>/tinker/perf.jsonl`. Disk-delta prints density/wire per version in the head log.
  platoon logs `timing/batch_total` and the rollout-side waits.
- **The gateway is single-tenant.** A second `create_model` resets to the base weights, and a resume needs a
  Tinker `save_state` (`checkpoint.every`).
- **platoon `TINKER_API_KEY`.** It must start with `tml-`; we use `tml-local`. miles-tinker has no auth, so the
  key only names the tenant.
