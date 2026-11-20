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
