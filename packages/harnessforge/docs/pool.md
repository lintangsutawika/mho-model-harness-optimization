# The model pool (`harnessforge.pool`) and launchers (`harnessforge.launch`)

A pool is N batch-job nodes of SGLang engines behind one sglang-router. It exposes one
OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/models`) that harness evaluations use,
and `load(policy)` swaps every engine to a new checkpoint (e.g. after a GRPO step) without
restarting anything.

```
 controller (any process: CLI, optimizer, trainer)          eval clients (other PBS jobs)
   SGLangPool.start/load/health/repair/stop                   endpoint_of(pool_id) -> URL
        |  qsub (Launcher)          |  HTTP control                 |  OpenAI HTTP
        v                           v                               v
 +---------------- node 0 job ------------------+   +-------- node 1..N-1 jobs --------+
 | node agent (python -m harnessforge.pool.node)|   | node agent                        |
 |   sglang-router :30000  <-- workers ---------+---+-- engines :31000+ (dp = gpus/tp)  |
 |   engines :31000+ (dp = gpus/tp)             |   |                                   |
 +----------------------------------------------+   +-----------------------------------+
                 \___________ shared state dir (~/hf-dev/pool/state/<pool_id>/) _________/
                 pool.json  router.json  nodes/<i>.json  logs/  STOP
```

## Design

- **One job per node** (`PoolSpec.nodes`, default 4), submitted through a `Launcher`. Each job runs
  the *node agent* inside the SGLang container (`singularity exec --nv <sif> python -m
  harnessforge.pool.node`). The agent starts `dp = gpus_per_node // tp` SGLang servers, one per
  GPU group (`CUDA_VISIBLE_DEVICES` split), on free ports, binding `0.0.0.0`.
- **Registration through the state dir.** The agent writes `nodes/<i>.json` (host, advertised IP,
  engine URLs, readiness, restarts) every few seconds. The directory is on a filesystem every node
  sees (`/home` or `/groups` on ABCI); writes are atomic renames.
- **Router on node 0, not a separate CPU job.** The router is light (one Rust process); putting
  it in node 0's job avoids a second scheduler slot and a separate failure domain that could be
  scheduled hours apart from the GPUs. Node 0's agent publishes `router.json` and reconciles the
  router's worker list with the ready engines of all nodes (`POST /workers`, `DELETE
  /workers/{id}`), so engines join as nodes come up and leave when a node dies or goes stale
  (no file refresh for 90 s). Cost: if node 0 dies, the endpoint moves (repair brings up a new
  router; clients re-read `router.json` via `endpoint_of`). Default routing policy: `cache_aware`
  (prefix-cache affinity, good for multi-turn agents).
- **The controller is stateless.** `SGLangPool` only writes `pool.json`, submits/cancels jobs and
  makes HTTP control calls. Anything long-lived is in the node jobs, so the process that called
  `start()` can exit; a later process `attach`es (by pool id) to `load`, `health` or `stop`.
- **Policy swap** (`load(policy, drain=True)`): `pool.json` records the target policy first (so
  an engine restarting mid-swap comes up on it), then every engine gets
  `POST /update_weights_from_disk {model_path, weight_version: policy.id, abort_all_requests:
  not drain}` in parallel; each is verified through `/model_info` (`weight_version ==
  policy.id`). With `drain=True` SGLang takes its model-update writer lock, so in-flight requests
  finish and new ones wait for the swap; `drain=False` aborts in-flight requests (they return
  `finish_reason: abort`). Any failure leaves `status=degraded`, raises, and lists the engines.
  The history of swaps is kept in `pool.json`.
- **Health / repair.** `health()` returns per node: job state (scheduler), agent status, and per
  engine: `/health`, `weight_version`, `model_path`, running/queued requests (Prometheus
  `/metrics`). The agent restarts a crashed engine (up to 3 times) on the *current* policy;
  `repair()` resubmits nodes whose job ended, whose agent reported failure, or whose file went
  stale; `restart_node(i)` does it for one node.
- `stop()` writes `STOP`, cancels the jobs, waits for them to end and moves the state dir to
  `<root>/.stopped/<pool_id>-<time>` (logs survive; the id can be reused).

### Launchers

`PBSLauncher`, `SlurmLauncher`, `LocalLauncher` implement `contracts.Launcher` (port of
`mho.jobs.Scheduler`). `JobSpec.options`: `queue`, `account`, `resource_type` (ABCI `-v RTYPE`;
with it the select string is just `select=<nodes>`), `select` (raw override), `host` (pin),
`mem`, `extra_args`, `cwd`. The child environment is the submitter's minus its own
`PBS_*`/`SLURM_*`/`RTYPE`/`CUDA_VISIBLE_DEVICES`/`TMPDIR`/... plus `JobSpec.env` (exported with
`qsub -V` / `--export=ALL`), so a CPU driver job can submit GPU jobs. Transient scheduler errors
("under maintenance", timeouts) are retried for up to 2 h. PBS client binaries are found via
`$PBS_EXEC/bin` or `/opt/pbs/bin` when a job's PATH lacks them. `LocalLauncher` runs
subprocesses and keeps exit codes on disk (tests, single machine).

## Usage

```python
from harnessforge import PolicyRef, PoolSpec
from harnessforge.pool import SGLangPool, endpoint_of

pool = SGLangPool(
    PoolSpec(nodes=4, gpus_per_node=8, tp=1, max_context=262144,
             engine_args={"tool-call-parser": "qwen3_coder", "mem-fraction-static": 0.85,
                          "enable-metrics": True},
             launcher={"kind": "pbs", "queue": "R9920261300", "account": "gai51740",
                       "resource_type": "rt_HF", "walltime": "24:00:00"}),
    PolicyRef("qwen3.5-4b", "Qwen/Qwen3.5-4B"),
    pool_id="main", sif="/path/to/miles_latest.sif",
    binds=["/groups", "/home/<me>/hf-dev/pool/emptyroot:/root"])
pool.start()                          # blocks until the router serves every engine
url = pool.endpoint()                 # http://10.0.80.x:30000/v1
pool.load(PolicyRef("grpo-step200", "/groups/.../hf/step200", base_model="Qwen/Qwen3.5-4B"))
# in an eval job elsewhere:
url = endpoint_of("main")             # reads <state>/main/router.json
```

CLI (JSON spec; YAML too when PyYAML is installed):

```bash
python -m harnessforge.pool start  --spec pool.json        # prints the endpoint
python -m harnessforge.pool status --spec pool.json [--json]
python -m harnessforge.pool load   --spec pool.json --id step200 --path /ckpt/step200 [--no-drain]
python -m harnessforge.pool wait|repair|endpoint|stop --spec pool.json
python -m harnessforge.pool restart-node --spec pool.json --node 1
```

Spec file: `{pool_id, state_root, sif, binds, router_args, engine_env, advertise, policy: {id,
path, base_model}, spec: {nodes, gpus_per_node, tp, served_name, max_context, engine_args,
launcher: {kind, queue, account, resource_type, walltime, engine_port, router_port, ...}}}`.

### On ABCI

- No compute on the login node: run the controller in an `rt_HC` job (it only submits and makes
  HTTP calls). Host Python on compute nodes is 3.9; use `~/.local/bin/python3.12` with
  `PYTHONPATH=<package>/src` (the core is stdlib-only). The node agent itself runs on the SIF's
  Python 3.12.
- Batch-job PATH has no `/opt/pbs/bin`; the launcher handles it. PBS writes `-o` logs only at job
  end, so the agent keeps a live log in `logs/node<i>.agent.log` (engines: `node<i>-engine<k>.log`,
  router: `router.log`).
- **Mask `/root` in the miles images**: SGLang imports `megatron.core` through an editable finder
  that stats `/root/Megatron-LM/...`; as a non-root user that raises `PermissionError` (not
  `ImportError`) and the scheduler process dies. Binding an empty dir over `/root`
  (`binds: [".../emptyroot:/root"]`) makes it a clean `ModuleNotFoundError`.
- `--language-model-only` is not supported for Qwen3.5 in this SGLang (only for one other
  architecture); the vision tower is loaded (small for 4B).
- Addresses: nodes advertise `gethostbyname(hostname)` = the `10.0.x.y` compute network; other
  compute nodes reach it directly (verified from a different node, see below).
- Engine ports start at 31000 (+200 per node index, +20 per engine), the router at 30000, each
  moved up if taken (rt_HG jobs share nodes with other jobs).

## Qwen3.5-4B on SGLang vs our vLLM eval server

Versions (both `miles_latest.sif` and `miles.sif`): SGLang `0.5.20.dev56+g3283911`,
sglang-router `0.3.2`, torch `2.13.0+cu130`, transformers `5.12.1`, Python 3.12.3.
vLLM baseline: `vllm-openai-cuda-0.28.0.sif` (vLLM 0.28.0) with exactly the flags of
`mho-rsft/configs/serve/Qwen/Qwen3.5-4B.json` (`--tool-call-parser qwen3_coder
--enable-auto-tool-choice`, no reasoning parser, prefix caching, chunked prefill 32768,
`--max-model-len 262144`, `--gpu-memory-utilization 0.85`).

SGLang flags: `--tool-call-parser qwen3_coder --context-length 262144 --mem-fraction-static 0.85
--enable-metrics`, the model's own chat template (default), **no** `--reasoning-parser` (same as
vLLM). SGLang logs "Auto-detected template features: reasoning_parser=qwen3,
tool_call_parser=qwen3_coder" but only *applies* them when the flag is `auto`; without the flag
reasoning stays in `content`, as in our vLLM setup (thinking text is not split into
`reasoning_content`).

### Tool-call parity (20 agent prompts with 4 tools, temperature 0, thinking on, max 8192 tokens)

Script: `bench.py parity` (system prompt + task, 5 of 20 multi-turn with a prior tool call and
tool result; tools `bash`, `str_replace`, `write_file`, `submit`).

| server | tool calls parsed | `<tool_call>` text left in content | finish | same first tool as vLLM | identical first call (name+args) | text similarity |
|---|---|---|---|---|---|---|
| vLLM 0.28 (`qwen3_coder`, no reasoning parser) | 20/20 | 0 | 20 `tool_calls` | - | - | - |
| SGLang via router (`qwen3_coder`, no reasoning parser) | 20/20 | 0 | 20 `tool_calls` | 20/20 | 14/20 | 0.80 |
| SGLang + `--reasoning-parser qwen3` | 20/20 | 0 | 20 `tool_calls` | 19/20 | 12/20 | 0.78 |

Text similarity = difflib ratio over the first 600 characters of reasoning+content. Greedy
outputs of two different engines/kernels are not bit-identical; agreement drops as the text
gets longer, the chosen tool agrees everywhere.

Differences found:
- **Prompt length**: SGLang renders the same request about 44 tokens longer (mean 648 vs 604
  prompt tokens; 2358 vs 2314 in the throughput set). Cause: SGLang's chat serving passes
  `tool.model_dump()` to the template, which adds `"strict": false` to each of the 4 function
  definitions. Harmless for parsing, but the model sees a slightly different prompt than under
  vLLM.
- **`</think>` in content**: without a reasoning parser, SGLang leaves the `</think>` marker in
  `content` (16/20 responses) while vLLM drops it (special token skipped on detokenization). An
  agent that does not strip `<think>...</think>` itself sees one extra tag.
- **Reasoning parser**: with `--reasoning-parser qwen3` SGLang moved the thinking into
  `reasoning_content` (20/20) and still parsed every tool call; the vLLM failure mode we avoid
  (a `<tool_call>` emitted before a missing `</think>` swallowed into reasoning) did not occur in
  these 20 prompts, but they do not provoke it. Default stays *off* to match the vLLM setup the
  harnesses were tuned on.

### Throughput (one H200 per engine)

`bench.py throughput`: agent-like requests (system prompt + tools + a ~2.3k-token repository
listing, thinking on, temperature 0.6, max 1024 tokens; outputs end at the first tool call, mean
~90 tokens). `throughput-long`: decode-heavy (short prompt, 1024 output tokens, thinking off).
Clients ran in a CPU job on another node, through the router for SGLang.

| setup | concurrency | requests | req/s | output tok/s | total tok/s | p50 / p90 latency |
|---|---|---|---|---|---|---|
| vLLM 0.28, 1 GPU | 64 | 256 | 21.0 | 1915 | 50.6k | 2.6 / 4.6 s |
| SGLang pool N=1 x 1 GPU | 64 | 256 | 22.0 | 1972 | 53.9k | 2.5 / 4.7 s |
| SGLang pool N=2 x 1 GPU | 64 | 256 | 38.2 | 3419 | 93.4k | 1.4 / 2.6 s |
| SGLang pool N=2 x 1 GPU | 128 | 512 | 43.7 | 3983 | 107.0k | 2.5 / 4.6 s |
| vLLM 0.28, 1 GPU, decode-heavy | 64 | 128 | 8.1 | 8307 | | 9.2 / 9.2 s |
| SGLang pool N=1, decode-heavy | 64 | 128 | 8.3 | 8482 | | 9.0 / 9.0 s |

SGLang matches vLLM per GPU on both workloads (within 5%), and two nodes scale ~2x at matched per-engine load
(128 concurrent on 2 engines vs 64 on 1).

### Policy swap

Checkpoint: `~/hf-dev/pool/ckpt-perturbed`, a copy of Qwen3.5-4B whose 2-D language weights
were multiplied by `1 + 0.02 * N(0,1)` (seed 1234, made in a CPU job; `PERTURBATION.json` in the
dir). On the N=2 pool (5 prompts, temperature 0, 4 calls each so both engines answer):

- `load(perturbed, drain=True)`: 8.4 s for both engines, including waiting for an in-flight
  1492-token request, which completed normally (`finish_reason: stop`).
- Both engines then reported `weight_version=perturbed-2pct`, `model_path=<ckpt>`;
  `pool.policy()` and `pool.json` history recorded it; `/v1/models` through the router still
  listed `policy`.
- Outputs changed for 3/5 prompts (long ones diverge; short factual ones such as "17*23" stay
  the same under this small perturbation). Both engines gave identical greedy outputs for every
  prompt, before and after.
- `load(base, drain=False)`: 4.0 s; the in-flight request returned `finish_reason: abort`.
  Outputs then reproduced the pre-swap outputs exactly.

### Startup and repair

- N=2 x rt_HG: `start()` returned after 2 min 10 s (job queueing + model load + CUDA graphs;
  the router came up on node 0 and added node 1's engine once ready).
- Killing node 1's job (`qdel`): its agent shut its engine down, node 0 removed it from the
  router (1 worker), the endpoint kept serving. `repair()` resubmitted node 1; the router had
  both workers again 1 min 51 s later, on the active policy.

## Known limits / open problems

- After `update_weights_from_disk`, SGLang's tokenizer manager sets its `served_model_name` to
  the new `model_path` (sglang `_update_model_path_info`). Through the router `/v1/models` still
  lists `policy`, and requests with `model: "policy"` keep working (SGLang does not check the
  name), but the `model` field echoed by an engine may change. Clients should not key on it.
- During a swap engines change policy independently (all in parallel); for a few seconds
  different engines may serve different policies. `TrialRecord.policy` should be read from
  `pool.policy()` after the trial, or the swap done between evaluation phases.
- `load()` swaps weights only. Tokenizer/chat template/parsers stay those of the policy the
  engines started with; a policy with a different architecture or tokenizer needs a restart
  (`stop`/`start`).
- Router on node 0: if node 0 dies the endpoint moves until `repair()`; clients that cached the
  URL must re-read it.
- The agent restarts crashed engines up to 3 times; a node whose engines all fail reports
  `failed` and exits (the job ends), and `repair()` resubmits it.
- `SlurmLauncher` is untested on a real SLURM cluster (unit-tested flag rendering only).

## Full size: N = 4 x 8 GPUs (rt_HF)

Nothing in the code is specific to N=2 x 1 GPU: set `nodes: 4, gpus_per_node: 8,
resource_type: rt_HF` (the select string is then `select=1` per job). Each node agent starts 8
engines (tp=1) on ports 31000..31140 and the router gets 32 workers. Not yet run: it needs 4
rt_HF nodes (32 GPUs). Things to check then: startup time with 8 engines loading at once per node
from the shared HF cache, router throughput at ~32 workers (cache-aware policy), and `load()`
time with 32 parallel `update_weights_from_disk` calls reading the same checkpoint from `/groups`.
