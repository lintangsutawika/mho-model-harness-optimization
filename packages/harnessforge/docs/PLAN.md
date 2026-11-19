# harnessforge: plan for the platoon port

Branch `platoon-port`. Every workstream is a PR into it. The old code (`src/mho`,
`src/utils`, `scripts/`) keeps running the current experiments until harnessforge replaces it.

## Goals

1. **A library people can depend on.** `harnessforge` lives in `packages/harnessforge`. Its core
   has no third-party dependencies; platoon and pool clients are optional extras.
2. **Pythonic orchestration.** No hand-written PBS scripts: jobs are `JobSpec`s submitted by
   a `Launcher` (PBS, SLURM or local).
3. **Sequentially tested harness search.** Every comparison is a paired, sequential test with
   early stopping. Candidates race; losers stop early.
4. **One persistent model pool.** N nodes (configurable) of SGLang engines behind one
   OpenAI-compatible endpoint. It serves harness evaluation and doubles as the rollout engine,
   and `load(policy)` swaps it to a newer policy (e.g. after GRPO).
5. **Training on platoon**: GRPO, DPPO and RSFT through the Tinker API, via a miles-tinker
   server we extend.
6. **A gold-harness toy benchmark.** A micro-swe-agent with 10 known features, from which a
   starting harness is derived by removing them. The optimizer must rediscover them, so we can
   measure discovery directly.

## Architecture

```
            harnessforge.search  (proposer + race/gate loop; later PR)
                 |  Candidate[]                 ^ TrialRecord[]
                 v                              |
   harnessforge.stats  <---- records ----  harnessforge.evals (EvalBackend)
   (sequential tests)                       platoon eval runner, harbor tasks,
                                            hpc-sandbox Singularity, hosted inference
                                                 |  OpenAI HTTP
                                                 v
                                     harnessforge.pool (ModelPool)
                                     N x SGLang nodes + router, load(policy)
                                                 ^  weights (training)
                                     harnessforge.train  (platoon RL/SFT via Tinker API
                                                          -> miles-tinker; later PR)
   harnessforge.launch (Launcher: PBS/SLURM/local) runs pool nodes, eval clients, trainers
```

Contracts: `src/harnessforge/contracts.py` (Candidate, PolicyRef, PoolSpec, ModelPool,
EvalRequest, EvalRun, EvalBackend, Order, JobSpec, Launcher) and `records.py` (TrialRecord,
the only result format). Change a contract only in a PR of its own.

## Workstreams

| | Workstream | Depends on | Where it runs |
|---|---|---|---|
| A | `stats`: sequential paired tests and racing | contracts | CPU, local |
| B | `pool` + `launch`: SGLang pool on N nodes, policy swap, PBS launcher | contracts | ABCI GPU |
| C | `evals`: platoon eval backend for micro-swe-agent candidates | contracts | ABCI CPU+GPU |
| D | `bench`: gold-harness toy (10 features, probes, tasks) | contracts | CPU, then ABCI evals |
| E | `train`: GRPO/DPPO/RSFT via platoon + miles-tinker | B | ABCI multi-node |
| F | `search`: the optimizer loop on A+B+C | A, B, C | ABCI |

### A. Sequential testing (`harnessforge.stats`)
- Paired per-task statistics (port `utils/gate.py`: paired, decide) over `TrialRecord`s.
- Looks for both `Order`s:
  - TASK_MAJOR: task prefixes of the request's task order;
  - ATTEMPT_MAJOR: first r trials of every task (the current `utils/seq.py`, done on mho-rsft).
  - A look is ready only when every unit it covers has finished (no completion-time bias).
- Group-sequential boundaries:
  - efficacy (O'Brien-Fleming shape) and non-binding futility;
  - calibrated by simulation from real per-task pass rates, with a CLI to recalibrate for
    other look schedules or trial counts.
  - Reference: c = 1.72 at K = 5 attempt looks gives about 5% one-sided false accepts.
- Racing: K candidates vs one reference A, all evaluated concurrently on the same tasks.
  - Each candidate is stopped independently (pass / futility).
  - Selection among passers by t. The reference keeps running while any candidate is alive.
- Val gate as a sequential test too (replaces the 2-look adaptive val).

### B. Model pool (`harnessforge.pool`, `harnessforge.launch`)
- One launcher job per node (N configurable, default 4).
- Each node: SGLang engines (dp = gpus/tp), registering with an SGLang router (sglang-router)
  that exposes `/v1/chat/completions`.
- `load(policy)`: SGLang `update_weights_from_disk` on every engine, with drain semantics;
  the pool records the active policy id.
- Health checks, replacing a dead node, a persistent state file so clients find the endpoint.
- Validate Qwen3.5-4B serving on SGLang, including tool-call and reasoning parsers, against
  our current vLLM behaviour (same prompts, temperature 0 comparisons, tool-call parse rate).
- Launcher: port `mho.jobs.Scheduler` (PBS/SLURM) to `JobSpec`/`Launcher`, plus a local one.
- ABCI: rt_HG is always 1 GPU, rt_HF always a full node (8 GPUs). Containers come from
  existing SIFs (miles images ship SGLang).

### C. Platoon eval backend (`harnessforge.evals`)
- A micro-swe-agent candidate dir becomes a platoon harness: a generated TOML spec with a
  toolchain and the candidate dir mounted, or a function harness. It must not install from the
  network inside sandboxes.
- Harbor tasks (tmax, terminal-bench) on Singularity via hpc-sandbox, hosted inference pointed
  at the pool endpoint.
- Several candidates in one run (one platoon `environments` entry each, same tasks).
- A disk-polling reader turns `rollouts/<task>/rollout_<k>/metadata.json` into `TrialRecord`s
  while the run is going, including in-flight trials.
- Per-candidate stop: run one platoon process per candidate (stop = SIGTERM, which reaps its
  sandboxes) unless a cleaner in-process way is found.
- Pilot: the same tasks and harness through harbor (old path) vs platoon. Accuracy must agree
  within noise before anything depends on this.

### D. Gold-harness benchmark (`harnessforge.bench`)
- 10 features, each a small, separable change to micro-swe-agent. Draw them from what our
  optimizer actually discovered: truncation recovery, loop escape, format-error repair,
  output trimming, finish discipline, and so on.
- The gold harness has all 10; the start harness has none.
- Per-feature **probes**: deterministic tests with a scripted model and environment that
  detect whether a harness has the behaviour. Discovery is then measured directly, not
  inferred from accuracy.
- Tasks where each feature matters (task families that trigger the behaviour), small and fast.
  Measure each feature's ablation effect (gold minus feature i) so we know which ones are
  discoverable at a given eval budget.

### E. Training (after B)
- Validate miles-tinker full fine-tuning with Qwen3.5-4B.
- Multi-node (>2) launcher.
- New losses and inputs: DPPO, Dr.GRPO reducer, KL with reference logprobs.
- HF export of checkpoints so the pool can `load` them.
- Platoon RL (GRPO/DPPO) and SFT (RSFT) configs.
- Acceptance: reproduce the naive-split GRPO run (val200 .492 -> ~.59 at step 200).
- Open question: can the pool's SGLang engines be miles-tinker's rollout engines (one fleet),
  or does training get its own engines with checkpoint `load`s between phases?

## Conventions for every PR

- Python >= 3.12. Code in `packages/harnessforge/src/harnessforge/<module>`, tests in
  `packages/harnessforge/tests/`.
  Run `uv run --directory packages/harnessforge --extra dev pytest`.
- Pure logic gets unit tests with fakes; cluster behaviour gets a recorded smoke test (command +
  output summary in the PR body).
- ABCI rules:
  - no compute on the login node: use `qsub` jobs;
  - reservation queue `R9920261300`, project `gai51740`, `-v RTYPE=rt_HC|rt_HG|rt_HF`;
  - never print secrets or `.env` values;
  - never read other users' directories;
  - don't modify files of running jobs in place;
  - move data aside rather than delete it.
- PR into `platoon-port`; the description says what was verified and how.
