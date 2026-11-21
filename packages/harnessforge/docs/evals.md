# harnessforge.evals: evaluating harness candidates on platoon

Workstream C. `harnessforge.evals` implements `contracts.EvalBackend` on `platoon eval` (v0.6) for micro-swe-agent
candidates on Harbor tasks, plus a `FakeBackend` for other workstreams' tests.

| Module | What | Needs |
|---|---|---|
| `backend` | `PlatoonBackend` / `PlatoonRun`: one `platoon eval` process per candidate, TASK_MAJOR (or look-major via unit aliases), per-candidate stop | stdlib (spawns platoon) |
| `reader` | rollout directories -> `TrialRecord`s, finished and in-flight; the outcome -> `env_failed` mapping | stdlib |
| `harness` | a candidate dir as a platoon harness spec (toolchain + read-only mount); `assets/` holds the runner and the toolchain recipe | stdlib |
| `pods` | a platoon sandbox provider that runs every task sandbox as a one-service hpc-sandbox pod | platoon[hpc] |
| `dataset` | copies the requested tasks, giving Dockerfile-only tasks their `FROM` image | stdlib |
| `fake` | `FakeBackend`: scripted rewards, simulated clock, task-major, stop | stdlib |
| `jobs` | `eval_node_job`: the PBS job an evaluation runs in (`contracts.JobSpec`) | stdlib |
| `ctxcap` | an OpenAI-compatible proxy that enforces a context window (training's end-at-limit rule) in front of a hosted endpoint | aiohttp, transformers (in the evals venv) |
| `harbor_export` | platoon run dirs -> harbor-compatible trial dirs (result.json, agent/, verifier/) for harbor-era readers | stdlib |
| `__main__` | `python -m harnessforge.evals run REQUEST.json [--harbor-export DIR]` / `export` / `stage-images` | stdlib |

## How a candidate runs

A candidate is a micro-swe-agent checkout (`src/micro_swe/`, `pyproject.toml`). Our harbor path
(`mho-rsft/src/harness/micro_swe_agent.py`) ran it **inside the task sandbox**: it uploaded the dir, `uv tool
install`ed it with litellm/mcp/orjson/fastapi from the network, and ran
`mini-swe-agent --yolo --model=litellm_proxy/<served> --task=... --cost-limit 0 -c mini -c model.model_kwargs.timeout=3600
-c <sampling yaml> --exit-immediately` in the task's working directory, where the agent's `LocalEnvironment` runs each
bash action as a subprocess.

The platoon backend keeps all of that and drops the network install:

- **Toolchain** `hf-micro-swe-1` (`assets/hf-micro-swe-toolchain.toml`, lock `assets/hf-micro-swe-1.txt`): a uv-managed
  CPython 3.12.14 with micro-swe-agent's dependencies and the same `--with` extras, hash-locked. Built once per site
  (`PLATOON_SITE=site.yaml platoon toolchain build .../hf-micro-swe-toolchain.toml`); platoon copies it to node-local
  disk once per node and mounts it read-only into every sandbox.
- **Candidate dir** mounted read-only at `/opt/harnessforge/candidate` (the Harbor service's `mounts`, one platoon
  process per candidate). `assets/hf_micro_runner.py` puts its `src/` first on `sys.path` and calls the candidate's
  own `micro_swe.__main__.main` with harbor's command line. So prompts, tools, the loop and parsing are the
  candidate's code, run where they ran before; nothing is installed for the agent.
- **Model**: the agent calls platoon's gateway as `litellm_proxy/policy` (`LITELLM_PROXY_API_BASE={base_url}/v1`); the
  gateway forwards each request unchanged (sampling parameters, `extra_body`, tools) to `inference.hosted.base_url` =
  `EvalRequest.endpoint` with `model` set to the served name, and records usage per call. `hosted.timeout_s` is 3600
  s, the agent's own litellm timeout.
- `check_candidate` refuses a dir without `src/micro_swe/__main__.py` or whose `[project] dependencies` go beyond the
  toolchain's packages (such a candidate needs a new toolchain version).

Why in the sandbox rather than a function harness driving the Agent class from the platoon process with platoon
tools: the candidate's tool is its own `LocalEnvironment` (subprocess per action, its own output formatting,
timeouts, env vars such as `PAGER=cat`); re-implementing it as platoon tool calls would change exactly the code the
optimizer edits (several accepted candidates changed how actions run and how output is shown). Running the
unmodified candidate where it ran before is what makes harbor and platoon scores comparable.

### What else the old path did to the environment (parity shims)

Found by reading the old trial logs and probing the pilot images (`probe.py` in the PR notes):

1. **System packages.** harbor's `MiniSweAgent.install` calls `ensure_system_dependencies(curl, bash, build_tools,
   git, python3, python_pip)`, which in *every* trial ran `apt-get update && apt-get install -y curl bash
   build-essential git python3 python3-pip` (every pilot image lacks curl and git; most lack gcc and make). The agent
   then had git/curl/gcc in its environment. `harbor_system_deps` (default on) runs the same check-and-install as the
   harness's `[install] commands`, as root, before the agent starts. With prepared images (`image_map`, below) the
   check finds everything and the step installs nothing.
2. **Dockerfile steps.** The tmax tasks name their image only as the Dockerfile's `FROM`; `hpc_sandbox_env` ran that
   image and replayed the remaining `RUN` (`mkdir -p /workspace /output /logs/verifier /root && ([ -d /app ] || ln -sf
   /workspace /app)`). platoon runs a dataset directory's task only from `[environment] docker_image`, so `dataset.py`
   copies the requested tasks into the run directory with `docker_image` set to the `FROM`, and the RUN lines (one
   distinct command across the tasks, else the backend refuses) are replayed by the same `[install] commands`.
3. **Network.** `hpc_network=pod`: each trial a pod with its own network namespace and slirp4netns egress. platoon's
   singularity provider runs single-container tasks in the node's network namespace, where concurrent trials of a
   task-major evaluation (every trial of a task at once, and every racing candidate on the same task) collide on the
   fixed ports many tmax tasks serve on (8080, 5000, ...). `pods.PodSingularityProvider` starts each sandbox as a
   one-service pod with egress and relays platoon's gateway socket into it (the harness reaches the gateway through
   the sandbox's route). No-network tasks keep platoon's own isolated sandbox.
4. **Users.** The tasks set `[agent] user = "root"`. Our pods run `main` as root (`run_as=subids`, root that can switch
   users, as the old pods did). platoon's `agent_user: honor` needs an image owner manifest that only a fetched
   dataset has, so the backend uses `agent_user: ignore`: the agent and the verifier run as the sandbox's own user,
   which is root here, as before.
5. **Timeouts and grading.** `agent_timeout_multiplier` as harbor's `--agent-timeout-multiplier` (3 in the optimizer),
   the spec's own limit high enough that the task's limit binds; `on_harness_exit: {timeout: grade, crash: grade}`:
   like harbor, the verifier grades whatever the agent left after a timeout or a crash. The harness's processes count
   against the task's memory limit, as in harbor (`[resources] memory_mb` 0).
6. **Processes the agent leaves running.** Many tmax tasks say "fix it and leave the service running on :8080", and
   their tests call that service. harbor grades with whatever the agent left running. platoon kills those processes
   twice, both on purpose (nothing the agent leaves behind can tamper with grading): (a) when the harness exits, every
   sandbox process whose environment carries the launch marker `PLATOON_HARNESS_AGENT` (all the agent's commands
   inherit it), and (b) before the tests, every process in the sandbox (`Sandbox.kill_processes`). With
   `harbor_keep_processes` (default on) the runner gets `--keep-children`, which unsets the marker before the agent
   starts, so only the agent loop itself is marked (a timeout still ends it); and the pod provider's
   `keep_agent_processes` makes (b) a no-op. Pilot task 000032 scored 0/8 on platoon with only (b) disabled and 4/4 on
   harbor; the bench task `background_service` scored 0 for gold and start the same way (see below).

## Launch order, concurrency and stopping

`platoon eval` creates one asyncio task per (task, rollout index) in dataset order and gates them with a FIFO
semaphore of `num_concurrent_workers`, so trials launch **task-major** in the order of the split manifest, which the
backend writes as `EvalRequest.tasks`. `PlatoonRun.order` is `Order.TASK_MAJOR`; a look's task prefix is a prefix of
`EvalRequest.tasks`. Shuffle that list once, up front.

**Attempt-major / look-major launch** (`options: {"order": "attempt_major", "trial_looks": [2, 4, 8, 24]}`, or a
`"plan"` as in `stats.LookSchedule`): platoon has no attempt-major mode, so the backend gives every unit (task, trial
index) a task name of its own, `<task>__hfa<index>`, in a `registry.json` of the materialized dataset that points each
alias at the task's one directory (platoon resolves dataset names through the registry, so nothing is copied). The
split lists the aliases in `LookSchedule.launch_order()` (look by look; within a look by trial index, then task) with
`num_rollouts_per_task = 1`, so the FIFO semaphore launches the units in exactly that order. `RolloutReader` maps each
alias back to (task, index); `PlatoonRun.order` is `Order.ATTEMPT_MAJOR`; request.json records `order` and the looks,
which `race_run` and `FileRun` read. Needs `materialize` (the default). Not yet: `harbor_export` of such a run (its
trial and task paths would be the aliases). Not validated against a real platoon yet (the fake platoon fixture runs it):
check on the first ABCI run that platoon 0.6's `HarborDataset` accepts the registry's many names for one directory
(its code does: `read_registry` + `_dirs[name]`, no uniqueness check on paths) and that the rollout directories are
named by alias.

**One platoon process per candidate**, each with `EvalRequest.concurrency` trials in flight. platoon's runner has no
way to stop one environment entry while the others go on (its stop ends the evaluation), and its SIGTERM path stops
the sandboxes of that process (labelled with its output directory), then exits 143. So `stop(candidate)` sends that
candidate's process SIGTERM, a second SIGTERM after `stop_grace_s / 2` (platoon's second Ctrl+C: stop waiting for
in-flight trials), SIGKILL to its process group after `stop_grace_s` (180 s), then `platoon sandbox reap` on its output
directory; it touches nothing else. The pilot ran base and candidate side by side this way. Cost per candidate: a
gateway and a task listing; the toolchain is staged once per node.

Verified on ABCI (base and a second candidate, 3 tasks x 2 trials, concurrency 2): stopping a candidate before its
agents started exited 143 in ~2 s; stopping it with two agents mid-run needed SIGKILL after 180 s with a single
SIGTERM (`stoptest2`), and exits 143 in ~90 s with the escalation (`stoptest3`: platoon reaped the one sandbox left,
no process left in the group). In both, the stopped candidate's in-flight trials became `error="stopped"`
(env_failed) and the other candidate finished all 6 trials.

A node runs about one sandbox per core (tmax tasks ask for 1 CPU): `concurrency x candidates <= cores - 4`
(`jobs.max_concurrency`). On ABCI an `rt_HC` node has 32 cores here; a full `rt_HF` node 192.

## Per-trial setup on hpc-sandbox: no network installs, capped starts

Both eval paths (this backend, and harbor's `scripts/optimize/eval.sh` with `HARBOR_ENV=hpcsb`) share these:

- **Prepared images** (`scripts/hpcsb/prepare_images.py`, at the repo root; run once per dataset in a CPU job). It
  pulls every image the dataset's tasks name into the shared hpc-sandbox cache, probes each for the commands
  harbor's agent setup ensures (`curl bash gcc make git python3 pip3`), builds a derived SIF with exactly the
  packages harbor's per-trial apt would add for each one that lacks any (`singularity build --fakeroot` from a
  node-local copy of the cached SIF; the build has network, the trials then need none), and writes
  `/groups/.../mho/prepared-images/image-map.json` = {image reference: SIF}. The dataset is not changed. Here:
  provider option `image_map` (backend option `image_map`, default `$HPCSB_IMAGE_MAP`); in harbor's path:
  `HPCSB_IMAGE_MAP`, which `eval.sh` sets to that file when it exists. A mapped SIF is copied to node-local scratch
  once per node (harbor path). Unmapped images start as before.
- **Start cap**: at most `max_starting` sandboxes (`$HPCSB_MAX_STARTING`, default 16, 0 = none) of the batch job are
  being started at once. The slots are `flock`s in `$PBS_LOCALDIR/hpcsb-start-slots`, so the cap spans platoon's
  rollout worker processes and both paths' code (`harnessforge.evals.start_gate`, `mho.backends.start_gate`).
  platoon times no single sandbox start (hpc-sandbox's `start_timeout_s` starts after the wait); the wait counts only
  against whole-rollout deadlines, which allow `sandbox_start_budget_s` (600 s) per start. In harbor's path the
  slot is held from the environment start through the agent setup and is taken in a wrapper of `Trial._prepare`,
  i.e. before harbor's `wait_for` on `environment.start` (task `build_timeout_sec`, 600 s) and on `agent.setup`
  (360 s): a trial waiting for a slot never times out for it, and the wait is not part of its `environment_setup`
  timing (it is the gap between the trial's `started_at` and `environment_setup.started_at`).
- **Sandboxes on other jobs**: `options.sandboxes = {mode: remote, hosts_spec: FLEET, local_slots: N}` starts the
  pods on sandbox host jobs (`harnessforge.sandboxhost`, docs/sandboxhost.md) instead of, or besides, this job's
  node; a trial whose host dies is env_failed (`unexpected:SandboxLostError`).
- **Agent toolchain**: this backend already runs candidates from the `hf-micro-swe-1` toolchain. harbor's path now
  does too: `scripts/hpcsb/build_toolchain.sh` builds the same lock (`assets/hf-micro-swe-1.txt`, wheels for glibc
  >= 2.28) into `/groups/.../mho/toolchains/hf-micro-swe-1`; `hpc_sandbox_env` copies it node-local once and mounts
  it read-only at `/opt/mho/toolchain`; `MicroSweAgent.install` uploads the snapshot as before plus a
  `$HOME/.local/bin/mini-swe-agent` launcher that runs the candidate's `[project.scripts] mini-swe-agent` entry
  point with the toolchain's CPython (`-I`, candidate `src/` first on `sys.path`), checks `--help` and the model
  stack's import, and otherwise (no toolchain, a candidate depending on a package the toolchain lacks, a failed
  check, `MHO_AGENT_INSTALL=uv`) falls back to the old per-trial `uv tool install`. `run()`'s command line is
  unchanged.

**TB2.1 (89 images, `terminal-bench-2-1-t7200`), probed in a CPU job**: lacking curl 71, gcc 60, make 72, git 71,
python3 26, pip3 35, bash 0; 87 of 89 lack at least one (`git-multibranch` and `sam-cell-seg` have all). All are
apt-based glibc images (Debian 11/12/13, Ubuntu 24.04); the toolchain imports in all of them (a first build with the
build node's wheels failed on the two Debian 11 / glibc 2.31 images through cryptography's manylinux_2_34 wheel,
hence the glibc-2.28 wheels). 85 derived images were built (5 Ubuntu ones only on a second try: apt hung past the
build timeout with ten builds at once); the two `qemu-*` images (Debian 11) cannot be prepared: their mirror 404s on
python3-pip's packages, so their per-trial apt fails exactly as before. The map holds all 89: 85 derived, 4 plain
cached SIFs (which still save the per-job pull).

**Before and after** (harbor path, base micro-swe-agent, Qwen3.5-4B on the 4-node SGLang pool, all 89 TB2.1 tasks
x 1 attempt at 96 concurrent on ONE `rt_HC` job, 32 cores; per-trial seconds, median / p90 / max). Before = the
four identical jobs `base-hpcsb-t7200/a1..a4` of the same day; after = `sandbox-setup/full96` (`HPCSB_MAX_STARTING=16`,
82 of 89 images prepared at the time: the five Ubuntu retries and the two `qemu-*` ones still apt-installed per trial).

| | before (a1..a4) | after |
|---|---|---|
| environment start | 189-274 / 313-550 / 454-633 | 8 / 82 / 84 |
| agent setup | 203-298 / 312-360 / 350-360 | 10 / 12 / 360 (the one timeout below) |
| wait for a start slot (untimed) | - | 109 / 153 / 172 |
| EnvironmentStartTimeoutError (600 s) | 0, 0, 7, 4 | 0 |
| AgentSetupTimeoutError (360 s) | 0, 1, 19, 14 | 1 (`adaptive-rejection-sampler`, unprepared then: its apt hung; prepared since) |
| agent installs from the network | 89 (uv + PyPI, plus apt in 87) | 0 uv; apt in the 7 unprepared |
| reward (sum over 89) | 15, 15, 11, 13 | 12 |

The rest of the trial is unchanged (NonZeroAgentExitCodeError 45 vs 38-44, AgentTimeoutError 6 vs 0-4: the
agent's own behaviour). A trial now reaches its agent about two minutes after launch at worst instead of up to ten.

## A context-window cap for hosted evaluations (`ctxcap`)

Training enforces its window at rollout in platoon's gateway (`train.context_window_length`; with
`platoon-tito2.patch` a prompt that leaves less than `min(2048, max_tokens)` ends the trajectory, graded). A hosted
evaluation has no cap: platoon leaves the window to the provider, and the pool's engines serve 262144 tokens.
`python -m harnessforge.evals.ctxcap` is an OpenAI-compatible proxy that applies the training rule per request, so an
evaluation runs at the training window without restarting (or capping) the pool:

```bash
python -m harnessforge.evals.ctxcap --upstream http://<router>:30000 --tokenizer <HF snapshot of the served model> \
    --window 65536 --min-completion 2048 --port 0 --url-file ctxcap.url --stats ctxcap.json &
# EvalRequest.endpoint = $(cat ctxcap.url)   (http://127.0.0.1:<port>/v1)
```

- Each chat request is rendered and counted with the model's chat template as SGLang renders it (tool-call arguments
  parsed, tools as function dicts, the request's `chat_template_kwargs`), in `--workers` processes.
- `room = window - prompt - margin`. `room < min(min_completion, max_tokens)`: HTTP 400 with the OpenAI/vLLM wording
  ("This model's maximum context length is ..."). platoon's hosted gateway counts it as a context overflow and grades
  the rollout (`scored:context_overflow`); litellm raises `ContextWindowExceededError`, on which micro-swe-agent aborts.
  Otherwise `max_tokens` is clamped to `room`, as platoon's `fit_max_tokens` does.
- `--stats` reports calls, clamped, refused, render time and the count's error against the engine's own
  `usage.prompt_tokens` (`est_minus_engine`), which says whether a `--margin` is needed.

## Records and the outcome -> env_failed mapping

`reader.RolloutReader` polls `<out>/<run>/<candidate>/platoon/rollouts/<task>/rollout_<k>/`:

- a directory exists from the moment the trial takes its slot (launch); without `metadata.json` it is **in flight**:
  `finished=False`, `started_at` = the directory's mtime;
- `metadata.json` (written when the trial ends) gives `wall_time_seconds`, `created_at` (= `finished_at`; `started_at`
  = `finished_at - wall`), `outcome_status`, `error` and `usage` (`output_tokens` = policy+aux `completion_tokens`,
  `input_tokens` = their `prompt_tokens`); the reward and the outcome's `failure_kind`/`reason` come from the root
  trajectory of `trajectory_collection.json` (metadata.json has no reward);
- once a candidate was stopped and its process has exited, its unfinished directories become `finished=True,
  env_failed=True, error="stopped"`, so looks never wait for them;
- finished records are cached (their files never change), so polling a long run is cheap.

`env_failed` follows the old harbor rule (`mho-rsft/src/utils/gate.py:env_failed`: a trial counts once the agent got
to run; harbor's missing reward counted 0):

| platoon result | env_failed | reward | `error` |
|---|---|---|---|
| `scored` | no | the verifier's reward | none, or `scored:<reason>` (e.g. `scored:agent_timeout`, `scored:harness_crash`: graded after a timeout/crash) |
| no outcome: the rollout raised (session/sandbox never opened, worker died) | **yes** | - | first line of platoon's error |
| `error` with failure_kind `sandbox_start`, `env_setup`, `env_config`, `env_healthcheck`, `sidecar_start`, `harness_setup`, `harness_config` (before the agent ran) | **yes** | - | `error:<kind>` |
| `error` with `harness_crash_gateway`, `env_busy`, `session_closed` (the evaluation's failure) | **yes** | - | `error:<kind>` |
| any status with `backend_unavailable` (model calls failed after retries) | **yes** | - | `backend_unavailable` |
| `cancelled` | **yes** | - | `cancelled` |
| `error` with any other kind (`harness_crash`, `context_overflow`, `verifier_timeout`, `verifier_no_reward`, `verifier_output`, `verifier_setup`, `artifact_collect`, `main_exited`, ...), `invalid`, `timeout` | no | 0 | `<status>:<kind>` |

Differences from harbor's rule, on purpose: a failed `harness_setup` (our toolchain or the parity install) is
env_failed because it is not the candidate's code (harbor counted a failed agent install as 0, since there the
candidate's own package was installed); model-backend failures are env_failed (no evidence about the harness).
Other workstreams that want harbor's exact rule can recompute it from `extra` (`outcome_status`, `failure_kind`).

## Running it

On ABCI (once): a venv with `platoon[tinker,ors,hpc]>=0.6,<0.7`, a site profile (`cache: ~/hf-dev/evals/platoon-cache`)
named by `PLATOON_SITE`, and the toolchain built under it. Then, inside an eval-node job (`jobs.eval_node_job`):

```bash
export PLATOON_NODE_SCRATCH=$PBS_LOCALDIR HF_EVAL_IMAGE_CACHE=$PBS_LOCALDIR/images
python -m harnessforge.evals stage-images request.json --from /groups/.../hpc-sandbox-images --to $HF_EVAL_IMAGE_CACHE
python -m harnessforge.evals run request.json      # records.jsonl every --poll s; <cand>.stop files stop one
```

or from Python: `PlatoonBackend(platoon=..., site=...).start(EvalRequest(...))`. `request.json` is EvalRequest's
fields plus `platoon` and `site`; options are listed in `backend.py`'s docstring.

`FakeBackend` (tests): `FakeBackend({"A": {"t1": 0.7, "t2": [1, 0]}}, durations={"t2": 3}).start(request)`; each
`records()` call advances a simulated clock one tick (or `auto_advance=False` and `step()`), launches are task-major
with `request.concurrency` per candidate, `stop()` ends a candidate's in-flight trials as `env_failed` `"stopped"`,
`env_failures` scripts sandbox failures, rewards are deterministic per seed.

## Harbor-compatible export (`harbor_export`)

The old optimizer loop, the proposer, `utils/gate.py` / `utils/seq.py`, `scripts/optimize/aggregate_results.py` and
the ad-hoc analysis scripts (`tb21cmp.py`, `speed.py`) read harbor's per-trial layout. `harbor_export` writes that
layout from a platoon run, so they read platoon runs unchanged:

```bash
python -m harnessforge.evals run request.json --harbor-export DIR    # incremental: every --poll, plus a final pass
python -m harnessforge.evals export <out_dir>/<run_id> DIR            # once, over an existing run dir
```

`DIR/<candidate>/` is a harbor job dir (what `eval.sh`'s `<cand>/eval/` was; point the loop's `eval_before/` /
`eval_after/`, `gate.per_task` or a `*/result.json` glob at it):

```
DIR/<candidate>/config.json, result.json      job level (n_attempts, agents, datasets / n_total_trials, stats.evals)
DIR/<candidate>/<task>__r<kkk>/                one trial; kkk = platoon's rollout index (= TrialRecord.index)
    config.json                                task.path, trial_name, agent; its mtime is the launch time
    result.json                                harbor's TrialResult (below); written last, once the trial finished
    agent/mini-swe-agent.trajectory.json       the agent's own trajectory (collected), else rebuilt from platoon's
    agent/mini-swe-agent.txt                   a readable transcript of that trajectory
    agent/platoon-harness.log                  platoon's harness log (install output, the agent's stdout/stderr)
    verifier/test-stdout.txt                   the tests' output, when known (collected, else platoon's 4 KB tail)
    verifier/reward.txt                        a scored trial's reward
    harnessforge.json                          the TrialRecord the trial was exported from
```

In-flight trials get `config.json` only (as harbor's running trials have no finished `result.json`), so
`seq.launched` sees them; once a stopped candidate's process has exited (or the export is `--final`, or the run's
`status.json` says done) its unfinished trials are written as env-failed `stopped`, like the TrialRecords. Trials are
exported once; a restarted exporter skips trials whose `result.json` exists.

**`result.json`** carries what the readers use: `task_name` (task.toml `[task].name`, else the task id: harbor's
name, so platoon and harbor runs of a dataset pair by task), `trial_name`, `config.task.path`,
`verifier_result.rewards.reward` (scored trials only; the readers count a missing one as 0, as they did for
harbor), `agent_result.n_input_tokens` / `n_output_tokens` / `n_cache_tokens` / `cost_usd`, `exception_info`
(`exception_type`, `exception_message`), `started_at` / `finished_at` and the phases `environment_setup`,
`agent_setup`, `agent_execution`, `verifier`, plus a `harnessforge` block (candidate, index, env_failed, error,
outcome status, failure kind, which sources the trajectory and the test output came from).

**Exceptions** are chosen so harbor's rules give the TrialRecord's answers: `gate.env_failed(result)` (an exception
and no `agent_setup`) == `TrialRecord.env_failed`, the readers' reward == `TrialRecord.reward or 0`, and a script that
drops harbor's two "never started" types (`tb21cmp.py`'s `INFRA`) drops exactly the env-failed trials. The message
starts with `[platoon <TrialRecord.error>]`, so the real cause is never lost.

| TrialRecord (`error`) | `exception_type` | `agent_setup` |
|---|---|---|
| scored, no reason | none | yes |
| `scored:agent_timeout` (or harness timed out) | `AgentTimeoutError` | yes |
| `scored:harness_crash` / nonzero harness exit | `NonZeroAgentExitCodeError` ("Command failed (exit N)") | yes |
| `*context_overflow` | `NonZeroAgentExitCodeError` (message names `ContextWindowExceededError`) | yes |
| `<status>:verifier_timeout` | `VerifierTimeoutError` | yes |
| `<status>:verifier_no_reward`, `verifier_output` | `RewardFileNotFoundError` | yes |
| other post-agent failure (reward 0) | `PlatoonError` | yes |
| env_failed: `error:harness_setup`, `harness_config` | `AgentSetupTimeoutError` | **no** |
| env_failed: every other (sandbox/env start, gateway, `backend_unavailable`, `cancelled`, `stopped`, no outcome) | `EnvironmentStartTimeoutError` | **no** |

**Phase times are estimates.** platoon records a trial's start (`created_at - wall_time_seconds`) and end, and each
model call's `perf_counter` span. The calls are put on the wall clock by the agent's own message timestamps (collected
trajectory), else by the harness log's last write (= the agent's end), else by the trial's end minus the tests' time;
`agent_execution` = first call start .. last call end, `environment_setup` ends when the sandbox was up (collect hook)
or where the agent starts (then `agent_setup` has zero length), `verifier` = agent end .. trial end. Totals
(`started_at`..`finished_at`) are exact; use `agent_execution` as "about how long the agent ran". Env-failed trials
have only `environment_setup`.

### Test output and the agent's trajectory

platoon's rollout directory holds neither. Three sources, best first:

1. **Collect hook** (no platoon change; on by default, backend option `collect`). `PodSingularityProvider`'s sandboxes
   copy, just before they stop, `/logs/verifier/test-stdout.txt` (last 4 MB) and the agent's own trajectory
   (`/tmp/platoon-harness/<agent>/trajectory.json`, which `render_spec`'s `--output` writes; micro-swe-agent rewrites
   it after every step, so a timed-out trial's is there too) to `<run>/<candidate>/collected/<rollout id>/`, with a
   `sandbox.json` of start/stop times. The rollout id is the sandbox's `platoon.rollout_id` label and the id of the
   rollout's `trajectory_collection.json`. Best effort: bounded (60 s), logged, never fails the stop.
2. **Patched platoon** (`src/harnessforge/evals/patches/platoon-outcome-metadata.patch`): keeps the outcome's metadata
   in `misc.outcome.metadata` (`test_output`, the tests' 4 KB tail; `verifier_exit_code`; `verifier_seconds`; ...),
   so the 0 of a trial whose sandbox was lost before the hook ran can still be explained. The reader puts
   `verifier_exit_code` in `TrialRecord.extra`.
3. **Rebuilt** from `trajectory_collection.json`: the task text and every model reply (content, tool calls, usage).

**Trajectory parity** (harbor pilot p3 vs platoon p3, base, task_000663): harbor's
`agent/mini-swe-agent.trajectory.json` is micro-swe-agent's own `save()` (system prompt, templated task, every
assistant message with its raw response and timestamp, every tool output, the exit message with `exit_status`,
`info.config`), `agent/mini-swe-agent.txt` is the agent's stdout (empty under `--exit-immediately`), `agent/trajectory.json`
is harbor's ATIF conversion. The proposer stages `result.json`, `agent/mini-swe-agent.txt` and
`agent/mini-swe-agent.trajectory.json`; the harness proposer also browses the trial dirs (`verifier/test-stdout.txt`).

| | harbor | platoon rollout dir (rebuilt) | collected |
|---|---|---|---|
| assistant replies, tool calls, per-call usage | yes | yes | yes |
| tool outputs (what the commands printed) | yes | **no** | yes |
| system prompt, templated instance prompt | yes | no (raw task text only) | yes |
| exit status / submission | yes | from the harness exit (`Submitted`, `AgentTimeout`, `ExitCode<n>`) | yes |
| per-message wall-clock timestamps | yes | no (call spans only) | yes |
| agent/model config (`info.config`) | yes | no | yes |
| tests' output | `verifier/test-stdout.txt` | no (patched platoon: 4 KB tail) | full (last 4 MB) |
| ATIF `agent/trajectory.json` | yes | not written (nothing here reads it) | not written |

So with the collect hook the proposer sees what it saw under harbor (plus a rendered transcript where harbor's was
empty). Exports of runs made before the hook have rebuilt trajectories without tool outputs and no test output.

### Validation on ABCI

Exports of existing platoon runs (made before the hook and the patch): pilot p3 (2 candidates x 32 tasks x 2), the
race pilot `race1-A` (439 trials, 55 env-failed, stopped mid-run) and `race1-B` (426, 57), and `bench2` (2 x 10).
For every candidate: one `result.json` per finished trial; `gate.env_failed` == `TrialRecord.env_failed` on every
trial; `gate.env_failures` and `gate.per_task` (reward, n, token cost per task) equal the records'; `seq.launched`
sees every launched trial in index order and `seq.look_pt` at the last full look agrees with `gate.per_task`; the
`tb21cmp.py` reader drops exactly the env-failed trials and gets the same accuracy; `aggregate_results.aggregate`
runs; the proposer's `_stage_prior_results` stages `result.json` + both agent files. tb21cmp-style pairing of the
harbor pilot p3 against the platoon p3 export matches all 32 task names (base: harbor 0.438, platoon 0.469, d +0.031
SE 0.059; cand: 0.578 vs 0.531, d -0.047 SE 0.052, as in "Pilot" above).

End to end with the hook and the patched platoon (`~/hf-dev/harbor-export/venv`): a real `platoon eval` (3 pilot tasks
x 2 trials, pods on rt_HC) against a scripted OpenAI-compatible stand-in for the model, run with `--harbor-export`:
6/6 trials exported while running, each with the agent's own trajectory (system, user, assistant, tool and exit
messages) and the full `verifier/test-stdout.txt`, `misc.outcome.metadata` holding `test_output` /
`verifier_exit_code` / `verifier_seconds`; the same consumer checks pass.

### The platoon patch

`patches/platoon-outcome-metadata.patch` (to `platoon/gateway/trace.py`, platoon 0.6.0): `_OutcomeFields.of` adds
the outcome's JSON-valued metadata to the summary it records as `misc["outcome"]`. Apply with `patch -p1` in the
site-packages dir. On ABCI it is applied only in a separate venv, `~/hf-dev/harbor-export/venv` (same package set as
`~/hf-dev/evals/venv`, from its `uv pip freeze`); `~/hf-dev/evals/venv` (used by the evals and the live GRPO run) is
unchanged. Point a request's `platoon` at `~/hf-dev/harbor-export/venv/bin/platoon` to use it.

Draft upstream PR (not opened):

> **gateway/trace: keep the outcome's metadata in the trajectory's `misc["outcome"]`**
>
> An env's `Outcome.metadata` (for Harbor tasks: `test_output`, the tail of the tests' output; `verifier_exit_code`;
> `verifier_seconds`; `artifacts`) is built on every rollout but dropped when the gateway records the outcome as
> `misc["outcome"]`, so it never reaches `trajectory_collection.json`, the rollout events or the reports. A 0 from a
> Harbor task then cannot be explained from the eval output (Harbor's own runs keep `verifier/test-stdout.txt`).
> This adds `misc["outcome"]["metadata"]`: the metadata's JSON-serializable entries (others are skipped, so the
> record stays JSON). Readers that look up `status` / `rewards` / `reason` / `failure_kind` are unaffected; the size
> cost is the 4 KB output tail per rollout. Alternative: a `test_output` key only, if the full metadata is too broad.

## Running the gold-harness benchmark (`harnessforge.bench`)

The bench tasks and harnesses go through this backend unchanged (`test_bench_tasks_and_harnesses_fit_the_platoon_backend`;
on ABCI, `bench1`/`bench2` below). What to know:

- **Image.** `task.toml` names the image as `[environment].docker_image`, by default the tag `harnessforge-bench:0.1`,
  which no registry serves: build the SIF once (`singularity build --fakeroot hf-bench-0.1.sif <out>/image/lab.def`,
  about a minute in a CPU job) and generate with `--image /abs/path/hf-bench-0.1.sif`. platoon runs a `.sif` path as
  is (`docker_image_reference`), `stage-images` skips it, nothing is replayed (`materialize` returns `{}`).
- **platoon's Harbor loader** reads every bench `task.toml` (`schema_version = "1.4"`, legacy `memory = "1G"` ->
  1024 MB, no `workdir` -> platoon's `/app`, which is the image's `WORKDIR`; no `[agent] user`, no steps). Checked
  with `HarborTask.load(..., agent_user="ignore")` on the cluster for all 10 families.
- **Options.** Set `harbor_system_deps: false`: the bench image has what its tasks need, and the apt step would put
  network installs and a minute per trial back. Keep `harbor_keep_processes` on: `background_service` asks the
  agent to leave `labctl serve` running (with it off, or before the marker fix, it scores 0 for every harness).
- The gold, start and ablation harnesses pass `check_candidate` (same dependencies as micro-swe-agent).

Smoke on ABCI (10 tasks, one per family, 1 trial, Qwen3.5-4B on vLLM, gold and start racing, concurrency 5 each):
all 20 trials graded, none env_failed, ~5 min. `bench1` (before the marker fix): gold 9/10, start 8/10, both 0 on
`background_service`. `bench2` (after): gold 10/10, start 9/10 (start's miss is `huge_log`, `scored:context_overflow`:
the output-truncation feature is what that family probes). One trial per task says nothing about effect sizes; it
shows the pipeline works.

## Pilot: harbor (old path) vs platoon

32 tmax-v11-gate200 tasks (fixed subset) x 2 trials, base micro-swe-agent and one accepted candidate
(`candidate_6_2_3d`), Qwen3.5-4B on one vLLM 0.28 GPU (the eval.sh serving block), agent timeout x3, concurrency 12
per harness, all trials on one `rt_HC` node per path. harbor = `mho-rsft/scripts/optimize/eval.sh` unmodified (group
mode, hpc-sandbox pods). Harbor ran twice (p1, p3) to measure its own run-to-run noise; platoon ran three times, each
time with the next parity fix (p1: none, p2: + keep processes before the tests, p3: + unmarked agent commands, the
final code). Accuracy is the mean over tasks of the per-task mean; d and SE are paired over the 32 tasks.

| | harbor p1 | harbor p3 | platoon p3 | platoon p3 - harbor (p1+p3 pooled) |
|---|---|---|---|---|
| base acc | 0.562 | 0.438 | 0.469 | -0.031 (SE 0.049) |
| cand acc | 0.562 | 0.562 | 0.531 | -0.031 (SE 0.046) |
| cand - base | 0.000 | +0.125 | +0.062 (SE 0.058) | effect difference +0.000 (SE 0.072) |
| env_failed | 0 / 128 | 0 / 128 | 0 / 128 | |
| agent timeouts (graded) | 18 | 22 | 20 | |
| other errors | 2 nonzero agent exit | 3 nonzero exit, 1 setup timeout | 1 harness crash (graded) | |
| output tokens / trial, median (base, cand) | 7.5k, 7.9k | 6.2k, 6.5k | 6.4k, 8.8k | |
| wall / trial, median (base, cand) | 215 s, 250 s | 208 s, 213 s | 182 s, 214 s | |
| span of the run | 68 min | 68 min | 70 min | |

- **Agreement.** platoon p3 is within noise of harbor: -0.031 for both harnesses (|t| < 0.7), and the candidate's
  effect is the same on both paths (+0.062 vs +0.062 pooled harbor). For scale, harbor against itself (p3 - p1) gave
  -0.125 (SE 0.071) on base. The first two platoon runs were 0.08-0.14 low (|t| up to 2.1) mostly through task 000032
  (below) plus that same run-to-run noise.
- **Task 000032** ("fix the API and leave it running on :8080"): harbor 8/8; platoon p1 0/4 and p2 0/4 (the agent's
  server was killed before the tests, first by grading's kill, then by the launch-marker kill); p3 4/4.
- **Tokens.** Prompt tokens of the first call are identical on both paths (same system prompt, tools and task text:
  1276 / 1485 / 1291 for the first three tasks). Means of input tokens are dominated by a few long trials and differ
  between harbor's own two runs by as much as between paths; medians agree.
- **Errors map one to one**: harbor `AgentTimeoutError` = platoon `scored:agent_timeout` (graded after the timeout, as
  harbor does), harbor `NonZeroAgentExitCodeError` ~ platoon `scored:harness_crash`.

The driver scripts (`harbor_pilot.sh`, `platoon_pilot.sh`, the comparison) are throwaway and not in the package; the
backend calls they make are the ones in "Running it".

## Known gaps

- **Verifier output and tool outputs need the collect hook (or the patch).** platoon itself keeps neither the tests'
  output nor the agent's tool outputs in the rollout dir. The pod provider's collect hook (default on) copies both;
  the platoon patch keeps the 4 KB test-output tail in `misc.outcome.metadata`; runs made before either have
  rebuilt trajectories (no tool outputs) and no `verifier/test-stdout.txt` in their harbor export. The hook only
  exists in `PodSingularityProvider` (another provider collects nothing). See "Harbor-compatible export".
- **Harbor export approximations.** Phase times are estimates (totals are exact); `agent/trajectory.json` (ATIF) is
  not written; `task_checksum` is null; `agent_result.metadata` is null (as in our harbor runs, so
  `aggregate_results.py`'s `n_turns` is null on both).
- **Grading guard off.** Parity keeps the agent's processes alive through grading (both kills off), so a process the
  agent left behind could rewrite the reward files; harbor had the same exposure. Turn `harbor_keep_processes` off
  for tasks that do not need a running service once we no longer compare with harbor.
- **Network installs.** `harbor_system_deps` (on by default, for parity) still apt-installs curl/git/gcc in a trial
  whose image is not in the prepared-images map (a dataset nobody ran `prepare_images.py` on, or an image whose
  derived build failed); prepare the dataset once to remove it.
- **Multi-step tasks** are refused by platoon's Harbor loader; no-network tasks use platoon's own isolated sandbox
  (needs `allow_network_none`), not a pod; tasks whose Dockerfile has more than one distinct RUN replay are refused.
- **One toolchain.** A candidate that adds a dependency beyond micro-swe-agent's needs a new toolchain version
  (`check_candidate` refuses it); candidates that change only code run as is.


