# harnessforge.evals: evaluating harness candidates on platoon

Workstream C. `harnessforge.evals` implements `contracts.EvalBackend` on `platoon eval` (v0.6) for micro-swe-agent
candidates on Harbor tasks, plus a `FakeBackend` for other workstreams' tests.

| Module | What | Needs |
|---|---|---|
| `backend` | `PlatoonBackend` / `PlatoonRun`: one `platoon eval` process per candidate, TASK_MAJOR, per-candidate stop | stdlib (spawns platoon) |
| `reader` | rollout directories -> `TrialRecord`s, finished and in-flight; the outcome -> `env_failed` mapping | stdlib |
| `harness` | a candidate dir as a platoon harness spec (toolchain + read-only mount); `assets/` holds the runner and the toolchain recipe | stdlib |
| `pods` | a platoon sandbox provider that runs every task sandbox as a one-service hpc-sandbox pod | platoon[hpc] |
| `dataset` | copies the requested tasks, giving Dockerfile-only tasks their `FROM` image | stdlib |
| `fake` | `FakeBackend`: scripted rewards, simulated clock, task-major, stop | stdlib |
| `jobs` | `eval_node_job`: the PBS job an evaluation runs in (`contracts.JobSpec`) | stdlib |
| `__main__` | `python -m harnessforge.evals run REQUEST.json` / `stage-images` | stdlib |

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
   harness's `[install] commands`, as root, before the agent starts. This is the one network step left; baking these
   packages into the task images would remove it (open problem).
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

## Launch order, concurrency and stopping

`platoon eval` creates one asyncio task per (task, rollout index) in dataset order and gates them with a FIFO
semaphore of `num_concurrent_workers`, so trials launch **task-major** in the order of the split manifest, which the
backend writes as `EvalRequest.tasks`. `PlatoonRun.order` is `Order.TASK_MAJOR`; a look's task prefix is a prefix of
`EvalRequest.tasks`. Shuffle that list once, up front.

**One platoon process per candidate**, each with `EvalRequest.concurrency` trials in flight. platoon's runner has no
way to stop one environment entry while the others go on (its stop ends the evaluation), and its SIGTERM path stops
the sandboxes of that process (labelled with its output directory), then exits 143. So `stop(candidate)` sends that
candidate's process SIGTERM (SIGKILL to its process group after `stop_grace_s`, 180 s) and touches nothing else; the
pilot ran base and candidate side by side this way. Cost per candidate: a gateway and a task listing; the toolchain is
staged once per node.

A node runs about one sandbox per core (tmax tasks ask for 1 CPU): `concurrency x candidates <= cores - 4`
(`jobs.max_concurrency`). On ABCI an `rt_HC` node has 32 cores here; a full `rt_HF` node 192.

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

PILOT_PLACEHOLDER

