# harnessforge.bench: the gold-harness toy benchmark

A controlled testbed for the harness optimizer. A **gold** micro-swe-agent has ten known features.
Removing all ten gives the **start** harness. The optimizer (an LLM proposer that edits the
harness, with candidates accepted by sequential paired tests) runs from start, and we measure:

- **which** features it rediscovers (behavioural probes, no evaluation needed);
- **how fast**, in proposals and in evaluation trials;
- **how much** of the start-to-gold accuracy gap its best harness closes.

Workstream D of [PLAN.md](PLAN.md). Everything here runs on a CPU without a model. The
evaluations themselves need workstream C's backend and the model pool, so they are defined here
(see [Planned experiments](#planned-experiments)) but not run yet.

```bash
python -m harnessforge.bench build --out runs/bench/harnesses      # gold/, start/, ablate_<f>/
python -m harnessforge.bench probe runs/bench/harnesses/start       # 0/10 present
python -m harnessforge.bench tasks --out data/hf-bench              # 200 Harbor tasks + image recipe
python -m harnessforge.bench plan --harnesses runs/bench/harnesses --tasks data/hf-bench
```

Probing needs the harness's runtime dependencies in the probing interpreter:
`pip install 'harnessforge[bench]'` (jinja2, pydantic, pyyaml, tenacity; `litellm` is stubbed).

## The ten features

`kind` is what a feature is expected to buy. On tmax, the accepted candidates these features
come from were mostly accepted for being cheaper. Only 6_2_3d, 15_7_6a, 22_10_2a, 28_12_5a,
29_13_1d and 30_14_1b passed on the "better" (accuracy) path. Seven of the ten features are chosen
or adapted for a clear accuracy effect. `length_recovery`, `loop_warning` and `context_trim` are
efficiency features: their tmax ancestors were accepted for cost, and they are kept because they
give the optimizer cheap, well-defined targets and they also prevent hard failures
(RepeatedFormatError, wall-clock timeouts, context overflow).

| # | feature | hook | behaviour | why it should help | probe | source | kind |
|---|---|---|---|---|---|---|---|
| 1 | `output_truncation` | tools/bash.py: format_toolcall_observation_messages | A command output longer than 10,000 chars is shown as its first and last 5,000 chars with a note saying how much was elided and how to see the rest. | One huge log otherwise fills (or overflows) the context window; a context overflow aborts the run. Build and test logs keep the useful parts at the start and the end. | Run a command printing ~60k chars with markers at the start and end; the next observation the model sees is <= 20k chars and contains both markers. | generic (micro-swe-agent's own 10k head/tail template) | accuracy |
| 2 | `timeout_guidance` | environments/local.py: LocalEnvironment.execute | When a command hits the environment timeout the model sees its partial output plus a note that it timed out and was killed, and how to background long-running commands. | Without it a timed-out command looks like a bare failure (returncode -1); small models re-run the same foreground server or build until the wall clock runs out. | Run `echo MARK; sleep 20` with a 2 s timeout; the next observation keeps MARK and says the command timed out / was killed (and a normal command's observation does not). | generic (mini-swe-agent's timeout_template) | accuracy |
| 3 | `persistent_cwd` | environments/local.py: LocalEnvironment.execute (+ the prompt's subshell rule in config/mini.yaml) | The environment remembers the shell's final directory and starts the next command there; the prompt says so. | Models habitually `cd` once and then use relative paths; in a fresh subshell per command those paths silently point elsewhere. | Run `mkdir -p probe_sub && cd probe_sub`, then `pwd`; the second observation contains probe_sub. | generic | accuracy |
| 4 | `length_recovery` | models/litellm.py: Model.query | A response with finish_reason=length and no tool call is re-queried once with a doubled max_tokens (bounded) instead of becoming a format error. | Truncated responses were the main format-error type on tmax (16% of trials); three in a row kill the run with RepeatedFormatError. | The scripted model is cut off (finish_reason=length, no tool call) at max_tokens=1000; the harness re-queries with a larger max_tokens or tells the model it hit the length limit. | candidate_1_0_1b (length-truncation recovery) | efficiency |
| 5 | `loop_warning` | agents/default.py: Agent.execute_actions | After the same command with the same output 3 times in a row, the observation carries a note that repeating it gives no new information and suggests changing approach. | Byte-identical re-issue loops were 19% of tmax trials and 59% of output tokens, all at 0 reward. | Run the same command 5 times; a later repetition's observation differs from the first one's (or the sampling parameters change). | candidate_4_1_5a / 7_3_1b / 18_8_5c (loop escapes) | efficiency |
| 6 | `format_streak_trim` | agents/default.py: Agent.query (messages sent to the model) | During consecutive format errors, only the most recent rejection message is sent to the model; earlier ones are dropped from the prompt (kept in the trajectory). | A wall of identical rejections makes a collapsed small model repeat the collapse. | Two malformed tool calls in a row, then a good one: the prompt after the second error is no longer than the prompt after the first. | candidate_28_12_5a (format-streak context trim) | accuracy |
| 7 | `missing_command_repair` | models/litellm.py: Model.query | A bash tool call whose `command` is missing or empty is re-asked once with an explicit instruction (not persisted); if the repaired call is valid the turn proceeds with no format error in the history. | The `{}` empty-arguments collapse was the most frequent RepeatedFormatError cause on tmax (14.5% of trials). | One `{}` call followed by a good one: the repair query carries an instruction about the command, and no trace of the bad call is in the following prompt. | candidate_22_10_2a (missing-command repair re-query) | accuracy |
| 8 | `submit_confirmation` | agents/default.py: Agent.execute_actions | The first submit command is held back once with a short checklist (deliverables exist, result was run after the last edit, services still up); the second submit ends the run. | Submitted-but-wrong was the largest 0-reward bucket on tmax (decide-when-done procedure). | Submit once; the run continues, and a second submit ends it with exit status Submitted. | candidate_6_2_3d (decide-when-done working procedure) | accuracy |
| 9 | `context_trim` | agents/default.py: Agent.query (messages sent to the model) | Command results older than the 6 most recent ones and longer than 1,200 chars are shortened to their first 400 chars plus a note before each query. | Long sessions otherwise grow the prompt until it no longer fits. | 30 commands printing 4k chars each: the final prompt is under half the total output size and the newest observation is intact. | generic (stale-observation trimming) | efficiency |
| 10 | `raw_observation` | config/mini.yaml: model.observation_template | Command output is shown verbatim inside <returncode>/<output> tags instead of as a JSON-escaped string. | The 4B model misread the JSON wrapper (\n, \", \u003c escapes) as file content and copied escapes into deliverables (~79 tmax runs, 30% vs 40% solve rate). | Print a line with a quote, a backslash and a newline; the observation contains those characters unescaped. | candidate_30_14_1b (tool-result wrapper annotation) | accuracy |

Not used, and why: 11_5_3d (hard decision-time budget), 12_6_2b (information-stall guard) and
15_7_6a (no-write stall guard) terminate runs, so their value depends on the outer timeout more
than on the harness. 18_8_5c, 21_9_7c and 25_11_6d (output budgets for re-walk turns) are pure
token savers with no accuracy effect. 9_4_3a (economical-output directive) is a prompt wording
change that a behavioural probe cannot pin down. 29_13_1d (transcript-shape hardening) only
matters for externally supplied histories, which these tasks never produce.

## Gold, start and ablations

The annotated source is `src/harnessforge/bench/gold_harness/` (micro-swe-agent, MIT, with
mini-swe-agent's license kept). Each feature's code is a delimited block in `.py` and `.yaml`
files:

```python
# >>> feature:loop_warning
...gold code...
# --- else                      (optional) the start code this block replaces
...start code...
# <<< feature:loop_warning
```

Blocks may name several features (`feature:a,b`, kept if any is on) or `*` (the shared flag
plumbing, kept if any feature is on). Blocks nest. `src/micro_swe/features.py` is one
whole-file block. `builder.build(on, out)` resolves every block and writes a clean tree
with no markers:

| variant | features | what it is for |
|---|---|---|
| `gold/` | all 10 | upper reference |
| `start/` | none: the feature code is **absent**, not switched off | where the optimizer starts |
| `ablate_<f>/` | all but f | per-feature effect (gold minus f) |

Absence matters because the optimizer edits code. A start harness carrying dormant feature code
or flag names would give the answer away. Tests check that no feature name, flag call or
distinctive feature string (e.g. `characters elided`, `timed out after`, `SUBMIT_CHECKLIST`)
appears anywhere in `start/`, and that `ablate_<f>/` differs from `gold/` only in lines of f's
blocks.

The gold tree also has runtime switches. A `features:` mapping in `config/mini.yaml` (default all
true, e.g. `-c features.loop_warning=false`) or `MICRO_SWE_FEATURES_OFF=a,b` turns a feature off
without rebuilding. The materialized ablations are the official ones. The switches are for quick
checks, and a test confirms that switching a feature off at runtime makes its probe fail.

The start harness is micro-swe-agent minus the features. Three behaviours of upstream
micro-swe-agent are themselves features here, so start lacks them: the 10k head/tail
truncation in its observation template (`output_truncation`), the finish_reason=length
message (`length_recovery`), and showing `exception_info` (which named timeouts) in observations
(`timeout_guidance`). Start renders observations as upstream's JSON `{"returncode", "output"}`
block, without truncation.

## Probes

`probes.probe(harness_dir)` returns `{feature: ProbeResult(present, evidence, error)}`. Each
feature has one scenario, and each scenario runs in its own `python -I` subprocess started in an
empty temp dir (`_probe_runner.py`). Every candidate's `micro_swe` package is therefore imported
fresh, even though all candidates share the package name.

The harness runs end to end through **its own CLI** (`micro_swe.__main__.main`, the entrypoint an
eval runner shells out to, with `-c mini`), so its own config, agent loop, model wrapper and
local environment all take part. Only the LLM is fake. `litellm` is replaced by a scripted stub
that records every request (messages and sampling kwargs) and answers from the scenario script.
The commands are real but harmless (`echo`, `python3 -c`, `sleep`, `mkdir`) and run in the temp
dir. When the script runs out, the model submits. A runaway harness is stopped after 200 calls,
and a probe times out after 180 s.

Judges inspect only behaviour: what the model was shown, the kwargs it was queried with, and
the trajectory's exit status. They never inspect code, so the optimizer's re-implementations in
different code still count. The judges are deliberately mechanism-agnostic where a feature has
several reasonable implementations:

- `length_recovery` passes if the harness re-queries with a larger budget *or* tells the model
  it was cut off at the length limit (upstream micro-swe-agent's message).
- `loop_warning` passes if a repeated observation changes *or* the sampling parameters change
  (4_1_5a's higher-temperature resample).
- `format_streak_trim` and `missing_command_repair` compare prompt lengths, not message text.
- `submit_confirmation` checks that the first submit does not end the run and a second one does.

Calibration (`tests/test_bench_probes.py`, about 7 s for all 12 variants):

| harness | present |
|---|---|
| gold | 10/10 |
| start | 0/10 |
| ablate_f (each of the 10) | 9/10, exactly f absent |

A harness that crashes or cannot be imported reports every feature absent, with the error
attached.

## Task families

`python -m harnessforge.bench tasks --out DIR [--per-family 20] [--seed 0] [--image REF]`
writes 10 families x 20 = **200 Harbor tasks** (`task.toml` with `[metadata].family/feature`,
`instruction.md`, `environment/`, and `tests/test.sh` writing `/logs/verifier/reward.txt`), plus
`manifest.json`. Output is deterministic: the same arguments give byte-identical trees.

**One shared image.** `python:3.12-slim` plus `curl`, `procps` and `labctl`, a stdlib toolkit
(`tasks/labctl.py`). The recipe is in `image/Dockerfile` and `image/lab.def`: build it once and
pass the ref or `.sif` path as `--image`. Task fixtures are created inside the container by the
task's first step (`labctl setup <opaque id>`). Fixtures and expected answers are pure functions of
(family, seed) in `labctl.py`, which the generator imports too, so they cannot drift. Verifiers
are stdlib Python. Tasks do not need network access.

| family | n | exercises | the task | how the feature helps |
|---|---|---|---|---|
| `huge_log` | 20 | output_truncation | run a tool printing ~400k chars; report the run id (first line) and checksum (last lines) | head+tail keeps both; without it one call can overflow the context |
| `background_service` | 20 | timeout_guidance | even seeds: start `labctl serve` and leave it running, save `/health`; odd seeds: a 60 s computation (> the 30 s command timeout) | the timeout note says to background with redirected output |
| `deep_dirs` | 20 | persistent_cwd | configure and build three modules whose scripts only work from the module dir | `cd` then `./build` works across commands |
| `bulk_write` | 20 | length_recovery | write a given 120-180 line file verbatim | a long heredoc cut off at max_tokens is retried with a bigger budget |
| `stuck_poll` | 20 | loop_warning | wait for a queued job; it is held until approved (documented in `labctl job --help` and the queue README) | polling `status` repeats identically; the warning says to change approach |
| `json_escape` | 20 | format_streak_trim, missing_command_repair | write a JSON config whose values contain quotes, backslashes, `$`, backticks, newlines and unicode | the doubly escaped tool-call arguments provoke malformed calls |
| `verify_first` | 20 | submit_confirmation | fix the reported bug; the provided self-check also covers a second, unreported case | the checklist makes the model run the check before submitting |
| `many_files` | 20 | context_trim | follow a 22-note chain (~3k chars per note) to a final code | old notes are stubbed; the prompt stays small |
| `exact_text` | 20 | raw_observation | count backslashes, double quotes and `<` in a printed line | JSON rendering shows `\\`, `\"`, `<` instead of the characters |
| `control` | 20 | none | sum, count or longest word from a data file | no-harm check |

Tests run an oracle solution for two seeds of every family against the real verifier (with
`/app` pointed at a temp dir) and check that it scores 1 while a no-op scores 0. The
`verify_first` test also checks that the shipped code fails its self-check.

Eval settings that decide whether a family bites (fix them in the eval config and record them):
the environment command timeout (30 s default, needed for `background_service`), `max_tokens`
(about 2-4k for `bulk_write`), and the served context window (32k tokens or less for `huge_log`
and `many_files`).

## Metrics

`harnessforge.bench.metrics`:

- `discovery_report(candidate)`: probe results for one candidate, with counts by kind.
- `discovery_timeline(steps)`: for a search lineage (`LineageStep`: candidate, features present,
  accepted, cumulative proposals, cumulative trials), each feature's first proposal that had it,
  the first accepted candidate that kept it, and whether the final frontier lost it.
  `discovery_curve(steps, by="trials"|"proposals")` gives features kept vs budget.
- `accuracy(records, cand)` (task-mean) and `gap_closed(start, gold, cand)`.
- `paired_effect` and `ablation_effects(records, families=...)`: paired per-task deltas of
  gold minus `ablate_f`, on all tasks and on f's own family.

## Planned experiments

`metrics.experiments(harness_root, tasks_root)` returns `EvalPlan`s. Each has `.units()`, a list
of (harness dir, tasks, trials), and `.request(endpoint)`, a `contracts.EvalRequest` for any
`EvalBackend`. Task order is shuffled once (seed 0), so task-prefix looks are unbiased.

| plan | candidates | tasks | trials/task | total trials | question |
|---|---|---|---|---|---|
| `gap_start_vs_gold` | gold, start | 200 | 8 | 3,200 | how big is the gap the optimizer should close (accuracy and tokens)? |
| `ablation_all` | gold + 10 ablations | 200 | 8 | 17,600 | per-feature effect on the whole set, including spill-over to other families |
| `ablation_<f>` (x10) | gold, ablate_f | f's family + control (40) | 16 | 1,280 each | is f detectable by our sequential paired test at a given budget? |

Run order: the gap plan first. If gold does not beat start clearly, adjust the eval settings
above before spending on ablations. Then the targeted ablations, which are cheap. Features whose
targeted ablation shows no significant effect at 16 trials are undiscoverable by an
accuracy-gated optimizer at that budget, and are counted separately when reporting rediscovery.
Then `ablation_all`, for interactions and spill-over.

Then the optimizer run itself (workstream F): start from `start/` on a train split of these
tasks, record a `LineageStep` per proposal (probe every proposed candidate, which takes about
5 s on a CPU), and report `discovery_timeline`, `discovery_curve` and `gap_closed` for the final
frontier against `gap_start_vs_gold`. Keep a held-out split (e.g. `--seed 1` tasks) for the final
gap number.
