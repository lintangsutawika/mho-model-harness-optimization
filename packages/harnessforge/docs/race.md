# harnessforge.race_run: driving a sequential race on a live evaluation

`harnessforge.stats.Race` is a pure state machine (docs/stats.md section 5). `harnessforge.race_run` is the loop around it,
the core of the harness search: start one evaluation with the reference and every candidate, poll its records, stop each
candidate as soon as its test decides, stop the reference once no candidate is alive, and write down what happened.

```python
from harnessforge.race_run import RaceDriver, replay
from harnessforge.stats import LookSchedule, Race

run = backend.start(request)                                   # reference + candidates, tasks shuffled once
race = Race("A", ["B"], LookSchedule(run.order, request.tasks, request.trials))   # c = recommended_c
summary = RaceDriver(run, race, poll_s=60, summary_path="race.json").drive()
reverse = replay(records, Race("B", ["A"], race.schedule))     # the same records, roles swapped, offline
```

| piece | what |
|---|---|
| `RaceDriver(run, race, poll_s, drain_timeout_s, summary_path, records_path)` | the loop: every `poll_s` `race.update(run.records())`; `run.stop(c)` for each `newly_decided`; `run.stop(reference)` once `reference_can_stop`; then polls every `drain_poll_s` (10 s) until every side has no trial in flight (or `drain_timeout_s`), to time the stops |
| timeline | one entry per (candidate, look) when both sides' units of that look are launched and finished: tasks, paired tasks, D, SE, t, efficacy and futility bounds, the design's verdict, elapsed time. Recomputed per look from the same units, so a poll that makes several looks ready at once still logs each one, in order, up to the decision |
| summary JSON | design, decisions (`RaceStatus.as_dict`), timeline, per side planned / launched / finished / scored / env_failed / cancelled / never launched, stops (when, why, in flight at the stop, settled after how many seconds), a compact poll log; rewritten every poll |
| `replay(records, race)` | the race offline on a finished evaluation's records, with a snapshot at every finish time (by `started_at`/`finished_at`): what a continuously polling driver would have decided, and when |
| `FileRun({cand: run_dir})` | an `EvalRun` over `python -m harnessforge.evals run` jobs, one per side on its own node: reads each `records.jsonl`, stops a side by creating `<run_dir>/<cand>.stop`, done when every job's `status.json` says `done` (the evals CLI now writes it every poll, `done: true` after the final records) |

CLI:

```bash
# one `evals run` job per side, the driver attached to their run dirs (what the pilot ran)
python -m harnessforge.race_run drive --side A=runs/race1-A --side B=runs/race1-B --reference A --candidates B \
    --out runs/race1.json --poll 60 --reverse
# or everything in one process on one node
python -m harnessforge.race_run drive --request request.json --reference A --out race.json
# offline, any orientation
python -m harnessforge.race_run replay runs/race1.records.jsonl --tasks runs/race1-A/request.json --reference B
```

Attempt-major (look-major) evaluations: `--trial-looks 2,4,8,24` or `--plan '100x1;200x2;200x8;200x24'` (default: the
request's own `trial_looks` / `plan` options), `--tau2/--sigma2` the planning variance components (default val200), and
`--design auto|obf|spending` (`auto`: `race.default_design`, Lan-DeMets OBF spending on observed information for trial
looks), `--alpha`, `--spending obf|pocock|power:R`, `--planned` (planned instead of observed information). Replay reads
`order` from the request JSON (or `--order`). See docs/stats.md section 7.

The val1000 search gate (docs/stats.md section 8) is task-major with `--design spending` (Lan-DeMets OBF on observed
information; `--tau2/--sigma2` set I_max) and `--mode all` (the reference's val1000 trials are on file). A frontier-weighted
order is made before launching, `python -m harnessforge.stats.sampling --tasks val1000.json --labels labels.json --tilt T
--out sampling.json` (or `--lineage lineage.json` for label chaining), its `tasks` become the request's task order, and the
race reads it with `--sampling sampling.json` (or the request's `options.sampling`). After an accept, `race_run accept
--lineage lineage.json --race race.json --candidate B --records B.records.jsonl --tasks request.json --fresh
never|marginal|always --out-topup topup.json` moves the lineage on and writes the top-up plan (only missing units, or a
full fresh plan); `race_run topup` writes a plan for any reference.

`--c` overrides the efficacy constant (default `recommended_c`: 1.77 for TASK_MAJOR / SAME_UNITS), `--mode all` pairs with
all of the reference's trials, `--max-env-fail` skips infra-suspect interim looks, `--checkpoints` sets the look
fractions.

## Pilot: candidate_1_0_2d vs base on 128 tmax train tasks

Setup: reference **A = `candidate_1_0_2d`** (tmax-v11-perfonly, "mandatory pre-submit verification checkpoint"; in the
old loop -5.4 points vs base on this train subset, t = -1.78 at look 1, harbor + vLLM, 5 trials), candidate **B = base
micro-swe-agent** (the harness the perf-only run's `candidate_1_0_1` A side ran, `mini_fork_local`). Tasks: that run's
128-task train subset (`samples/candidate_1_0_1-train_sample`, the one 2d was gated on), shuffled once (seed 20261119).
4 trials per task, TASK_MAJOR, looks at 32/64/96/128 tasks, `Race("A", ["B"])`, `SAME_UNITS`, c = 1.77 (efficacy -, 2.50,
2.04, 1.77; linear non-binding futility -0.375, 0.25, 0.875). Qwen3.5-4B on the 4-node SGLang pool `n4hf` (32 engines,
served name `policy`), sampling config `Qwen3.5-4B.yaml`, agent timeout x3, parity options on. One rt_HC job per side
(`evals run`, concurrency 48 each), the driver attached with `FileRun` in side A's job (poll 60 s).

| look | tasks (paired) | A acc | B acc | D = B - A +- SE | t | efficacy | futility | verdict | elapsed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 (32) | 0.458 | 0.458 | 0.000 +- 0.042 | 0.00 | - | -0.375 | continue | 54 min |
| 2 | 64 (64) | 0.448 | 0.428 | -0.020 +- 0.034 | -0.58 | 2.50 | 0.25 | **fail** (futility) | 70 min |

- **Decision**: B failed at look 2 on futility; B was stopped, then A ("no candidate alive"). The -5.4-point gap of the
  old loop did not reproduce: on every non-cancelled trial (101 tasks with data on both sides) B - A = -0.008 +- 0.028
  (t = -0.29). 2d and base are indistinguishable here. The old result was either noise (t = -1.78 is p of about 0.04
  one-sided, after a selection over 4 candidates per round) or an artifact of harbor + vLLM vs platoon + SGLang. On this
  pair the race saved trials, but its verdict is "no evidence B is better", not "A is worse".
- **Trials** (planned 512 per side):

  | side | launched | finished, scored | env_failed | cancelled by the stop | never launched |
  |---|---|---|---|---|---|
  | A | 439 | 384 | 7 | 48 | 73 |
  | B | 426 | 369 | 9 | 48 | 86 |

  So 159 of 1024 trials were never launched and 96 were cut off mid-run. The race ended at 70 min. Running to the end
  would have taken roughly 1.6x as long, the full prefix plus the timeout tail.
- **Stopping**: the stop files were written at the look-2 poll. Each `evals run` picked them up within its 15 s poll,
  and both platoon processes exited on the first SIGTERM about 10-13 s later ("the evaluation stopped on SIGTERM", no
  SIGKILL, 0 sandboxes left to reap). The driver saw both sides settled 20 s after the stop (its 10 s drain poll). Both
  jobs exited on their own about 20 s after the decision.
- **Look latency is the timeout tail.** Median trial 157-162 s, but p90 28-30 min and max 34-47 min (agent timeout x3).
  A task-major look waits for the slowest trial of its prefix, so look 1 (25%) came at 54 min and look 2 only 16 min
  later. The first look is bounded by the timeout, not the throughput.
- **env failures** (16, 1.8%): all `harness_setup` from the `harbor_system_deps` apt step (an Ubuntu mirror mid-sync,
  "File has unexpected size"). They were classified env_failed and left out of the pairing, as designed (env-failure
  share 6% at look 1 and 3% at look 2, `max_env_fail` off).
- **Reverse orientation** (offline, `replay` on the same records, `Race("B", ["A"])`): look 1 t = 0.00 (continue), look 2
  t = +0.58 (continue: above futility 0.25, below 2.50); then **undecided**, because the forward stop cut off the data for
  look 3. The expected "futility stop of A at look 1" assumed the old -5.4-point gap. With D about 0 the orientations are
  mirror images (t = -0.58 vs +0.58), and only the forward one crosses its futility bound (0.25), which is asymmetric by
  design.
- **Offline replay = live**: replaying the forward race on the records gives the same verdicts and t at each look.
  Decision time from the first launch is 68.6 min (live: 70 min, 60 s polling).

Bug found and fixed while doing this: `replay` first counted trials cancelled by the stop as finished env-failures,
so in the reverse race look 3 became "ready" on 94 tasks of truncated data and "failed" A. Cancelled units now never
finish in a replay (`snapshot_at`). The live race was not affected: a side is stopped only after it is decided or no
candidate is alive. The same caution applies to any consumer of a stopped run's records: `error="stopped"` units are
not data.

Files on ABCI: `~/hf-dev/race/` (`runs/race1.json` summary and timeline, `runs/race1.records.jsonl`,
`runs/race1.reverse.json` the corrected reverse, `logs/driver.log`, per-side run dirs `runs/race1-{A,B}`,
`race1-side.sh` the job script, `prep.py` task order and requests).
