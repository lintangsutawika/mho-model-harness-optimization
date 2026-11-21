# harnessforge.stats: sequential paired testing

`harnessforge.stats` decides whether a candidate harness beats a reference while both are still
being evaluated, and stops trials as soon as the answer is clear. It reads `TrialRecord`s only,
so it does not depend on the backend. It is pure and stdlib-only; numpy is an optional speed-up
for the calibration CLI.

| module | what it does |
|---|---|
| `paired.py` | per-task paired statistics (`paired`), fixed-sample `decide` (better path; optional cheaper path) |
| `looks.py` | `LookSchedule` for both `Order`s (trial looks and general look plans for ATTEMPT_MAJOR), `collect` (one candidate's data for one look, with readiness) |
| `information.py` | `Information(tau2, sigma2)`: planned information I(k) = n / (tau2 + sigma2 / k), `VAL200` |
| `boundaries.py` | `Design`: O'Brien-Fleming efficacy c / sqrt(tau), non-binding futility, final look; `LookBounds`, `Verdict` |
| `spending.py` | `SpendingDesign`: Lan-DeMets alpha spending on planned or observed information (section 7) |
| `calibrate.py` | Monte Carlo calibration (`simulate`, `Simulation.c_for`, `.operating`) and the CLI |
| `race.py` | `Race`: K candidates vs one reference, run concurrently, as a pure state machine |
| `gate.py` | `ValGate`: candidate vs current best on the val split (a one-candidate race) |

## 1. Paired statistic

Candidate B and reference A run the same tasks. Each side's trials are averaged per task first,
so the unit of variation is the **task**:

    D = mean_i(B_i - A_i),  SE = sd_i(B_i - A_i) / sqrt(n),  t = D / SE

Pairing removes task difficulty, which is most of the variance of a plain accuracy. Only tasks
where both sides have scored (non-env-failed) trials are paired.

## 2. Looks: launch order, not completion order

A look is a fixed set of units `(task, trial index)`, where `index` is the launch order of the
trial within (candidate, task). A look is **ready** only when every unit has a record (it was
launched; a trial never launched has no record) and that record is finished.

| `Order` | look j uses | information fraction |
|---|---|---|
| `ATTEMPT_MAJOR` (harbor) | trials with index < r of **every** task, r = 1..K | r / K |
| `TASK_MAJOR` (platoon) | **all** trials of the first n_j tasks of `EvalRequest.tasks` (checkpoints, default 25/50/75/100%) | n_j / N |
| `ATTEMPT_MAJOR` + `trial_looks=(2, 4, 8, 24)` | trials with index < k_j of every task | I(k_j) / I(K) with `information`, else k_j / K |
| `ATTEMPT_MAJOR` + `plan` | per look, k trials of each task listed (segments over the task order; looks add tasks, trials, or both) | sum of per-task information, section 7 |

**The bias argument.** Trial duration and outcome are correlated: slow trials are mostly
timeouts, and timeouts are mostly failures. If a look used "whatever has finished so far", early
looks would hold an excess of fast successes and overstate the candidate, worst exactly where
the boundaries are most permissive about stopping. Selecting units by launch order makes the
look's sample independent of the outcomes. Waiting for every unit of the look costs some time
(the slowest trial of the look), but never biases the answer. For `TASK_MAJOR`, the request's
task list should be shuffled once, up front, so that task prefixes are random samples of tasks.

Env-failed trials (the sandbox never started) count as finished, so they never block a look, but
they carry no evidence (`TrialRecord.scored` is False). `LookData.env_fail_frac` reports their
share so a caller can treat a look as infra-suspect (`Race(max_env_fail=...)`).

## 3. Boundaries

At look j with information fraction tau_j:

    efficacy  t > c / sqrt(tau_j)     from look 2 on (first_efficacy), O'Brien-Fleming shape
    futility  t <= f_j                interim looks; non-binding
    final     pass iff t > c, else fail

With `ATTEMPT_MAJOR`, K = 5, c = 1.72: efficacy (none), 2.72, 2.22, 1.92, 1.72. The default
futility is `f(tau) = -1 + 2.5 tau`, which is (-0.5, 0, 0.5, 1.0) at K = 5, the rule validated
on the old loop. Futility is **non-binding**: c is calibrated without it, so a futility stop can
only lower the false-accept rate, and a caller may ignore one without breaking alpha.

## 4. Calibration

`calibrate.py` simulates evaluations from real per-task pass rates p_i (smoothed
`(successes + 0.5) / (trials + 1)`). The candidate's rates are `logistic(logit(p_i) + shift)`;
shift = 0 is the null. Without futility a candidate is accepted iff
`M = max_j t_j sqrt(tau_j) > c`, so c for one-sided alpha is just the (1 - alpha) quantile of M
under the null, and any alpha (e.g. Bonferroni alpha/K) is read off one simulation.

    python -m harnessforge.stats.calibrate                       # bundled 128-task tmax sample
    python -m harnessforge.stats.calibrate --reference same      # reference raced concurrently
    python -m harnessforge.stats.calibrate --order task_major --trials 2 --checkpoints .25,.5,.75,1
    python -m harnessforge.stats.calibrate --rewards records.jsonl --candidate h0 --k 4

`--rewards` takes `{task: [rewards]}` / `{task: rate}` JSON or a TrialRecord JSONL. `--backend
numpy` (extra `harnessforge[calibrate]`) does 20k simulations in about a second; the pure-python
backend takes about 15 s.

Results on the bundled sample (128 tmax tasks, mean p 0.443, 20 000 simulations; c spread over
seeds 0-2):

| order, reference | c for alpha 0.05 | alpha at c=1.72, no futility | with linear futility | B's share of units under the null |
|---|---|---|---|---|
| ATTEMPT_MAJOR K=5, `all` | 1.71-1.74 | 0.053 | 0.051 | 0.50 |
| ATTEMPT_MAJOR K=5, `same` | 1.76-1.78 | 0.055 | 0.049 | 0.49 |
| TASK_MAJOR 4 looks, 5 trials | 1.73-1.77 | 0.057 | 0.051 | 0.53 |
| TASK_MAJOR 4 looks, 2 trials | 1.75-1.76 | 0.054 | 0.049 | 0.54 |

Bonferroni: c for alpha 0.05/3 is about 2.2 and for 0.05/8 about 2.57-2.60 in every row.

Operating characteristics at K = 5 attempt looks (`all`, c = 1.74, linear futility): accuracy
gain +1.4 points: power 0.15 (fixed test 0.16), 3.1 of 5 trials per task; +2.9 points: power 0.36
(0.38), 3.5 trials; +4.3 points: power 0.62 (0.65), 3.6 trials; a 2.8-point worse candidate
uses 1.6 trials per task. Futility costs 1-2 points of power and halves the trials of null
candidates, as on the old loop.

`Race` and `ValGate` pick c by default with `race.recommended_c(schedule, mode)`: 1.72 for
ATTEMPT_MAJOR with `ReferenceMode.ALL_TRIALS` (the validated reference, `DEFAULT_C`), and 1.77
for `ReferenceMode.SAME_UNITS` or a `TASK_MAJOR` schedule, so alpha <= 0.05 holds without relying
on futility. Pass an explicit `Design` to override, or recalibrate on your own pass rates.

## 5. Racing

```python
from harnessforge.stats import Design, LookSchedule, Race

sched = LookSchedule(run.order, request.tasks, request.trials)
race = Race("A", [c.id for c in candidates], sched, Design.for_schedule(sched, c=1.77))
while True:
    st = race.update(run.records())      # snapshot: finished + in-flight trials
    for c in st.newly_decided:           # passed or failed: its remaining trials are waste
        run.stop(c)
    if st.reference_can_stop:            # no candidate alive
        run.stop("A")
        break
    run.wait(timeout=60)
winner = st.winner                       # max-t passer (better path first), or None
```

* Each candidate is tested on its own: looks in order, the first pass/fail is final. A
  candidate with several looks ready at once is still evaluated look by look.
* `update` is idempotent: repeating a snapshot (or a later one) never changes a decided verdict
  and never re-reports it in `newly_decided`. `stop` lists every decided candidate.
* The reference keeps running while any candidate is alive.
* **Pairing.** `SAME_UNITS` (default): candidate look j vs the reference's own look j, the same
  `(task, index)` units. Both sides have equal information, early looks happen while everything
  is running, and the null is symmetric. `ALL_TRIALS`: vs all of the reference's trials of the
  look's tasks, for a reference already evaluated in full. With a concurrent `ATTEMPT_MAJOR`
  reference this mode cannot look until the reference has finished. In both modes, a look is
  decided only when both sides' units are launched and finished.
* **Winner.** Among passers the largest t (at its deciding look), then larger D, then input
  order. Picking the max of K inflates the winner's apparent effect, and with K null candidates
  the chance that one passes is up to K alpha. The race is a screen. Before adoption the winner
  must pass a held-out `ValGate`. For K-way control on the race itself use a Bonferroni c,
  `simulation.c_for(alpha / K)` (CLI `--k K`).
* `max_env_fail`: interim looks whose env-failure share exceeds it are skipped. Skipping an
  efficacy look only lowers alpha. The final look is decided anyway and flagged `infra_suspect`.
* `cheaper=GateConfig(cheaper_delta=...)`: a candidate failing at the final look may pass via
  non-inferiority plus lower cost (`paired.decide`). This test is fixed-sample, outside the
  sequential calibration, and off by default.

## 6. Val gate

`ValGate(best, candidate, schedule)` is a one-candidate race on the val split. It defaults to
`ALL_TRIALS` because the current best's val results are usually on file; use `SAME_UNITS` when
the best is re-evaluated concurrently. `update(records)` returns the `CandidateState`.

## 7. Trial looks, look plans and information-based boundaries

**Why.** With k trials per task on each side, the paired per-task difference has variance tau2 + sigma2 / k (tau2:
effect heterogeneity across tasks; sigma2: per-trial Bernoulli variance summed over both sides). On val200
(`information.VAL200`: sigma2 = 0.249 = 2 x mean_i s_i(8 - s_i)/(8*7) on the seed harness, tau2 = 0.0081 from orig vs
stderr) the information I(k) = n / (tau2 + sigma2 / k) grows 7.2x from 2 to 24 trials (not 12x) and 2.1x from 8 to 24
(not 3x). Boundaries indexed by k / K put far too little alpha on early looks: an O'Brien-Fleming boundary at "1/12 of
the information" is 6.1, at the real fraction 0.139 it is 5.1.

**Schedules.** `LookSchedule(Order.ATTEMPT_MAJOR, tasks, 24, trial_looks=(2, 4, 8, 24), information=VAL200)`: look j uses
the trials with index < k_j of every task. The general form is a `plan`: per look, segments (end, k) over the shuffled
task order, so a look may add tasks, add trials to tasks it has, or both, e.g.

    (a)  ((50, 2),), ((50, 2), (100, 4)), ((50, 2), (100, 4), (150, 8)), ((50, 2), (100, 4), (150, 8), (200, 24))
    (b)  ((100, 1),), ((200, 2),), ((200, 8),), ((200, 24),)

Look j's planned information is I_j = sum over its tasks of 1 / (tau2 + sigma2 / k_task), its fraction I_j / I_J.
**Estimator** for a look whose tasks have unequal k: the inverse-variance weighted mean D = sum w_i d_i / sum w_i with
w_i = 1 / (tau2 + sigma2 / k_i) from the planning values (fixed, not estimated), and the robust (sandwich) standard error
SE = sqrt(n / (n - 1) sum w_i^2 (d_i - D)^2) / sum w_i (`paired(weights=)`, `LookSchedule.weights`), so the SE is right
even when the planning values are not. With equal k it is the plain mean and sd / sqrt(n). Readiness and units are as
before (launch order; env-failed units finished but not evidence). `launch_order()` is the order a backend should launch
the units: look by look, within a look by trial index then task. `PlatoonBackend` does this with
`options.order = "attempt_major"` (+ `trial_looks` / `plan`), see docs/evals.md.

**Designs.**

* `Design` (c / sqrt(tau), c calibrated by simulation) works unchanged on the new fractions.
* `SpendingDesign` (`spending.py`): Lan-DeMets alpha spending, by default the one-sided O'Brien-Fleming type
  alpha(t) = 2 - 2 Phi(z_{1-alpha/2} / sqrt(t)) (also `pocock`, `power:R`). Boundaries from the cumulative spend under
  the canonical joint normal, corr(Z_i, Z_j) = sqrt(I_i / I_j), by recursive numerical integration (pure python;
  reproduces gsDesign's sfLDOF bounds 4.3326, 2.9631, 2.3590, 2.0141 for 4 equal looks at alpha 0.025). Fractions
  are *planned* (fixed boundaries) or *observed*: t_j = (1 / SE_j^2) / I_J,planned, capped at 1. An interim look whose
  observed information reaches the plan becomes the final look and spends the remainder; the last look always spends
  the remainder (below or above plan). Skipped (infra-suspect) looks and looks before `first_efficacy` spend nothing.
  Efficacy from look 1 by default.
* Futility: non-binding; the old linear rule -1 + 2.5 t on the new (observed or planned) fractions.
* `Race` hands every look's observed information and the candidate's earlier looks to `design.bounds`, so each
  candidate gets the boundaries of the information it actually has. For a schedule with `trial_looks` or a `plan` the
  default design is OBF spending on observed information at alpha 0.05 (`race.default_design`); without them nothing
  changes. `race_run` (`--trial-looks`, `--plan`, `--design`, `--spending`, `--planned`) and `replay` work with it.

Planned boundaries (alpha 0.05 one-sided, val200 planning values):

| schedule | fractions | efficacy | futility | final SE (points) |
|---|---|---|---|---|
| (2,4,8,24) | .139 .263 .471 1 | 5.12 3.65 2.63 1.66 | -0.65 -0.34 0.18 | 0.96 |
| (2,4,8,16,24) | .139 .263 .471 .781 1 | 5.12 3.65 2.63 1.96 1.73 | -0.65 -0.34 0.18 0.95 | 0.96 |
| (4,8,16,24) | .263 .471 .781 1 | 3.65 2.63 1.96 1.73 | -0.34 0.18 0.95 | 0.96 |
| (4,8,16,32) | .226 .405 .671 1 | 3.96 2.87 2.14 1.70 | -0.44 0.01 0.68 | 0.89 |
| (a) 50x2 +50x4 +50x8 +50x24 | .074 .215 .466 1 | 7.09 4.07 2.65 1.66 | -0.81 -0.46 0.17 | 1.40 |
| (a2) 50x4 +50x8 +50x16 +50x48 | .090 .253 .521 1 | 6.41 3.73 2.48 1.67 | -0.77 -0.37 0.30 | 1.13 |
| (b) 100x1, 200x2, 200x8, 200x24 | .036 .139 .471 1 | - 5.12 2.63 1.66 | -0.91 -0.65 0.18 | 0.96 |

### 7.1 Simulation (`scripts/trial_looks_sim.py`)

Model: 200 base rates resampled from the val200 seed-harness results (`stats._val200`, successes of 8, shrunk
(s + a) / (8 + 2a), a = 0.246, so that 2 mean p(1 - p) = 0.249); the candidate's rates logistic(logit p + s + b z),
z ~ N(0, 1), with (s, b) solved for mean difference delta and true-difference variance tau2 (at +14 points the shift
alone gives tau2 = 0.0097 > 0.0081, so b = 0 there); every trial Bernoulli; SAME_UNITS pairing (reference raced on the
same units). "identical" is the sharp null (q = p). 20 000 simulations for each null, 10 000 for each alternative (MC se
of alpha 0.0015, of power <= 0.005). The observed-information designs run through `SpendingDesign.bounds` with the
observed fractions rounded to 2e-3 (library default 1e-3; it moves a boundary by < 0.01 at the earliest look). Units
= trials per arm at the deciding look (4800 = 200 x 24). Spending rows are "observed information, linear futility"
unless marked.

Tau2 as planned:

| design | alpha identical | alpha delta 0, tau2 | power +2.4 | +5 | +14 | E[units] at -2 | 0 | +2.4 | +14 | max | P(look 1) at +14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (2,4,8,24) | 0.039 | 0.047 | 0.750 | 0.994 | 1.000 | 860 | 2267 | 3671 | 513 | 4800 | 0.72 |
| (2,4,8,24) no futility | 0.051 | 0.052 | 0.804 | 1.000 | 1.000 | 4799 | 4778 | 4156 | 513 | 4800 | 0.72 |
| (2,4,8,24) planned info | 0.040 | 0.046 | 0.753 | 0.995 | 1.000 | 869 | 2312 | 3751 | 529 | 4800 | 0.68 |
| (2,4,8,24) c/sqrt(t), c = 1.687 | 0.040 | 0.045 | 0.749 | 0.995 | 1.000 | 869 | 2302 | 3575 | 457 | 4800 | 0.86 |
| (2,4,8,16,24) | 0.041 | 0.046 | 0.736 | 0.994 | 1.000 | 800 | 1879 | 2931 | 513 | 4800 | 0.72 |
| (2,4,8,16,24) planned info | 0.041 | 0.046 | 0.745 | 0.995 | 1.000 | 807 | 1914 | 3001 | 529 | 4800 | 0.68 |
| (4,8,16,24) | 0.043 | 0.047 | 0.754 | 0.998 | 1.000 | 1056 | 2078 | 3041 | 800 | 4800 | 1.00 |
| (4,8,16,24) planned info | 0.043 | 0.047 | 0.765 | 0.999 | 1.000 | 1064 | 2116 | 3118 | 800 | 4800 | 1.00 |
| **(4,8,16,32)** | 0.042 | 0.046 | **0.805** | 0.999 | 1.000 | 1121 | 2618 | 3923 | 800 | 6400 | 1.00 |
| (4,8,16,32) no futility | 0.048 | 0.052 | 0.847 | 1.000 | 1.000 | 6388 | 6321 | 4406 | 800 | 6400 | 1.00 |
| (a) 50x2 +50x4 +50x8 +50x24 | 0.043 | 0.042 | 0.469 | 0.922 | 1.000 | 485 | 895 | 1383 | 392 | 1900 | 0.00 |
| (a) no futility | 0.049 | 0.050 | 0.537 | 0.974 | 1.000 | 1899 | 1891 | 1790 | 392 | 1900 | 0.00 |
| (a2) 50x4 +50x8 +50x16 +50x48 | 0.042 | 0.044 | 0.623 | 0.974 | 1.000 | 729 | 1657 | 2708 | 588 | 3800 | 0.03 |
| (a2) no futility | 0.046 | 0.051 | 0.694 | 0.998 | 1.000 | 3798 | 3775 | 3311 | 588 | 3800 | 0.03 |
| (b) 100x1, 200x2, 200x8, 200x24 | 0.037 | 0.044 | 0.706 | 0.967 | 1.000 | 877 | 2208 | 3457 | 739 | 4800 | 0.00 |
| current: task-major 4 looks x 4 trials, c = 1.77 | 0.045 | 0.041 | 0.318 | 0.811 | 1.000 | 320 | 426 | 571 | 400 | 800 | 0.00 |
| current ATTEMPT_MAJOR r/K, 24 looks, c = 1.77 | 0.044 | 0.046 | 0.737 | 0.988 | 1.000 | 634 | 1649 | 2626 | 550 | 4800 | 0.00 |
| k/K fractions at (2,4,8,24), c = 1.77 | 0.033 | 0.037 | 0.745 | 0.997 | 1.000 | 1074 | 2788 | 4206 | 800 | 4800 | 0.00 |
| fixed 24 trials, z > 1.645 | 0.050 | 0.051 | 0.806 | 1.000 | 1.000 | 4800 | 4800 | 4800 | 4800 | 4800 | - |

Sensitivity: true tau2 = 0.5x and 2x the planning value (designs still planned with 0.0081):

| design | alpha 0.5x | power +2.4 0.5x | E[units] 0 / +2.4, 0.5x | alpha 2x | power +2.4 2x | E[units] 0 / +2.4, 2x |
|---|---|---|---|---|---|---|
| (2,4,8,24) | 0.040 | 0.798 | 2137 / 3430 | 0.047 | 0.643 | 2481 / 3982 |
| (2,4,8,24) planned info | 0.040 | 0.814 | 2318 / 3680 | 0.046 | 0.637 | 2318 / 3779 |
| (2,4,8,16,24) | 0.043 | 0.756 | 1707 / 2586 | 0.047 | 0.642 | 2199 / 3534 |
| (4,8,16,24) | 0.044 | 0.777 | 1892 / 2688 | 0.049 | 0.655 | 2427 / 3664 |
| (4,8,16,24) planned info | 0.043 | 0.827 | 2121 / 3021 | 0.047 | 0.636 | 2128 / 3233 |
| (4,8,16,32) | 0.042 | 0.835 | 2291 / 3275 | 0.049 | 0.692 | 3169 / 4923 |
| (4,8,16,32) planned info | 0.042 | 0.878 | 2666 / 3811 | 0.046 | 0.680 | 2700 / 4265 |
| (a) | 0.044 | 0.512 | 866 / 1351 | 0.043 | 0.403 | 952 / 1441 |
| (a2) | 0.044 | 0.677 | 1513 / 2471 | 0.043 | 0.516 | 1868 / 2961 |
| (b) | 0.038 | 0.758 | 2084 / 3262 | 0.045 | 0.612 | 2396 / 3763 |
| current task-major 4x4 | 0.044 | 0.328 | 431 / 572 | 0.043 | 0.296 | 427 / 565 |
| fixed 24 | 0.049 | 0.885 | 4800 / 4800 | 0.051 | 0.670 | 4800 / 4800 |

Reading it:

* Alpha holds everywhere (<= 0.052 without futility, 0.037-0.049 with it), also when tau2 is off by 2x.
* Power is capped by the 24-trial maximum: the fixed 24-trial test has 0.806 at +2.4, and every sequential
  24-trial design loses 2-5 points to early looks and about 5 more to futility (0.73-0.77). **Only (4,8,16,32) meets
  "about 80%" with futility (0.805).** With tau2 twice the plan nothing does (fixed 24: 0.67).
* +14 points stops at look 1 always when look 1 has 4 trials per task, 68-72% of the time with 2, never with plans
  (a), (a2), (b) (their first look has too little information: boundary 7.1 and 6.4; (b)'s look 1 at t = 0.036
  has alpha(t) below 1e-15, so no efficacy boundary at all).
* Neutral and worse candidates: -2 points stops after 800-1100 trials per arm (2-5 per task), delta 0 after
  1900-2600 (futility; without it they run to the end).
* Observed vs planned information: the same at the planned tau2. When tau2 is smaller than planned, observed
  information reaches I_max early and that look becomes final, so it saves trials but gives back the extra power
  (e.g. (4,8,16,24): 0.777 vs 0.827); when tau2 is larger it spends alpha by the real information and keeps a
  little more power. The observed version keeps the boundaries valid whatever tau2 is; the planned one is the
  cheaper choice of fixed boundaries if tau2 is trusted.
* k / K fractions with c = 1.77 (today's code on trial looks) are conservative (alpha 0.033) and spend 20-35% more
  trials than the information-based designs at the same power.
* Plan (a) (max 1 900 per arm): final SE 1.40 points, fixed-test power at +2.4 about 0.53, sequential 0.47. Plan (a2)
  doubles every count (max 3 800): SE 1.13 points, power 0.62. Neither reaches 80%: they spend trials on depth for a
  quarter of the tasks, where breadth (all 200 tasks) is worth more because tau2 > 0.

### 7.2 Wall time and cost (`scripts/trial_looks_cost.py`)

Model: every arm is one platoon process with a fixed concurrency of 192 / arms (2 rt_HC sandbox hosts x 96 slots, pool
node 0 = 8 GPUs); units launched in the schedule's launch order (task-major for the task-major baseline); each trial's
duration drawn from the empirical durations of that task (9 649 trials of the val200 `main` and `stack-r2` runs: median
107 s, p90 794 s, max 2 107 s), task order reshuffled per replicate; a look is ready when all its units of both arms have
finished, the driver sees it within its 60 s poll, the stop takes 20 s. The stop look comes from 7.1 (tau2 as planned),
assumed independent of durations. Calibration: on the measured runs' setups the simulated task-major look times are
57/95/135/176 min (main: measured about 62/100/143/184) and 51/79/107/139 min (stack-r2: 56-74/79-104/119/127): the model
is 5-10% optimistic. GPU-h = 8 x node-hours, rt_HC node-h = 2 x node-hours. 40 duration replicates x 20 stop draws.

One candidate + reference alone on the node (2 arms x 96 slots); hours to decision, mean / p90:

| design | delta -2 | 0 | +2.4 | +14 | GPU-h at 0 / +2.4 | rt_HC node-h at 0 | trials launched / arm at 0 | waste at 0 |
|---|---|---|---|---|---|---|---|---|
| (2,4,8,24) | 1.06 / 1.71 | 2.22 / 4.34 | 3.39 / 4.39 | 0.79 / 1.05 | 17.8 / 27.1 | 4.4 | 2642 | 29% |
| (2,4,8,16,24) | 1.07 / 1.71 | 1.89 / 4.21 | 2.75 / 4.32 | 0.81 / 1.06 | 15.1 / 22.0 | 3.8 | 2387 | 33% |
| (4,8,16,24) | 1.26 / 1.71 | 2.16 / 4.25 | 2.88 / 4.29 | 1.03 / 1.07 | 17.3 / 23.0 | 4.3 | 2712 | 26% |
| (4,8,16,32) | 1.30 / 1.71 | 2.52 / 5.59 | 3.62 / 5.62 | 1.04 / 1.08 | 20.2 / 29.0 | 5.0 | 3114 | 24% |
| (a) | 0.79 / 1.06 | 1.12 / 1.97 | 1.53 / 2.00 | 0.68 / 0.94 | 9.0 / 12.3 | 2.2 | 1306 | 43% |
| (a2) | 1.00 / 1.56 | 1.77 / 3.56 | 2.56 / 3.61 | 0.87 / 0.95 | 14.2 / 20.4 | 3.5 | 2137 | 35% |
| (b) | 1.08 / 1.73 | 2.24 / 4.32 | 3.19 / 4.38 | 1.00 / 1.70 | 17.9 / 25.5 | 4.5 | 2669 | 33% |
| current task-major 4 x 4, c = 1.77 | 0.65 / 0.86 | 0.73 / 1.00 | 0.84 / 1.05 | 0.70 / 0.74 | 5.8 / 6.7 | 1.5 | 794 | 46% |
| current r/K, 24 looks | 0.91 / 1.52 | 1.74 / 3.39 | 2.52 / 3.97 | 0.83 / 0.91 | 13.9 / 20.1 | 3.5 | 2243 | 38% |
| k/K at (2,4,8,24), c = 1.77 | 1.25 / 1.75 | 2.46 / 4.35 | 3.81 / 4.37 | 1.03 / 1.07 | 19.6 / 30.5 | 4.9 | 2881 | 26% |
| fixed 24 | 4.30 / 4.37 | 4.30 / 4.37 | 4.30 / 4.37 | 4.30 / 4.37 | 34.4 / 34.4 | 8.6 | 4800 | 0% |

Waste = share of the trials an arm launched that the deciding look did not use (cancelled in flight or finished beyond
the look). A look waits for its slowest unit: with p90 trials of 13 min and a max of 35 min, every look costs at least
one timeout tail, which is why look 1 at 2 trials (400 units per arm) takes about 0.7-0.8 h, not 400 / 960 per h.

Search round: 4 candidates + reference on one node (5 arms x 38 slots); a round ends when the last candidate is decided:

| design | round h, mean / p90: 0, 0, -2, -2 | 0, 0, -2, +2.4 | 0, 0, -2, +14 | rounds / 24 h (0, 0, -2, +2.4) | GPU-h / round | GPU-h / candidate |
|---|---|---|---|---|---|---|
| (2,4,8,24) | 7.42 / 10.35 | 9.39 / 10.38 | 7.14 / 10.33 | 2.6 | 75.1 | 18.8 |
| (2,4,8,16,24) | 5.91 / 10.21 | 7.81 / 10.31 | 6.12 / 10.24 | 3.1 | 62.5 | 15.6 |
| (4,8,16,24) | 6.39 / 10.26 | 7.93 / 10.32 | 6.27 / 10.26 | 3.0 | 63.5 | 15.9 |
| (4,8,16,32) | 8.10 / 13.61 | 10.65 / 13.67 | 8.41 / 13.63 | 2.3 | 85.2 | 21.3 |
| (a) | 3.32 / 4.42 | 3.91 / 4.47 | 3.15 / 4.39 | 6.1 | 31.3 | 7.8 |
| (a2) | 5.69 / 8.60 | 7.15 / 8.70 | 5.60 / 8.56 | 3.4 | 57.2 | 14.3 |
| (b) | 7.59 / 10.39 | 9.16 / 10.44 | 7.12 / 10.37 | 2.6 | 73.3 | 18.3 |
| current task-major 4 x 4, c = 1.77 | 1.56 / 2.02 | 1.75 / 2.06 | 1.55 / 2.03 | 13.8 | 14.0 | 3.5 |
| current r/K, 24 looks | 5.63 / 9.09 | 7.13 / 9.92 | 5.37 / 9.14 | 3.4 | 57.0 | 14.3 |
| fixed 24 | 10.36 / 10.53 | 10.36 / 10.53 | 10.36 / 10.53 | 2.3 | 82.8 | 20.7 |

A round is as slow as its slowest candidate, and with fixed per-arm concurrency the slots of decided candidates stay
idle. Handing them to the arms still running (one platoon process per arm cannot do that today) would shorten rounds
substantially; it is not modelled.

### 7.3 Recommendation

**(4,8,16,32) with OBF spending on observed information and linear non-binding futility** if a +2.4-point fix must be
accepted with about 80% power: alpha 0.042-0.049, power 0.805 (0.835 / 0.692 at tau2 0.5x / 2x), +14 always at look 1
(1.0 h), -2 stopped after about 5.6 trials per task (1.3 h), 2.5 h (p90 5.6 h) for a neutral candidate alone on a node;
about 2.3 search rounds of 4 candidates per node-day. It needs `trials = 32`.

If 75% power is acceptable, **(4,8,16,24)** is the 24-trial choice: power 0.754, the same look-1 stop at +14, 2.2 h for a
neutral candidate, 3.0 rounds per node-day. It dominates (2,4,8,24) (same power, +14 always at look 1, cheaper rounds) and
(b); (2,4,8,16,24) is about 10% cheaper but has 2 points less power and stops +14 at look 1 only 72% of the time.
Plans (a) and (a2) are cheaper but cannot reach the power target (0.47, 0.62).

The current task-major 4 x 4 design costs a fifth as much (13.8 rounds per node-day) but has power 0.32 at +2.4: it
mostly finds large fixes.
