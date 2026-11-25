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
| `gate.py` | `ValGate`: candidate vs current best on the val split (a one-candidate race); `val1000_schedule` (section 8) |
| `sampling.py` | `TaskSampling`, `frontier_order`: frontier-weighted task-major look orders, Horvitz-Thompson weights (section 8) |
| `reference.py` | reference reuse on the gate: `topup_plan`, `apply_topup`, `Lineage` (label chaining, fresh-after-marginal) |

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

## 8. The val1000 search gate

**Design (decided).** The search gate compares a candidate with the current reference on **val1000**: 1 000 held-out
tmax tasks x 4 trials per side, TASK_MAJOR, looks after 250 / 500 / 750 / 1 000 tasks, Lan-DeMets alpha spending with the
O'Brien-Fleming shape (one-sided alpha 0.05) on **observed** information, efficacy from look 1, non-binding linear futility
-1 + 2.5 t. It replaces the val200 attempt-major (4, 8, 16, 32) design for the search gate; that code stays (val200 is kept
for checkpoint evals and confirmatory tests).

```python
from harnessforge.stats import Race, ReferenceMode, val1000_schedule
sched = val1000_schedule(tasks)                       # tasks: val1000 in launch order; or sampling=TaskSampling
gate = Race(ref, [cand], sched, reference_mode=ReferenceMode.ALL_TRIALS)   # default design: LD-OBF, observed info
```

`LookSchedule` now accepts planning `information` for TASK_MAJOR (I_j = n_j / (tau2 + sigma2 / 4), I_max = 1000 / 0.0704
= 14 215 at the val200 planning values, final SE 0.84 points), and `race.default_design` then picks
`SpendingDesign.for_schedule` (observed information), as it does for attempt-major trial looks. Planned boundaries for 4
equal looks: **3.75 / 2.54 / 2.02 / 1.72** (futility -0.375 / 0.25 / 0.875); exact normal-theory alpha 0.0500. CLI:
`race_run drive ... --mode all --design spending [--sampling sampling.json]`.

### 8.1 Frontier-weighted looks (`sampling.py`, toggle; default off = uniform order)

With `--sampling` / `LookSchedule(sampling=TaskSampling)` every look is a **stratified sample of the full set** instead of a
uniform prefix. Tasks are put in difficulty bands by a label pass rate, [0, .25) [.25, .5) [.5, .75) [.75, 1] (plus an
"unlabeled" band), and each look takes from every band, over-sampling the frontier. Band priority: 0.25-0.5 > 0.5-0.75 >
0-0.25 > 0.75-1, multipliers (2, 4, 3, 1) ** tilt for the bands in rate order (tilt 0 = proportional, tilt 1 = the
0.25-0.5 band at 4x the sampling rate of the 0.75-1 band). At look j the cumulative per-band fraction is
f_bj = clip(c_j m_b, floor x n_j / N, 1), c_j solved so the look has exactly n_j tasks; `floor` (default 0.5) keeps every
band at no less than half its proportional share from look 1 on. Counts are integerised by largest remainders and never
decrease; the last look has every task. Within a band the tasks are a seeded random permutation, so look j's tasks of
band b are a simple random sample of the band and the inclusion probability is exactly pi_i = n_bj / N_b.

**Estimator.** Hajek / Horvitz-Thompson with w_i = 1 / pi_i: D_j = sum w_i d_i / sum w_i = sum_b (N_b / N) mean_b(d), the
full-set mean, at every look (sum w = N exactly, so Hajek = HT). SE: the stratified with-replacement (sandwich) variance
`paired(weights=, strata=)`, SE^2 = sum_b (N_b / N)^2 s_b^2 / n_bj (no finite-population correction: tasks are treated
as a sample from a super-population, as the plain statistic does). Observed information for the spending function is
1 / SE^2 of this estimator. Looks whose weights are all equal (the last look; tilt 0 up to rounding) use the plain
paired mean and sd / sqrt(n), so **the final look is the uniform full-set analysis** and a uniform order reproduces the
existing statistic exactly. Planned fractions with a tilt include the design effect n_j sum_b (N_b / N)^2 / n_bj of
unequal weights (equal per-task variance); the observed information decides the boundaries anyway.

**Labels** (`load_labels`, `labels_from_records`): any per-task pass rate from data **not** used in the comparison:
(a) a separate pass of a harness on val1000 (1 trial is enough to run, but see 8.3: 1-trial labels only separate 0 from
1), (b) the previous reference's val1000 pass rates (label chaining, `Lineage.labels()`), or at step 0 the base-model
sweep (`labels.json` of the val1000 draw, 8.4). HT stays design-unbiased for any labels (the labels fix pi, the
within-band sampling is random); labels taken from the comparison data itself make the weights a function of the
reference's noise, and 8.3 shows that this biases the SE downward.

```bash
python -m harnessforge.stats.sampling --tasks val1000.json --labels labels.json --tilt 0.5 --seed S --out sampling.json
python -m harnessforge.stats.sampling --tasks val1000.json --lineage lineage.json --tilt 0.5 --seed S --out sampling.json
```

`sampling.json["tasks"]` is the launch order for the request; the race reads the design from `--sampling` (or
`options.sampling` in the request).

### 8.2 Reference reuse with top-up (`reference.py`)

The gate's reference is compared on all its val1000 trials (`ReferenceMode.ALL_TRIALS`; in TASK_MAJOR this is the same
unit set as SAME_UNITS). When candidate B is **accepted at look k**, its gate trials become the new reference's val1000
trials: `topup_plan(records, B, tasks, 4)` lists the missing units (tasks after look k's prefix, trials cancelled by the
stop, env failures; a unit counts when it has a finished, scored, not-cancelled record) and `apply_topup` re-indexes a
top-up run's trials onto them (provenance in `extra`). `Lineage` (JSON) tracks the current reference, the previous one
(its records give the next gate's labels: label chaining) and one history entry per accept (look, t, marginal, fresh,
units reused / missing). `Lineage.accept(..., fresh=never|marginal|always)`: a fresh full evaluation of the new
reference instead of a top-up, never / after a marginal accept (at the final look, or t - efficacy < 0.5) / always.

```bash
python -m harnessforge.race_run accept --lineage lineage.json [--reference A --step0-labels labels.json] \
    --race race.json --candidate B --records B.records.jsonl --tasks request.json --fresh marginal --out-topup topup.json
python -m harnessforge.race_run topup --records REF.records.jsonl --reference REF --tasks request.json --out topup.json
```

### 8.3 Simulation (`scripts/val1000_sim.py`)

Model: per replicate 1 000 true reference rates from a smoothed NPMLE (EMS on a logit grid) of the val200 seed-harness
results (8 trials per task): 35.6 always solved / 45.5 never per 200 at 8 trials (val200: 35 / 47; 118 mixed), mean
0.503, 2 E p(1 - p) = 0.248 (planning sigma2 0.249); true-rate band mix .30 / .18 / .17 / .35. Candidate rates
logistic(logit p + s + b z) with (s, b) solved for the mean difference and tau2 in {0.0025 (measured), 0.0081 (planning)}
(at +14 points the shift alone gives tau2 = 0.0119, b = 0). Mixed null: +6 points on true-p [.25, .75) tasks, easy tasks
(p >= .75) lowered so that the 1 000-task mean is exactly 0 per replicate; mixed harmful: the same with mean -1 point.
4 Bernoulli trials per task and side. Labels from an independent pass of the reference with 4 trials (L4) or 1 trial (L1);
"own-ref labels" = the reference's own comparison trials (a diagnostic of what not to do). Design: LD-OBF, observed
information, I_max at the planning values, linear futility, through `SpendingDesign.bounds` (observed fractions rounded
to 5e-3: moves the look-1 boundary by <= 0.02, later ones by < 0.01). 10 000 replicates per cell, common random numbers
across designs (MC se: alpha 0.002, power 0.004). Units = trials per arm at the deciding look (max 4 000 = 1 000 x 4).

**Sanity.** LD-OBF boundaries for 4 equal looks at one-sided 0.05: 3.7496 / 2.5399 / 2.0161 / 1.7202 (exact
normal-theory alpha 0.0500). The classic c / sqrt(t) with c = 1.77 (efficacy from look 2, today's task-major default):
- / 2.503 / 2.044 / 1.770, exact alpha 0.0459 (0.0460 with a look-1 boundary 3.54); the exact OBF constant for alpha 0.05
is c = 1.733 (from look 1 or look 2). So 1.77 was conservative by about 0.004 in alpha on equal task-major looks.

tau2 = 0.0025 (measured):

| design | alpha null | alpha mixed null | accept mixed harmful | power +1.6 | +2.4 | +5 | +14 | E[units] null | -2 | +2.4 | +14 | E[looks] +2.4 | P(look 1) +14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **uniform** | 0.043 | 0.041 | 0.001 | 0.558 | **0.856** | 1.000 | 1.000 | 1989 | 1190 | 2629 | 1000 | 2.63 | 1.00 |
| frontier L4 tilt 0.5 | 0.042 | 0.041 | 0.001 | 0.558 | 0.857 | 0.999 | 1.000 | 2000 | 1200 | 2656 | 1000 | 2.66 | 1.00 |
| frontier L4 tilt 1 | 0.041 | 0.041 | 0.002 | 0.558 | 0.853 | 0.999 | 1.000 | 2166 | 1280 | 2861 | 1000 | 2.86 | 1.00 |
| frontier L4 tilt 2 | 0.042 | 0.041 | 0.002 | 0.559 | 0.854 | 0.998 | 1.000 | 2196 | 1300 | 2916 | 1000 | 2.92 | 1.00 |
| frontier L1 tilt 1 | 0.042 | 0.043 | 0.002 | 0.566 | 0.859 | 0.999 | 1.000 | 2155 | 1258 | 2872 | 1000 | 2.87 | 1.00 |
| frontier L1 tilt 2 | 0.042 | 0.042 | 0.001 | 0.561 | 0.853 | 0.998 | 1.000 | 2262 | 1353 | 2995 | 1001 | 3.00 | 1.00 |
| frontier own-ref labels tilt 1 | **0.054** | **0.053** | 0.004 | 0.575 | 0.860 | 0.999 | 1.000 | 2059 | 1230 | 2629 | 1000 | 2.63 | 1.00 |
| uniform, c/sqrt(t) c = 1.77 | 0.041 | 0.039 | 0.001 | 0.565 | 0.869 | 1.000 | 1.000 | 2145 | 1208 | 2790 | 2000 | 2.79 | 0.00 |

tau2 = 0.0081 (planning; the nulls do not depend on tau2):

| design | power +1.6 | +2.4 | +5 | +14 | E[units] -2 | +2.4 | E[looks] +2.4 |
|---|---|---|---|---|---|---|---|
| **uniform** | 0.545 | **0.840** | 1.000 | 1.000 | 1229 | 2762 | 2.76 |
| frontier L4 tilt 0.5 | 0.548 | 0.840 | 1.000 | 1.000 | 1246 | 2791 | 2.79 |
| frontier L4 tilt 1 | 0.546 | 0.840 | 1.000 | 1.000 | 1332 | 2995 | 3.00 |
| frontier L4 tilt 2 | 0.545 | 0.841 | 0.999 | 1.000 | 1348 | 3036 | 3.04 |
| frontier L1 tilt 1 | 0.547 | 0.845 | 0.999 | 1.000 | 1300 | 3010 | 3.01 |
| frontier L1 tilt 2 | 0.543 | 0.840 | 0.999 | 1.000 | 1407 | 3114 | 3.11 |
| frontier own-ref labels tilt 1 | 0.563 | 0.848 | 1.000 | 1.000 | 1268 | 2763 | 2.76 |
| uniform, c/sqrt(t) c = 1.77 | 0.539 | 0.839 | 1.000 | 1.000 | 1234 | 2816 | 2.82 |

Without futility, alpha null / mixed null: uniform 0.049 / 0.046, frontier L4 tilt 1 0.048 / 0.047, own-ref labels
0.061 / 0.061. The look-1 estimate is centred on the full-set mean in every design (|mean D| <= 0.02 points under both
nulls) and its SE is calibrated (null, sd of D vs mean SE in points: uniform 1.576 / 1.570, L4 tilt 1 1.731 / 1.736, L1
tilt 2 1.840 / 1.852; mixed null L4 tilt 1 1.837 / 1.836), except with own-ref labels (mixed null 1.716 / 1.679).
Mean observed information fraction at looks 1-3 under the null: uniform .29 / .57 / .86, L4 tilt 0.5 .28 / .56 / .84,
L4 tilt 1 .24 / .47 / .77, L1 tilt 2 .21 / .41 / .74.

Reading it:

* **Alpha holds** for every valid design (0.041-0.043 with futility, <= 0.050 without), under the strong null AND the
  mixed null (gains on mid-difficulty tasks cancelled by losses on easy ones): the HT estimator targets the full-set mean
  at every look, so over-sampling the bands where the candidate gains does not leak into a false accept. A mixed
  candidate with a full-set loss of 1 point is accepted 0.1-0.2% of the time.
* **Frontier weighting buys no power and costs trials.** Power at +1.6 / +2.4 / +5 is the same as uniform within MC
  error for every tilt and label source; expected trials grow with the tilt (+0.5-1% at tilt 0.5, +8-9% at tilt 1,
  +10-14% at tilt 2, under the null and at +2.4). Unequal weights inflate the interim variance (design effect 1.28 at tilt 1 on the real val1000
  labels, 8.4) by more than over-sampling the higher-variance frontier tasks saves (the Neyman gain over proportional
  allocation is small here: the per-task variance differs between bands by about 2x, and the 0-0.25 band, which the
  priority order puts third, is mostly near-zero-variance never-solved tasks). Less information at looks 1-2 means fewer
  early stops. 1-trial labels can only separate 0 from 1, so with L1 "frontier" just means "label 0 before label 1"; it
  behaves like L4.
* **Labels must not come from the comparison data**: with the reference's own trials as labels the SE is biased low and
  alpha is 0.054 (0.061 without futility): the weights then depend on the reference's noise in d_i. With independent
  labels (a separate pass, or label chaining from the previous reference) everything is calibrated.
* So the toggle stays **off by default** (uniform order). It is valid and can be switched on (tilt <= 0.5 costs nothing
  measurable) when early looks are wanted to contain the frontier tasks for diagnostics, but not for efficiency.
* Against the classic c = 1.77 (efficacy from look 2): LD-OBF can stop a +14-point candidate at look 1 (always, 1 000
  trials per arm vs 2 000), uses 7% fewer trials under the null and 2-6% fewer at +2.4, at 1 point less power at +2.4
  with tau2 = 0.0025 (0.856 vs 0.869; the same at 0.0081) and alpha 0.043 vs 0.041.
* val1000 at 4 trials (max 4 000 trials per arm) has power 0.84-0.86 at +2.4 points and 0.55 at +1.6, and decides a
  -2-point candidate after about 1 200 trials per arm (1.2 looks), a neutral one after about 2 000.

**Winner's curse of reference reuse** (`--curse`, 20 000 replicates per design, tau2 = 0.0081): B1 at +2.4 over A (power
0.85), then B2 gated against B1, where B1's reference trials are either reused (its gate trials on the tasks of looks
<= k, the rest topped up with fresh trials) or a fresh full evaluation. B2 is +2.2 points over B1 (power row) or identical
to B1 (alpha row). Bias = mean over the 1 000 tasks of the reused reference's observed rate minus B1's true rate
(points; the final SE is about 0.84). Uniform order (frontier L4 tilt 0.5 / tilt 1 in the JSON: the same pattern):

| B1 accepted at | share of accepts | B1's D at accept (true 2.4) | ref bias, reused | B2 power, reused / fresh | B2 alpha, reused / fresh |
|---|---|---|---|---|---|
| look 1 | 0.02 | 6.1 | +0.40 | 0.55 / 0.78 | 0.010 / 0.067 (n = 313) |
| look 2 | 0.37 | 3.6 | +0.26 | 0.70 / 0.81 | 0.013 / 0.049 |
| look 3 | 0.43 | 2.5 | +0.05 | 0.83 / 0.81 | 0.028 / 0.049 |
| look 4 (final) | 0.18 | 1.9 | **-0.20** | 0.90 / 0.81 | **0.071** / 0.048 |

Over all accepts, by policy (`Lineage.accept(fresh=...)`, `race_run accept --fresh`); "fresh share" = share of accepts
that re-run the reference in full; reference trials run per accept (uniform; 4 000 = a full evaluation):

| policy | B2 power | B2 alpha | worst alpha by accept look | fresh share | reference trials per accept |
|---|---|---|---|---|---|
| never (top-up only) | 0.789 | 0.030 | 0.071 (look 4) | 0 | 1 220 |
| marginal (final look or t - eff < 0.5) | 0.777 | 0.034 | - | 0.57 | - |
| early (look <= 2) | 0.834 | 0.044 | 0.071 (look 4) | 0.39 | 1 980 |
| **looks 1, 2, 4** (`--fresh looks`) | 0.817 | 0.040 | 0.049 | 0.57 | 2 700 |
| always | 0.811 | 0.049 | 0.049 | 1 | 4 000 |

* Reuse after an **early** accept (looks 1-2, about 40% of accepts at +2.4) leaves the reference too good by 0.26-0.40
  points (0.3-0.5 final SEs) and costs the next +2.4 candidate 11-23 points of power; it also makes the next gate
  conservative (alpha about 0.01).
* Reuse after a **final-look** accept leaves it too bad by 0.20 points: the candidate reached look 4 because its early
  looks came out low. The next gate is then anti-conservative: alpha 0.071 for a null next candidate (power 0.90 instead
  of 0.81). Reuse after a look-3 accept is nearly unbiased (+0.05; alpha 0.028 because B2 is compared on the same tasks
  whose noise B1's acceptance selected).
* "Fresh after a **marginal** accept" (as first proposed) points the fresh runs at the wrong accepts: most marginal
  accepts are final-look ones (bias down, but fresh helps alpha) while the early, non-marginal accepts carry the upward
  bias; net power 0.777, below never-fresh.
* **Recommendation: `--fresh looks --fresh-looks 1,2,4`** (fresh unless accepted at look 3): every accept look keeps
  alpha <= 0.05 for the next gate (worst 0.049), power 0.82, and 57% of accepts re-run the reference (2 700 reference
  trials per accept vs 4 000 for "always"). "always" is the simple, fully clean alternative. "never" saves the most but
  lets a final-look accept inflate the next gate's alpha to 0.07.

Reproduce (about 25 min on 8 cores; `--only-curse` for the second table):

    uv run --no-project -p 3.12 --with numpy --with-editable packages/harnessforge \
        python packages/harnessforge/scripts/val1000_sim.py --sims 10000 --curse-sims 20000 --resolution 0.005 \
        --curse-designs "uniform,frontier L4 tilt 0.5,frontier L4 tilt 1" --workers 8 --out val1000_sim.json

### 8.4 The val1000 taskset (`scripts/val1000_draw.py`)

On ABCI: `/groups/gai51740/aci18914wh/mho/tasksets/val1000/` (drawn in an rt_HC job; stdlib only).

* **Train pool**: the 6 726 tasks of the GRPO v2 run's `dataset/` (`runs/platoon-grpo200-tmax-naive-uniform-noKL-64kcap-v2-ext--Qwen3.5-4B/dataset`,
  = `data-harbor/tmax-v11-train-naive`). **val200** = `data-harbor/tmax-v11-val200` (200 tasks); it does not overlap the pool
  (checked: 0 shared tasks).
* **Difficulty data**: the Qwen3.5-4B pass-rate sweep on all validated TMax v1.1 tasks (`sweeps/tmax-passrate-hpcsb/Qwen3.5-4B/shard{0..3}`:
  harbor, base mini-swe-agent harness, 4 attempts (one per shard), 1 800 s agent timeout, vLLM, no 64k cap). An attempt is
  scored when its verifier wrote a reward (agent timeouts are scored: the verifier runs after them); 6 554 of the pool's
  tasks have 4 scored attempts, 164 have 3, 7 have 1-2, 1 has none. The sweep's settings differ from the current eval
  setup (platoon, SGLang, 64k cap, keep-thinking): that affects label quality, not validity (8.1).
* **Draw**: strata on the sweep pass rate, 0 | (0, .5) | .5 | (.5, 1) | 1 | unlabeled; proportional allocation (largest
  remainder) of 1 000; seeded simple random sample within each stratum (seed 20261122); the list then shuffled once
  (its order is the uniform task-major launch order). The remaining 5 726 tasks are the screen pool.

| stratum (4B sweep rate) | 0 | (0, .5) | .5 | (.5, 1) | 1 | unlabeled | mean rate |
|---|---|---|---|---|---|---|---|
| pool (6 726) | 2 173 | 1 021 | 921 | 1 069 | 1 541 | 1 | 0.455 |
| val1000 | 323 | 152 | 137 | 159 | 229 | 0 | 0.455 |
| screen pool (5 726) | 1 850 | 869 | 784 | 910 | 1 312 | 1 | |

Files: `val1000.json` (tasks in launch order, per-task label rate and stratum, provenance), `val1000.tasks.txt`,
`screen_pool.json` / `.tasks.txt`, `labels.json` (successes / scored / attempts / exception counts for all 6 926 sweep
tasks: the step-0 difficulty labels), `manifest.json` (counts, mixes, sweep exception summary). In the gate's bands
([0, .25) [.25, .5) [.5, .75) [.75, 1]) val1000 is 323 / 152 / 140 / 385. Frontier orders on these labels
(`python -m harnessforge.stats.sampling --tasks val1000.json --labels val1000.json --tilt T`), tasks per band at look 1
(250 tasks) and the look-1 design effect: tilt 0.5: 82 / 55 / 44 / 69, 1.07; tilt 1: 78 / 73 / 51 / 48, 1.28; tilt 2:
52 / 99 / 51 / 48, 1.43 (proportional: 81 / 38 / 35 / 96).

Note: val1000's mean 4B sweep rate (0.455) is below val200's seed-harness rate under the eval setup (0.50); the
simulation (8.3) uses the val200 mix, as specified.

### 8.5 Summary

* Gate: val1000, 4 trials, task-major, LD-OBF on observed information (3.75 / 2.54 / 2.02 / 1.72 planned), linear
  non-binding futility; uniform task order (the frontier toggle is valid but gives no efficiency gain on this population).
* Power at +2.4 points 0.84-0.86, at +1.6 0.55, at +5 and +14 1.00 (+14 always at look 1); alpha 0.041-0.043 under the
  strong and the mixed-effect null; about 2 000 trials per arm for a neutral candidate, 1 200 for a -2-point one.
* Difficulty labels only from data outside the comparison (own-reference labels: alpha 0.054).
* Reference reuse: top up after a look-3 accept, re-run the reference after an accept at look 1, 2 or 4.
