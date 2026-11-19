# harnessforge.stats: sequential paired testing

`harnessforge.stats` decides whether a candidate harness beats a reference while both are still
being evaluated, and stops trials as soon as the answer is clear. It reads `TrialRecord`s only,
so it does not depend on the backend. It is pure and stdlib-only; numpy is an optional speed-up
for the calibration CLI.

| module | what it does |
|---|---|
| `paired.py` | per-task paired statistics (`paired`), fixed-sample `decide` (better path; optional cheaper path) |
| `looks.py` | `LookSchedule` for both `Order`s, `collect` (one candidate's data for one look, with readiness) |
| `boundaries.py` | `Design`: O'Brien-Fleming efficacy, non-binding futility, final look; `Verdict` |
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
