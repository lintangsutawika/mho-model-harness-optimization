Follow the **propose-harness** skill (available in <available_skills>) for the
scaffold-evolution workflow: analyze, state ONE falsifiable hypothesis, implement a single
minimal additive change, validate the package, and write CANDIDATE.md.

## Your workspace
Your working directory is the run's `candidates/` archive -- it gives you access to ALL
optimization information in one place:
- `frontier.json` -- the current best candidate and its val score.
- `summary.json` -- the history of every proposal and its outcome (and val score, when it
  reached val).
- `<tag>/harness/` -- each candidate's harness code.
- `<tag>/status.json` -- that candidate's outcome: the same record as its summary row.
- `<tag>/eval_before/` and `<tag>/eval_after/` -- the real train-sample execution traces
  (result.json + agent transcripts) for each candidate's base (A) and proposed (B) harness.

You build ON the frontier-best harness. Your task: add ONE improvement to the CURRENT
candidate, editing ONLY `./<tag>/harness/` (the installable scaffold with pyproject.toml +
src/). Treat all OTHER candidates' dirs + `frontier.json` + `summary.json` + `status.json`
as READ-ONLY reference -- never edit them. Your proposal's diff must be confined to
`./<tag>/harness/` (or `./<tag>/model_specs/` for a model proposal).

## HARD CONSTRAINTS (a candidate that violates these will be REJECTED)
- Make MINIMAL, mostly-ADDITIVE edits for your ONE mechanism. Change only the files that
  mechanism needs.
- NEVER delete existing files, NEVER empty or rewrite the src/ tree, NEVER recreate the
  package from scratch. The snapshot must remain the FULL working scaffold PLUS your change.
- Keep it an installable package that still imports. Do not run destructive shell commands
  (rm -rf, git clean, moving the whole tree, etc.).

## Prior results (read these, then propose something novel)
Start by reading `./frontier.json`, `./summary.json`, and the recent candidates'
`<tag>/harness/` + `<tag>/status.json` + `<tag>/eval_before/` + `<tag>/eval_after/`
traces to identify the single most promising failure mode. Especially read `./summary.json`
-- it is the history of every proposal so far, ONE row per candidate (the same record as
that candidate's `<tag>/status.json`), labeled with its `outcome`.

How a candidate is judged: the same gate runs twice, first on the train sample (B = the
candidate vs A = its parent, on the SAME sampled tasks, `subset_seed`), then on the val split
(vs the current best). Either way B passes if it is
  - "better": clearly more accurate, judged per task (B minus A, averaged over the tasks,
    beyond the noise of the task-to-task differences), or
  - "cheaper": about as accurate (not worse by more than a small margin) AND clearly fewer
    agent output tokens.
A higher score alone is NOT enough: a small gain within the noise is a fail. Several sibling
candidates are proposed at once; only the one with the strongest train-gate result goes to val.
- `accepted`     -- passed on the train sample and on val: it is the new best (your parent).
                    `path` says how ("better" or "cheaper").
- `val_fail`     -- passed on the train sample, not on val. Its `val` score and `val_gate`
                    show by how much it missed (a val score above the best's can still fail).
- `not_selected` -- passed on the train sample, but a sibling with stronger evidence went to
                    val instead. Not a verdict on the idea.
- `train_fail`   -- did not pass on the train sample.
- `no_proposal`  -- the proposer produced no valid harness.
- `no_score`     -- an evaluation failed for infrastructure reasons (not a verdict on the idea).
- `baseline`     -- the starting harness.
Each row also has the hypothesis, the ts_A/ts_B train-sample scores and the `train_gate`
details; rows that reached val also have `val` and `val_gate`.

**AVOID REPEATING PRIOR ATTEMPTS.** Your proposed mechanism MUST be **novel** -- it must
not re-propose the same underlying fix that already appears in `./summary.json` or in a
prior candidate's harness. If an idea is already there and failed (REJECTED), or is a
re-wording of one that was tried, pick a different failure mode. Build on ACCEPTED winners,
avoid repeating rejected regressions.

NOTE on val: you see each candidate's val SCORE (in summary.json / status.json /
frontier.json) -- that is the signal of which ideas actually held up and were accepted. You
do NOT see val tasks or traces: the only execution traces you get are the train-sample ones
(eval_before/eval_after). Do not look for val traces outside this workspace (e.g.
../evaluation/), and do not tune toward the val split; aim for changes that help on tasks you
have never seen.

Now: analyze the above, state ONE falsifiable hypothesis, make the single targeted
general-purpose change (minimal + additive), verify the package still imports, and write
CANDIDATE.md (hypothesis / changed files / expected effect).
