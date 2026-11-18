Follow the **propose-rubric** skill (available in <available_skills>).

## Your workspace
Your working directory is the run's `candidates/` archive:
- `candidate_0_0_0/` -- the evaluated train-sample tasks: `task_rewards.json`
  ({task: [reward per trial]}), `tasks/<task>.md` (instruction), `eval_before/<task>__<id>/`
  (trials).
- `summary.json` -- one row per earlier rubric candidate: its proposal cost and, per judge
  run, the yield (tasks kept by `threshold` = score >= 0.85 and by `top` = best 10%) and the
  rewards of the kept vs dropped design tasks. Absent on the first iteration.
- `<tag>/rubric.md`, `<tag>/CANDIDATE.md`, `<tag>/curated/*.json` -- earlier candidates.

A task's score is the share of rubric questions answered "yes". Tasks are then kept either by
score >= 0.85 or as the top 10% by score, so questions that every task passes, or none does,
select nothing. Treat every existing file as read-only.

Do not repeat an earlier rubric's idea: if `summary.json` shows a rubric kept as many all-0 /
all-1 tasks as it dropped, try a different property of the task text.
