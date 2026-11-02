Follow the **propose-harness** skill (available in <available_skills>) for the
scaffold-evolution workflow: analyze, state ONE falsifiable hypothesis, implement a single
minimal additive change, validate the package, and write CANDIDATE.md.

## Your workspace
The micro-swe-agent scaffold snapshot is your working directory -- it already contains
the accumulated changes from the best iteration so far (you are building ON TOP of it,
not from scratch). Edit files here in place to add ONE more improvement; keep it an
installable package (pyproject.toml + src/).

## HARD CONSTRAINTS (a candidate that violates these will be REJECTED)
- Make MINIMAL, mostly-ADDITIVE edits for your ONE mechanism. Change only the files that
  mechanism needs.
- NEVER delete existing files, NEVER empty or rewrite the src/ tree, NEVER recreate the
  package from scratch. The snapshot must remain the FULL working scaffold PLUS your change.
- Keep it an installable package that still imports. Do not run destructive shell commands
  (rm -rf, git clean, moving the whole tree, etc.).

## Prior results
Prior eval results are mounted under `eval_prior/` in your workspace -- read them directly
with the file editor for full detail:
- `eval_prior/frontier.json`
- `eval_prior/evolution_summary.jsonl`
- `eval_prior/aggregated.jsonl`
- `eval_prior/trials/` (one dir per trial, named `<candidate>__<trial>` so you can tell
  which prior proposal it came from; each holds `result.json` + `agent/mini-swe-agent.txt` +
  `agent/mini-swe-agent.trajectory.json`). These are REAL execution traces staged from PRIOR
  candidates (their train-sample eval_after trials) -- study them to understand exactly how
  prior harness edits behaved, not just their aggregate scores.

Start by reading these **especially `eval_prior/evolution_summary.jsonl`** to identify the
single most promising failure mode.

**AVOID REPEATING PRIOR ATTEMPTS.** `evolution_summary.jsonl` is the history of every
proposal so far: each row's `outcome` tells you whether the attempt `train_fail`ed,
`train_pass_val_fail`ed (improved on the train sample but not on the held-out val set),
`train_pass_val_pass`ed (improved on both, accepted), or `failed_both`, plus its
`hypothesis` (what was actually changed) and the `ts_A`/`ts_B` train-sample scores.

Your proposed mechanism MUST be **novel** -- it must not re-propose the same underlying
fix that already appears in `evolution_summary.jsonl`. If an idea is already there and
failed, or is a re-wording of one that was tried, pick a different failure mode. Study the
`outcome` column to learn what worked vs. what regressed, build on winners, and avoid
repeating regressions.

NOTE: you see only the training-sample eval (eval_prior/trials). The validation set is
deliberately NOT shown to you -- it is the held-out accept/reject gate. Do NOT attempt to
reconstruct or infer val results.
Do NOT edit anything under `eval_prior/` -- it is read-only reference material.

Now: analyze the above, state ONE falsifiable hypothesis, make the single targeted
general-purpose change (minimal + additive), verify the package still imports, and write
CANDIDATE.md (hypothesis / changed files / expected effect).
