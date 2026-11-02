Follow the **propose-harness** skill (available in <available_skills>) for the
scaffold-evolution workflow: analyze, state ONE falsifiable hypothesis, implement a single
minimal additive change, validate the package, and write CANDIDATE.md.

## Your workspace
Your working directory is the run's `candidates/` archive -- it gives you access to ALL
optimization information in one place:
- `frontier.json` -- the current best candidate and its val score (the accept bar).
- `summary.json` -- the history of every proposal and its outcome.
- `<tag>/harness/` -- each candidate's harness code.
- `<tag>/status.json` -- whether that harness was ACCEPTED or REJECTED (with outcome + score).
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
-- it is the history of every proposal so far, each row labeled with its `outcome`
(`train_fail`, `train_pass_val_fail`, `train_pass_val_pass` = accepted, or `failed_both`),
plus the hypothesis, ts_A/ts_B scores, and (via `<tag>/status.json`) whether each harness
was ACCEPTED or REJECTED on the held-out val gate.

**AVOID REPEATING PRIOR ATTEMPTS.** Your proposed mechanism MUST be **novel** -- it must
not re-propose the same underlying fix that already appears in `./summary.json` or in a
prior candidate's harness. If an idea is already there and failed (REJECTED), or is a
re-wording of one that was tried, pick a different failure mode. Build on ACCEPTED winners,
avoid repeating rejected regressions.

NOTE: you see only the training-sample evals (eval_before/eval_after). The held-out
validation set is deliberately NOT shown to you -- it is the accept/reject gate. Do NOT
attempt to reconstruct or infer val results.

Now: analyze the above, state ONE falsifiable hypothesis, make the single targeted
general-purpose change (minimal + additive), verify the package still imports, and write
CANDIDATE.md (hypothesis / changed files / expected effect).
