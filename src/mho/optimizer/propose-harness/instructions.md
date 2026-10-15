Follow the **propose-harness** skill (available in <available_skills>) for the
scaffold-evolution workflow: analyze, state ONE falsifiable hypothesis, implement a single
minimal additive change, validate the package, and write CANDIDATE.md.

## Your workspace
The micro-swe-agent scaffold snapshot is your working directory -- it already contains
the accumulated changes from the best iteration so far (you are building ON TOP of it,
not from scratch). Edit files here in place to add ONE more improvement; keep it an
installable package (pyproject.toml + src/).

## HARD CONSTRAINTS (a candidate that violates these is REJECTED)
- Make MINIMAL, mostly-ADDITIVE edits for your ONE mechanism. Change only the files that
  mechanism needs.
- NEVER delete existing files, NEVER empty or rewrite the src/ tree, NEVER recreate the
  package from scratch. The snapshot must remain the FULL working scaffold PLUS your change.
- Keep it an installable package that still imports. Do not run destructive shell commands
  (rm -rf, git clean, moving the whole tree, etc.).

## Prior results
{{prior_results}}

The same prior results are ALSO mounted as files under `eval_prior/` in your workspace --
open them directly with the file editor for full detail:
`eval_prior/frontier.json`, `eval_prior/evolution_summary.jsonl`, `eval_prior/aggregated.jsonl`,
and `eval_prior/trials/` (one dir per trial, with `result.json` + `agent/mini-swe-agent.txt` +
`agent/mini-swe-agent.trajectory.json`).

NOTE: you see only the training-sample eval (eval_prior/trials). The validation set is
deliberately NOT shown to you -- it is the held-out accept/reject gate. Do NOT attempt to
reconstruct or infer val results.
Do NOT edit anything under `eval_prior/` -- it is read-only reference material.

Now: analyze the above, state ONE falsifiable hypothesis, make the single targeted
general-purpose change (minimal + additive), verify the package still imports, and write
CANDIDATE.md (hypothesis / changed files / expected effect).