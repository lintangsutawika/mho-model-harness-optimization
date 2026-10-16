Follow the **propose-model** skill (available in <available_skills>) for the
model-improvement-specification workflow: analyze, state ONE falsifiable hypothesis per
capability, write the model specs, validate them, and leave them under `model_specs/`.

## Your workspace
This is the candidate unit directory. The model archive is mounted under `eval_prior/` --
read it with the file editor for full detail:
- `eval_prior/frontier.json`
- `eval_prior/evolution_summary.jsonl`
- `eval_prior/aggregated.jsonl`
- `eval_prior/trials/` -- **the primary evidence.** One dir per trial, with `result.json` +
  `agent/mini-swe-agent.txt` (readable turn-by-turn transcript) + `agent/mini-swe-agent.trajectory.json`
  (full traced rollout). Read several to see WHY the model failed, not just whether.

Write your proposal as one `.md` file per capability under **`model_specs/`** in this workspace.
The harness snapshot (`harness/`) is read-only reference for you; do not edit it.

## HARD CONSTRAINTS (a proposal that violates these will be REJECTED)
- Write 1-3 focused, general-purpose capability specs; ONE capability per file.
- Each spec MUST carry: `hypothesis` (a falsifiable statement of what behavior change raises
  reward and why), `evidence` (the specific `eval_prior/trials/` trajectories that support it),
  `spec` (the precise capability σ the model must learn), and `counterexample`.
- NEVER hardcode task/dataset specifics, answer-format special-casing beyond what's already in
  the harness, or references to task names/families. Keep the capability general.
- Do NOT edit anything under `eval_prior/` or `harness/` -- read-only reference.

## Prior results
Prior eval results are mounted under `eval_prior/` -- read them directly with the file editor
(trajectories first) to identify the single most promising model-side failure mode.
NOTE: you see only the training-sample eval (eval_prior/trials). The validation set is
deliberately NOT shown to you -- it is the held-out accept/reject gate. Do NOT attempt to
reconstruct or infer val results.

Now: analyze the trajectories, state ONE falsifiable hypothesis per capability, and write 1-3
focused, general-purpose model specs under `model_specs/` (name / hypothesis / evidence / spec /
counterexample / realization), each about a behavior the model should learn that raises verified
reward on unseen tasks. Validate each with
`python -m mho.optimizer.model_spec validate model_specs/<name>.md` and fix any rejection before
finishing. Write each field as YAML frontmatter: use block scalars (`key: |`) or quotes for any
value that contains `:` or spans lines (the validator's YAML parser will reject loose `key: value: more`
lines).