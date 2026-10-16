Follow the **propose-model** skill (available in <available_skills>) to propose what the
model should learn next. You read the mounted harness archive and write model specs; a
downstream realization step turns each spec into verifiable training data. You do NOT run
training or benchmarks.

## Your workspace
This is the candidate unit directory. The harness archive is mounted under `eval_prior/` --
read it with the file editor for full detail:
- `eval_prior/frontier.json`
- `eval_prior/evolution_summary.jsonl`
- `eval_prior/aggregated.jsonl`
- `eval_prior/trials/` (one dir per trial, with `result.json` + `agent/mini-swe-agent.txt` +
  `agent/mini-swe-agent.trajectory.json`)

Write your proposal as one `.md` file per capability under **`model_specs/`** in this workspace.

NOTE: you see only the training-sample eval (eval_prior/trials). The validation set is
deliberately NOT shown to you -- it is the held-out accept/reject gate. Do NOT attempt to
reconstruct or infer val results.
Do NOT edit anything under `eval_prior/` -- it is read-only reference material.

Now: analyze the archive, state ONE falsifiable hypothesis per capability, and write 1-3
focused, general-purpose model specs under `model_specs/` (name / hypothesis / evidence /
spec / counterexample), each about a behavior the model should learn that raises verified
reward on unseen tasks.