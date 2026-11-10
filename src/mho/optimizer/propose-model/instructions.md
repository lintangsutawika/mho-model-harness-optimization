Follow the **propose-model** skill (available in <available_skills>) for the
model-improvement-specification workflow: analyze, state ONE falsifiable hypothesis, write the
model spec AND the check functions that turn it into a training reward, screen them, and leave
both under your candidate's `model_specs/`.

## Your workspace
This is the run's candidate archive (`candidates/`). Your own candidate dir is named below
("Your candidate"); everything else is read-only reference:
- `../frontier.json`, `../evolution_summary.jsonl`: scores so far.
- `<tag>/status.json`, `<tag>/CANDIDATE.md`: what each earlier candidate tried and whether it was kept.
- `<tag>/eval_before/<trial>/` and `<tag>/eval_after/<trial>/`: **the primary evidence**, the
  train-sample rollouts. Each trial has `result.json` (its reward under `verifier_result`),
  `agent/mini-swe-agent.txt` (readable turn-by-turn transcript) and
  `agent/mini-swe-agent.trajectory.json` (the full trajectory: the same object your check
  functions receive). Read several to see WHY the model failed, not just whether.

## What you write (two files, same name)
1. `<tag>/model_specs/<name>.md`: YAML frontmatter with `name` (= `<name>`), `hypothesis`,
   `evidence`, `spec`, `counterexample`, `realization`, and `checks` (a list of function names).
2. `<tag>/model_specs/<name>.py`: those functions. Each scores ONE rollout's trajectory:
   ```python
   from sigma_checklib import *   # messages, assistant_texts, reasoning_texts, commands,
                                  # tool_results, n_steps, task_prompt

   def checks_work_before_answering(traj: dict) -> bool:
       ...
   ```
   Return a bool, or a float in [0, 1] for partial credit. Standard library only. Each check
   gets a 10 s limit per rollout.

Training reward per rollout = 0.5 × answer correctness (the task's own verifier) + 0.5 × the
mean of your checks. The weights are fixed: your checks can move at most half the reward, and
they cannot make a wrong answer look right.

## What makes a good check
- It measures the behavior your spec names, read from what the agent actually did (commands,
  tool results, its messages), and it is true for some rollouts and false for others.
- **A check that is the same on every rollout is useless**: it adds the same reward to every
  rollout in a group and changes no advantage. **Screen before you finish:**
  `python -m mho.data.reward_synthesis screen --spec ./<tag>/model_specs/<name>.md --traces .`
  runs every check over this run's train-sample trajectories exactly as training will, and
  rejects a check that raises or never varies. A rejected spec is not trained. It also reports
  each check's correlation with the trial reward: a check that rises with correctness reinforces
  it; one that falls with it trades correctness away. Fix and re-screen until it passes.
- General, not task-specific: no task names, dataset names, or expected answers in the checks.

## HARD CONSTRAINTS (a proposal that violates these will be REJECTED)
- Exactly ONE capability spec (the single most-predictive failure mode) and its checks module.
- The spec must validate: `python -m mho.optimizer.model_spec validate ./<tag>/model_specs/<name>.md`
  (schema, `name` matches the file, every listed check is defined in the .py).
- Never hardcode task/dataset specifics or answer-format special-casing beyond what the harness
  already does. Keep the capability general.
- Write only under your own `<tag>/model_specs/`. Everything else is read-only.

## Prior results
You see each candidate's val SCORE (summary / status / frontier): that is the signal of which
ideas held up. You do NOT see val tasks or traces; the only traces are the train-sample ones.
Do not look for val traces, and do not tune toward the val split.

Now: analyze the trajectories, identify the single most-predictive model-side failure mode,
write the spec and its checks, validate, screen, and fix until both pass. Write each YAML field
as a block scalar (`key: |`) or quoted when it contains `:` or spans lines.
