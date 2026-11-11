---
name: propose-model
description: >
  One iteration of model-improvement specification for the MHO project. Use when asked to
  propose what the model should learn: analyze prior eval results + failed trajectories,
  form ONE falsifiable hypothesis per capability, and write the model specs (capability
  specifications) that a realization step turns into verifiable training data.
license: MIT
compatibility: Python / OpenHands SDK
metadata:
  author: mho
  version: "1.0"
triggers:
  - propose model
  - model improvement
  - what should the model learn
  - optimize model
---

# Model spec proposal (one iteration)

Run **ONE** iteration of model-improvement specification. The model counterpart of
`propose-harness`: the harness step runs first and produces the archive; **you** read that
archive, form ONE hypothesis about what the model should learn, and write it as model specs.
**You do NOT run training or benchmarks.** The outer loop (`optimize_loop.py`) realizes each
spec into verifiable training data and trains the model; you only specify what to learn.

## CRITICAL CONSTRAINTS
- You MUST produce **exactly one** focused model spec this iteration (one capability, one hypothesis). Do not
  conclude "the model is already good enough" or stop early — always ship a proposal.
- **Ability-first / hypothesis-first.** Identify a concrete model-side failure mode from the
  trajectories/metrics, then specify the capability that targets it. No speculative specs.
- **One capability per spec.** Each spec targets a single failure mode / hypothesis. If
  tempted to add "and also…", that's a separate spec — split it.
- **Write ONLY under your own `<tag>/model_specs/`.** Other candidates, their traces and
  every `harness/` are read-only reference.

### Anti-overfitting (critical)
- Whatever the current task family is, the model must stay **general-purpose**.
  **Do NOT hardcode domain knowledge or task-specific hints** — no
  answer-format special-casing beyond what's already in the harness, no "if the problem
  mentions…" behaviors, no dataset-specific heuristics.
- **Never reference task names / families, datasets, or specific prompts** in the specs.
- General guidance is OK: capabilities that help the model solve *many* unfamiliar tasks
  (know when to stop, recover from an error, structure a long multi-step solution, keep
  context, format a final answer that a verifier can parse). Test: "would this help a
  competent model on tasks it has never seen?"
- If in doubt, make it more general.

## CONTEXT
You are proposing what the **model** (Qwen3.5-4B) should learn, given the current harness.
The model is RL-trained on verified rollouts produced by (model, harness); the harness is
improved in the same loop. Your job is to specify the *capabilities* (behaviors) the model
should gain so the system gets more reward on tasks the harness's failures expose.

**The search space is model behavior.** You do not edit the harness; you write a capability
specification plus the **check functions** that make it a training reward. The model is then
RL-trained on the train tasks with reward = 0.5 × answer correctness + 0.5 × mean(your checks),
so your checks decide which behavior the update reinforces. See instructions.md for the check
contract (`def <check>(traj: dict) -> bool | float`, helpers in `sigma_checklib`).

**Key things to read first (use the file editor / terminal in your workspace):**
- `../frontier.json`, `../evolution_summary.jsonl` — the incumbent and the history.
- `*/eval_before/*/agent/mini-swe-agent.txt` and `*/eval_after/*/...` — **why** trials failed
  (the actual turn-by-turn model behavior); `result.json` has each trial's reward, and
  `mini-swe-agent.trajectory.json` is the exact object your checks receive.

## WORKFLOW
1. **Analyze.** Read the frontier, the iteration summaries and the train-sample trials. Identify the most promising
   **model-side** failure mode: e.g. the model stops prematurely and never reaches the answer;
   emits a malformed tool call and never retries; floods context by repeating commands; loses
   the goal mid-trail; formats the final answer so the verifier rejects it; loops without
   progress; abandons after a single error. Attribute the failure to the MODEL (a missing
   capability) versus a harness artifact — you propose what the model should learn.
2. **State a falsifiable hypothesis** — one sentence per capability on what behavior change
   will raise verified reward and why.
3. **Write the spec** — `<tag>/model_specs/<name>.md`: `name`, `hypothesis`, `evidence`
   (which trajectories support it), `spec` (the precise capability σ), `counterexample` (an
   input the current model gets wrong), `realization: select`, `checks` (function names), and
   `rubric` (criteria + keywords that pick the train tasks exercising the capability; see
   instructions.md).
4. **Write the checks** — `<tag>/model_specs/<name>.py`, one function per name in `checks`,
   each reading ONE rollout's trajectory and returning bool or a float in [0, 1]. A check must
   separate rollouts that show the behavior from ones that don't; prefer what the agent did
   (`commands`, `tool_results`) over what it said.
5. **Validate** — `python -m mho.optimizer.model_spec validate ./<tag>/model_specs/<name>.md`.
6. **Screen** — `python -m mho.data.reward_synthesis screen --spec ./<tag>/model_specs/<name>.md --traces .`
   runs the checks on this run's trajectories exactly as training will. It rejects a check that
   raises or returns the same value everywhere (such a check changes no advantage). Read the
   mean and the correlation with reward; fix and re-screen until it passes.
7. **Screen the rubric** — `python -m mho.data.task_select screen --spec ./<tag>/model_specs/<name>.md --traces .`
   scores the train-sample tasks and rejects a rubric that keeps none or all of them. Aim for a
   kept set with fewer always-failed (all_0) and always-solved (all_1) tasks than the dropped set.

## OUTPUT CONTRACT
When you finish, `<tag>/model_specs/<name>.md` and `<tag>/model_specs/<name>.py` **are** the
proposal. The loop re-runs the validation and the screen; a spec that fails either is not
trained. There is no separate submission file.
