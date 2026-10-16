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
- You MUST produce at least one, ideally **1-3 focused** model specs this iteration. Do not
  conclude "the model is already good enough" or stop early — always ship a proposal.
- **Ability-first / hypothesis-first.** Identify a concrete model-side failure mode from the
  trajectories/metrics, then specify the capability that targets it. No speculative specs.
- **One capability per spec.** Each spec targets a single failure mode / hypothesis. If
  tempted to add "and also…", that's a separate spec — split it.
- **Do NOT edit anything under `eval_prior/`** (read-only reference). Write ONLY under
  `model_specs/`; do not touch the harness snapshot (`harness/`).

### Anti-overfitting (critical)
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

**The search space is model behavior, not code.** You do not edit source; you write
capability specifications. Each spec is a concise, testable statement of a behavior the
model must learn — precise enough that a downstream realization map `f` can turn it into
verifiable training tasks (select matching ones from the dataset or synthesize new ones).

**Key things to read first (use the file editor / terminal in your workspace):**
- `eval_prior/aggregated.jsonl` and `evolution_summary.jsonl` — which trials failed.
- `eval_prior/trials/*/agent/mini-swe-agent.txt` and `trajectory.json` — **why** they failed
  (the actual turn-by-turn model behavior).
- `eval_prior/frontier.json` — the incumbent score to beat.

## WORKFLOW
1. **Analyze.** Read the prior-results context mounted under `eval_prior/` (frontier +
   recent iteration summaries + trial trajectories/metrics). Identify the most promising
   **model-side** failure mode: e.g. the model stops prematurely and never reaches the answer;
   emits a malformed tool call and never retries; floods context by repeating commands; loses
   the goal mid-trail; formats the final answer so the verifier rejects it; loops without
   progress; abandons after a single error. Attribute the failure to the MODEL (a missing
   capability) versus a harness artifact — you propose what the model should learn.
2. **State a falsifiable hypothesis** — one sentence per capability on what behavior change
   will raise verified reward and why.
3. **Write the spec(s)** — one `.md` file per capability under `model_specs/` in your
   workspace. Each file: `name`, `hypothesis`, `evidence` (which trajectories support it),
   `spec` (the precise capability σ the model must learn, written so a realization map can
   turn it into training tasks), and `counterexample` (an input the current model gets wrong).
4. **Validate** — confirm each spec is standalone, one-capability-per-file, non-overlapping,
   and general-purpose (no task/dataset references). Fix anything that isn't.
5. **Record** — keep each spec self-contained; the set of spec files IS the proposal.

## OUTPUT CONTRACT
When you finish, the `model_specs/` directory **is** the proposal: one `.md` file per
capability, each a standalone capability specification (name / hypothesis / evidence / spec /
counterexample). There is no separate submission file — the realization step reads
`model_specs/` directly to build the model's next training data.
