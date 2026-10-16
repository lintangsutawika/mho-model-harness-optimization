---
name: propose-model
description: >
  Propose what the model should learn next. Use when asked to propose the model
  side of the MHO optimization: analyze the mounted harness archive (eval_prior/:
  frontier, summary, aggregated trials, trajectories), identify the model's failure
  modes, and write ONE or more model specs (capability specifications) that a
  realization step turns into verifiable training data for RL fine-tuning.
license: MIT
compatibility: Python / OpenHands SDK
metadata:
  author: mho
  version: "1.0"
triggers:
  - propose model
  - model improvement
  - what should the model learn
---

# Model Spec Proposal (one iteration)

Propose what the **model** should learn, given the current harness archive. Adapted from
the MHO paper's model-side proposer Ψ: after a harness step, read the tested-harness archive
(failed trajectories, scores) and write a set of model specs S_k = {σ_k,1, ..., σ_k,M} --
each σ_j is a **capability specification**: a concise, testable statement of a behavior the
model should learn, enough that a realization map `f` can later select or synthesize
verifiable training tasks from it.

**You do NOT run training or benchmarks.** You read the archive, propose what to learn, and
write it as model spec files. A downstream step realizes each spec into training data.

## Work
1. **Read the archive** (mounted at `eval_prior/`): `frontier.json`, `evolution_summary.jsonl`,
   `aggregated.jsonl`, and `eval_prior/trials/*` (result.json + trajectory transcripts).
   Identify the model's recurring failure modes -- e.g. the model stops early, mangles a
   multi-line command, fails to format the final answer, loses context, loops without progress.
2. **State a falsifiable hypothesis** for one capability the model should gain.
3. **Write model specs** -- one `.md` file per capability, under `model_specs/` in your workspace:
   - `name` -- short capability name.
   - `hypothesis` -- what behavior change should raise verified reward and why.
   - `evidence` -- which failures in `eval_prior/trials/` support this (trial ids, patterns).
   - `spec` -- the precise, testable capability the model should learn (the σ itself):
     a behavioral rule the model must exhibit, written so a task generator / data selector
     can realize it into verifiable training tasks.
   - `counterexample` -- an example input the current model gets wrong (qualitative).
4. **Validate** -- confirm each spec is standalone, non-overlapping with the others (one
   capability each), and NOT task-specific (no dataset/paper references, no answer-format
   hardcoding beyond what's already in the harness). Keep the model capability general.

## Constraints
- **General-purpose capabilities only.** The model must improve on *many* unseen tasks, not
  memorize this benchmark. Never reference task names/families, datasets, or specific prompts.
- Write at least one but ideally 1-3 focused, high-leverage specs.
- Do NOT edit anything under `eval_prior/` (read-only reference).
- Keep `model_specs/*.md` self-contained; no external files needed.

## Output contract
The workspace `model_specs/` dir **is** the proposal: one `.md` per capability spec. A
downstream realization step reads them to build the model's next training data.