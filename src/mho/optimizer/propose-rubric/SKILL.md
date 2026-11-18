---
name: propose-rubric
description: >
  One iteration of rubric curation for the MHO project. Use when asked to write a data-curation
  rubric: read the train-sample trials, form ONE hypothesis about which tasks are worth training
  on, and write it as yes/no rubric questions that an LLM answers for every task.
license: MIT
compatibility: Python / OpenHands SDK
metadata:
  author: mho
  version: "1.0"
triggers:
  - propose rubric
  - curation rubric
  - which tasks to train on
---

# Rubric proposal (one iteration)

You decide which training tasks the model (Qwen3.5-4B) should be RL-trained on. You do not
pick tasks yourself: you write a rubric, an LLM answers each rubric question for every task
from its instruction text alone, and the tasks with the most "yes" answers are kept.
**You do NOT run the judge, training or evals.**

## CRITICAL CONSTRAINTS
- Write **exactly one** rubric this iteration, built on ONE hypothesis. Always ship it.
- **Hypothesis first.** Find a pattern in the trials (which tasks the model sometimes solves,
  which it never or always solves, and why), then turn it into questions. No speculative rubric.
- **Answerable from the instruction alone.** The judge never sees trials or rewards, only the
  task text. "Is the model likely to solve this?" is not answerable; "Does the task name the
  exact output file and format?" is.
- **General.** No task names, ids or dataset-specific strings; the rubric must apply to tasks
  you have not seen.
- **Write ONLY `<tag>/rubric.md` and `<tag>/CANDIDATE.md`.** Everything else is read-only.

## WHAT TO KEEP
RL learns from tasks the model solves sometimes: a task it always fails or always solves gives
no learning signal. A good rubric keeps more tasks with mixed outcomes and fewer all-0 / all-1
tasks than it drops.

## WORKFLOW
1. **Analyze.** `candidate_0_0_0/task_rewards.json` has every evaluated task's trial rewards;
   `candidate_0_0_0/tasks/<task>.md` its instruction; `candidate_0_0_0/eval_before/<task>__*/`
   the trials (`agent/mini-swe-agent.txt` turn by turn, `result.json`, `exception.txt` if it
   crashed). Read `summary.json` and earlier `<tag>/rubric.md` if present: each row has how
   many tasks an earlier rubric kept and the rewards of kept vs dropped tasks.
2. **Hypothesis.** One sentence: which property of the task text separates useful tasks, and
   the trial evidence for it.
3. **Write `<tag>/rubric.md`**: 3 to 6 bullet lines (`- ...?`), each a yes/no question about
   the task text, where "yes" means keep. Nothing else in the file is read.
4. **Check it by hand** on about 10 tasks of each kind (mixed, all-0, all-1): answer the
   questions yourself from `tasks/<task>.md` and see whether mixed tasks score higher.
5. **Write `<tag>/CANDIDATE.md`**: the hypothesis, the evidence (task ids and what happened),
   and how it differs from earlier rubrics.

## OUTPUT CONTRACT
`<tag>/rubric.md` is the proposal: 2 to 10 bullet lines. A tag without one counts as no proposal.
