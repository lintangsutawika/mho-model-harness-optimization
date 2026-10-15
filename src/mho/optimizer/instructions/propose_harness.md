# Meta-Harness — scaffold evolution (one iteration)

Run **ONE** iteration of agent-scaffold evolution. Adapted from stanford-iris-lab/meta-harness
for the MHO project: the scaffold is **micro-swe-agent** (not Terminus2), the workspace is a
full snapshot you edit in place, and the outer loop (`optimize_loop.py`) runs all benchmarks.

**You do NOT run benchmarks.** You analyze prior results + failed trajectories, form ONE
hypothesis, and implement it by editing the scaffold in your workspace. The loop evaluates it.

## CRITICAL CONSTRAINTS
- You MUST produce exactly **one** improved scaffold this iteration. Do not conclude "the
  frontier is optimal" or stop early — always ship a change.
- **Minimal, additive edits only.** Change ONLY the files your one mechanism needs.
  **NEVER delete existing files, NEVER empty or rewrite the `src/` tree, NEVER recreate the
  package from scratch, NEVER run destructive shell commands** (`rm -rf`, `git clean`, moving the
  whole tree). The snapshot must stay the FULL working scaffold PLUS your change — a candidate
  that deletes the scaffold is **rejected** (a file-count + parse check enforces this).
- **One mechanism per iteration.** Target a single failure mode / hypothesis. If tempted to
  add "and also…", that's a second candidate — save it.
- **Mechanism-first.** Identify a concrete failure mode from the trajectories/metrics, then
  make the change that targets it. No speculative changes.

### Anti-overfitting (critical)
- The current eval task is **math** (DAPO-Math-17k), but the harness must stay
  **general-purpose**. **Do NOT hardcode math knowledge or task-specific hints** — no
  answer-format special-casing beyond what's already there, no "if the problem mentions…"
  branches, no dataset-specific heuristics.
- **Never reference task names / families** in code, prompts, or comments.
- General guidance is OK: changes that help an agent solve *many* unfamiliar tasks (better
  tool-use discipline, cleaner reasoning scaffolding, more robust command execution, smarter
  context handling). Test: "would this help a competent agent on tasks it has never seen?"
- If in doubt, make it more general.

## CONTEXT
You are evolving the **micro-swe-agent** scaffold — the agent that drives a shell inside a
sandbox to solve a task and emit a final answer. The model behind it (Qwen3.5-4B) is being
RL-trained in the same loop; your job is to improve the *scaffold* (prompts, agent loop,
tool use, parsing, context handling) so the agent gets more reward.

**The search space is arbitrary Python in the snapshot.** You may edit ANY file — the agent
loop, prompt templates, tool definitions, command execution, output handling — anything.
The only invariant: the snapshot must remain a valid **installable package** (a `pyproject.toml`
at the root + its `src/`), because the eval `uv tool install`s it into each sandbox.

**Key things to read first (use the file editor / terminal in your workspace):**
- `pyproject.toml` and `src/` — the package layout; find the agent loop + prompt templates.
- The system/prompt templates the agent uses each turn.
- The main agent loop (how it calls the model, parses tool calls, executes commands,
  decides when to stop, and produces the final answer).

## WORKFLOW
1. **Analyze.** Read the prior-results context provided below (frontier + recent iteration
   summaries + any trajectory/metrics digest). Identify the single most promising failure
   mode: e.g. the agent stops too early, mangles multi-line commands, floods context, fails
   to put the final answer in the expected form, loops without progress, etc.
2. **State a falsifiable hypothesis** — one sentence on what change will raise reward and why.
3. **Implement it** — make targeted edits to the scaffold in your workspace. Keep the change
   focused on the one mechanism. Do not break the package layout.
4. **Validate** — from the workspace root, confirm the package still imports/builds, e.g.
   `python -c "import tomllib,sys; tomllib.load(open('pyproject.toml','rb'))"` and import the
   agent module. Fix anything you broke. The snapshot MUST end valid.
5. **Record** — write a short `CANDIDATE.md` at the workspace root with: `hypothesis`,
   `changes` (files + what/why), and `expected_effect` (predicted reward/behavior impact).

## OUTPUT CONTRACT
When you finish, the workspace directory **is** the candidate: a valid installable
micro-swe-agent snapshot with your one change applied and `CANDIDATE.md` describing it. There
is no separate submission file — the loop picks up the snapshot directly.
