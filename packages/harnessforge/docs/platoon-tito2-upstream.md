# Upstream PR draft: chat-template override and a context limit for platoon

Draft text for a PR to platoon. It has **not** been opened. The change is
`src/harnessforge/train/patches/platoon-tito2.patch` (against platoon 0.6.0 plus `platoon-tito.patch`). The validation
numbers are in `docs/train.md` ("Keep-thinking template and the context limit").

---

**Title:** renderer `chat_template` override; end rollouts at a context limit instead of clamping `max_tokens`

**1. `chat_template` override for the `chat_template` renderer.**
- *Problem.* There is no way to render with a template other than the tokenizer's. Qwen3.5's template drops the
  thinking of every assistant turn before the last user query. For agent RL with interleaved thinking you usually
  want it kept (TMax: "We edit chat templates such that thinking is preserved on intermediate turns"). The
  `preserve_thinking` kwarg does nothing for templates that do not read it.
- *Change.* `ChatTemplateRenderer(chat_template=...)`, also settable through `renderer_kwargs.chat_template`, takes a
  `.jinja` path or the template text.
  - It is passed to `apply_chat_template(chat_template=...)` and is part of the tool-call parser's cache key, so the
    parser probe runs against the template actually used.
- *Consideration.* A run that evaluates through a hosted endpoint should give that server the same template
  (SGLang/vLLM `--chat-template`). Otherwise training and evaluation prompts differ.

**2. A context limit (`min_completion_tokens`).**
- *Problem.* `fit_max_tokens` clamps `max_tokens` to whatever room the window leaves, down to 1 token. Near the window
  an agent gets replies cut after a few tokens. A harness such as mini/micro-swe-agent turns them into format errors
  and continues, often several times.
  - With TITO each such turn also starts a new datum of almost the whole window.
  - On a Qwen3.5-4B run at 64k, ending rollouts there instead cut trainer tokens 28–34 % beyond TITO, and 12–17 % of
    rollouts ended this way.
- *Change.* `ModelGateway(min_completion_tokens=...)`, default from `PLATOON_MIN_COMPLETION_TOKENS` (we use 2048; an
  upstream default of 0 keeps today's behaviour). A prompt leaving less than `min(min_completion_tokens, max_tokens)`
  raises `ContextLimitError`.
  - It subclasses `ContextWindowExceededError`, so every transport reports the native context-length error.
  - The agent counts it (`context_limits`) and records `misc.context_limit`.
- *Grading.* `harness.rollout` and `inline.failure_outcome` grade such a run with reason `context_limit`, whether the
  harness then failed or finished. `on_context_overflow: exclude` excludes it instead.
- *Metrics.* `TrajectoryStats.context_limit` (0/1) is summarized with the rollout stats (`.../context_limit/avg`).
  The outcome stays `scored`.

**Open questions.**
- A config key (`rollout_config.min_completion_tokens`) instead of the variable.
- Whether `context_limit` should count under its own `outcome/` key, which would need the reason plumbed through
  `RolloutWorkerResponse`.

**Tests:**
- the template differs from the stock one only in the thinking condition;
- the override keeps thinking after a user message, and set as a `renderer_kwargs` attribute it is honoured;
- prompts no longer shrink after a micro-swe-agent format error, and TITO still stitches;
- `fit_max_tokens` limit, clamp and off cases, and the variable;
- the agent records the limit;
- `failure_outcome` reasons;
- the trajectory stat;
- end to end with micro-swe-agent through litellm hitting the limit (the harness exits on the context error after
  the calls that fit, with one datum).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_016TXQPBtsBokDCfCuLCEbik
