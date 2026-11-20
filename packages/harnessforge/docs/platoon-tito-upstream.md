# Upstream PR draft: token-in/token-out prompts in platoon's model gateway

Draft text for a PR to platoon. It has **not** been opened. The change is `src/harnessforge/train/patches/platoon-tito.patch`
(against platoon 0.6.0). The validation numbers are in `docs/train.md` ("Token-in/token-out prompts").

---

**Title:** gateway: token-in/token-out prompts (an agent's call extends the exact tokens of its previous call)

**Summary**

`ModelGateway` renders every call's messages from scratch with the chat template. An agent's previous turn then
comes back as the template's rendering of the message parsed from its sampled tokens. That rendering differs from the
sampled tokens in several cases:
- the model wrote whitespace the template normalizes, e.g. `\n\n</think>` or one `\n` before a `<tool_call>`
  (`\n` and `\n\n` are single tokens in Qwen's vocabulary, so the token count does not even change);
- a tool call did not parse, so its raw text is appended to the content;
- the reply has several tool calls;
- the generation was cut at `max_tokens`;
- the same text tokenizes differently in context.

The next prompt then no longer extends the previous prompt + completion. `trajectory_to_data` starts a new datum and
repeats the whole shared prefix in the batch. On a Qwen3.5-4B GRPO run (micro-swe-agent, 64k context), step 1 trained
759 datums and 21.3M tokens for 256 trajectories: 3 datums per trajectory.

With this PR, the gateway keeps each agent's exact token stream: the prompt + the sampled tokens of its last call, as
sampled. A request whose messages are, in order,
- the previous request's messages,
- the assistant message the gateway returned (compared by role, trimmed content and reasoning, tool-call names and
  arguments),
- at least one new message,

is sampled on that stream plus the template's rendering of the new messages and the generation prompt. That suffix is
taken from a render of the whole conversation, after the last assistant turn's end marker. Earlier tokens are never
re-rendered. A completion cut at `max_tokens` gets the end token appended; it is a prompt token, so it is masked.
Datum construction is unchanged: `trajectory_to_data` already merges a call whose prompt extends the accumulated
sequence, and now every call that extends its conversation does.

Anything else is rendered from scratch exactly as before, and starts a new stream, so a new datum:
- history dropped or rewritten by the harness (compaction, a dropped reply);
- changed tools or chat-template kwargs;
- a template that renders the earlier turns differently once the new messages follow (Qwen3.5 drops past thinking
  after a new user query).

**What the model sees.** Its earlier turns are its own sampled tokens instead of a re-rendering of them. Everything else
is token for token the template's rendering. With canonical samples the stitched prompt is identical to the
from-scratch one (tested).

**Scope**
- `platoon/gateway/tito.py` (new): `TitoStitcher`, `TitoSession` (per agent), canonical message/reply comparison,
  counts.
- `platoon/gateway/core.py`:
  - `ModelGateway(tito=None)`, from `PLATOON_TITO` (on unless `0`);
  - `render_call` / `tito_record`;
  - `prepare(..., agent=None)`;
  - `chat` passes the agent and records the stream;
  - `PreparedCall.tito`, `AgentSession.tito`.
- `platoon/gateway/server.py`: the HTTP path passes the agent to `prepare`.
- `platoon/gateway/hosted.py`: signature only.
- Only renderers that render whole conversations as text with token stop sequences (`ChatTemplateRenderer`) are
  stitched. Cookbook renderers render as before.

**Observability.** Each agent's trajectory misc gets `tito: {stitched, fresh, reasons{first, stitched,
history_changed, reply_dropped_or_changed, rerender_differs, request_changed, ...}}`. The gateway logs totals every
2000 calls. `PLATOON_TITO_VERIFY=1` also decodes every stitched prompt and counts whether it equals the from-scratch
render as text (`text_equal` / `text_differs`).

**Cost.** Per call, two template renders (the conversation, and the history up to the last reply) and an encode of the
suffix only, instead of encoding the whole prompt. The agent keeps one token list in memory.

**Tests** (real Qwen3.5 tokenizer and template; a scripted sampler):
- whitespace variants: one datum with TITO, a datum per call without;
- canonical samples: the stitched prompts equal the from-scratch prompts token for token;
- loss mask = exactly the sampled tokens, sampler logprobs aligned;
- multiple tool calls, an echoed unparsable call and a cut generation (masked end token) stay in one datum;
- a dropped reply, truncated history and changed tools render from scratch;
- `PLATOON_TITO=0`; the HTTP server path;
- end to end with micro-swe-agent through litellm 1.110.0 against `GatewayServer`.

**Open questions for upstream**
- Default on or off? This draft reads `PLATOON_TITO` (default on). A `train.tito` / `inference.tito` config key may
  fit platoon's config better.
- The reply comparison ignores fields the chat templates we know do not render (tool-call ids, `function_call`,
  provider fields). A template that renders other assistant fields would need them added.
- Harnesses that drop a reply on a format error (mini/micro-swe-agent) still start a new datum there. Keeping the
  failed turn in the stream would change what the model sees, so this PR does not.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_016TXQPBtsBokDCfCuLCEbik
