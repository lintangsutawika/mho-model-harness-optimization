"""Modal reverse-relay so a private node vLLM is reachable from Modal sandboxes.

Ported from tts-tokens-that-suffice. modal_relay.py = the public Modal app (HTTPS +
WebSocket bridge endpoint); bridge.py = the node-side leg that dials out and proxies to
local vLLM. Driven by scripts/eval/eval_terminal_bench_2.sh (RELAY=1)."""
