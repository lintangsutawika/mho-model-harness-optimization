"""Stdlib OpenAI client checks against a pool endpoint: models, chat, tool call."""
import json, sys, time, urllib.request
ep = sys.argv[1].rstrip("/")
model = sys.argv[2] if len(sys.argv) > 2 else "policy"
def call(path, body=None, timeout=600):
    req = urllib.request.Request(ep + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
print("models:", [m["id"] for m in call("/models")["data"]])
t = time.time()
r = call("/chat/completions", {"model": model, "temperature": 0, "max_tokens": 64,
     "messages": [{"role": "user", "content": "Say hello in five words."}],
     "chat_template_kwargs": {"enable_thinking": False}})
print(f"chat ({time.time()-t:.1f}s):", json.dumps(r["choices"][0]["message"])[:400], r.get("usage"))
tools = [{"type": "function", "function": {"name": "bash", "description": "Run a bash command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                         "required": ["command"]}}}]
r = call("/chat/completions", {"model": model, "temperature": 0, "max_tokens": 2048,
     "tools": tools, "messages": [{"role": "user", "content": "List the files in /tmp using the bash tool."}]})
m = r["choices"][0]["message"]
print("tool:", json.dumps({"tool_calls": m.get("tool_calls"), "content": (m.get("content") or "")[-300:],
      "reasoning": (m.get("reasoning_content") or "")[:100], "finish": r["choices"][0].get("finish_reason")}))
