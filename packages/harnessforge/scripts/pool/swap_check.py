"""Policy swap check against a running pool: outputs before/after load(), drain semantics."""
import json, sys, threading, time, urllib.request, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from harnessforge.contracts import PolicyRef
from harnessforge.pool import SGLangPool, from_config, load_config

spec_path, ckpt = sys.argv[1], sys.argv[2]
cfg = load_config(spec_path)
pool = SGLangPool.attach(cfg["pool_id"], os.path.expanduser(cfg["state_root"]),
                         launcher=from_config(cfg).launcher)
ep = pool.endpoint()
base = pool.policy()
PROMPTS = ["Write one sentence about the ocean.", "What is 17 * 23? Answer with the number only.",
           "Name three primary colors.", "Write a short story about a lighthouse keeper.",
           "Explain how a hash map works."]

def chat(text, max_tokens=160, timeout=1800):
    body = {"model": "policy", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(ep + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def outputs():
    # several calls per prompt so every engine behind the router answers
    res = {}
    for p in PROMPTS:
        res[p] = sorted({chat(p)["choices"][0]["message"]["content"] for _ in range(4)})
    return res

def versions():
    h = pool.health()
    return [(e["url"], e.get("weight_version"), e.get("model_path")) for n in h["nodes"] for e in n["engines"]]

print("endpoint", ep, "policy", base.id)
print("engines", versions())
before = outputs()
print("BEFORE", json.dumps({k: [x[:120] for x in v] for k, v in before.items()}, indent=1))

# drain: a long request in flight while load(drain=True) runs must complete normally
long_res = {}
def long_req():
    t = time.time()
    try:
        r = chat("Count from 1 to 400, separated by spaces.", max_tokens=1500)
        long_res.update(finish=r["choices"][0]["finish_reason"], tokens=r["usage"]["completion_tokens"],
                        seconds=round(time.time() - t, 1))
    except Exception as e:
        long_res.update(error=repr(e))
th = threading.Thread(target=long_req); th.start(); time.sleep(1.0)
new = PolicyRef("perturbed-2pct", ckpt, base_model="Qwen/Qwen3.5-4B")
t = time.time(); pool.load(new, drain=True); t_load = time.time() - t
th.join()
print(f"LOAD drain=True took {t_load:.1f}s; in-flight long request: {long_res}")
print("policy now", pool.policy().id)
with urllib.request.urlopen(ep + "/models", timeout=30) as r:
    print("/v1/models after load:", [m["id"] for m in json.loads(r.read())["data"]])
print("engines", versions())
after = outputs()
print("AFTER", json.dumps({k: [x[:120] for x in v] for k, v in after.items()}, indent=1))
print("outputs changed for", sum(before[p] != after[p] for p in PROMPTS), "/", len(PROMPTS), "prompts")

# abort semantics: drain=False aborts in-flight requests
long_res.clear()
th = threading.Thread(target=long_req); th.start(); time.sleep(1.0)
t = time.time(); pool.load(base, drain=False); t_load = time.time() - t
th.join()
print(f"LOAD back to {base.id} drain=False took {t_load:.1f}s; in-flight long request: {long_res}")
back = outputs()
print("back to base reproduces BEFORE:", back == before)
print("history", json.dumps(pool.dir.pool()["history"], indent=1))
