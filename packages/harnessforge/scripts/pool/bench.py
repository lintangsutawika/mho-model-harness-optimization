"""Serving comparisons (stdlib only).

  bench.py parity <base_url_v1> <model> <out.jsonl>     20 tool prompts, temperature 0
  bench.py throughput <base_url_v1> <model> <conc> <n>  agent-like load, prints req/s, tok/s
  bench.py compare a.jsonl b.jsonl                      parse rate + agreement
"""
import concurrent.futures as cf
import difflib
import json
import random
import sys
import time
import urllib.request

SYSTEM = ("You are a helpful assistant that can interact with a computer to solve software "
          "engineering tasks. You work in a Linux sandbox with a git repository checked out at "
          "/testbed. Use the provided tools: run one tool call per turn, inspect results, and "
          "call `submit` when the task is complete. Do not ask the user questions.")
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a bash command in the sandbox and return stdout/stderr.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to run."},
                    "timeout": {"type": "integer", "description": "Seconds before the command is killed."}},
                    "required": ["command"]}}},
    {"type": "function", "function": {"name": "str_replace", "description": "Replace an exact string in a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                    "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "submit", "description": "Finish the task.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]
TASKS = [
    "Find all Python files under /testbed that import `requests` and count them.",
    "The test tests/test_math.py::test_div fails with ZeroDivisionError. Find the cause.",
    "Show the last 5 commits of the repository with their authors.",
    "Create a file /testbed/hello.py that prints 'hello world'.",
    "Check which Python version is installed in the sandbox.",
    "Replace the string 'DEBUG = True' with 'DEBUG = False' in /testbed/settings.py.",
    "Run the test suite with pytest and report how many tests fail.",
    "Find the definition of the function `parse_config` in the repository.",
    "List the 10 largest files in /testbed.",
    "Install nothing; just print the contents of /testbed/README.md.",
    "Count the lines of code in /testbed/src, excluding blank lines.",
    "Find TODO comments in the codebase and list the files that contain them.",
    "Check whether port 8080 is in use inside the sandbox.",
    "Create a git branch named fix/issue-42 and switch to it.",
    "Write a bash one-liner that prints the number of CPU cores, and run it.",
    "The file /testbed/data.csv has a header row. Print the column names.",
    "Find files modified in the last day under /testbed.",
    "Check the disk usage of /testbed in human-readable form.",
    "Search the repo for the environment variable API_URL and show where it is read.",
    "Verify that /testbed/setup.py exists and print its first 20 lines.",
]


def chat(base, body, timeout=1800):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def messages(i):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASKS[i]}]
    if i % 4 == 3:   # some multi-turn: a previous tool call + its result
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",
                  "function": {"name": "bash", "arguments": json.dumps({"command": "ls /testbed"})}}]},
                 {"role": "tool", "tool_call_id": "call_0", "content": "README.md\nsetup.py\nsrc\ntests\nsettings.py\ndata.csv"}]
    return msgs


def parity(base, model, out):
    def one(i):
        t = time.time()
        body = {"model": model, "messages": messages(i), "tools": TOOLS, "temperature": 0,
                "max_tokens": 8192, "chat_template_kwargs": {"enable_thinking": True}}
        try:
            r = chat(base, body)
        except Exception as e:
            return {"i": i, "error": repr(e)}
        c = r["choices"][0]
        m = c["message"]
        calls = m.get("tool_calls") or []
        ok_calls = []
        for tc in calls:
            try:
                args = json.loads(tc["function"]["arguments"])
                ok_calls.append({"name": tc["function"]["name"], "args": args})
            except (json.JSONDecodeError, KeyError, TypeError):
                ok_calls.append({"name": tc.get("function", {}).get("name"), "args": None})
        return {"i": i, "seconds": round(time.time() - t, 2), "finish": c.get("finish_reason"),
                "tool_calls": ok_calls, "content": m.get("content") or "",
                "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                "usage": r.get("usage")}
    with cf.ThreadPoolExecutor(20) as ex:
        rows = list(ex.map(one, range(len(TASKS))))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summarize(rows, out)


def parsed(r):
    names = {t["function"]["name"] for t in TOOLS}
    return bool(r.get("tool_calls")) and all(t["args"] is not None and t["name"] in names for t in r["tool_calls"])


def leaked(r):
    return "<tool_call>" in (r.get("content") or "") or "<function=" in (r.get("content") or "")


def summarize(rows, name):
    n = len(rows)
    print(f"{name}: tool-call parsed {sum(parsed(r) for r in rows)}/{n}; "
          f"unparsed <tool_call> text in content {sum(leaked(r) for r in rows)}; "
          f"errors {sum('error' in r for r in rows)}; finish={[r.get('finish') for r in rows]}; "
          f"reasoning field non-empty {sum(bool(r.get('reasoning')) for r in rows)}")


def compare(a, b):
    A = [json.loads(l) for l in open(a)]
    B = [json.loads(l) for l in open(b)]
    summarize(A, a); summarize(B, b)
    same_call = same_name = 0
    sims = []
    for x, y in zip(A, B):
        cx, cy = x.get("tool_calls") or [], y.get("tool_calls") or []
        if cx and cy and cx[0]["name"] == cy[0]["name"]:
            same_name += 1
            same_call += cx[0]["args"] == cy[0]["args"]
        tx = (x.get("reasoning") or "") + (x.get("content") or "")
        ty = (y.get("reasoning") or "") + (y.get("content") or "")
        sims.append(difflib.SequenceMatcher(None, tx[:600], ty[:600]).ratio())
        print(f"  {x['i']:2d} {[c['name'] for c in cx]} vs {[c['name'] for c in cy]} "
              f"args_equal={bool(cx and cy and cx[0]['args'] == cy[0]['args'])} text_sim={sims[-1]:.2f}")
    print(f"same first tool: {same_name}/{len(A)}; identical first call (name+args): {same_call}/{len(A)}; "
          f"mean text similarity (first 600 chars) {sum(sims)/len(sims):.2f}")


def throughput_long(base, model, conc, n, max_tokens=1024):
    """Decode-heavy: short prompts, ignore_eos-free long answers (thinking off)."""
    topics = ["TCP congestion control", "B-trees", "garbage collection", "the CAP theorem",
              "gradient descent", "virtual memory", "consensus protocols", "hash maps"]
    def one(i):
        body = {"model": model, "temperature": 0.6, "top_p": 0.95, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": f"Explain {topics[i % len(topics)]} in great detail, "
                              f"with examples, in at least 1500 words. (#{i})"}],
                "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        r = chat(base, body)
        u = r.get("usage") or {}
        return time.time() - t, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    return _run(base, model, conc, n, one)


def _run(base, model, conc, n, one):
    chat(base, {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t0
    lat = sorted(r[0] for r in res)
    pin, pout = sum(r[1] for r in res), sum(r[2] for r in res)
    print(json.dumps({"concurrency": conc, "requests": n, "wall_s": round(wall, 1),
                      "req_per_s": round(n / wall, 3), "output_tok_per_s": round(pout / wall, 1),
                      "total_tok_per_s": round((pin + pout) / wall, 1),
                      "mean_prompt_tokens": round(pin / n), "mean_output_tokens": round(pout / n),
                      "p50_latency_s": round(lat[n // 2], 1), "p90_latency_s": round(lat[int(n * .9)], 1)}))


def throughput(base, model, conc, n, max_tokens=1024):
    rnd = random.Random(0)
    filler = " ".join(f"file_{k}.py" for k in range(300))

    def one(i):
        msgs = messages(i % len(TASKS))
        msgs[1] = {"role": "user", "content": msgs[1]["content"] + f"\nRepository listing (#{i}, seed {rnd.random():.6f}):\n" + filler}
        body = {"model": model, "messages": msgs, "tools": TOOLS, "temperature": 0.6, "top_p": 0.95,
                "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": True}}
        t = time.time()
        r = chat(base, body)
        u = r.get("usage") or {}
        return time.time() - t, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    chat(base, {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t0
    lat = sorted(r[0] for r in res)
    pin, pout = sum(r[1] for r in res), sum(r[2] for r in res)
    print(json.dumps({"concurrency": conc, "requests": n, "wall_s": round(wall, 1),
                      "req_per_s": round(n / wall, 3), "output_tok_per_s": round(pout / wall, 1),
                      "total_tok_per_s": round((pin + pout) / wall, 1),
                      "mean_prompt_tokens": round(pin / n), "mean_output_tokens": round(pout / n),
                      "p50_latency_s": round(lat[n // 2], 1), "p90_latency_s": round(lat[int(n * .9)], 1)}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "parity":
        parity(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "throughput":
        throughput(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "throughput-long":
        throughput_long(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
