#!/usr/bin/env python3
"""Direct (no-scaffold) math eval: one model call per problem, graded with math_verify.

This bypasses the agent harness entirely -- no mini-swe-agent, no shell tool loop, no
sandbox. Each problem is sent as a single chat completion; the model reasons freely and
puts its final answer in \\boxed{...}; we grade with math_verify (same grader as the harbor
tasks). This measures the MODEL's native capability, so it's the right baseline to compare
against (a) our harnessed pass@1 and (b) the published thinking-mode numbers.

Problems come from either a local harbor dataset dir (--data-path, same problems the harness
eval uses) or a HF dataset (--dataset, columns problem/answer).

Run behind an OpenAI-compatible endpoint (e.g. the vLLM that direct_eval.sh serves):
  python direct_eval.py --model Qwen3.5-4B --base-url http://localhost:8000/v1 \
    --data-path data-harbor/hmmt_feb_2025 --n-attempts 4 --concurrency 32 --out out.json
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import time
from pathlib import Path

SYSTEM = (
    "You are an expert competition mathematician. Solve the problem, reasoning step by "
    "step. Give your final answer on the last line inside \\boxed{...}."
)


def load_problems(dataset: str, data_path: str, split: str, max_tasks: int):
    """Return list of (id, problem_text, gold_answer)."""
    probs = []
    if data_path:
        d = Path(data_path)
        dirs = sorted(p for p in d.iterdir() if (p / "task.toml").exists())
        for td in dirs:
            instr = (td / "instruction.md").read_text()
            # our instruction template puts the problem after this boilerplate sentence
            marker = "no extra words."
            problem = instr.split(marker, 1)[-1].strip() if marker in instr else instr.strip()
            tp = (td / "tests" / "test_outputs.py").read_text()
            m = re.search(r"EXPECTED_ANSWER\s*=\s*(.+)", tp)
            ans = ast.literal_eval(m.group(1).strip()) if m else None
            probs.append((td.name, problem, str(ans)))
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, split=split)
        for i, row in enumerate(ds):
            probs.append((str(row.get("problem_idx", i + 1)), row["problem"], str(row["answer"])))
    if max_tasks:
        probs = probs[:max_tasks]
    return probs


def grade(resp: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify
        g = parse(gold if gold.strip().startswith("$") else f"${gold}$")
        p = parse(resp)  # extracts \boxed{} / last answer from the full response
        return bool(verify(g, p))
    except Exception:
        return False


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-path", help="local harbor dataset dir (same problems as harness eval)")
    src.add_argument("--dataset", help="HF dataset id (columns: problem, answer)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--n-attempts", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=30000)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=a.base_url, api_key=a.api_key)

    probs = load_problems(a.dataset, a.data_path, a.split, a.max_tasks)
    jobs = [(pid, prob, ans, k) for (pid, prob, ans) in probs for k in range(a.n_attempts)]
    src_name = a.data_path or a.dataset
    print(f"[direct] {len(probs)} problems x {a.n_attempts} attempts = {len(jobs)} calls | "
          f"model={a.model} | src={src_name} | max_tokens={a.max_tokens} temp={a.temperature}")

    sem = asyncio.Semaphore(a.concurrency)
    results = []

    async def run(job):
        pid, prob, ans, k = job
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=a.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prob}],
                    temperature=a.temperature, max_tokens=a.max_tokens,
                )
                resp = r.choices[0].message.content or ""
                fr = r.choices[0].finish_reason
                ok = grade(resp, str(ans))
            except Exception as e:
                resp, fr, ok = f"<error: {e}>", "error", False
            results.append({"id": pid, "attempt": k, "solved": ok, "gold": str(ans),
                            "finish_reason": fr, "resp_tail": resp[-500:]})

    t0 = time.time()
    await asyncio.gather(*[run(j) for j in jobs])
    solved = sum(r["solved"] for r in results)
    errored = sum(r["finish_reason"] == "error" for r in results)
    truncated = sum(r["finish_reason"] == "length" for r in results)
    tot = len(results)
    passk = solved / tot if tot else 0.0
    print(f"[direct] pass@1 = {passk:.4f}  ({solved}/{tot})  "
          f"errored={errored} truncated={truncated}  in {time.time() - t0:.0f}s")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"pass@1": passk, "solved": solved, "total": tot, "errored": errored,
             "truncated": truncated, "model": a.model, "source": src_name,
             "n_attempts": a.n_attempts, "temperature": a.temperature,
             "max_tokens": a.max_tokens, "results": results}, indent=1))
        print(f"[direct] wrote {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
