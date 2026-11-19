"""Memory/time probe of a Tinker gateway: one synthetic datum of N tokens, forward_backward +
optim_step (learning rate 0: the weights do not move), then the model is unloaded so the next
client (platoon) can take the single full-fine-tuning slot.

    python -m harnessforge.train.probe --url http://host:port --model Qwen/Qwen3.5-4B --tokens 131072

Needs the tinker SDK (the platoon venv). Writes a JSON summary (--out); the trainer node's GPU
memory comes from the gateway job's logs/gpu_mem.<job>.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import time


def datum(types, n: int, seed: int = 0):
    rng = random.Random(seed)
    toks = [rng.randrange(1, 150_000) for _ in range(n + 1)]
    return types.Datum(model_input=types.ModelInput.from_ints(toks[:-1]),
                       loss_fn_inputs={"target_tokens": toks[1:], "logprobs": [-1.0] * n,
                                       "advantages": [1.0 / n] * n})


def main(argv=None) -> int:
    import httpx
    import tinker
    from tinker import types
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    n = a.tokens - 1                       # model_input + 1 target = tokens
    sc = tinker.ServiceClient(base_url=a.url)
    tc = sc.create_lora_training_client(base_model=a.model, rank=32)
    results = {"tokens": a.tokens, "steps": []}
    try:
        for i in range(a.steps):
            t0 = time.time()
            fb = tc.forward_backward([datum(types, n, seed=i)], loss_fn="ppo",
                                     loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}).result()
            t1 = time.time()
            op = tc.optim_step(types.AdamParams(learning_rate=0.0)).result()
            t2 = time.time()
            m = dict(getattr(op, "metrics", {}) or {})
            step = {"forward_backward_s": round(t1 - t0, 1), "optim_step_s": round(t2 - t1, 1),
                    "fb_metrics": dict(getattr(fb, "metrics", {}) or {}),
                    "optim_metrics": {k: m[k] for k in sorted(m)}}
            results["steps"].append(step)
            print(json.dumps(step), flush=True)
        results["ok"] = True
    except Exception as e:                  # an OOM comes back as a failed request
        results.update(ok=False, error=f"{type(e).__name__}: {str(e)[:2000]}")
        print(results["error"], flush=True)
    finally:
        try:                                  # free the single full-FT slot for platoon
            r = httpx.post(f"{a.url}/api/v1/unload_model", json={"model_id": tc.model_id},
                           headers={"X-API-Key": "tml-local"}, timeout=60)
            results["unload"] = r.status_code
        except Exception as e:
            results["unload"] = f"failed: {e}"
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    return 0 if results.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
