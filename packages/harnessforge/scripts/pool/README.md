Cluster scripts behind the numbers in `docs/pool.md` (ABCI). Stdlib only.

- `pool-abci-2xHG.json`: the N=2 x rt_HG smoke spec; `pool-abci-4xHF.json`: full size (untested).
- `smoke_n2.sh`: start, chat through the router, throughput, policy swap, node kill + repair, stop.
- `smoke_client.py <endpoint>`: /v1/models, a chat, a tool call.
- `bench.py parity|throughput|throughput-long|compare`: tool-call parity and load tests.
- `swap_check.py <spec> <ckpt>`: outputs before/after `load()`, drain vs abort, swap back.
- `perturb.py <out> [scale]`: a deterministic perturbed copy of Qwen3.5-4B (run in a job).
- `vllm_bench.sh`: the vLLM 0.28 baseline with the mho-rsft eval flags.
