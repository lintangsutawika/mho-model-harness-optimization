# Public Modal endpoints for SGLang models (`scripts/serve/`)

Serve models from an ABCI node as public OpenAI-compatible HTTPS endpoints, one Modal relay app per
model. The node only dials out (WebSocket), so nothing inbound is needed.

```
client --https /v1/*--> Modal app hf-serve-<name> --ws--> bridge.py (node) --> SGLang DP4 (127.0.0.1)
```

- `endpoint.sh`: one endpoint: SGLang (`--dp-size` = GPUs in `CUDA_VISIBLE_DEVICES`, tp 1) in
  `miles_latest.sif`, deploys `src/harness/relay/modal_relay.py` as `hf-serve-<name>`, runs
  `src/harness/relay/bridge.py`, probes `/v1/models` through the relay, supervises (bridge restarts;
  the job ends if SGLang dies).
- `pair.sh`: the PBS job: two `endpoint.sh` on one rt_HF node, A on GPUs 0-3, B on GPUs 4-7
  (defaults: served `Qwen/Qwen3.5-9B` as `qwen35-9b` on :30011, `Qwen/Qwen3.5-4B` as `qwen35-4b` on :30012).

Adapted from the MSL eval serve process (`msl-multilingual-self-learning`:
`scripts/serve/eval_pair.sh` -> `scripts/rollout.sh` with `CAPTURE=0` -> `src/relay/public_api.py`
+ `src/relay/node_forwarder.py`). Kept: one job, two SGLang servers on GPU halves, one relay per
model, SGLang on localhost with a per-launch API key that only the bridge holds. Changed: no
capture spool, weight watcher or dispatcher; our relay/bridge (`bridge.py` gained
`BACKEND_API_KEY`); harnessforge pool flags (`--tool-call-parser qwen3_coder
--mem-fraction-static 0.85 --enable-metrics`, context 262144); a stable relay key in a mode-600 key
file so the URL and key survive a restart; and the relay app is left deployed on exit (set
`RELAY_STOP_ON_EXIT=1` to stop it), so a later job reconnects to the same URL.

## Run

```bash
mkdir -p ~/hf-dev/serve/logs
qsub -N hf-serve -q R9920261300 -P gai51740 -v RTYPE=rt_HF -l select=1 -l walltime=24:00:00 \
     -j oe -o ~/hf-dev/serve/logs/ scripts/serve/pair.sh       # from the repo root (PBS_O_WORKDIR)
```

Walltime can be longer (the relay's WebSocket is capped at 24 h per connection; the bridge
reconnects on its own). Pass local snapshots with `-v MODEL_A=/path,MODEL_B=/path` to skip the
hub lookup.

Per endpoint, in `~/hf-dev/serve/` (override `SERVE_DIR`):
`<name>.key` (relay key, mode 600), `<name>.endpoint` (URL, served name, key path, job; no secret),
`logs/<job>-<name>.out`, `logs/<job>-<name>/{sglang,bridge}.log`.

## Use

```bash
KEY="$(cat ~/hf-dev/serve/qwen35-9b.key)"
curl -H "Authorization: Bearer $KEY" https://<workspace>--hf-serve-qwen35-9b-web.modal.run/v1/models
```

`OPENAI_BASE_URL=https://<workspace>--hf-serve-<name>-web.modal.run/v1`, `OPENAI_API_KEY=$KEY`,
model = the served name (`Qwen/Qwen3.5-9B` / `Qwen/Qwen3.5-4B`; litellm: `litellm_proxy/Qwen/Qwen3.5-9B`). The relay does not stream (`stream: false`).
Tool calls come back as structured `tool_calls` (qwen3_coder parser); thinking is on by default
(`chat_template_kwargs: {"enable_thinking": false}` turns it off).

The relay container asks Modal for `RELAY_CPU` CPUs (default 1). With 4 it sometimes waited
minutes for capacity after a redeploy, and every request hung meanwhile.

Stop: `qdel <job>` ends SGLang and the bridge; the relay apps stay (503 until a bridge connects)
until `modal app stop hf-serve-qwen35-9b` / `modal app stop hf-serve-qwen35-4b`.
