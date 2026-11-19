#!/bin/bash
# Serve Qwen3.5-4B with vLLM exactly like mho-rsft eval (configs/serve/Qwen/Qwen3.5-4B.json), run bench.
cd $HOME/hf-dev/pool
exec > logs/vllm_bench.live.log 2>&1
SIF=/home/aci18914wh/mho-model-harness-optimization/vllm-openai-cuda-0.28.0.sif
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PORT=18000
singularity exec --nv $SIF vllm --version
singularity exec --nv $SIF vllm serve Qwen/Qwen3.5-4B --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 1 --data-parallel-size 1 --max-model-len 262144 --gpu-memory-utilization 0.85 \
  --served-model-name Qwen3.5-4B --enable-chunked-prefill --max-num-batched-tokens 32768 \
  --enable-prefix-caching --long-prefill-token-threshold 0 --enable-auto-tool-choice \
  --language-model-only --tool-call-parser qwen3_coder > logs/vllm_server.log 2>&1 &
for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 5; done
echo "vllm ready after $((i*5))s"
PY=$HOME/.local/bin/python3.12
$PY scripts/bench.py parity http://127.0.0.1:$PORT/v1 Qwen3.5-4B results/vllm_parity.jsonl
$PY scripts/bench.py throughput http://127.0.0.1:$PORT/v1 Qwen3.5-4B 64 256
echo BENCH-DONE
kill %1; wait
