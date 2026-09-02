#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CACHE_BASE=$TMPDIR/cache_dir/
APPTAINER_TMPDIR=$TMPDIR/apptainer_tmp/
BASE_SIF=${BASE_PATH}/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron.sif

mkdir -p $CACHE_BASE
mkdir -p $APPTAINER_TMPDIR

singularity exec --nv \
	--writable-tmpfs \
	--workdir "${APPTAINER_TMPDIR}" \
	--bind "${CACHE_BASE}/tmp:/tmp_work" \
	--bind "${CACHE_BASE}/uv_cache:/root/.cache/uv" \
	--bind "${CACHE_BASE}/hf_cache:/root/.cache/huggingface" \
	--env CPATH= \
	--env TMPDIR=/tmp_work \
	--env UV_CACHE_DIR=/root/.cache/uv \
	--env HF_HOME=/root/.cache/huggingface \
	--env RAY_worker_register_timeout_seconds=600 \
	"${BASE_SIF}" \
		bash examples/train/megatron/run_megatron_lora_qwen3-30b-a3b.sh
