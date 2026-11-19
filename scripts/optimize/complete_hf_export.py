"""Make miles' HF exports loadable: fill each <hf_dir>/step_N from the base checkpoint.

miles' Megatron->HF export (hf_export.export_hf_model_direct) writes only what the Megatron model
holds -- the language model -- and copies config/tokenizer files only when --hf-checkpoint is a
local dir. For Qwen3.5 that leaves a step_N with no config.json, no vision tower (model.visual.*)
and no MTP head (mtp.*), while config.json declares Qwen3_5ForConditionalGeneration. The loop
needs config.json to find the checkpoint, and vLLM needs the full weight set to serve it.

Training never touches the missing tensors, so the base's copies are the trained model's too.
For each export this script copies the base's metadata files that are missing, writes the base
tensors the export lacks into one extra shard, and adds them to model.safetensors.index.json.
Exported tensors are never overwritten. Idempotent.

Usage (container python, HF cache bound as in train.sh):
    python -I complete_hf_export.py <hf_dir> <base: local dir or hub id>
"""

import json
import shutil
import sys
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
INDEX = "model.safetensors.index.json"
EXTRA_SHARD = "model-base-extra.safetensors"


def resolve_base(base: str) -> Path:
    p = Path(base)
    if p.is_dir():
        return p
    # Resolve the cached snapshot via refs/main by hand: snapshot_download(local_files_only=True)
    # raises IncompleteSnapshotError when non-model files (README, LICENSE) were never fetched.
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = Path(HF_HUB_CACHE) / ("models--" + base.replace("/", "--"))
    snap = repo / "snapshots" / (repo / "refs" / "main").read_text().strip()
    if not snap.is_dir():
        raise FileNotFoundError(f"no cached snapshot for {base} at {snap}")
    return snap


def base_weight_map(base: Path) -> dict[str, str]:
    idx = base / INDEX
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    from safetensors import safe_open

    out = {}
    for f in sorted(base.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            out.update({k: f.name for k in h.keys()})
    return out


def complete(step: Path, base: Path, bmap: dict[str, str]) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    copied = []
    for f in base.iterdir():
        if f.is_file() and f.suffix not in WEIGHT_SUFFIXES and not f.name.endswith(".index.json") \
                and not (step / f.name).exists():
            shutil.copy2(f, step / f.name)  # follows HF-cache symlinks to the blob
            copied.append(f.name)

    index = json.loads((step / INDEX).read_text())
    wmap = index["weight_map"]
    missing = sorted(set(bmap) - set(wmap))
    if missing:
        by_shard: dict[str, list[str]] = {}
        for k in missing:
            by_shard.setdefault(bmap[k], []).append(k)
        tensors = {}
        for shard, keys in by_shard.items():
            with safe_open(str(base / shard), "pt") as h:
                for k in keys:
                    tensors[k] = h.get_tensor(k)
        save_file(tensors, str(step / EXTRA_SHARD), metadata={"format": "pt"})
        for k, t in tensors.items():
            wmap[k] = EXTRA_SHARD
        index.setdefault("metadata", {})
        index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + sum(
            t.numel() * t.element_size() for t in tensors.values())
        tmp = step / (INDEX + ".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(step / INDEX)

    extra = set(wmap) - set(bmap)
    print(f"[complete_hf_export] {step}: copied {len(copied)} metadata files, "
          f"filled {len(missing)} base tensors"
          + (f" ({', '.join(sorted({k.split('.')[1] if k.startswith('model.') else k.split('.')[0] for k in missing}))})"
             if missing else "")
          + (f"; WARNING {len(extra)} exported tensors not in base, e.g. {sorted(extra)[:3]}" if extra else ""))


def main() -> int:
    hf_dir, base_arg = Path(sys.argv[1]), sys.argv[2]
    # Only finished exports: miles writes .complete after the last shard and the index, so a
    # step still being written (e.g. while ckpt_eval_sidecar runs mid-training) is skipped.
    steps = [d for d in sorted(hf_dir.glob("step_*"))
             if (d / INDEX).exists() and (d / ".complete").exists()]
    if not steps:
        print(f"[complete_hf_export] no exports under {hf_dir}; nothing to do")
        return 0
    base = resolve_base(base_arg)
    bmap = base_weight_map(base)
    for step in steps:
        complete(step, base, bmap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
