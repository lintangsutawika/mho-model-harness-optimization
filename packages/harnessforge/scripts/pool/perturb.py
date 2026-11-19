"""Copy Qwen3.5-4B HF dir to OUT with a tiny deterministic multiplicative weight perturbation."""
import glob, json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file
src = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/"))[0]
out = sys.argv[1]; scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
os.makedirs(out, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith(".safetensors"):
        shutil.copy(os.path.realpath(os.path.join(src, f)), os.path.join(out, f))
g = torch.Generator().manual_seed(1234)
n = 0
for f in sorted(glob.glob(src + "*.safetensors")):
    sd = load_file(f)
    for k, v in sd.items():
        if v.is_floating_point() and v.dim() == 2 and "visual" not in k:
            noise = torch.randn(v.shape, generator=g, dtype=torch.float32)
            sd[k] = (v.float() * (1 + scale * noise)).to(v.dtype); n += 1
    save_file(sd, os.path.join(out, os.path.basename(f)), metadata={"format": "pt"})
json.dump({"source": src, "scale": scale, "seed": 1234, "tensors_perturbed": n},
          open(os.path.join(out, "PERTURBATION.json"), "w"), indent=1)
print("perturbed", n, "tensors ->", out)
