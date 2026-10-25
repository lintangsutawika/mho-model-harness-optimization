from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

PKG = Path(__file__).parent

DATASET_NAME = "nvidia/Nemotron-Terminal-Synthetic-Tasks"
SPLIT = "skill_based/mixed"
CATEGORIES = ("data_processing", "data_science", "debugging",
              "file_operations", "scientific_computing", "security")
ARCHIVE_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/{SPLIT}/{{category}}.tar.gz"
DATA_DIR = PKG / "data"

# Every upstream task.toml points docker_image at an NVIDIA-internal registry. With the line
# set, harbor pulls that image instead of building environment/Dockerfile (a public
# ubuntu-24-04 base), so the tasks fail on docker, singularity and modal alike.
_DOCKER_IMAGE_RE = re.compile(r"^docker_image\s*=.*\n", re.MULTILINE)


# ---------------------------------------------------------------- data loading
def fetch_archive(category: str) -> Path:
    """Download ``<category>.tar.gz`` once into data/ (a few MB each)."""
    path = DATA_DIR / f"{category}.tar.gz"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(ARCHIVE_URL.format(category=category), tmp)
        tmp.rename(path)
    return path


def task_ids(archive: Path, category: str) -> list[str]:
    """Sorted task directories ``./<category>/<task_id>/`` (images.json is not a task)."""
    ids = set()
    with tarfile.open(archive) as tar:
        for name in tar.getnames():
            parts = Path(name).parts
            if parts and parts[0] == ".":
                parts = parts[1:]
            if len(parts) >= 3 and parts[0] == category:
                ids.add(parts[1])
    return sorted(ids)


def select(category: str, pool: list[str], per_category: int, seed: int) -> list[str]:
    if per_category <= 0 or per_category >= len(pool):
        return list(pool)
    return sorted(random.Random(f"nemotron:{seed}:{category}").sample(pool, per_category))


# ---------------------------------------------------------------- generation
def extract(archive: Path, category: str, ids: list[str], output: Path) -> None:
    wanted = set(ids)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if parts and parts[0] == ".":
                parts = parts[1:]
            if len(parts) < 3 or parts[0] != category or parts[1] not in wanted:
                continue
            if not (member.isfile() or member.isdir()):
                continue
            dest = output.joinpath(*parts[1:])
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            dest.chmod(member.mode & 0o777 | 0o600)
    for task_id in ids:
        toml = output / task_id / "task.toml"
        toml.write_text(_DOCKER_IMAGE_RE.sub("", toml.read_text()))


def generate_all(output: Path | str, per_category: int = 50, seed: int = 1,
                 categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "archives": [f"{SPLIT}/{c}.tar.gz" for c in categories],
        "selection_seed": seed,
        "selection_method": (
            "per-category stratified. For each category c the pool is the sorted list of "
            "directory members at ./c/<task_id>/ in skill_based/mixed/c.tar.gz (the archive's "
            f"./c/images.json is excluded), and the sample is random.Random('nemotron:{seed}:'+c)"
            f".sample(pool, {per_category})."
        ),
        "tasks_per_category": per_category,
        "docker_image": "removed from every task.toml; environment/Dockerfile is built instead",
        "per_category": {},
        "tasks": [],
    }
    for category in categories:
        archive = fetch_archive(category)
        pool = task_ids(archive, category)
        ids = select(category, pool, per_category, seed)
        extract(archive, category, ids, output)
        manifest["per_category"][category] = {"archive_task_count": len(pool), "selected": len(ids)}
        manifest["tasks"].extend(ids)
        print(f"{category}: {len(ids)} of {len(pool)}")
    manifest["num_tasks"] = len(manifest["tasks"])
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest['num_tasks']} tasks to {output}")
    return manifest["tasks"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Harbor Nemotron-Terminal-Synthetic-Tasks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=50,
                        help="Tasks sampled per skill category (0 = all 1000).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--category", action="append", choices=CATEGORIES,
                        help="Repeatable; default all six.")
    args = parser.parse_args()
    generate_all(args.output_dir, args.per_category, args.seed,
                 tuple(args.category) if args.category else CATEGORIES)


if __name__ == "__main__":
    main()
