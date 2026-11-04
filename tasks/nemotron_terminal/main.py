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

# Per task, only instruction.md, environment/ (the category's Dockerfile plus files/) and the
# tests besides test.sh differ upstream; task.toml and tests/test.sh are the same for all 5,984
# tasks and live in task-template/. The upstream task.toml is dropped because it points
# docker_image at an NVIDIA-internal registry, which harbor would pull instead of building
# environment/Dockerfile.
FROM_ARCHIVE = ("environment/", "tests/")
SHARED = ("tests/test.sh",)


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
def _fill(path: Path, **kw: str) -> None:
    """Substitute {placeholders} in one pass, so braces inside an instruction stay literal."""
    pattern = re.compile(r"\{(" + "|".join(map(re.escape, kw)) + r")\}")
    path.write_text(pattern.sub(lambda m: str(kw[m.group(1)]), path.read_text()))


def extract(archive: Path, category: str, ids: list[str], output: Path) -> None:
    """Render each task from task-template/ plus its own files from the archive."""
    wanted = set(ids)
    instructions = {}
    for task_id in ids:
        shutil.copytree(PKG / "task-template", output / task_id, dirs_exist_ok=False)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if parts and parts[0] == ".":
                parts = parts[1:]
            if len(parts) < 3 or parts[0] != category or parts[1] not in wanted:
                continue
            if not member.isfile():
                continue
            task_id, rel = parts[1], "/".join(parts[2:])
            with tar.extractfile(member) as src:
                data = src.read()
            if rel == "instruction.md":
                instructions[task_id] = data.decode("utf-8")
                continue
            if rel in SHARED:
                if data != (PKG / "task-template" / rel).read_bytes():
                    raise ValueError(f"{task_id}: upstream {rel} differs from task-template/{rel}")
                continue
            if not rel.startswith(FROM_ARCHIVE):
                continue
            dest = output / task_id / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            dest.chmod(member.mode & 0o777 | 0o600)
    for task_id in ids:
        task = output / task_id
        _fill(task / "task.toml", name=f"mho/nemotron-terminal-{task_id}",
              category=category, task_id=task_id)
        _fill(task / "instruction.md", instruction=instructions[task_id])


def _write_tasks(output: Path, selection: list[tuple[str, Path, int, list[str]]],
                 per_category: int, seed: int) -> list[str]:
    """Render ``(category, archive, pool_size, ids)`` rows into one dataset directory."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "archives": [f"{SPLIT}/{category}.tar.gz" for category, *_ in selection],
        "selection_seed": seed,
        "selection_method": (
            "per-category stratified. For each category c the pool is the sorted list of "
            "directory members at ./c/<task_id>/ in skill_based/mixed/c.tar.gz (the archive's "
            f"./c/images.json is excluded), and the sample is random.Random('nemotron:{seed}:'+c)"
            f".sample(pool, {per_category})."
        ),
        "tasks_per_category": per_category,
        "task_toml": "task-template/task.toml; the upstream one (with an internal docker_image) is dropped",
        "per_category": {},
        "tasks": [],
    }
    for category, archive, pool_size, ids in selection:
        extract(archive, category, ids, output)
        manifest["per_category"][category] = {"archive_task_count": pool_size, "selected": len(ids)}
        manifest["tasks"].extend(ids)
        print(f"{category}: {len(ids)} of {pool_size}")
    manifest["num_tasks"] = len(manifest["tasks"])
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest['num_tasks']} tasks to {output}")
    return manifest["tasks"]


def generate_all(output: Path | str, per_category: int = 50, seed: int = 1,
                 categories: tuple[str, ...] = CATEGORIES, val_size: int = 0) -> list[str]:
    output = Path(output)
    if val_size < 0:
        raise SystemExit(f"--val-size must be >= 0, got {val_size}.")
    train_out = output.with_name(output.name + "-train-split")
    val_out = output.with_name(output.name + "-val-split")
    if val_size:
        for out in (train_out, val_out):
            if out.exists():
                raise SystemExit(f"Output directory already exists: {out}. Use a fresh --output-dir.")

    # N is a total: every category gets base, and the first `extra` categories get one more.
    base, extra = divmod(val_size, len(categories))
    train, val = [], []
    for i, category in enumerate(categories):
        archive = fetch_archive(category)
        pool = task_ids(archive, category)
        ids = select(category, pool, per_category, seed)
        k = base + (1 if i < extra else 0)
        # Check every category before writing anything; train must keep at least one task.
        if k and k >= len(ids):
            raise SystemExit(f"--val-size {val_size} takes {k} tasks from {category}, "
                             f"which only has {len(ids)} selected.")
        cut = len(ids) - k  # one index, so k == 0 needs no special case (ids[-0:] is all of ids)
        train.append((category, archive, len(pool), ids[:cut]))
        val.append((category, archive, len(pool), ids[cut:]))

    if not val_size:
        return _write_tasks(output, train, per_category, seed)

    train_ids = _write_tasks(train_out, train, per_category, seed)
    val_ids = _write_tasks(val_out, val, per_category, seed)
    (output.parent / f"{output.name}_split_manifest.json").write_text(json.dumps({
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "selection_seed": seed,
        "tasks_per_category": per_category,
        "val_size": val_size,
        "split_method": "per category, the last k of the sorted selected task_ids are val; "
                        "k = val_size // n_categories, plus 1 for the first val_size % n_categories",
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_out": str(train_out),
        "val_out": str(val_out),
        "val_task_ids": val_ids,
    }, indent=2) + "\n")
    return train_ids + val_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Harbor Nemotron-Terminal-Synthetic-Tasks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=50,
                        help="Tasks sampled per skill category (0 = all 1000).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=0,
                        help="Total tasks to hold out as val, split evenly across categories "
                             "(last task_ids of each, disjoint from train). Generates "
                             "<out>-train-split + <out>-val-split. 0 disables (generate one dir).")
    parser.add_argument("--category", action="append", choices=CATEGORIES,
                        help="Repeatable; default all six.")
    args = parser.parse_args()
    generate_all(args.output_dir, args.per_category, args.seed,
                 tuple(args.category) if args.category else CATEGORIES,
                 val_size=args.val_size)


if __name__ == "__main__":
    main()
