from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

LANGUAGES = ("python",)
PKG = Path(__file__).parent

DATASET_NAME = "MathArena/arxivmath-training"
SPLIT = "train"
DEFAULT_DATASET = PKG / "data" / "arxivmath_training.jsonl"

# ---------------------------------------------------------------- data loading
def row_to_problem(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """arxivmath-training columns differ from arxivmath-0826.

    training: {paper_id, question, answer, title, authors, license, categories}
    0826:     {problem_idx, problem, answer, source, title, authors}

    This dataset has no problem_idx. `problem_id` is therefore a synthetic row index
    (index + 1), used only for --val-size/--sample subsetting and ordering (stable given a
    fixed dataset). The REAL dataset key is `paper_id`, kept in `source` -- it is what names
    each task dir / instance_id (see generate()), matching the 0826 package's use of source.
    """
    return {
        "problem_id": index + 1,
        "problem": str(row["question"]).strip(),
        "answer": str(row["answer"]).strip(),
        "problem_type": [c for c in (row.get("categories") or ()) if c],
        "source": str(row.get("paper_id") or "").strip() or None,
        "title": str(row.get("title") or "").strip() or None,
    }


def build_dataset(
    dataset_path: Path | str | None = None,
    max_tasks: int = 0,
    split: str = SPLIT,
) -> list[dict[str, Any]]:
    """Assemble ``data/<task>.jsonl`` from HF (MathArena/arxivmath-training)."""
    from datasets import load_dataset

    out = Path(dataset_path or DEFAULT_DATASET)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_NAME, split=split)
    n = len(ds) if not max_tasks else min(max_tasks, len(ds))
    records = [row_to_problem(i, ds[i]) for i in range(n)]
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} arxivmath-training problem records to {out}")
    return records


def load_problems(
    dataset_path: Path | str | None = None,
    *,
    build: bool = True,
) -> list[dict[str, Any]]:
    out = Path(dataset_path or DEFAULT_DATASET)
    if not out.is_file():
        if not build:
            raise FileNotFoundError(f"Dataset JSONL not found: {out}. Run build_dataset() first.")
        return build_dataset(out)
    records = []
    with out.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_problem(dataset_path: Path | str, problem_id: int) -> dict[str, Any]:
    out = Path(dataset_path)
    records = build_dataset(out) if not out.is_file() else load_problems(out, build=False)
    for record in records:
        if int(record["problem_id"]) == problem_id:
            return record
    raise ValueError(f"Problem ID {problem_id} not found in {out}")


# ------------------------------------------------------------- task generation
def _dockerfile_to_def(dockerfile: Path) -> str:
    """Minimal Dockerfile -> Singularity recipe: our task Dockerfile is usually just
    FROM + WORKDIR, so we only carry RUN/COPY/ENV deps into a %post/%files/%environment
    fragment. None when there are no such layers."""
    if not dockerfile.exists():
        return ""
    posts, files, envs = [], [], []
    current: list[str] | None = None
    def flush():
        nonlocal current
        if current:
            block = " ".join(ln.strip() for ln in current).strip()
            if block:
                posts.append(block)
            current = None
    lines = dockerfile.read_text().splitlines()
    i = 0
    # Light translation: only capture RUN (continuations) and COPY/ENV simply.
    while i < len(lines):
        raw = lines[i]; line = raw.strip(); i += 1
        if not line or line.startswith("#") or line.upper().startswith("FROM ") or line.upper().startswith("WORKDIR"):
            flush(); continue
        if line.upper().startswith("RUN "):
            flush()
            # Fold backslash continuations, dropping each trailing `\` so the %post line is one
            # clean command (no literal backslashes that apt/bash mis-parse).
            parts = [line[4:].strip()]
            while parts[-1].endswith("\\") and i < len(lines):
                parts[-1] = parts[-1][:-1].rstrip()
                parts.append(lines[i].strip()); i += 1
            if parts and not parts[-1].endswith("\\"):
                posts.append(" ".join(p for p in parts if p))
            flush()
        elif line.upper().startswith("ENV "):
            envs.append(line[4:].strip())
        elif line.upper().startswith("COPY "):
            files.append(line[5:].strip())
    parts = []
    if posts: parts += ["%post"] + ["    " + p for p in posts]
    if files: parts += ["%files"] + ["    " + f for f in files]
    if envs:  parts += ["%environment"] + ["    export " + e for e in envs]
    return "\n".join(parts)


def prebuild_sif(dockerfile: Path, out_sif: Path, *,
                 container_bin: str = "singularity", force: bool = False) -> Path:
    """Build a Singularity sif from the task's environment/Dockerfile into out_sif.
    Idempotent (skip if exists unless force). Mirrors msl's prebuild_sif."""
    dockerfile, out_sif = Path(dockerfile).resolve(), Path(out_sif).resolve()
    if out_sif.is_file() and not force:
        print(f"[prebuild] {out_sif.name} exists; skipping (--prebuild-force to rebuild)")
        return out_sif
    out_sif.parent.mkdir(parents=True, exist_ok=True)
    # The Singularity base is taken SOLELY from the Dockerfile's FROM (no hardcoded default).
    # A Dockerfile that says `FROM scratch` builds an empty image with no python3, which breaks
    # harbor's sandbox bootstrap ([ ! -x /usr/bin/python3 ] -> cannot install). The task recipe
    # must specify a real python base (e.g. FROM python:3.11-slim), and that is honored verbatim.
    base_image = ""
    try:
        for raw in dockerfile.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and line.upper().startswith("FROM "):
                ref = line.split(None, 1)[1].strip()
                base_image = ref.split()[0]
                break
    except OSError:
        pass
    if not base_image:
        raise ValueError(
            f"prebuild_sif: {dockerfile} has no FROM line. The sif base must come from the "
            f"Dockerfile (e.g. FROM python:3.11-slim), not a hardcoded default.")
    fragment = _dockerfile_to_def(dockerfile)
    def_path = out_sif.with_name(out_sif.name + ".def")
    # The sif is defined ENTIRELY by the task's Dockerfile (its FROM base + the translated
    # RUN/COPY/ENV layers as %post). No hardcoded %post here: the Dockerfile is the single
    # source of truth for the sandbox runtime (python base, uvicorn/fastapi, agent tools), so
    # any task built from its own Dockerfile yields a self-contained, offline-capable image.
    def_lines = ["Bootstrap: docker", f"From: {base_image}", ""]
    if fragment:
        def_lines.append(fragment)
    def_path.write_text("\n".join(def_lines))
    singular_cmd = [container_bin, "build", "--fakeroot"]
    if force:
        singular_cmd.append("--force")   # singularity build refuses to overwrite an existing .sif
    singular_cmd += [str(out_sif), str(def_path)]
    subprocess.run(singular_cmd, check=True, cwd=str(dockerfile.parent))
    print(f"[prebuild] built {out_sif}")
    return out_sif


def _install_sif(gen_dir: Path, sif: Path) -> Path:
    """Place the shared prebuilt sif as <gen_dir>/image.sif and relink every task's
    task.toml docker_image to it. Runs AFTER generation so <gen_dir> is fresh."""
    dest = gen_dir / "image.sif"
    shutil.copy2(sif, dest)
    for task in gen_dir.iterdir():
        toml = task / "task.toml"
        if toml.is_file():
            txt = toml.read_text()
            txt = txt.replace(f'docker_image = "{sif}"', f'docker_image = "{dest}"')
            toml.write_text(txt)
    print(f"[prebuild-sif] installed {dest}")
    return dest


def _fill(path: Path, **kw: str) -> None:
    text = path.read_text()
    for key, value in kw.items():
        text = text.replace("{" + key + "}", str(value))
    path.write_text(text)


_PAPER_COUNTER: dict[str, int] = {}


def _sanitize(paper_id: str) -> str:
    """Turn a paper_id (e.g. arXiv id / URN) into a safe, readable dir component."""
    s = str(paper_id).strip().split("/")[-1]      # last path segment (arXiv id)
    s = "".join(c for c in s if c.isalnum() or c in "._-")
    return s or "paper"


def generate(
    problem: dict[str, Any],
    language: str = "python",
    output: Path | str = "tasks",
    docker_image: str = "python:3.11-slim",
) -> Path:
    """Render one arxivmath-training task into a fresh subdir of ``output``.

    The task-dir name (== Miles' ``instance_id``) is the dataset's real key,
    ``paper_id`` (sanitized) -- this dataset has no problem_idx, one row maps to
    one paper, and every task is python-only (no language suffix needed). If a
    paper_id repeats (dataset changes / multiple questions from one paper), a
    ``-qN`` suffix keeps each dir unique."""
    if language != "python":
        raise ValueError(f"HMMT math only supports 'python', got {language!r}")
    output = Path(output)
    base = _sanitize(problem.get("source") or "")
    n = _PAPER_COUNTER.get(base, 0)
    _PAPER_COUNTER[base] = n + 1
    task = output / (f"{base}-q{n}" if n else base)
    if task.exists():
        raise FileExistsError(f"Use a fresh output directory: {task}")
    tpl = PKG / "task-template"
    shutil.copytree(tpl, task, dirs_exist_ok=False)
    fills = {
        "question_id": problem.get("source") or problem["problem_id"],
        "language": language,
        "problem": problem["problem"].strip(),
        "answer": json.dumps(problem.get("answer") or "", ensure_ascii=False),
        "name": f"mho/arxivmath-training-{_sanitize(problem.get('source') or '')}",
        "docker_image": docker_image,
    }
    # A prebuilt .sif already carries the full toolchain -> the task's environment/Dockerfile
    # must NOT re-trigger runtime image layering; replace it with a minimal no-RUN recipe that
    # keeps WORKDIR /workspace (msl leetcode adapter pattern).
    if docker_image.lower().endswith(".sif"):
        df = task / "environment" / "Dockerfile"
        if df.exists():
            df.write_text("FROM scratch\nWORKDIR /workspace\n")
    _fill(task / "task.toml", **fills)
    _fill(task / "instruction.md", **fills)
    _fill(task / "tests" / "test.py", **fills)
    return task


def generate_all(
    problems: list[dict[str, Any]],
    output: Path | str,
    languages: tuple[str, ...] = LANGUAGES,
    skip_unsupported: bool = False,
    docker_image: str = "python:3.11-slim",
) -> tuple[list[dict[str, Any]], int]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _PAPER_COUNTER.clear()
    exclusions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "tasks"
        staged.mkdir()
        for row in problems:
            for language in languages:
                try:
                    generate(row, language, staged, docker_image=docker_image)
                except (NotImplementedError, ValueError) as exc:
                    if not skip_unsupported:
                        raise
                    exclusions.append(
                        {"problem_id": row["problem_id"], "language": language, "reason": str(exc)}
                    )
        if exclusions:
            (staged / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        staged.rename(output)
    _emit_seed(output)
    return exclusions, sum(1 for p in output.iterdir() if p.is_dir())

def _emit_seed(output: Path) -> Path:
    """Write ``<output>/seed.jsonl``: one row per generated task dir with
    ``{prompt, metadata.instance_id}``. `instance_id` == the task-dir name, so Miles'
    harbor connector can resolve each prompt to its task dir via
    HARBOR_TASKS_DIR/<instance_id>. This SEEDS the rollout dataset (RL produces the
    rollouts); it is not the rollouts themselves."""
    seed_path = output / "seed.jsonl"
    n = 0
    with seed_path.open("w", encoding="utf-8") as f:
        for d in sorted(p for p in output.iterdir() if p.is_dir()):
            inst = d / "instruction.md"
            if not inst.exists():
                continue
            f.write(json.dumps({"prompt": inst.read_text().strip(),
                                "metadata": {"instance_id": d.name}}) + "\n")
            n += 1
    print(f"[seed] wrote {n} seed rows to {seed_path}")
    return seed_path


# ------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Harbor arxivmath-training tasks.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"Problems JSONL (default: {DEFAULT_DATASET}); built on first use.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to tasks/.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Generate only the first N selected problems.")
    parser.add_argument("--question-id", type=int, action="append",
                        help="Generate only this problem ID (repeatable).")
    parser.add_argument("--sample", type=int, default=None,
                        help="Draw N problems at random instead of taking the first N. "
                             "Requires --seed so the draw is reproducible.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for --sample. Two subsets drawn with different seeds are "
                             "independent; --limit would return the same leading rows.")
    parser.add_argument("--val-size", type=int, default=None,
                        help="Split the selected problems into a held-out val of this many "
                             "(last N by problem_id, disjoint from train) and generate "
                             "<out>-train-split + <out>-val-split. 0 disables (generate one dir).")
    parser.add_argument("--skip-unsupported", action="store_true",
                        help="Record unsupported problems in exclusions.json instead of aborting.")
    parser.add_argument("--image", default="python:3.11-slim",
                        help="docker_image for the generated tasks: either a registry image (default "
                             "python:3.11-slim) or a prebuilt .sif path. A .sif already carries the "
                             "toolchain (no runtime install/network needed), so the task's "
                             "environment/Dockerfile is neutralized (msl leetcode --images pattern).")
    parser.add_argument("--prebuild-sif", action="store_true",
                        help="Build a prebuilt Singularity sif (from the task's environment/Dockerfile) "
                             "into the output dir and use it as this dataset's docker_image -- no "
                             "runtime install/network needed in the sandbox. Idempotent; "
                             "--prebuild-force rebuilds. Mirrors msl leetcode --prebuild-sif.")
    parser.add_argument("--prebuild-force", action="store_true",
                        help="With --prebuild-sif, rebuild the sif even if it already exists.")
    parser.add_argument("--container-bin", default="singularity",
                        help="Container binary for --prebuild-sif (default: singularity).")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("tasks")
    docker_image = args.image
    built_sif = None
    if args.prebuild_sif:
        # Build the shared sif into a staging sibling (<out>.sif) so output_dir stays fresh for
        # generation; after generation, _install_sif places it as <out>/image.sif and relinks.
        built_sif = prebuild_sif(
            PKG / "task-template" / "environment" / "Dockerfile",
            output_dir.with_name(output_dir.name + ".sif"),
            container_bin=args.container_bin, force=args.prebuild_force,
        )
        docker_image = str(built_sif)
    if not args.dataset.is_file():
        build_dataset(args.dataset)
    problems = load_problems(args.dataset, build=False)

    if args.question_id:
        wanted = set(args.question_id)
        problems = [p for p in problems if int(p["problem_id"]) in wanted]
        found = {int(p["problem_id"]) for p in problems}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"Problem ID(s) absent from {args.dataset}: {missing}")

    problems.sort(key=lambda p: int(p["problem_id"]))
    if args.sample is not None:
        if args.seed is None:
            raise SystemExit("--sample requires --seed, otherwise the subset cannot be rebuilt.")
        if args.limit is not None:
            raise SystemExit("--sample and --limit both choose the subset; pass only one.")
        if args.sample > len(problems):
            raise SystemExit(f"--sample {args.sample} exceeds the {len(problems)} problems available.")
        problems = random.Random(args.seed).sample(problems, args.sample)
        problems.sort(key=lambda p: int(p["problem_id"]))
    elif args.limit is not None and args.limit >= 0:
        problems = problems[: args.limit]

    # Held-out val split: val = last args.val_size (deterministic by problem_id order,
    # disjoint from train), train = the rest. Generates <out>-train-split + <out>-val-split.
    if args.val_size:
        if args.val_size < 0 or args.val_size > len(problems):
            raise SystemExit(
                f"--val-size {args.val_size} must be in [0, {len(problems)}] "
                f"(got {len(problems)} selected problems)."
            )
        val = problems[-args.val_size:] if args.val_size else []
        train = problems[:-args.val_size] if args.val_size else problems
        train_out = output_dir.with_name(output_dir.name + "-train-split")
        val_out = output_dir.with_name(output_dir.name + "-val-split")
        for out_, subset, label in ((train_out, train, "train"), (val_out, val, "val")):
            if out_.exists():
                raise SystemExit(f"Output directory already exists: {out_}. Use a fresh --output-dir.")
            _ex, _count = generate_all(subset, output=out_, docker_image=docker_image)
            if built_sif:
                _install_sif(out_, built_sif)          # do NOT unlink here -- val-split still needs it
            print(f"[split] generated {_count} {label} tasks in {out_}; excluded {len(_ex)}")
        (output_dir.parent / f"{output_dir.name}_split_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": DATASET_NAME,
                    "split": SPLIT,
                    "val_size": args.val_size,
                    "train_count": len(train),
                    "val_count": len(val),
                    "train_out": str(train_out),
                    "val_out": str(val_out),
                    "problem_ids": [int(p["problem_id"]) for p in problems],
                    "val_problem_ids": [int(p["problem_id"]) for p in val],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}. Use a fresh --output-dir.")
    exclusions, count = generate_all(problems, output=output_dir, docker_image=docker_image)
    if built_sif:
        _install_sif(output_dir, built_sif)
        built_sif.unlink(missing_ok=True)              # staging sif was copied in as image.sif
    if args.sample is not None:
        (output_dir / "selection_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": DATASET_NAME,
                    "split": SPLIT,
                    "seed": args.seed,
                    "sample": args.sample,
                    "problem_ids": [int(p["problem_id"]) for p in problems],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Generated {count} tasks in {output_dir}; excluded {len(exclusions)}")


if __name__ == "__main__":
    main()
