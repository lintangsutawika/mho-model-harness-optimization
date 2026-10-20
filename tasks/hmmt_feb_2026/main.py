from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

LANGUAGES = ("python",)
PKG = Path(__file__).parent

DATASET_NAME = "MathArena/arxivmath-0826"
SPLIT = "train"
DEFAULT_DATASET = PKG / "data" / "arxivmath_0826.jsonl"

# ---------------------------------------------------------------- data loading
def row_to_problem(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """MathArena columns: problem_idx (int), problem (text), answer (exact
    arxivmath-0826 is an arXiv-math slice: columns {problem_idx, problem, answer,
    source, title, authors} -- no problem_type. source/title are carried through for
    provenance (unused by the verifier, which grades answer-string match like HMMT)."""
    idx = row.get("problem_idx", index + 1)
    return {
        "problem_id": int(idx),
        "problem": str(row["problem"]).strip(),
        "answer": str(row["answer"]).strip(),
        "problem_type": [t for t in (row.get("problem_type") or ()) if t],
        "source": str(row.get("source") or "").strip() or None,
        "title": str(row.get("title") or "").strip() or None,
    }


def build_dataset(
    dataset_path: Path | str | None = None,
    max_tasks: int = 0,
    split: str = SPLIT,
) -> list[dict[str, Any]]:
    """Assemble ``data/<task>.jsonl`` from HF (MathArena/arxivmath-0826)."""
    from datasets import load_dataset

    out = Path(dataset_path or DEFAULT_DATASET)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_NAME, split=split)
    n = len(ds) if not max_tasks else min(max_tasks, len(ds))
    records = [row_to_problem(i, ds[i]) for i in range(n)]
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} HMMT problem records to {out}")
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
def _fill(path: Path, **kw: str) -> None:
    text = path.read_text()
    for key, value in kw.items():
        text = text.replace("{" + key + "}", str(value))
    path.write_text(text)


def generate(
    problem: dict[str, Any],
    language: str = "python",
    output: Path | str = "tasks",
    docker_image: str = "python:3.11-slim",
) -> Path:
    """Render one HMMT math task into a fresh subdir of ``output``."""
    if language != "python":
        raise ValueError(f"HMMT math only supports 'python', got {language!r}")
    output = Path(output)
    task = output / f"{problem['problem_id']}-{language}"
    if task.exists():
        raise FileExistsError(f"Use a fresh output directory: {task}")
    tpl = PKG / "task-template"
    shutil.copytree(tpl, task, dirs_exist_ok=False)
    fills = {
        "question_id": problem["problem_id"],
        "language": language,
        "problem": problem["problem"].strip(),
        "answer": json.dumps(problem.get("answer") or "", ensure_ascii=False),
        "docker_image": docker_image,
        "name": f"mho/arxivmath-0826-{problem['problem_id']}",
    }
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
    parser = argparse.ArgumentParser(description="Generate Harbor HMMT math tasks.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"Problems JSONL (default: {DEFAULT_DATASET}); built on first use.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to tasks/.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Generate only the first N selected problems.")
    parser.add_argument("--question-id", type=int, action="append",
                        help="Generate only this problem ID (repeatable).")
    parser.add_argument("--skip-unsupported", action="store_true",
                        help="Record unsupported problems in exclusions.json instead of aborting.")
    parser.add_argument("--image", default="python:3.11-slim",
                        help="docker_image for the generated tasks: a registry image (default "
                             "python:3.11-slim) or a prebuilt .sif path (toolchain baked in, no "
                             "runtime install/network; environment/Dockerfile is neutralized).")
    parser.add_argument("--prebuild-sif", action="store_true",
                        help="Build a prebuilt Singularity sif (from the task environment Dockerfile) "
                             "and use it as this dataset's docker_image -- no runtime install/network. "
                             "Idempotent; --prebuild-force rebuilds.")
    parser.add_argument("--prebuild-force", action="store_true")
    parser.add_argument("--container-bin", default="singularity")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("tasks")
    docker_image = args.image
    if args.prebuild_sif:
        built = prebuild_sif(
            PKG / "task-template" / "environment" / "Dockerfile",
            output_dir.with_name(output_dir.name + ".sif"),
            container_bin=args.container_bin, force=args.prebuild_force,
        )
        docker_image = str(built)


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
    if args.limit is not None and args.limit >= 0:
        problems = problems[: args.limit]

    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}. Use a fresh --output-dir.")
    exclusions, count = generate_all(problems, output=output_dir, docker_image=docker_image)
    print(f"Generated {count} tasks in {output_dir}; excluded {len(exclusions)}")


if __name__ == "__main__":
    main()


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
            flush(); current = [line[4:].strip()]
            while lines and lines[i-1].rstrip().endswith("\\") and i < len(lines):
                current.append(lines[i].strip()); i += 1
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
    base_image = "python:3.11-slim"
    try:
        for raw in dockerfile.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and line.upper().startswith("FROM "):
                ref = line.split(None, 1)[1].strip()
                base_image = ref.split()[0]
                break
    except OSError:
        pass
    fragment = _dockerfile_to_def(dockerfile)
    def_path = out_sif.with_name(out_sif.name + ".def")
    def_lines = ["Bootstrap: docker", f"From: {base_image}", ""]
    if fragment:
        def_lines.append(fragment)
    def_path.write_text("\n".join(def_lines))
    subprocess.run([container_bin, "build", "--fakeroot", str(out_sif), str(def_path)],
                   check=True, cwd=str(dockerfile.parent))
    print(f"[prebuild] built {out_sif}")
    return out_sif
