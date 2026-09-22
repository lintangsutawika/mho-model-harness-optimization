from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

LANGUAGES = ("python",)
PKG = Path(__file__).parent

DATASET_NAME = "MathArena/hmmt_feb_2025"
SPLIT = "train"
DEFAULT_DATASET = PKG / "data" / "hmmt_feb_2025.jsonl"

# ---------------------------------------------------------------- data loading
def row_to_problem(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """MathArena columns: problem_idx (int), problem (text), answer (exact
    expression, often LaTeX), problem_type (list of str)."""
    idx = row.get("problem_idx", index + 1)
    return {
        "problem_id": int(idx),
        "problem": str(row["problem"]).strip(),
        "answer": str(row["answer"]).strip(),
        "problem_type": [t for t in (row.get("problem_type") or ()) if t],
    }


def build_dataset(
    dataset_path: Path | str | None = None,
    max_tasks: int = 0,
    split: str = SPLIT,
) -> list[dict[str, Any]]:
    """Assemble ``data/<task>.jsonl`` from HF (MathArena/hmmt_feb_2025)."""
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
        "name": f"mho/hmmt-feb-2025-{problem['problem_id']}",
    }
    _fill(task / "task.toml", **fills)
    _fill(task / "instruction.md", **fills)
    _fill(task / "tests" / "test.py", **fills)
    return task


def generate_all(
    problems: list[dict[str, Any]],
    output: Path | str,
    languages: tuple[str, ...] = LANGUAGES,
    skip_unsupported: bool = False,
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
                    generate(row, language, staged)
                except (NotImplementedError, ValueError) as exc:
                    if not skip_unsupported:
                        raise
                    exclusions.append(
                        {"problem_id": row["problem_id"], "language": language, "reason": str(exc)}
                    )
        if exclusions:
            (staged / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        staged.rename(output)
    return exclusions, sum(1 for p in output.iterdir() if p.is_dir())


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
    args = parser.parse_args()

    output_dir = args.output_dir or Path("tasks")
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
    exclusions, count = generate_all(problems, output=output_dir)
    print(f"Generated {count} tasks in {output_dir}; excluded {len(exclusions)}")


if __name__ == "__main__":
    main()
