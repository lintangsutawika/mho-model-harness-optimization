from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PKG = Path(__file__).parent

DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
DEFAULT_DATASET = PKG / "data" / "swe_rebench_v2.jsonl"

# The dataset names a log parser per instance (install_config.log_parser) but ships the parser
# code in its GitHub repo, so it is vendored into data/ (gitignored like the JSONL) and copied
# into every task's tests/ -- the verifier then runs without network.
PARSER_DIR = PKG / "data" / "log_parsers"
PARSER_SOURCES = {
    "log_parsers.py": "https://raw.githubusercontent.com/SWE-rebench/SWE-rebench-V2/main/lib/agent/log_parsers.py",
    "swe_constants.py": "https://raw.githubusercontent.com/SWE-rebench/SWE-rebench-V2/main/lib/agent/swe_constants.py",
}

# Paging the datasets-server costs a few hundred KB; `datasets` pulls the 409 MB parquet.
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
ROWS_PAGE = 100
ROWS_SERVER_MAX = 2000

REQUIRED = ("instance_id", "repo", "base_commit", "image_name", "problem_statement",
            "patch", "test_patch", "test_cmd", "log_parser", "fail_to_pass")

_DIFF_HEADER_RE = re.compile(r'^diff --git ("?)a/(?P<old>.+?)\1 ("?)b/(?P<new>.+?)\3$', re.MULTILINE)


# ---------------------------------------------------------------- data loading
def row_to_problem(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """SWE-rebench-V2 is a raw SWE dataset, not a harbor one.

    One row carries the environment (a prebuilt per-instance ``image_name`` with the repo
    already cloned at ``base_commit`` and its dependencies installed), the issue text, the
    gold ``patch``, the reference ``test_patch``, the test command / log parser, and the test
    IDs that decide resolution.

    `problem_id` is a synthetic row index (index + 1) used only for --sample/--limit
    subsetting and ordering; the REAL dataset key is `instance_id`, which names each task dir.
    """
    install = row.get("install_config") or {}
    return {
        "problem_id": index + 1,
        "instance_id": str(row.get("instance_id") or "").strip(),
        "repo": str(row.get("repo") or "").strip(),
        "base_commit": str(row.get("base_commit") or "").strip(),
        "image_name": str(row.get("image_name") or "").strip(),
        "language": str(row.get("language") or "").strip(),
        "license": str(row.get("license") or "").strip(),
        "problem_statement": str(row.get("problem_statement") or "").strip(),
        "patch": str(row.get("patch") or ""),
        "test_patch": str(row.get("test_patch") or ""),
        "fail_to_pass": [t for t in (row.get("FAIL_TO_PASS") or ()) if t],
        "pass_to_pass": [t for t in (row.get("PASS_TO_PASS") or ()) if t],
        "test_cmd": str(install.get("test_cmd") or "").strip(),
        "log_parser": str(install.get("log_parser") or "").strip(),
        # The images do NOT ship the repo installed: `import <pkg>` fails and the test command
        # dies in conftest. The verifier runs these first.
        "install": [c for c in (install.get("install") or ()) if c],
    }


def _rows_from_server(split: str, max_tasks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while len(rows) < max_tasks:
        query = urllib.parse.urlencode({
            "dataset": DATASET_NAME, "config": "default", "split": split,
            "offset": len(rows), "length": min(ROWS_PAGE, max_tasks - len(rows)),
        })
        with urllib.request.urlopen(f"{ROWS_ENDPOINT}?{query}", timeout=60) as resp:
            page = json.loads(resp.read().decode("utf-8"))["rows"]
        if not page:
            break
        rows.extend(item["row"] for item in page)
    return rows


def build_dataset(
    dataset_path: Path | str | None = None,
    max_tasks: int = 0,
    split: str = SPLIT,
) -> list[dict[str, Any]]:
    """Assemble ``data/<task>.jsonl`` from HF (nebius/SWE-rebench-V2)."""
    out = Path(dataset_path or DEFAULT_DATASET)
    out.parent.mkdir(parents=True, exist_ok=True)
    if 0 < max_tasks <= ROWS_SERVER_MAX:
        rows = _rows_from_server(split, max_tasks)
    else:
        from datasets import load_dataset

        ds = load_dataset(DATASET_NAME, split=split)
        n = len(ds) if not max_tasks else min(max_tasks, len(ds))
        rows = [ds[i] for i in range(n)]
    records = [row_to_problem(i, row) for i, row in enumerate(rows)]
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} SWE-rebench-V2 problem records to {out}")
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


def load_problem(dataset_path: Path | str, question_id: str) -> dict[str, Any]:
    out = Path(dataset_path)
    records = build_dataset(out) if not out.is_file() else load_problems(out, build=False)
    for record in records:
        if question_id in (record["instance_id"], str(record["problem_id"])):
            return record
    raise ValueError(f"Question ID {question_id} not found in {out}")


def fetch_parsers(force: bool = False) -> Path:
    """Vendor SWE-rebench-V2's log parsers into ``data/log_parsers/``."""
    PARSER_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in PARSER_SOURCES.items():
        dest = PARSER_DIR / name
        if dest.is_file() and not force:
            continue
        with urllib.request.urlopen(url, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        # Upstream imports it as lib.agent.swe_constants; in a task both files sit in tests/.
        dest.write_text(text.replace("from lib.agent.swe_constants import",
                                     "from swe_constants import"), encoding="utf-8")
        print(f"[parsers] fetched {dest}")
    return PARSER_DIR


# ------------------------------------------------------------- task generation
def _fill(path: Path, **kw: str) -> None:
    """Substitute {placeholders} in one pass -- an issue body can itself contain text that
    looks like a placeholder, and a second pass would substitute into it."""
    pattern = re.compile(r"\{(" + "|".join(map(re.escape, kw)) + r")\}")
    path.write_text(pattern.sub(lambda m: str(kw[m.group(1)]), path.read_text()))


def _sanitize(instance_id: str) -> str:
    s = "".join(c for c in str(instance_id).strip() if c.isalnum() or c in "._-")
    return s or "instance"


def _patched_files(diff: str) -> list[str]:
    """Repo-relative paths the test patch touches; the verifier restores them first."""
    paths = set()
    for match in _DIFF_HEADER_RE.finditer(diff):
        paths.update((match.group("old"), match.group("new")))
    return sorted(paths)


def generate(problem: dict[str, Any], output: Path | str = "tasks") -> Path:
    """Render one SWE-rebench-V2 instance into a fresh subdir of ``output``.

    The task-dir name (== Miles' ``instance_id``) is the dataset's own key. Everything the
    verifier needs travels with the task: tests/spec.json, tests/test_patch.diff and the
    vendored log parsers; the gold patch goes to solution/ as the oracle.
    """
    missing = [k for k in REQUIRED if not problem.get(k)]
    if missing:
        raise ValueError(f"{problem.get('instance_id') or problem['problem_id']}: "
                         f"row is missing {', '.join(missing)}")
    output = Path(output)
    task = output / _sanitize(problem["instance_id"])
    if task.exists():
        raise FileExistsError(f"Use a fresh output directory: {task}")
    shutil.copytree(PKG / "task-template", task, dirs_exist_ok=False)

    # The instance image clones the repo to /<repo-name> and WORKDIRs there (SWE-rebench-V2's
    # combine.Dockerfile.j2); the agent and the verifier both work in that checkout.
    workdir = "/" + problem["repo"].split("/")[1]
    fills = {
        "name": f"mho/swe-rebench-v2-{_sanitize(problem['instance_id'])}",
        "instance_id": problem["instance_id"],
        "docker_image": problem["image_name"],
        "repo": problem["repo"],
        "base_commit": problem["base_commit"],
        "language": problem["language"] or "unknown",
        "license": json.dumps(problem["license"], ensure_ascii=False),
        "log_parser": problem["log_parser"],
        "workdir": workdir,
        "test_cmd": problem["test_cmd"],
        "problem_statement": problem["problem_statement"],
    }
    _fill(task / "task.toml", **fills)
    _fill(task / "instruction.md", **fills)
    _fill(task / "environment" / "Dockerfile", **fills)
    _fill(task / "solution" / "solve.sh", **fills)

    # git apply rejects a diff whose last line has no newline.
    for dest, diff in (("solution/patch.diff", problem["patch"]),
                       ("tests/test_patch.diff", problem["test_patch"])):
        (task / dest).write_text(diff if diff.endswith("\n") else diff + "\n", encoding="utf-8")
    (task / "tests" / "spec.json").write_text(json.dumps({
        "instance_id": problem["instance_id"],
        "repo": problem["repo"],
        "base_commit": problem["base_commit"],
        "workdir": workdir,
        "test_cmd": problem["test_cmd"],
        "install": problem["install"],
        "log_parser": problem["log_parser"],
        "test_files": _patched_files(problem["test_patch"]),
        "fail_to_pass": problem["fail_to_pass"],
        "pass_to_pass": problem["pass_to_pass"],
    }, indent=2) + "\n", encoding="utf-8")
    for name in PARSER_SOURCES:
        shutil.copy2(PARSER_DIR / name, task / "tests" / name)
    return task


def generate_all(
    problems: list[dict[str, Any]],
    output: Path | str,
    skip_unsupported: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fetch_parsers()
    exclusions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "tasks"
        staged.mkdir()
        for row in problems:
            try:
                generate(row, staged)
            except (NotImplementedError, ValueError) as exc:
                if not skip_unsupported:
                    raise
                exclusions.append({"problem_id": row["problem_id"],
                                   "instance_id": row.get("instance_id"), "reason": str(exc)})
        if exclusions:
            (staged / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        staged.rename(output)
    _emit_seed(output)
    return exclusions, sum(1 for p in output.iterdir() if p.is_dir())


def _emit_seed(output: Path) -> Path:
    """Write ``<output>/seed.jsonl``: one row per generated task dir with
    ``{prompt, metadata.instance_id}``. `instance_id` == the task-dir name, so Miles'
    harbor connector can resolve each prompt to its task dir via
    HARBOR_TASKS_DIR/<instance_id>."""
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
    parser = argparse.ArgumentParser(description="Generate Harbor SWE-rebench-V2 tasks.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"Problems JSONL (default: {DEFAULT_DATASET}); built on first use.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to tasks/.")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help=f"Rows to pull from HuggingFace when building the JSONL. Up to "
                             f"{ROWS_SERVER_MAX} are paged from the datasets-server; a larger "
                             f"number, or 0 for every row, downloads the 409 MB parquet.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Generate only the first N selected problems.")
    parser.add_argument("--question-id", action="append",
                        help="Generate only this instance_id (e.g. elastic__synthetics-316) or "
                             "row problem_id (repeatable).")
    parser.add_argument("--language", action="append",
                        help="Keep only rows in this language (repeatable). The dataset spans "
                             "20 languages; the task image is per-instance either way.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Draw N problems at random instead of taking the first N. "
                             "Requires --seed so the draw is reproducible.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for --sample. Two subsets drawn with different seeds are "
                             "independent; --limit would return the same leading rows.")
    parser.add_argument("--skip-unsupported", action="store_true",
                        help="Record unusable rows in exclusions.json instead of aborting.")
    parser.add_argument("--refresh-parsers", action="store_true",
                        help="Re-download the vendored SWE-rebench log parsers before generating.")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("tasks")
    if args.refresh_parsers:
        fetch_parsers(force=True)
    if not args.dataset.is_file():
        build_dataset(args.dataset, max_tasks=args.max_tasks)
    problems = load_problems(args.dataset, build=False)

    if args.question_id:
        wanted = set(args.question_id)
        problems = [p for p in problems
                    if wanted & {p["instance_id"], str(p["problem_id"])}]
        found = {p["instance_id"] for p in problems} | {str(p["problem_id"]) for p in problems}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"Question ID(s) absent from {args.dataset}: {missing}")
    if args.language:
        keep = {lang.lower() for lang in args.language}
        problems = [p for p in problems if p["language"].lower() in keep]
        if not problems:
            raise SystemExit(f"No {sorted(keep)} rows in {args.dataset}.")

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

    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}. Use a fresh --output-dir.")
    exclusions, count = generate_all(problems, output=output_dir,
                                     skip_unsupported=args.skip_unsupported)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "split": SPLIT,
                "seed": args.seed,
                "sample": args.sample,
                "limit": args.limit,
                "languages": args.language,
                "count": count,
                "problem_ids": [int(p["problem_id"]) for p in problems],
                "instance_ids": [p["instance_id"] for p in problems],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated {count} tasks in {output_dir}; excluded {len(exclusions)}")


if __name__ == "__main__":
    main()
