"""Convert BytedTsinghua-SIA/DAPO-Math-17k into Harbor local-task directories.

Each row becomes one Harbor task directory usable as a *local task* (``task:
  path: <dir>``) by the HarborGenerator / harbor trial config and by the
``HarborTaskDataset``:

    <out>/<index>/
        instruction.md      # the DAPO problem prompt (single user turn)
        task.toml           # harbor TaskConfig: environment, agent, metadata
E.g. ``<out>/0000000000/instruction.md``.

The problem statement already carries DAPO's answer protocol ("The last line of
your response should be of the form Answer: $Answer"), which the math verifier
(``harbor.math_verifier:MathVerifier``) parses. The ground-truth answer is
stored in ``[metadata] answer`` for the verifier.

Args mirror :mod:`prepare_harbor_dataset` so the two converters stay parallel.
Run (subset for a smoke test):
    python -m harbor.prepare_math_tasks \
        --max-tasks 16 --out ~/data/harbor/DAPO-Math-17k --split train
Which 16 rows are picked is deterministic via --seed for reproducible subsets.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

_ROLE_CONTENT = re.compile(r"netloc|message|role|content", re.I)


def _require_parquet_deps():
    """Lazily import the heavy deps so --help / import stay cheap."""
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Missing 'datasets'. Install it in the training env (e.g. 'uv pip install datasets')."
        ) from e


def _prompt_text(messages: list) -> str:
    """Flatten a DAPO ``prompt`` (list of {content, role}) to plain text."""
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
        parts.append(str(c))
    text = "\n\n".join(p for p in parts if p.strip())
    return text


def prepare(
    out_dir: str | Path,
    max_tasks: int | None = None,
    split: str = "train",
    seed: int = 0,
    dataset_name: str = "BytedTsinghua-SIA/DAPO-Math-17k",
    overwrite: bool = False,
) -> Path:
    """Write DAPO problems as Harbor local-task dirs. Returns the out dir."""
    _require_parquet_deps()
    from datasets import load_dataset

    out = Path(out_dir).expanduser()  # NOTE: no .resolve() -- must not follow a symlink
    if out.is_symlink():
        raise FileExistsError(
            f"{out} is a symlink; refusing to write through it. Remove it and retry "
            "(e.g. 'rm {out}')."
        )
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists; pass overwrite=True (or --overwrite) to replace it."
        )
    if overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, split=split)
    if max_tasks is not None:
        rng = __import__("random").Random(seed)
        ds = ds.shuffle(seed=seed).select(range(min(max_tasks, len(ds))))

    n = 0
    for i, row in enumerate(ds):
        prompt_text = _prompt_text(row.get("prompt") or [])
        if not prompt_text.strip():
            continue
        rm = row.get("reward_model") or {}
        answer = rm.get("ground_truth")
        task_dir = out / f"{i:010d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "instruction.md").write_text(prompt_text + "\n")
        (task_dir / "task.toml").write_text(_task_toml(answer))
        _write_stub_tests(task_dir)
        n += 1
        if max_tasks is not None and n >= max_tasks:
            break
    print(f"Wrote {n} DAPO task dirs to {out}")
    return out


def _write_stub_tests(task_dir) -> None:
    """Write a stub tests/test.sh so the Task constructor passes validation.

    The math verdict comes from the custom verifier (harness.math_verifier),
    which reads the agent trajectory directly and never executes this script;
    harbor only requires a tests file to exist for a shared-mode verifier.
    """
    tests = task_dir / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test.sh").write_text(
        "#!/bin/sh\n# DAPO math grading is done by harness.math_verifier:MathVerifier;\n"
        "# this stub only satisfies harbor's task test-file requirement.\nexit 0\n"
    )


def _task_toml(answer: str | None) -> str:
    """Harbor TaskConfig TOML for a DAPO math task.

    - [environment]  plain python image (agent computes with python).
    - [agent]        override timeout for a short single-turn math problem.
    - [metadata]     ground-truth answer for the verifier.
    The agent (micro-swe-agent via AgentHarness) is selected by the *trial*
    config (harbor_trial_config/default.yaml), not the task, so task.toml
    leaves [agent] minimal per-task overrides.
    """
    answer_toml = f'answer = "{answer}"' if answer is not None else "# no answer"
    return f"""\
schema_version = "1.4"

[environment]
docker_image = "python:3.11-slim"
# The agent needs network during setup (apt/uv install of the harness) and to
# call the served vLLM at api_base. Math *solving* needs none, but the runtime
# does; a no-network policy would break setup and inference.
network_mode = "public"

[agent]
# Single-turn-ish reasoning problem; keep per-task agent timeout tight.
override_timeout_sec = 1800.0

[verifier]
environment_mode = "shared"

[metadata]
task_family = "DAPO-Math-17k"
ability = "MATH"
{answer_toml}
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="~/data/harbor/DAPO-Math-17k", help="output task-dir root")
    p.add_argument("--max-tasks", type=int, default=None, help="cap rows (subset); None=all")
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="BytedTsinghua-SIA/DAPO-Math-17k")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    prepare(
        out_dir=args.out,
        max_tasks=args.max_tasks,
        split=args.split,
        seed=args.seed,
        dataset_name=args.dataset,
        overwrite=args.overwrite,
    )