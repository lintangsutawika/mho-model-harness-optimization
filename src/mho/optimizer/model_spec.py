"""Model-spec schema + validator.

A model-spec is the model-side proposal output (the paper's capability specification sigma):
a concise, testable statement of a behavior the model should learn, written so a downstream
realization map `f` can turn it into verifiable training data (select matching tasks from the
dataset, or synthesize new ones with an auto-verifier).

Format: each spec lives in ``model_specs/<name>.md`` with a YAML frontmatter block carrying the
structured fields; the body is free-form elaboration (optional). The frontmatter must schema:
  name:            short capability name (also the file name, sanitized)
  hypothesis:      one falsifiable sentence: what behavior change raises verified reward and why
  evidence:        list of the trajectories that support it (paths under eval_prior/trials/)
  spec:            the precise capability the model must learn (the sigma)
  counterexample:  an input the current model gets wrong (concrete)
  realization:     select | synthesize   (how f turns spec into training data; default synthesize)
  checks:          names of the check functions that score a rollout for this capability. They
                   are defined in model_specs/<name>.py beside the spec:
                       def <check>(traj: dict) -> bool | float   (float in [0, 1])
                   See mho.data.sigma_checklib for the trajectory helpers and
                   mho.data.reward_synthesis for how they become the training reward.
  rubric:          (required for realization: select) the criteria a train task must meet to
                   exercise this capability, each with keywords looked for in its instruction:
                       - criterion: reads and cleans a CSV
                         keywords: [csv, read_csv, missing values]
                         weight: 1        # optional, default 1
                   A task's score is the weighted share of criteria with a keyword hit; tasks
                   scoring >= the threshold (0.5) are kept. See mho.data.task_select.

Usage (agent-invoked, gives write-time feedback):
  python -m mho.optimizer.model_spec validate <spec.md>...
  python -m mho.optimizer.model_spec check <dir>          # validate every *.md in a dir
Exit 0 if all valid, 2 on any validation error (printed to stderr).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class RubricItem(BaseModel):
    criterion: str
    keywords: list[str] = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)

    @field_validator("keywords")
    @classmethod
    def _nonblank(cls, v: list[str]) -> list[str]:
        v = [str(k).strip() for k in v]
        if not all(v):
            raise ValueError("keywords must not be blank")
        return v


class ModelSpec(BaseModel):
    name: str
    hypothesis: str
    evidence: list[str] = Field(default_factory=list)
    spec: str
    counterexample: str
    realization: str = Field(default="synthesize")
    # Names of the check functions (in model_specs/<name>.py) that score each rollout's
    # trajectory for this capability. The training reward is 0.5 * correctness + 0.5 * their mean
    # (mho.data.reward_synthesis / sigma_rule).
    checks: list[str] = Field(min_length=1)
    # Which train tasks to keep (realization: select); scored by mho.data.task_select.
    rubric: list[RubricItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _realization(self) -> "ModelSpec":
        if self.realization not in ("select", "synthesize"):
            raise ValueError(f"realization must be select or synthesize, not {self.realization!r}")
        if self.realization == "select" and not self.rubric:
            raise ValueError("realization: select needs a `rubric` (criteria + keywords)")
        return self

    @field_validator("checks")
    @classmethod
    def _identifiers(cls, v: list[str]) -> list[str]:
        bad = [c for c in v if not str(c).isidentifier()]
        if bad:
            raise ValueError(f"checks must be function names (identifiers), not expressions: {bad}")
        if len(set(v)) != len(v):
            raise ValueError("checks must not repeat a name")
        return v


def parse_spec(path: Path) -> tuple[ModelSpec, str]:
    """Parse a model-spec .md into (ModelSpec, body). Raises ValueError on malformed input."""
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path}: missing YAML frontmatter (--- block) at the top")
    try:
        front = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        # Tip: multi-line/colon-containing values must be quoted or block scalars (|).
        raise ValueError(
            f"{path}: YAML frontmatter does not parse: {e}. "
            "Put multi-line / colon-containing values in block scalars (`key: |`) or quotes."
        ) from e
    try:
        spec = ModelSpec(**front)
    except ValidationError as e:
        raise ValueError(f"{path}: invalid frontmatter: {e}") from e
    # The `name` must match the file name (sanitized) so the loop can key on it.
    expect = "".join(c for c in spec.name if c.isalnum() or c in "._-") or "spec"
    if path.stem != expect:
        raise ValueError(
            f"{path}: `name: {spec.name!r}` does not match file stem {path.stem!r} "
            f"(expected {expect!r})"
        )
    body = text[m.end() :].strip()
    return spec, body


def validate_file(path: Path) -> None:
    """Schema + the checks module: model_specs/<name>.py must define every listed check.
    Raises ValueError on any violation. (Whether the checks run and vary is the screen's job:
    python -m mho.data.reward_synthesis screen --spec <spec.md>.)"""
    from mho.data import reward_synthesis
    spec, _ = parse_spec(path)
    reward_synthesis.produce_config(spec, reward_synthesis.checks_module(path))


def check_dir(d: Path) -> list[Path]:
    """List the spec files under a dir (no validation -- main validates per file and collects)."""
    return sorted(p for p in d.glob("*.md") if "eval_prior" not in str(p))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate one or more spec files")
    v.add_argument("files", nargs="+", type=Path)
    c = sub.add_parser("check", help="validate every *.md in a directory")
    c.add_argument("dir", type=Path)
    args = p.parse_args(argv)

    errors: list[str] = []
    targets: list[Path] = []
    if args.cmd == "validate":
        targets = args.files
    else:
        targets = check_dir(args.dir) or []
        if not targets:
            print(f"check: no *.md files under {args.dir}", file=sys.stderr)
            return 2
    for f in targets:
        try:
            validate_file(f)
            print(f"OK  {f}")
        except (ValueError, OSError, yaml.YAMLError, ValidationError) as e:
            errors.append(str(e))
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    if errors:
        return 2
    print(f"validated {len(targets)} spec(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())