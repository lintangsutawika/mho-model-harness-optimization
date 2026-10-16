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
from pydantic import BaseModel, Field, ValidationError

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class ModelSpec(BaseModel):
    name: str
    hypothesis: str
    evidence: list[str] = Field(default_factory=list)
    spec: str
    counterexample: str
    realization: str = Field(default="synthesize")


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
    parse_spec(path)  # raises on any violation


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