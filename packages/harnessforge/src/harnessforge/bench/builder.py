"""Materialize harness variants from the annotated gold source.

The gold harness (`bench/gold_harness/`) is a micro-swe-agent fork in which every feature's code
sits in a delimited block (whole lines, any indentation, `#` comments in .py/.yaml/.toml/.md):

    # >>> feature:loop_warning            kept iff loop_warning is on
    ...code...
    # --- else                            optional: what replaces the block when it is off
    ...code...
    # <<< feature:loop_warning

A block may name several features (`feature:a,b`: kept iff any of them is on) or `*` (kept iff
any feature is on: the shared flag plumbing). Blocks nest. A file whose first line is
`# @feature-file: <names>` is dropped entirely unless one of the names is on.

`build(on, out)` writes a clean tree (no markers) with exactly the features in `on`:
gold = all, start = none (the feature code is ABSENT, not flagged off, so the optimizer cannot
find it), ablate_<f> = all but f.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from harnessforge.bench.features import FEATURE_NAMES

GOLD_SOURCE = Path(__file__).parent / "gold_harness"
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".cfg", ""}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", ".git"}

_OPEN = re.compile(r"^\s*# >>> feature:(?P<names>[\w*,]+)\s*$")
_ELSE = re.compile(r"^\s*# --- else\s*$")
_CLOSE = re.compile(r"^\s*# <<< feature:(?P<names>[\w*,]+)\s*$")
_FILE = re.compile(r"^# @feature-file:\s*(?P<names>[\w*,]+)\s*$")


class MarkerError(ValueError):
    pass


def _names(spec: str) -> frozenset[str]:
    names = frozenset(n.strip() for n in spec.split(",") if n.strip())
    unknown = names - set(FEATURE_NAMES) - {"*"}
    if unknown:
        raise MarkerError(f"unknown feature name(s) in marker: {sorted(unknown)}")
    return names


def _kept(names: frozenset[str], on: frozenset[str]) -> bool:
    return bool(on) if "*" in names else bool(names & on)


@dataclass
class Block:
    """One feature block of one file: its names and the raw lines of each branch."""
    path: str
    names: frozenset[str]
    on_lines: list[str] = field(default_factory=list)
    off_lines: list[str] = field(default_factory=list)


def strip(text: str, on: Iterable[str], *, path: str = "<text>") -> str:
    """Resolve every feature block of `text` for the feature set `on`."""
    on = frozenset(on)
    out: list[str] = []
    # stack of (names, branch: "on"|"off", emitting)
    stack: list[tuple[frozenset[str], str, bool]] = []
    for lineno, line in enumerate(text.splitlines(keepends=True), 1):
        body = line.rstrip("\n")
        if m := _OPEN.match(body):
            names = _names(m["names"])
            parent = stack[-1][2] if stack else True
            stack.append((names, "on", parent and _kept(names, on)))
            continue
        if _ELSE.match(body):
            if not stack or stack[-1][1] != "on":
                raise MarkerError(f"{path}:{lineno}: stray '# --- else'")
            names, _, _ = stack.pop()
            parent = stack[-1][2] if stack else True
            stack.append((names, "off", parent and not _kept(names, on)))
            continue
        if m := _CLOSE.match(body):
            if not stack or stack[-1][0] != _names(m["names"]):
                raise MarkerError(f"{path}:{lineno}: unbalanced '{body.strip()}'")
            stack.pop()
            continue
        if not stack or stack[-1][2]:
            out.append(line)
    if stack:
        raise MarkerError(f"{path}: unclosed feature block {sorted(stack[-1][0])}")
    return "".join(out)


def blocks(text: str, *, path: str = "<text>") -> list[Block]:
    """Every (possibly nested) block of `text`, with its own lines (markers excluded)."""
    found: list[Block] = []
    stack: list[tuple[Block, str]] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        if m := _OPEN.match(body):
            b = Block(path, _names(m["names"]))
            found.append(b)
            stack.append((b, "on"))
            continue
        if _ELSE.match(body):
            stack[-1] = (stack[-1][0], "off")
            continue
        if _CLOSE.match(body):
            stack.pop()
            continue
        for b, branch in stack:
            (b.on_lines if branch == "on" else b.off_lines).append(line)
    return found


def file_names(text: str) -> frozenset[str] | None:
    """Feature names of a whole-file marker, or None."""
    first = text.split("\n", 1)[0]
    m = _FILE.match(first)
    return _names(m["names"]) if m else None


def _iter_source(src: Path):
    for p in sorted(src.rglob("*")):
        if p.is_file() and not (set(p.relative_to(src).parts) & SKIP_DIRS) and p.suffix not in {".pyc"}:
            yield p


def build(on: Iterable[str], out: str | Path, *, src: str | Path = GOLD_SOURCE,
          overwrite: bool = False) -> Path:
    """Write the harness with features `on` to `out` (a fresh directory)."""
    on = frozenset(on)
    bad = on - set(FEATURE_NAMES)
    if bad:
        raise ValueError(f"unknown features: {sorted(bad)}")
    src, out = Path(src), Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    for p in _iter_source(src):
        rel = p.relative_to(src)
        dest = out / rel
        if p.suffix not in TEXT_SUFFIXES and p.name != "LICENSE":
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            continue
        text = p.read_text()
        whole = file_names(text)
        if whole is not None:
            if not _kept(whole, on):
                continue
            text = text.split("\n", 1)[1] if "\n" in text else ""
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(strip(text, on, path=str(rel)))
    return out


def variants() -> dict[str, frozenset[str]]:
    """{variant name: features on}: gold, start and one ablation per feature."""
    allf = frozenset(FEATURE_NAMES)
    out = {"gold": allf, "start": frozenset()}
    out.update({f"ablate_{f}": allf - {f} for f in FEATURE_NAMES})
    return out


def build_all(root: str | Path, *, src: str | Path = GOLD_SOURCE, overwrite: bool = False,
              names: Iterable[str] | None = None) -> dict[str, Path]:
    """Materialize gold/, start/ and every ablate_<feature>/ under `root`."""
    root = Path(root)
    wanted = variants()
    if names is not None:
        wanted = {k: v for k, v in wanted.items() if k in set(names)}
    return {name: build(on, root / name, src=src, overwrite=overwrite) for name, on in wanted.items()}


def source_blocks(src: str | Path = GOLD_SOURCE) -> list[Block]:
    """All feature blocks of the gold source (whole-file markers become one block)."""
    src = Path(src)
    found: list[Block] = []
    for p in _iter_source(src):
        if p.suffix not in TEXT_SUFFIXES:
            continue
        text = p.read_text()
        rel = str(p.relative_to(src))
        whole = file_names(text)
        if whole is not None:
            found.append(Block(rel, whole, on_lines=text.splitlines(keepends=True)[1:]))
        found.extend(blocks(text, path=rel))
    return found
