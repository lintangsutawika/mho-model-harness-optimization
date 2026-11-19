"""Build a PYTHONPATH overlay so host venvs only FILL IN what the trainer container lacks.

Usage (run with the CONTAINER's python, PYTHONPATH unset, -I):
    python -I container_pyoverlay.py <overlay_dir> <site-packages> [<site-packages> ...]

Putting a host venv's site-packages on PYTHONPATH makes it shadow the container's own packages
(PYTHONPATH comes before site-packages), so e.g. a newer tokenizers/numpy in the host venv breaks
the container's transformers / Megatron stack. Instead, <overlay_dir> gets a symlink for each
top-level module of the given site-packages dirs that the container cannot already import (first
dir wins on duplicates), plus the matching *.dist-info so importlib.metadata still works.
Namespace packages are not linked wholesale: the overlay gets a real dir for them (no
__init__.py, so it still merges with the container's portion) and the same rule applies one
level down, so the host never shadows a submodule the container already has (e.g. a host
opentelemetry-api upgrade shadowing the container's opentelemetry/util/types.py broke Ray).
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import re
import shutil
import sys
from pathlib import Path

_SUFFIX = re.compile(r"(\.cpython-[^.]+)?(\.abi3)?\.(py|so|pyd)$")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _top(entry: Path) -> str | None:
    n = entry.name
    if n in ("__pycache__", "_virtualenv.py") or n.endswith((".pth", ".dist-info", ".egg-info", ".data")):
        return None
    if entry.is_dir():
        return n if n.isidentifier() else None
    m = _SUFFIX.sub("", n)
    return m if m != n and m.isidentifier() else None


def _container_state(mod: str) -> str:
    """'has' (regular module/package), 'namespace' (a namespace portion only) or 'missing'."""
    try:
        spec = importlib.util.find_spec(mod)
    except (ImportError, ValueError, AttributeError):
        return "missing"
    if spec is None:
        return "missing"
    return "namespace" if spec.origin in (None, "namespace") else "has"


def _container_has(mod: str) -> bool:
    return _container_state(mod) == "has"


def _link_tree(entry: Path, dest: Path, dotted: str, linked_mods: list[str], skipped: list[str]) -> None:
    """Link `entry` (module `dotted`) into `dest` unless the container has it; recurse into
    namespace packages so only the submodules the container lacks are linked."""
    state = _container_state(dotted)
    if state == "has":
        skipped.append(dotted)
        return
    if state == "missing" or not entry.is_dir() or (entry / "__init__.py").exists():
        dest.symlink_to(entry.resolve())
        linked_mods.append(dotted)
        return
    dest.mkdir(exist_ok=True)  # container namespace portion + host dir: merge child by child
    for child in sorted(entry.iterdir()):
        top = _top(child)
        if top is None or (dest / child.name).exists():
            continue
        _link_tree(child, dest / child.name, f"{dotted}.{top}", linked_mods, skipped)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    have_dists = {_norm(d.metadata["Name"] or "") for d in md.distributions()}
    linked: set[str] = set()
    skipped: list[str] = []
    for sp in map(Path, argv[1:]):
        if not sp.is_dir():
            print(f"[pyoverlay] skip missing {sp}", file=sys.stderr)
            continue
        for entry in sorted(sp.iterdir()):
            if entry.name.endswith(".dist-info"):
                proj = _norm(entry.name[: -len(".dist-info")].rsplit("-", 1)[0])
                if proj not in have_dists and entry.name not in linked:
                    (out / entry.name).symlink_to(entry.resolve())
                    linked.add(entry.name)
                continue
            top = _top(entry)
            if top is None or entry.name in linked:
                continue
            linked_mods: list[str] = []
            _link_tree(entry, out / entry.name, top, linked_mods, skipped)
            if linked_mods:
                linked.add(entry.name)
    mods = sum(1 for n in linked if not n.endswith(".dist-info"))
    print(f"[pyoverlay] {out}: {mods} host modules linked, "
          f"{len(set(skipped))} already in the container (container wins)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
