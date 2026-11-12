"""agent._derive_paths: the parent harness a proposal starts from."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mho.optimizer import agent  # noqa: E402


def test_no_frontier_falls_back_to_base_not_cwd(tmp_path, monkeypatch):
    run, base, cwd = tmp_path / "run", tmp_path / "base", tmp_path / "repo"
    (run / "candidates").mkdir(parents=True)
    base.mkdir()
    cwd.mkdir()
    (cwd / "pyproject.toml").write_text("")      # cwd looks like a harness (the repo root)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("MHO_RUN_DIR", str(run))
    monkeypatch.setenv("MICRO_SCAFFOLD_BASE", str(base))
    paths = agent._derive_paths(run / "candidates" / "candidate_1_1")
    assert paths["parent"] == base
