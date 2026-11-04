"""optimize_loop.main(): every --mode with --val-data runs mho.loop.Loop with that mode's schedule."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import optimize_loop as ol  # noqa: E402


@pytest.mark.parametrize("mode,flags,schedule", [
    ("harness-only", [], [("harness", 1)]),
    ("model-only", [], [("model", 1)]),
    ("model-harness", ["--harness-proposals", "4", "--model-proposals", "1"],
     [("harness", 4), ("model", 1)]),
])
def test_mode_sets_the_schedule(tmp_path, monkeypatch, mode, flags, schedule):
    seen = {}

    class FakeLoop:
        def __init__(self, args, run_dir, sched, *, kinds, schedule, **kw):
            seen.update(run_dir=run_dir, kinds=kinds, schedule=schedule)

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(ol, "Loop", FakeLoop)
    monkeypatch.setattr(ol, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(ol, "_tee_driver_log", lambda path: None)
    train = tmp_path / "train"
    (train / "p0").mkdir(parents=True)
    ol.main(["--mode", mode, "--model", "Qwen/Qwen3.5-4B", "--train-data", str(train),
             "--val-data", str(tmp_path / "val"), "--run-name", "r", *flags])
    assert seen["ran"] and seen["schedule"] == schedule
    assert set(seen["kinds"]) == {"harness", "model"} and seen["run_dir"] == tmp_path / "runs" / "r"


def test_model_is_required_with_val_data(tmp_path, monkeypatch):
    monkeypatch.delenv("MHO_MODEL", raising=False)
    train = tmp_path / "train"
    (train / "p0").mkdir(parents=True)
    with pytest.raises(SystemExit):
        ol.main(["--mode", "model-harness", "--model", "", "--train-data", str(train),
                 "--val-data", str(tmp_path / "val")])


def test_val_data_is_required(tmp_path):
    train = tmp_path / "train"
    (train / "p0").mkdir(parents=True)
    with pytest.raises(SystemExit):
        ol.main(["--mode", "harness-only", "--model", "m", "--train-data", str(train),
                 "--val-data", ""])
