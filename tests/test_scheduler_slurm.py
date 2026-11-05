"""SLURM side of Scheduler with stub sbatch/squeue/sacct on PATH (no cluster needed)."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mho import jobs as ol  # noqa: E402


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "sbatch", f'echo "$@" > {tmp_path}/cmd; env > {tmp_path}/env; echo "4242;cluster"')
    _stub(bin_dir, "squeue", f'cat {tmp_path}/squeue 2>/dev/null; exit 0')
    _stub(bin_dir, "sacct", f'cat {tmp_path}/sacct 2>/dev/null; exit 0')
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return tmp_path


def test_submit_flags_log_file_and_env(slurm, monkeypatch):
    monkeypatch.setenv("SBATCH_PARTITION", "cpu")     # how the driver itself was launched
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    log = slurm / "run" / "logs" / "c-propose.out"
    jid = ol.Scheduler("slurm").submit(Path("/bin/true"), {"MHO_TAG": "c"}, name="mho-propose-c",
                                       cpus=8, walltime="02:00:00", queue="gpu", out_dir=log)
    assert jid == "4242"
    cmd = (slurm / "cmd").read_text().split()
    assert cmd[:3] == ["--parsable", "--export=ALL", "-J"] and "mho-propose-c" in cmd
    assert cmd[cmd.index("-p") + 1] == "gpu" and cmd[cmd.index("-o") + 1] == str(log)
    assert "--cpus-per-task=8" in cmd and log.parent.is_dir()
    env = (slurm / "env").read_text()
    assert "MHO_TAG=c" in env
    assert "SBATCH_PARTITION=" not in env and "SLURM_JOB_ID=" not in env


@pytest.mark.parametrize("row,expected", [
    ("COMPLETED|0:0", ("DONE", 0)),
    ("FAILED|2:0", ("DONE", 2)),
    ("TIMEOUT|0:15", ("DONE", 143)),
    ("CANCELLED by 1234|0:15", ("DONE", 143)),
    ("OUT_OF_MEMORY|0:125", ("DONE", 253)),
    ("NODE_FAIL|0:0", ("DONE", 1)),
    ("RUNNING|0:0", ("RUNNING", None)),
])
def test_final_state_from_sacct(slurm, row, expected):
    (slurm / "sacct").write_text(row + "\n")
    assert ol.Scheduler("slurm").state("4242") == expected


def test_live_job_comes_from_squeue(slurm):
    (slurm / "squeue").write_text("RUNNING\n")
    assert ol.Scheduler("slurm").state("4242") == ("RUNNING", None)


def test_missing_from_sacct_waits_then_gives_up(slurm, monkeypatch):
    monkeypatch.setattr(ol, "SACCT_MAX_MISSES", 2)
    s = ol.Scheduler("slurm")
    assert [s.state("4242") for _ in range(3)] == [
        ("PENDING", None), ("PENDING", None), ("DONE", None)]
