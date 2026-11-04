"""The env a child job inherits from a driver that itself runs inside a PBS job."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.env import child_job_env  # noqa: E402

DRIVER_ENV = {  # an rt_HC driver launched via submit_pbs.sh
    "PATH": "/home/u/.local/bin:/usr/bin", "HOME": "/home/u", "DATA_DIR": "/data/",
    "MHO_EVAL_RTYPE": "rt_HF", "MHO_MODEL": "Qwen/Qwen3.5-9B", "HF_DIR": "/hf",
    "RTYPE": "rt_HC", "PBS_JOBID": "1.pbs1", "PBS_NODEFILE": "/x", "PBS_O_WORKDIR": "/r",
    "PBS_RTYPE": "rt_HC", "PBS_SELECT": "select=1:ncpus=8", "JOB_NAME": "mho-h-4b-math",
    "CUDA_VISIBLE_DEVICES": "", "TMPDIR": "/local/1.pbs1", "OMP_NUM_THREADS": "8",
    "TMUX": "/tmp/tmux", "HOSTNAME": "hnode008",
}


def test_drops_job_local_and_sets_own_rtype():
    env, dropped = child_job_env(DRIVER_ENV, {"MHO_OUT": "/o"}, resource_type="rt_HF")
    assert env["RTYPE"] == "rt_HF"
    for k in ("PBS_JOBID", "PBS_NODEFILE", "PBS_O_WORKDIR", "PBS_RTYPE", "PBS_SELECT",
              "JOB_NAME", "CUDA_VISIBLE_DEVICES", "TMPDIR", "OMP_NUM_THREADS", "TMUX",
              "HOSTNAME"):
        assert k not in env and k in dropped
    # config the child needs is kept; this job's own settings are added
    for k in ("PATH", "HOME", "DATA_DIR", "MHO_EVAL_RTYPE", "MHO_MODEL", "HF_DIR"):
        assert env[k] == DRIVER_ENV[k]
    assert env["MHO_OUT"] == "/o"


def test_no_resource_type_means_no_rtype():
    env, _ = child_job_env(DRIVER_ENV, {}, resource_type="")
    assert "RTYPE" not in env


def test_extra_wins_over_driver_env():
    env, _ = child_job_env(DRIVER_ENV, {"MHO_MODEL": "other"})
    assert env["MHO_MODEL"] == "other"
