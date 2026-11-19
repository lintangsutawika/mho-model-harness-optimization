import time
from pathlib import Path

import pytest

from harnessforge import JobSpec, JobState, Launcher
from harnessforge.launch import LocalLauncher, PBSLauncher, SlurmLauncher, make_launcher
from harnessforge.launch.base import Result, child_job_env
from harnessforge.launch.pbs import parse_qstat_f, pbs_select


class FakeRunner:
    def __init__(self, *responses):
        self.calls = []
        self.responses = list(responses)

    def __call__(self, argv, env):
        self.calls.append((list(argv), dict(env) if env is not None else None))
        r = self.responses.pop(0) if self.responses else Result(0, "")
        return r


ENV = {"PATH": "/usr/bin", "HOME": "/home/u", "PBS_JOBID": "1.pbs", "RTYPE": "rt_HC",
       "CUDA_VISIBLE_DEVICES": "GPU-x", "SLURM_JOB_ID": "9", "TMPDIR": "/local/1", "KEEP": "1"}


def test_child_job_env_strips_submitter_job_vars():
    env, dropped = child_job_env(ENV, {"A": 1}, "rt_HG")
    assert env == {"PATH": "/usr/bin", "HOME": "/home/u", "KEEP": "1", "A": "1",
                   "RTYPE": "rt_HG"}
    assert "PBS_JOBID" in dropped and "CUDA_VISIBLE_DEVICES" in dropped


def test_protocols():
    assert isinstance(PBSLauncher(), Launcher)
    assert isinstance(SlurmLauncher(), Launcher)
    assert isinstance(LocalLauncher(), Launcher)


def test_pbs_render_abci_resource_type():
    run = FakeRunner(Result(0, "123.pbs1\n"))
    L = PBSLauncher(runner=run, environ=ENV,
                    defaults={"queue": "R99", "account": "gai", "resource_type": "rt_HG"})
    spec = JobSpec(name="hfpool-x-0", command=["python", "-m", "x", "--a", "b c"], gpus=1,
                   walltime="2:00:00", env={"FOO": "bar"}, log=Path("/tmp/hf-test/x.log"))
    assert L.submit(spec) == "123.pbs1"
    argv, env = run.calls[0]
    assert argv == ["qsub", "-V", "-v", "RTYPE=rt_HG", "-N", "hfpool-x-0", "-q", "R99", "-P",
                    "gai", "-l", "select=1", "-l", "walltime=2:00:00", "-o", "/tmp/hf-test/x.log",
                    "-j", "oe", "--", "python", "-m", "x", "--a", "b c"]
    assert env["FOO"] == "bar" and env["RTYPE"] == "rt_HG"
    assert "PBS_JOBID" not in env and "CUDA_VISIBLE_DEVICES" not in env


def test_pbs_select_variants():
    s = JobSpec(name="j", command=["x"], nodes=2, gpus=8, cpus=192)
    assert pbs_select(s) == "select=2:ngpus=8:ncpus=192"
    assert pbs_select(s, mem="100gb") == "select=2:ngpus=8:ncpus=192:mem=100gb"
    assert pbs_select(s, select="1:ncpus=192:ngpus=8") == "select=1:ncpus=192:ngpus=8"
    assert pbs_select(s, resource_type="rt_HF") == "select=2"
    assert pbs_select(s, resource_type="rt_HG", host="hnode001") == "select=2:host=hnode001"
    assert (pbs_select(s, select="select=1:ngpus=8", host="h1")
            == "select=1:host=h1:ngpus=8")


def test_pbs_options_override_defaults_and_cwd():
    run = FakeRunner(Result(0, "5.pbs1"))
    L = PBSLauncher(runner=run, environ={}, defaults={"queue": "Q1", "account": "A"})
    L.submit(JobSpec(name="j", command=["run.sh"], walltime="",
                     options={"queue": "Q2", "host": "hnode7", "extra_args": ["-W", "x"],
                              "cwd": "/w d"}))
    argv, _ = run.calls[0]
    assert argv[argv.index("-q") + 1] == "Q2"
    assert "select=1:host=hnode7" in argv
    assert argv[-6:] == ["-W", "x", "--", "/bin/bash", "-c", "cd '/w d' && exec run.sh"]
    assert not any(a.startswith("walltime") for a in argv)


def test_pbs_submit_retries_transient_then_fails_hard():
    sleeps = []
    run = FakeRunner(Result(1, "", "qsub: PTL is under maintenance"), Result(0, "7.pbs1"))
    L = PBSLauncher(runner=run, environ={}, sleep=sleeps.append, retry_interval=1)
    assert L.submit(JobSpec(name="j", command=["x"])) == "7.pbs1"
    assert sleeps == [1]
    run = FakeRunner(Result(1, "", "qsub: illegal -l value"))
    with pytest.raises(RuntimeError, match="illegal"):
        PBSLauncher(runner=run, environ={}).submit(JobSpec(name="j", command=["x"]))


QSTAT = """Job Id: 2989395.pbs1
    Job_Name = hfpool-probe
    job_state = F
    exec_host = hnode008/0*16+hnode009/0*16
    Exit_status = 0
    Variable_List = PBS_O_HOME=/home/u,PBS_O_LANG=C,
\tPBS_O_PATH=/usr/bin
"""


def test_pbs_status_parsing_and_cancel():
    st = parse_qstat_f(QSTAT)
    assert st.state == JobState.DONE and st.exit_code == 0
    assert st.hosts == ("hnode008", "hnode009")
    assert parse_qstat_f("    job_state = R\n    exec_host = hnode001/1*16\n").state \
        == JobState.RUNNING
    assert parse_qstat_f("    job_state = Q\n").state == JobState.PENDING
    run = FakeRunner(Result(35, "", "qstat: Unknown Job Id"))
    assert PBSLauncher(runner=run).status("1.pbs1").state == JobState.DONE
    run = FakeRunner(Result(1, "", "qdel: Unknown Job Id 1.pbs1"))
    PBSLauncher(runner=run).cancel("1.pbs1")          # already gone: fine
    run = FakeRunner(Result(1, "", "qdel: Unauthorized Request"))
    with pytest.raises(RuntimeError):
        PBSLauncher(runner=run).cancel("1.pbs1")


def test_slurm_render_and_status():
    run = FakeRunner(Result(0, "4242;cluster\n"))
    L = SlurmLauncher(runner=run, environ=ENV, defaults={"queue": "gpu", "account": "acct"})
    jid = L.submit(JobSpec(name="j", command=["python", "a b.py"], nodes=1, gpus=8, cpus=32,
                           walltime="1:00:00", log=Path("/tmp/hf-test/j.out"),
                           options={"host": "n1", "resource_type": "rt_HG"}))
    assert jid == "4242"
    argv, env = run.calls[0]
    assert argv == ["sbatch", "--parsable", "--export=ALL", "-J", "j", "-p", "gpu", "-A", "acct",
                    "--nodes=1", "--gres=gpu:8", "--cpus-per-task=32", "-t", "1:00:00", "-w",
                    "n1", "-o", "/tmp/hf-test/j.out", "--wrap=python 'a b.py'"]
    assert "SLURM_JOB_ID" not in env and "RTYPE" not in env
    run = FakeRunner(Result(0, "RUNNING|n[1-2]\n"), Result(0, "n1\nn2\n"))
    st = SlurmLauncher(runner=run).status("4242")
    assert st.state == JobState.RUNNING and st.hosts == ("n1", "n2")
    run = FakeRunner(Result(0, ""), Result(0, "COMPLETED|3:0|n1\n"), Result(0, "n1\n"))
    st = SlurmLauncher(runner=run).status("4242")
    assert st.state == JobState.DONE and st.exit_code == 3 and st.hosts == ("n1",)


def test_local_launcher_runs_cancels_and_reports(tmp_path):
    L = LocalLauncher(tmp_path / "jobs")
    log = tmp_path / "out.log"
    jid = L.submit(JobSpec(name="j", command=["sh", "-c", "echo hi $FOO; exit 3"],
                           env={"FOO": "bar"}, log=log))
    for _ in range(100):
        if L.status(jid).state == JobState.DONE:
            break
        time.sleep(0.05)
    st = L.status(jid)
    assert st.state == JobState.DONE and st.exit_code == 3
    assert "hi bar" in log.read_text()
    # another process (fresh launcher) still sees the exit code
    assert LocalLauncher(tmp_path / "jobs").status(jid).exit_code == 3
    jid = L.submit(JobSpec(name="k", command=["sleep", "30"]))
    assert L.status(jid).state == JobState.RUNNING
    L.cancel(jid)
    assert L.status(jid).state == JobState.DONE


def test_make_launcher():
    assert isinstance(make_launcher("pbs", {"queue": "q"}), PBSLauncher)
    assert make_launcher("pbs", {"queue": "q"}).defaults == {"queue": "q"}
    assert isinstance(make_launcher("slurm"), SlurmLauncher)
    with pytest.raises(ValueError):
        make_launcher("lsf")
