import os
import platform
import signal
import subprocess
from typing import Any

from pydantic import BaseModel

# >>> feature:*
from micro_swe import features
# <<< feature:*
from micro_swe.exceptions import Submitted
from micro_swe.utils.serialize import recursive_merge


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)
        # >>> feature:persistent_cwd
        self._cwd: str | None = None
        # <<< feature:persistent_cwd

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        # >>> feature:persistent_cwd
        track_cwd = features.on("persistent_cwd")
        if track_cwd:
            cwd, command = self._cwd or cwd, _with_cwd_report(command)
        # <<< feature:persistent_cwd
        try:
            result = _run(command, cwd, os.environ | self.config.env, timeout or self.config.timeout)
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
            # >>> feature:persistent_cwd
            if track_cwd:
                output["output"], self._cwd = _split_cwd_report(output["output"], self._cwd)
            # <<< feature:persistent_cwd
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
            # >>> feature:persistent_cwd
            if track_cwd:
                output["output"], self._cwd = _split_cwd_report(output["output"], self._cwd)
            # <<< feature:persistent_cwd
            # >>> feature:timeout_guidance
            if features.on("timeout_guidance") and isinstance(e, subprocess.TimeoutExpired):
                output["output"] += TIMEOUT_NOTE.format(timeout=timeout or self.config.timeout)
            # <<< feature:timeout_guidance
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ, kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


# >>> feature:timeout_guidance
TIMEOUT_NOTE = (
    "\n[The command timed out after {timeout} seconds and was killed; the output above is what it printed "
    "before that. Commands that run for a long time or never exit (servers, watchers, long builds) must be "
    "started in the background with their output redirected, e.g. `nohup <command> > /tmp/job.log 2>&1 &`, "
    "then checked with `tail /tmp/job.log`. Avoid commands that wait for interactive input.]"
)
# <<< feature:timeout_guidance
# >>> feature:persistent_cwd
_CWD_MARK = "__MICRO_SWE_CWD__"


def _with_cwd_report(command: str) -> str:
    """Make the shell report its final working directory so the next command starts there."""
    return f"trap 'printf \"\\n{_CWD_MARK}%s\" \"$PWD\"' EXIT\n{command}"


def _split_cwd_report(output: str, previous: str | None) -> tuple[str, str | None]:
    head, sep, tail = output.rpartition("\n" + _CWD_MARK)
    if not sep or "\n" in tail:
        return output, previous
    return head, tail if os.path.isdir(tail) else previous


# <<< feature:persistent_cwd
def _run(command: str, cwd: str, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Like subprocess.run, but kills the whole process group on timeout so no children are orphaned."""
    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)
