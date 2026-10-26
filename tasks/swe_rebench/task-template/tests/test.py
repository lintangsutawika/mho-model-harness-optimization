"""SWE-rebench-V2 verifier (standalone tests/ form).

Runs the dataset's own evaluation inside the task container: restore the files the reference
test patch touches, apply that patch, run the repository's own test command and read the
result with the log parser the dataset names for this instance. The task is resolved when
every FAIL_TO_PASS and PASS_TO_PASS test is reported PASSED -- the criterion SWE-rebench-V2's
scripts/eval.py uses. The agent's edits to the rest of the tree are what make that happen;
unlike a math task there is no answer to compare.

Everything it needs is in tests/: spec.json, test_patch.diff and the vendored parsers.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

SPEC = json.loads((TESTS / "spec.json").read_text(encoding="utf-8"))
LOG_DIR = Path("/logs/verifier")
GIT_APPLY = ["git", "apply", "-v", "--recount", "--ignore-space-change", "--whitespace=nowarn"]

# Some runners print per-test timings inside the test name; eval.py strips them on both sides
# so a slow run does not look like a different test.
_TIMING_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize(name: str) -> str:
    for pattern in _TIMING_RES:
        name = pattern.sub("", name)
    return name.strip()


def _run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=SPEC["workdir"], env=env,
                          capture_output=True, text=True, errors="replace")


def _restore_test_files() -> None:
    for rel in SPEC["test_files"]:
        if _run(["git", "checkout", SPEC["base_commit"], "--", rel]).returncode != 0:
            # Absent from the base commit -> the test patch creates it; drop the agent's copy.
            Path(SPEC["workdir"], rel).unlink(missing_ok=True)


def _apply_test_patch() -> subprocess.CompletedProcess:
    patch = str(TESTS / "test_patch.diff")
    applied = _run(GIT_APPLY + ["--3way", patch])
    if applied.returncode != 0:
        applied = _run(GIT_APPLY + [patch])  # --3way implies --index, which a dirty index fails
    return applied


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _restore_test_files()
    applied = _apply_test_patch()
    if applied.returncode != 0:
        print("X could not apply the reference test patch:\n" + applied.stderr)
        return 1

    env = dict(os.environ)
    env.setdefault("_JAVA_OPTIONS", "-Djava.net.preferIPv6Addresses=false")  # as in eval.py

    # install_config.install, which the image does not run for us: without it `import <pkg>`
    # fails, conftest raises and the parser sees an empty log rather than a failing test run.
    for cmd in SPEC.get("install") or ():
        done = _run(["bash", "-c", cmd], env=env)
        if done.returncode != 0:
            print(f"X install step failed: {cmd}\n{done.stdout[-2000:]}{done.stderr[-2000:]}")
            return 1

    proc = _run(["bash", "-c", SPEC["test_cmd"]], env=env)
    log = (proc.stdout or "") + (proc.stderr or "")
    (LOG_DIR / "test_output.txt").write_text(log, encoding="utf-8")

    import log_parsers

    statuses = {_normalize(k): v for k, v in log_parsers.NAME_TO_PARSER[SPEC["log_parser"]](log).items()}
    passed = {name for name, status in statuses.items() if status == "PASSED"}
    missing_f2p = sorted({_normalize(t) for t in SPEC["fail_to_pass"]} - passed)
    missing_p2p = sorted({_normalize(t) for t in SPEC["pass_to_pass"]} - passed)
    resolved = not missing_f2p and not missing_p2p

    (LOG_DIR / "report.json").write_text(json.dumps({
        "instance_id": SPEC["instance_id"],
        "resolved": resolved,
        "test_exit_code": proc.returncode,
        "tests_parsed": len(statuses),
        "missing_fail_to_pass": missing_f2p,
        "missing_pass_to_pass": missing_p2p,
    }, indent=2) + "\n", encoding="utf-8")

    print(("OK" if resolved else "X")
          + f" {SPEC['instance_id']}: {len(SPEC['fail_to_pass']) - len(missing_f2p)}"
            f"/{len(SPEC['fail_to_pass'])} fail_to_pass, "
            f"{len(SPEC['pass_to_pass']) - len(missing_p2p)}/{len(SPEC['pass_to_pass'])} pass_to_pass"
          + f" (log: {LOG_DIR / 'test_output.txt'})")
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main())
