"""Run a micro-swe-agent candidate from a read-only checkout, inside the task sandbox.

Launched by the harnessforge harness spec with the toolchain's interpreter (`{toolchain}/bin/python3 -I`), so no
package is installed in the sandbox: the candidate's `src/` (mounted read-only) goes first on sys.path and its own
`micro_swe.__main__.main` runs with the command line harbor's MiniSweAgent passes to the `mini-swe-agent` binary.

    hf_micro_runner.py [--keep-children] --candidate DIR --task-file FILE [mini-swe-agent args...]

`--keep-children` drops platoon's launch marker (PLATOON_HARNESS_AGENT) from the environment the agent's commands
inherit. platoon kills every sandbox process carrying the marker when the harness exits, which ends a server the
agent was asked to leave running before the tests can reach it; harbor's agent left such processes alive. This
process keeps its own marker, so a timeout still kills the agent loop.
"""
import os
import sys

MARKER_ENV = "PLATOON_HARNESS_AGENT"  # platoon.harness.runner.AGENT_MARKER_ENV


def main() -> None:
    args = sys.argv[1:]
    candidate = task_file = None
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--candidate":
            candidate, i = args[i + 1], i + 2
        elif args[i] == "--keep-children":
            os.environ.pop(MARKER_ENV, None)  # unsetenv: children started from here on are not marked
            i += 1
        elif args[i] == "--task-file":
            task_file, i = args[i + 1], i + 2
        else:
            rest.append(args[i])
            i += 1
    if not candidate or not task_file:
        sys.exit("hf_micro_runner: --candidate and --task-file are required")
    src = os.path.join(candidate, "src")
    if not os.path.isdir(os.path.join(src, "micro_swe")):
        sys.exit(f"hf_micro_runner: {src} holds no micro_swe package")
    sys.dont_write_bytecode = True  # the checkout is read-only
    sys.path.insert(0, src)
    with open(task_file, encoding="utf-8") as fh:
        task = fh.read()
    from micro_swe.__main__ import main as micro_main

    micro_main([*rest, f"--task={task}"])


if __name__ == "__main__":
    main()
