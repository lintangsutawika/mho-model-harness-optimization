"""Run a micro-swe-agent candidate from a read-only checkout, inside the task sandbox.

Launched by the harnessforge harness spec with the toolchain's interpreter (`{toolchain}/bin/python3 -I`), so no
package is installed in the sandbox: the candidate's `src/` (mounted read-only) goes first on sys.path and its own
`micro_swe.__main__.main` runs with the command line harbor's MiniSweAgent passes to the `mini-swe-agent` binary.

    hf_micro_runner.py --candidate DIR --task-file FILE [mini-swe-agent args...]
"""
import os
import sys


def main() -> None:
    args = sys.argv[1:]
    candidate = task_file = None
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--candidate":
            candidate, i = args[i + 1], i + 2
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
