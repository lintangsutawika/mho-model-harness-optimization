"""python -m harnessforge.bench {build,probe,tasks,plan} ...

    build --out DIR [--only gold start ...]   materialize gold/, start/, ablate_<f>/
    probe HARNESS_DIR [--json]                behavioural probes for the ten features
    tasks --out DIR [--per-family 20]         generate the Harbor task set
    plan --harnesses DIR --tasks DIR          print the planned experiments as JSON
"""
import argparse
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "build":
        from harnessforge.bench import builder
        ap = argparse.ArgumentParser(prog="python -m harnessforge.bench build")
        ap.add_argument("--out", required=True)
        ap.add_argument("--only", nargs="*", default=None)
        ap.add_argument("--overwrite", action="store_true")
        a = ap.parse_args(rest)
        for name, path in builder.build_all(a.out, names=a.only, overwrite=a.overwrite).items():
            print(f"{name:32s} {path}")
        return 0
    if cmd == "probe":
        from harnessforge.bench.probes import main as probe_main
        return probe_main(rest)
    if cmd == "tasks":
        from harnessforge.bench.tasks.generate import main as tasks_main
        return tasks_main(rest)
    if cmd == "plan":
        from harnessforge.bench.metrics import main as metrics_main
        return metrics_main(["plan", *rest])
    print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
