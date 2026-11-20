"""Stack dumps on demand for the platoon processes of a training run (harnessforge.train.guard).

The driver puts this directory first on the platoon process's PYTHONPATH, with HF_STACKDUMP_DIR set,
so every Python process of `platoon train` registers two dump signals at start-up:

  SIGRTMIN+2  faulthandler: every thread's Python stack, written from C (works even when the main
              thread is blocked in native code);
  SIGRTMIN+3  a Python handler: every asyncio task's coroutine stack (a hung await shows here, not
              in the thread stacks, where the event loop just sits in select).

Both append to <HF_STACKDUMP_DIR>/stacks.<pid>.txt. platoon itself uses SIGINT/SIGTERM/SIGUSR1/
SIGUSR2 (SIGUSR1 = "exit at once"), so those are left alone. Any other sitecustomize on the path
still runs.
"""
import os
import sys


def _install() -> None:
    out_dir = os.environ.get("HF_STACKDUMP_DIR")
    if not out_dir or not hasattr(__import__("signal"), "SIGRTMIN"):
        return
    import faulthandler
    import signal
    import time
    try:
        os.makedirs(out_dir, exist_ok=True)
        f = open(os.path.join(out_dir, f"stacks.{os.getpid()}.txt"), "a", buffering=1)
    except OSError:
        return
    sys._hf_stackdump_file = f                      # faulthandler needs the file kept open
    faulthandler.register(signal.SIGRTMIN + 2, file=f, all_threads=True, chain=False)

    def _tasks(signum, frame):                       # noqa: ARG001
        try:
            import asyncio.tasks as T
            tasks = list(getattr(T, "_scheduled_tasks", ())) + list(getattr(T, "_eager_tasks", ()))
            f.write(f"\n=== asyncio tasks of pid {os.getpid()} at {time.strftime('%Y-%m-%d %H:%M:%S')}: "
                    f"{len(tasks)} ===\n")
            for t in tasks:
                if t.done():
                    continue
                f.write(f"--- {t!r}\n"[:2000] + "\n")
                t.print_stack(limit=40, file=f)
            f.flush()
        except Exception as e:                       # never take the process down
            try:
                f.write(f"asyncio task dump failed: {type(e).__name__}: {e}\n")
            except Exception:
                pass
    try:
        signal.signal(signal.SIGRTMIN + 3, _tasks)
    except (ValueError, OSError):                    # not the main thread
        pass


def _chain() -> None:
    """Run the next sitecustomize on sys.path, if any (this one shadows it)."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    for p in sys.path:
        if not p or os.path.abspath(p) == here:
            continue
        cand = os.path.join(p, "sitecustomize.py")
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("_hf_chained_sitecustomize", cand)
            if spec and spec.loader:
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return


_install()
try:
    _chain()
except Exception:
    pass
