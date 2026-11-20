"""Who cancels the train loop or a rollout (HF_TRACE_CANCEL=1; installed by this directory's sitecustomize).

The 64kcap hangs and the "rollout group cancelled from within" failures are cancellations that come from outside
the task they hit. This records the canceller: every event loop gets

* a task factory whose tasks are a ``Task`` subclass: its ``cancel()`` logs the caller's stack when the task is a
  target (the ``train_loop`` task, or a rollout / group-member task) and the caller is not that task itself;
* a ``create_future`` whose futures are a ``Future`` subclass: its ``cancel()`` logs the caller's stack when a
  target task is waiting on it (its wake-up callback) and the caller is not that task. A task's own ``cancel()``
  cancelling the future it waits on is the task's event, logged there, and skipped here.

Entries go to <HF_STACKDUMP_DIR>/cancels.<pid>.txt, at most MAX_ENTRIES per process. Off (nothing installed, no
cost) unless HF_TRACE_CANCEL=1.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import traceback

MAX_ENTRIES = 400
TARGET_NAMES = ("train_loop",)
TARGET_CORO_NAMES = ("harness_rollout", "_run_group_member", "run_group_guarded", "_fail_if_cancelled_from_within",
                     "_train_loop", "arun_episode", "roll_out")

_state = {"n": 0, "file": None}
_lock = threading.Lock()


def _coro_name(task) -> str:
    try:
        return getattr(task.get_coro(), "__qualname__", "?")
    except Exception:
        return "?"


def is_target(task) -> bool:
    if task is None:
        return False
    try:
        if task.get_name() in TARGET_NAMES:
            return True
    except Exception:
        pass
    name = _coro_name(task)
    return any(name == t or name.endswith("." + t) for t in TARGET_CORO_NAMES)


def _describe(task) -> str:
    if task is None:
        return "no task (a callback or another thread)"
    try:
        return f"{task.get_name()} <{_coro_name(task)}>"
    except Exception:
        return repr(task)[:200]


def _write(kind: str, target, msg) -> None:
    with _lock:
        if _state["n"] >= MAX_ENTRIES:
            return
        _state["n"] += 1
        f = _state["file"]
        if f is None:
            d = os.environ.get("HF_STACKDUMP_DIR") or "."
            os.makedirs(d, exist_ok=True)
            f = _state["file"] = open(os.path.join(d, f"cancels.{os.getpid()}.txt"), "a", buffering=1)
    try:
        cur = asyncio.current_task()
    except RuntimeError:
        cur = None
    stack = "".join(traceback.format_stack(sys._getframe(2), limit=25))
    f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {kind} cancel of {_describe(target)} "
            f"(msg={msg!r}) by {_describe(cur)} [thread {threading.current_thread().name}]\n{stack}")


class TracedTask(asyncio.Task):
    def cancel(self, msg=None):
        try:
            if not self.done() and is_target(self):
                cur = None
                try:
                    cur = asyncio.current_task()
                except RuntimeError:
                    pass
                if cur is not self:
                    _write("task", self, msg)
        except Exception:
            pass
        return super().cancel(msg)


def _waiting_tasks(fut) -> list:
    out = []
    for cb in (getattr(fut, "_callbacks", None) or ()):
        fn = cb[0] if isinstance(cb, tuple) else cb
        owner = getattr(fn, "__self__", None)
        if isinstance(owner, asyncio.Task):
            out.append(owner)
    return out


class TracedFuture(asyncio.Future):
    def cancel(self, msg=None):
        try:
            if not self.done():
                try:
                    cur = asyncio.current_task()
                except RuntimeError:
                    cur = None
                for t in _waiting_tasks(self):
                    if t is not cur and is_target(t) and not t.cancelling():
                        _write("future", t, msg)
                        break
        except Exception:
            pass
        return super().cancel(msg)


def _task_factory(loop, coro, **kwargs):
    return TracedTask(coro, loop=loop, **kwargs)


def install() -> bool:
    """Patch every event loop created from now on (and the running one, if any). Returns whether it installed."""
    if os.environ.get("HF_TRACE_CANCEL") != "1":
        return False
    import asyncio.base_events as be

    init = be.BaseEventLoop.__init__
    if getattr(init, "_hf_traced", False):
        return True

    def __init__(self, *a, **k):
        init(self, *a, **k)
        self.set_task_factory(_task_factory)
    __init__._hf_traced = True
    be.BaseEventLoop.__init__ = __init__
    be.BaseEventLoop.create_future = lambda self: TracedFuture(loop=self)
    return True
