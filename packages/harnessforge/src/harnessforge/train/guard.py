"""The driver's stall guard: notice a platoon trainer that stopped stepping, dump its stacks, stop it,
and have the gateway restart the driver from the latest recovery checkpoint (bounded).

Why: platoon's own watchdog is fed by every rollout group in flight, so a wedged training step stays
invisible for as long as straggler rollouts run (up to the agent timeout, 3600 s); the trial run of
2026-11-20 sat 86 min after its last Tinker call, the gateway reclaimed the model on its lease, and
the watchdog then exited rc=2 with no stack. The guard watches progress itself: platoon's
"| Step N/M |" log lines, and platoon's own "Still waiting for <call>" heartbeats while a
step waits on the training service (a long final forward_backward is not a stall: v2's step 200).

    driver.stall_timeout        seconds without a new step before the guard acts (default 1200; 0 = off)
    driver.stall_startup_grace  extra seconds allowed before the first step of a platoon process (default 3600)
    driver.max_auto_restarts    restarts the guard may ask for over the run's life (default 8)
    driver.stall_wait_ceiling   a "Still waiting for <call>" heartbeat counts as progress only while that one wait
                                is under this many seconds (default 3600): a call that never returns still fires

On a stall: dump every platoon Python process's thread stacks and asyncio task stacks
(stackdump_site), copy them to logs/stall.<n>.<time>.txt, stop platoon (SIGTERM; after
driver.stall_stop_wait seconds the second signal; then SIGKILL), and, within the budget, write
KEEP_GATEWAY so the gateway starts a fresh driver (release_previous_models + platoon resume). A
watchdog exit (rc=2) is handled the same way; its stacks are taken when platoon first warns
"WATCHDOG: No heartbeat". Any other non-zero exit nobody asked for (a crash) is too, with the end of
platoon's log (its traceback) as the record. Each auto-restart is recorded in auto_restarts.json, driver.events.log and
the run's W&B run (a tag auto-restart-<n> and a line in its notes).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

STEP_RE = re.compile(rb"\| Step (\d+)/\d+")
#: platoon's own "the trainer is waiting on the service" heartbeat (trainer/waiting.py, every 60 s while a
#: forward_backward / optim_step / save is outstanding): the trainer is alive, so it is progress, not a stall.
#: A wedged train loop logs neither (the 64kcap hangs).
ALIVE_RE = re.compile(rb"platoon\.train\.trainer\.waiting \| Still waiting for .*?\((\d+)s elapsed\)")
WATCHDOG_RE = re.compile(rb"WATCHDOG: No heartbeat for")
WATCHDOG_RC = 2
AUTO_RESTARTS = "auto_restarts.json"
STACKS_DIR = "stacks"
SITE_DIR = Path(__file__).resolve().parent / "stackdump_site"


def stack_signals() -> tuple[int, int]:
    """(thread stacks via faulthandler, asyncio task stacks)."""
    return signal.SIGRTMIN + 2, signal.SIGRTMIN + 3


class StallGuard:
    """Reads a platoon log incrementally; says when steps stop coming."""

    def __init__(self, log: Path, timeout: float, startup_grace: float = 3600.0,
                 clock: Callable[[], float] = time.time, wait_ceiling: float = 3600.0) -> None:
        self.log, self.timeout, self.grace, self.clock = Path(log), float(timeout), float(startup_grace), clock
        self.wait_ceiling = float(wait_ceiling)
        self.offset = self.log.stat().st_size if self.log.exists() else 0   # this process's lines only
        self.start_offset = self.offset
        self.started = self.last_progress = clock()
        self.last_step: int | None = None
        self.watchdog_warned = False
        self._tail = b""

    def poll(self) -> str | None:
        """None while fine; "watchdog" once, on platoon's first watchdog warning; "stall" once the
        trainer has gone stall_timeout (+ the start-up grace before its first step) without a step."""
        event = None
        if self.log.exists():
            with open(self.log, "rb") as f:
                f.seek(self.offset)
                data = f.read()
            self.offset += len(data)
            buf = self._tail + data
            if b"\n" in buf:
                data, self._tail = buf.rsplit(b"\n", 1)
            else:
                data, self._tail = b"", buf[-65536:]
            for m in STEP_RE.finditer(data):
                self.last_step, self.last_progress = int(m.group(1)), self.clock()
            for m in ALIVE_RE.finditer(data):
                # only while the one wait it reports is under the ceiling: a call that never returns server-side
                # keeps heartbeating, and must still count as a hang
                if int(m.group(1)) < self.wait_ceiling:
                    self.last_progress = self.clock()
            if not self.watchdog_warned and WATCHDOG_RE.search(data):
                self.watchdog_warned, event = True, "watchdog"
        if not self.timeout:
            return event
        limit = self.timeout + (self.grace if self.last_step is None else 0.0)
        if self.clock() - self.last_progress > limit:
            return "stall"
        return event

    def silent_for(self) -> float:
        return self.clock() - self.last_progress


def log_tail(log: Path, start: int = 0, max_lines: int = 400, max_bytes: int = 4 << 20) -> str:
    """The last lines this platoon process wrote to its log (from byte ``start``): a crash's traceback."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(start, f.tell() - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError as e:
        return f"could not read {log}: {e}\n"
    return "\n".join(data.splitlines()[-max_lines:]) + "\n"


def platoon_pids(pgid: int, proc_root: str = "/proc") -> list[int]:
    """The Python processes of platoon's process group (its supervisor and start). Only these get the
    dump signals: a real-time signal's default action terminates a process that has no handler."""
    out = []
    for d in Path(proc_root).iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            fields = stat[stat.rindex(")") + 2:].split()
            if int(fields[2]) != pgid:                      # field 5: pgrp
                continue
            cmd = (d / "cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError, IndexError):
            continue
        exe = os.path.basename(cmd[0].decode(errors="replace")) if cmd and cmd[0] else ""
        if exe.startswith("python") or any(b"platoon" in c for c in cmd[:2]):
            out.append(int(d.name))
    return sorted(out)


def dump_stacks(logs: Path, pgid: int, *, wait: float = 10.0, sleep=time.sleep,
                pids: Callable[[int], list[int]] = platoon_pids, kill=os.kill) -> list[int]:
    """Ask every platoon Python process for its thread and task stacks (appended under logs/stacks/).
    Returns the pids signalled."""
    threads, tasks = stack_signals()
    done = []
    for pid in pids(pgid):
        try:
            kill(pid, threads)
            kill(pid, tasks)
            done.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    if done:
        sleep(wait)
    return done


def collect_stacks(logs: Path, dest: Path, pids: list[int], since: float) -> Path:
    """Copy what the processes in pids wrote to their stack files since `since` into dest."""
    parts = []
    for pid in pids:
        f = logs / STACKS_DIR / f"stacks.{pid}.txt"
        if f.exists() and f.stat().st_mtime >= since - 1:
            parts.append(f"##### pid {pid} ({f})\n{f.read_text(errors='replace')[-2_000_000:]}")
    dest.write_text("\n".join(parts) or f"no stack output from pids {pids}\n")
    return dest


def stop_platoon(proc: subprocess.Popen, term: Callable[[], None], *, stop_wait: float,
                 sleep=time.sleep) -> None:
    """Graceful stop (platoon checkpoints if it can), then the second signal, then SIGKILL."""
    term()
    for limit, action in ((stop_wait, lambda: proc.send_signal(signal.SIGTERM)),
                          (120.0, lambda: os.killpg(proc.pid, signal.SIGKILL))):
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < limit:
            sleep(5)
        if proc.poll() is None:
            try:
                action()
            except ProcessLookupError:
                pass
    proc.wait(timeout=60)


def restarts(run_dir: Path) -> list[dict[str, Any]]:
    f = Path(run_dir) / AUTO_RESTARTS
    try:
        return json.loads(f.read_text())
    except (FileNotFoundError, ValueError):
        return []


def record_restart(run_dir: Path, event: dict[str, Any]) -> int:
    """Append an auto-restart; returns its number (1-based)."""
    ev = restarts(run_dir) + [event]
    tmp = Path(run_dir) / (AUTO_RESTARTS + ".tmp")
    tmp.write_text(json.dumps(ev, indent=1))
    tmp.replace(Path(run_dir) / AUTO_RESTARTS)
    return len(ev)


def wandb_run_id(platoon_dir: Path) -> str | None:
    rid = None
    for f in Path(platoon_dir).glob("*/*/checkpoints.jsonl"):
        for line in f.read_text().splitlines():
            try:
                rid = json.loads(line).get("wandb_run_id") or rid
            except ValueError:
                pass
    return rid


WANDB_NOTE = r"""
import json, os, sys
import wandb
a = json.loads(sys.argv[1])
api = wandb.Api()
path = "/".join(p for p in (os.environ.get("WANDB_ENTITY") or api.default_entity, a["project"], a["id"]) if p)
r = api.run(path)
if a["tag"] not in r.tags:
    r.tags = list(r.tags) + [a["tag"]]
r.notes = ((r.notes or "") + "\n" + a["note"]).strip()
r.update()
print("noted", a["tag"])
"""


def note_wandb(python: str, project: str, run_id: str, tag: str, note: str, env: dict[str, str],
               run_cmd=subprocess.run) -> str:
    """Tag the run's W&B run and add a line to its notes (best effort). Returns what happened."""
    arg = json.dumps({"project": project, "id": run_id, "tag": tag, "note": note})
    try:
        r = run_cmd([python, "-c", WANDB_NOTE, arg], env=env, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"W&B note failed: {type(e).__name__}"
    if r.returncode == 0:
        return f"W&B run {run_id}: tag {tag} added, note appended"
    last = (r.stderr or "").strip().splitlines()[-1:] or [""]
    return f"W&B note failed rc={r.returncode}: {last[0][:200]}"
