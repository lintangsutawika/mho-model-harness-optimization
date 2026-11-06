"""Adaptive val allocation: how many trials each val task gets.

A task's uncertainty is the spread of its pass/fail outcome, p(1 - p): 0 for a task that is
always failed or always solved (another trial says nothing new), largest at p = 0.5. In the
paired gate, task i adds p_A(1-p_A)/k_A + p_B(1-p_B)/k_B to the noise of its difference, so
for a fixed number of trials the noise is smallest when k_i grows with sqrt(p_i(1 - p_i))
(Neyman allocation). p is never known: it is estimated from trials that already exist, smoothed
so a few trials never give exactly 0 or 1:  p = (successes + 0.5) / (trials + 1).

Every task gets at least one trial, so a candidate that starts solving a task nothing solved
before is still seen (and that task then gets more trials in later comparisons).

An allocation is run as GROUPS: one eval job per distinct trial count c, over the tasks that get
c trials. Groups are independent eval jobs (they run in parallel).
"""
from __future__ import annotations

import heapq
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # py < 3.11
    import tomli as tomllib


def task_dirs(split_dir: str | Path) -> dict[str, Path]:
    """{task name as harbor reports it in result.json (task.toml [task].name): task dir}."""
    out: dict[str, Path] = {}
    for d in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        name = d.name
        toml = d / "task.toml"
        if toml.exists():
            try:
                name = (tomllib.loads(toml.read_text()).get("task") or {}).get("name") or name
            except tomllib.TOMLDecodeError:
                pass
        out[name] = d
    return out


def rate(successes: float, trials: float) -> float:
    """Smoothed pass rate (Jeffreys prior): never exactly 0 or 1."""
    return (successes + 0.5) / (trials + 1.0)


def spread(successes: float, trials: float) -> float:
    """sqrt(p (1 - p)) of the smoothed pass rate: the task's uncertainty weight."""
    p = rate(successes, trials)
    return math.sqrt(p * (1.0 - p))


def neyman(weights: dict[str, float], extra: int, *, start: dict[str, int] | None = None,
           cap: int) -> dict[str, int]:
    """Hand out `extra` trials on top of `start` (default 0 each) to minimise
    sum_i w_i^2 / k_i (the variance of a mean of per-task means): each next trial goes to the
    task with the largest gain w^2 / (k (k + 1)), never above `cap` per task in total.
    Returns the ADDED trials per task (0 for tasks that get none)."""
    start = start or {}
    k = {t: start.get(t, 0) for t in weights}
    heap = []
    for t, w in weights.items():
        if k[t] < cap:
            heapq.heappush(heap, (-_gain(w, k[t]), t))
    added = {t: 0 for t in weights}
    while extra > 0 and heap:
        _, t = heapq.heappop(heap)
        k[t] += 1
        added[t] += 1
        extra -= 1
        if k[t] < cap:
            heapq.heappush(heap, (-_gain(weights[t], k[t]), t))
    return added


def _gain(w: float, k: int) -> float:
    return math.inf if k == 0 else w * w / (k * (k + 1))


def first_look(weights: dict[str, float], budget: int, cap: int) -> dict[str, int]:
    """Trials per task for a candidate's first val look: 1 each, the rest of `budget` by
    uncertainty, at most `cap` per task."""
    floor = {t: 1 for t in weights}
    add = neyman(weights, max(0, budget - len(weights)), start=floor, cap=cap)
    return {t: 1 + add[t] for t in weights}


def fill_to(counts: dict[str, int], tasks: list[str], target: int) -> dict[str, int]:
    """Trials to add so every task has `target` (a uniform eval, for the accepted best)."""
    return {t: max(0, target - counts.get(t, 0)) for t in tasks}


def groups(k: dict[str, int]) -> dict[int, list[str]]:
    """{trial count c: tasks that get c trials} for c > 0. Each group is one eval job with
    c attempts per task (harbor's -k is per job, not per task); groups run in parallel."""
    out: dict[int, list[str]] = {}
    for t, n in k.items():
        if n > 0:
            out.setdefault(n, []).append(t)
    return {c: sorted(ts) for c, ts in sorted(out.items())}
