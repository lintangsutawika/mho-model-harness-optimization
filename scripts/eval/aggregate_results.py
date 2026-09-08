#!/usr/bin/env python3
"""Aggregate per-trial harbor results into one tidy JSONL (one row per trajectory).

Each harbor trial writes ``result.json`` (reward, timing, exception_info,
agent_result.metadata) and ``agent/mini-swe-agent.trajectory.json`` (turns + per-turn
rollout token IDs). SkyRL's own ``dumped_evals`` only records score + stop_reason + text;
this joins in n_turns, completion-token counts, timeout status, and phase timings so a run
can be summarized at a glance.

Usage:
    # newest ray session's trials, write next to it + print summary
    python scripts/eval/aggregate_results.py

    # a specific trials dir (e.g. one eval step) -> explicit output
    python scripts/eval/aggregate_results.py --trials <dir> --out <path.jsonl>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime


def _dur(info: dict | None) -> float | None:
    """Seconds between started_at/finished_at of a harbor TimingInfo dict."""
    if not info:
        return None
    s, f = info.get("started_at"), info.get("finished_at")
    if not (s and f):
        return None
    try:
        fmt = lambda t: datetime.fromisoformat(t.replace("Z", "+00:00"))
        return round((fmt(f) - fmt(s)).total_seconds(), 2)
    except Exception:
        return None


def _newest_trials_dir() -> str | None:
    sessions = sorted(
        glob.glob("/scratch/lsutawik/tmp/ray/session_2*"),
        key=os.path.getmtime,
        reverse=True,
    )
    for sess in sessions:
        hits = glob.glob(os.path.join(sess, "**", "trials"), recursive=True)
        if hits:
            return hits[0]
    return None


def _traj_token_stats(trial_dir: str) -> dict:
    """Turn count + completion-token totals from the fork trajectory, if present."""
    p = os.path.join(trial_dir, "agent", "mini-swe-agent.trajectory.json")
    out = {"exit_status": None, "total_completion_tokens": None, "max_turn_tokens": None}
    if not os.path.exists(p):
        return out
    try:
        d = json.load(open(p))
    except Exception:
        return out
    out["exit_status"] = d.get("info", {}).get("exit_status")
    comps = [
        len((m.get("extra") or {}).get("rollout", {}).get("completion_token_ids") or [])
        for m in d.get("messages") or []
        if m.get("role") == "assistant"
    ]
    comps = [c for c in comps if c]
    if comps:
        out["total_completion_tokens"] = sum(comps)
        out["max_turn_tokens"] = max(comps)
    return out


def aggregate(trials_dir: str, out_path: str) -> dict:
    rows = []
    for rp in sorted(glob.glob(os.path.join(trials_dir, "*", "result.json"))):
        try:
            r = json.load(open(rp))
        except Exception:
            continue
        exc = (r.get("exception_info") or {}).get("exception_type")
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        meta = (r.get("agent_result") or {}).get("metadata") or {}
        row = {
            "task_name": r.get("task_name"),
            "trial_name": r.get("trial_name"),
            "reward": reward,
            "solved": bool(reward and reward > 0),
            "n_turns": meta.get("n_episodes"),
            "exception_type": exc,
            "timed_out": exc == "AgentTimeoutError",
            "agent_exec_sec": _dur(r.get("agent_execution")),
            "agent_setup_sec": _dur(r.get("agent_setup")),
            "env_setup_sec": _dur(r.get("environment_setup")),
            **_traj_token_stats(os.path.dirname(rp)),
        }
        rows.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    n = len(rows)
    scored = [r for r in rows if r["reward"] is not None]
    solved = sum(r["solved"] for r in rows)
    timed = sum(r["timed_out"] for r in rows)
    turns = [r["n_turns"] for r in rows if r["n_turns"]]
    toks = [r["total_completion_tokens"] for r in rows if r["total_completion_tokens"]]
    summary = {
        "trials": n,
        "scored": len(scored),
        "solved_pass@1": round(solved / n, 4) if n else 0.0,
        "mean_reward": round(sum(r["reward"] for r in scored) / len(scored), 4) if scored else None,
        "timeouts": timed,
        "timeout_rate": round(timed / n, 4) if n else 0.0,
        "mean_turns": round(sum(turns) / len(turns), 2) if turns else None,
        "mean_completion_tokens": round(sum(toks) / len(toks), 1) if toks else None,
        "max_completion_tokens": max(toks) if toks else None,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default=None, help="trials dir (default: newest ray session)")
    ap.add_argument("--out", default=None, help="output JSONL path")
    args = ap.parse_args()

    trials_dir = args.trials or _newest_trials_dir()
    if not trials_dir or not os.path.isdir(trials_dir):
        raise SystemExit(f"no trials dir found (looked at: {trials_dir})")
    out_path = args.out or os.path.join(trials_dir, "aggregated.jsonl")

    summary = aggregate(trials_dir, out_path)
    print(f"trials_dir: {trials_dir}")
    print(f"wrote:      {out_path}  ({summary['trials']} rows)")
    print("summary:    " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
