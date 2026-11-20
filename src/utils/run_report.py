"""Read an optimization run dir into flat records for experiment tracking (W&B or anything else).

Everything here is read-only and pure (no wandb import), so it can run beside a live driver and
be unit-tested. Sources, all written by the loop:
  - candidates/summary.json   one row per finished candidate (outcome, train/val gate details)
  - candidates/frontier.json  the current best and the anchor
  - logs/run.out              the driver log: job submit/done times per stage
  - result.json per trial     in evaluation/<tag>[.<part>]/ (val) and candidates/<tag>/eval_after/

`candidate_rows` gives one record per candidate in the order they finished, each with the
cumulative state of the run at that point (best so far, outcome counts), so a record's index
is a natural x-axis (proposals used).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TRAIN_PASS = ("accepted", "val_fail", "not_selected")

# candidate_<iteration>_<parent iteration>_<round>[<sibling letter>]
TAG_RE = re.compile(r"^candidate_(\d+)_(\d+)_(\d+)([a-z]?)$")

# Rough mechanism families, from the CANDIDATE.md title. First match wins; order matters
# (e.g. "answer whitespace compaction" is answer-format, not context).
CATEGORIES = [
    ("answer-format", r"answer.*(hygiene|format|normali[sz]|artifact|whitespace)|bare.answer"),
    ("context", r"context|condens|compact|prun"),
    ("loop-guard", r"loop|repeat|repetit|stagna|no.op|runaway|dead.?man|deadlock|re.?affirm|re.?run"),
    ("budget", r"budget|turn.count|token.cap|output.token|truncat|hard.stop"),
    ("lookup", r"fetch|network|lookup|remote|search|web"),
    ("verification", r"verif|review|check|evidence|compute|symbolic|tool"),
]


# --------------------------------------------------------------------------- small parsers
def parse_tag(tag: str) -> dict[str, Any]:
    m = TAG_RE.match(tag or "")
    if not m:
        return {}
    it, parent, rnd, sib = m.groups()
    return {"iteration": int(it), "parent_iteration": int(parent), "round": int(rnd),
            "sibling": sib or "-"}


def title_of(hypothesis: str | None) -> str:
    """First markdown heading of CANDIDATE.md without the 'Candidate x_y_z --' prefix."""
    for line in (hypothesis or "").splitlines():
        line = line.strip()
        if line.startswith("#"):
            t = line.lstrip("#").strip()
            return re.split(r"\s+[\u2014\u2013-]+\s+", t, maxsplit=1)[-1].strip()
    return ""


def category_of(title: str) -> str:
    t = title.lower()
    for name, pat in CATEGORIES:
        if re.search(pat, t):
            return name
    return "other"


def _lower_bound(reason: str | None) -> float | None:
    m = re.search(r"lower bound ([-+]?\d*\.?\d+)", reason or "")
    return float(m.group(1)) if m else None


def gate_fields(prefix: str, g: dict | None) -> dict[str, Any]:
    """Flatten one gate decision (utils.gate.decide output) under `prefix/`."""
    if not g:
        return {}
    acc, cost, anchor = g.get("acc") or {}, g.get("cost") or {}, g.get("anchor") or {}
    out: dict[str, Any] = {
        f"{prefix}/pass": int(bool(g.get("accept"))),
        f"{prefix}/path": g.get("path") or "-",
        f"{prefix}/n": acc.get("n"),
        f"{prefix}/acc_A": acc.get("A"), f"{prefix}/acc_B": acc.get("B"),
        f"{prefix}/d": acc.get("d"), f"{prefix}/se": acc.get("se"), f"{prefix}/t": acc.get("t"),
        f"{prefix}/noninf_lb": _lower_bound(g.get("reason")),
    }
    if cost:
        a, b = cost.get("A"), cost.get("B")
        out.update({
            f"{prefix}/tok_A": a, f"{prefix}/tok_B": b,
            f"{prefix}/tok_delta_pct": (100.0 * (b - a) / a) if a and b is not None else None,
            # positive = B clearly cheaper (the gate prints it this way round)
            f"{prefix}/tok_cheaper_t": -cost["t"] if cost.get("t") is not None else None,
        })
    if anchor:
        out[f"{prefix}/anchor_d"] = anchor.get("d")
        out[f"{prefix}/anchor_se"] = anchor.get("se")
    return {k: v for k, v in out.items() if v is not None}


# --------------------------------------------------------------------------- trial stats
def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _is_timeout(exc: dict) -> bool:
    typ = exc.get("exception_type") or ""
    return "Timeout" in typ or "exit 124" in (exc.get("exception_message") or "")


def iter_trials(d: Path):
    """result.json dicts of the harbor trials under `d` (skips job-level result.json)."""
    for f in Path(d).glob("**/result.json"):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and "task_name" in r:
            yield r


def trial_stats(dirs: list[Path]) -> dict[str, Any]:
    """Trial-level health over one or more eval dirs: count, timeouts, failures without a
    score, durations (minutes), mean output tokens, per-task accuracy (mean of task means)."""
    n = timeouts = unscored = errors = 0
    durs, toks = [], []
    per_task: dict[str, list[float]] = defaultdict(list)
    for d in dirs:
        for r in iter_trials(d):
            n += 1
            exc = r.get("exception_info") or {}
            if exc:
                errors += 1
                timeouts += _is_timeout(exc)
            rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            if rew is None:
                unscored += 1
            else:
                per_task[r["task_name"]].append(float(rew))
            s, f = _ts(r.get("started_at")), _ts(r.get("finished_at"))
            if s and f:
                durs.append((f - s).total_seconds() / 60.0)
            t = (r.get("agent_result") or {}).get("n_output_tokens")
            if t is not None:
                toks.append(float(t))
    if not n:
        return {}
    task_means = [sum(v) / len(v) for v in per_task.values()]
    return {
        "trials": n, "tasks": len(per_task),
        "acc": sum(task_means) / len(task_means) if task_means else None,
        "timeout_rate": timeouts / n, "error_rate": errors / n, "unscored_rate": unscored / n,
        "dur_p50_min": _pct(durs, 0.5), "dur_p95_min": _pct(durs, 0.95),
        "dur_max_min": max(durs) if durs else None,
        "tokens_mean": sum(toks) / len(toks) if toks else None,
    }


def val_dirs(run_dir: Path, tag: str) -> list[Path]:
    """A candidate's val trials: evaluation/<tag> plus its parts evaluation/<tag>.<suffix>
    (adaptive val), including top-ups of it made while it was the best."""
    ev = Path(run_dir) / "evaluation"
    if not ev.is_dir():
        return []
    return sorted(p for p in ev.iterdir()
                  if p.is_dir() and (p.name == tag or p.name.startswith(tag + ".")))


# --------------------------------------------------------------------------- driver log
_TS = re.compile(r"^(\d\d):(\d\d):(\d\d) (.*)$")
_START = re.compile(r"^===== driver start (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .*?argv=(.*)$")
_SUBMIT = re.compile(r"^iter \d+ (\S+) (\S+)(?: \(x\d+\))? -> submit$")
_JOBID = re.compile(r"^\s+-> (\S+)$")
_DONE = re.compile(r"^\[(\S+)\] (?:(\S+) )?job (\S+) done \(rc=(-?\d+)\)")


def parse_argv(argv: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    toks = argv.split()
    i = 0
    while i < len(toks):
        k = toks[i]
        if k.startswith("--"):
            key = k[2:].replace("-", "_")
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                v: Any = toks[i + 1]
                for cast in (int, float):
                    try:
                        v = cast(v)
                        break
                    except ValueError:
                        pass
                out[key] = v
                i += 2
                continue
            out[key] = True
        i += 1
    return out


def parse_run_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(config from the last driver start's argv, jobs). Each job: {job_id, stage, tag,
    submit, done, minutes, rc}. Log lines only carry HH:MM:SS, so dates are carried from the
    'driver start' header and advanced when the clock goes backwards (midnight)."""
    config: dict[str, Any] = {}
    jobs: dict[str, dict[str, Any]] = {}
    pending: dict[str, Any] | None = None
    now: datetime | None = None
    if not Path(path).exists():
        return config, []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _START.match(line)
        if m:
            now = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            config = parse_argv(m.group(2))
            continue
        m = _TS.match(line)
        if m and now is not None:
            h, mi, s, rest = int(m[1]), int(m[2]), int(m[3]), m[4]
            t = now.replace(hour=h, minute=mi, second=s)
            if t < now - timedelta(hours=1):          # crossed midnight
                t += timedelta(days=1)
            now = t
            sm = _SUBMIT.match(rest)
            if sm:
                pending = {"stage": sm[1], "tag": sm[2], "submit": now}
                continue
            dm = _DONE.match(rest)
            if dm:
                j = jobs.get(dm[3])
                if j is not None:
                    j["done"] = now
                    j["rc"] = int(dm[4])
                    j["minutes"] = (now - j["submit"]).total_seconds() / 60.0
            continue
        m = _JOBID.match(line)
        if m and pending is not None:
            jobs[m[1]] = {"job_id": m[1], **pending, "done": None, "minutes": None, "rc": None}
            pending = None
    return config, list(jobs.values())


def job_owner(job: dict[str, Any]) -> str:
    """The candidate a job's cost belongs to. Val parts `<tag>.ak2` belong to <tag>; a top-up
    of the best made for candidate X (`<best>.t-<X short>-k2`) belongs to X."""
    tag = job.get("tag") or ""
    m = re.search(r"\.t-([0-9a-z_]+)-k\d+$", tag)
    if m:
        return "candidate_" + m.group(1)
    return tag.split(".", 1)[0]


# --------------------------------------------------------------------------- candidates
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def candidate_rows(run_dir: Path, *, with_trials: bool = True,
                   cache: dict | None = None) -> list[dict[str, Any]]:
    """One flat record per finished candidate (baseline first), with cumulative run state.

    `cache` (optional dict, reused across calls) memoises trial stats of eval dirs whose
    trial count has not changed, so repeated syncs of a long run stay cheap."""
    run_dir = Path(run_dir)
    rows = read_jsonl(run_dir / "candidates" / "summary.json")
    _, jobs = parse_run_log(run_dir / "logs" / "run.out")
    by_owner: dict[str, list[dict]] = defaultdict(list)
    for j in jobs:
        by_owner[job_owner(j)].append(j)
    cache = {} if cache is None else cache

    def stats(dirs: list[Path]) -> dict[str, Any]:
        if not with_trials or not dirs:
            return {}
        key = tuple(str(d) for d in dirs)
        sig = sum(1 for d in dirs for _ in d.glob("*/result.json"))
        hit = cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
        st = trial_stats(dirs)
        cache[key] = (sig, st)
        return st

    out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    by_path: Counter = Counter()
    best = best_tok = anchor = None
    best_tag = None
    for i, r in enumerate(rows):
        tag, outcome = r.get("candidate", ""), r.get("outcome") or "?"
        title = title_of(r.get("hypothesis"))
        rec: dict[str, Any] = {
            "step": i, "candidate": tag, "kind": r.get("kind"), "outcome": outcome,
            "title": title, "category": category_of(title) if tag != "baseline" else "baseline",
            "ts_A": r.get("ts_A"), "ts_B": r.get("ts_B"), **parse_tag(tag),
        }
        rec.update(gate_fields("train", r.get("train_gate")))
        rec.update(gate_fields("val", r.get("val_gate")))
        for k in ("val", "prev_best", "look", "val_trials"):
            if r.get(k) is not None:
                rec[f"val/{k}" if k != "val" else "val/acc"] = r[k]
        if r.get("path") and outcome == "accepted":
            by_path[r["path"]] += 1

        # cumulative run state
        if outcome == "baseline":
            best = anchor = r.get("val")
            best_tag = tag
        elif outcome == "accepted":
            best, best_tag = r.get("val"), tag
            best_tok = ((r.get("val_gate") or {}).get("cost") or {}).get("B", best_tok)
            if best is not None and (anchor is None or best > anchor):
                anchor = best
        if outcome != "baseline":
            counts[outcome] += 1
            counts["proposed"] += 1
            counts["train_pass"] += outcome in TRAIN_PASS
            counts["to_val"] += r.get("val") is not None
        rec.update({
            "best/candidate": best_tag, "best/val_acc": best, "best/val_tokens": best_tok,
            "best/anchor_acc": anchor,
            "best/anchor_margin": (best - anchor) if best is not None and anchor is not None else None,
            "cum/proposed": counts["proposed"], "cum/train_pass": counts["train_pass"],
            "cum/to_val": counts["to_val"], "cum/accepted": counts["accepted"],
            "cum/accepted_better": by_path["better"], "cum/accepted_cheaper": by_path["cheaper"],
            "cum/train_fail": counts["train_fail"], "cum/val_fail": counts["val_fail"],
            "cum/not_selected": counts["not_selected"], "cum/no_proposal": counts["no_proposal"],
            "cum/no_score": counts["no_score"],
        })
        if counts["proposed"]:
            rec["cum/train_pass_rate"] = counts["train_pass"] / counts["proposed"]
            rec["cum/accept_rate"] = counts["accepted"] / counts["proposed"]

        # cost: wall minutes per stage (sum over that candidate's jobs) and GPU-node minutes
        stage_min: dict[str, float] = defaultdict(float)
        for j in by_owner.get(tag, []):
            if j.get("minutes") is not None:
                stage_min[j["stage"]] += j["minutes"]
        for st, m in stage_min.items():
            rec[f"time/{st}_min"] = m
        vjobs = [j for j in by_owner.get(tag, []) if j["stage"] in ("val", "baseline")
                 and j.get("done")]
        if vjobs:
            rec["time/val_wall_min"] = (max(j["done"] for j in vjobs)
                                        - min(j["submit"] for j in vjobs)).total_seconds() / 60
            rec["time/val_node_min"] = sum(j["minutes"] for j in vjobs)

        # trial health of this candidate's own evals
        for name, dirs in (("eval_b", [run_dir / "candidates" / tag / "eval_after"]),
                           ("val_eval", val_dirs(run_dir, tag))):
            dirs = [d for d in dirs if d.is_dir()]
            for k, v in stats(dirs).items():
                if v is not None:
                    rec[f"{name}/{k}"] = v
        out.append({k: v for k, v in rec.items() if v is not None})
    return out


def round_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Funnel per round of sibling proposals: proposed / train pass / to val / accepted."""
    rounds: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        if "round" not in r:
            continue
        key = (r["iteration"], r["round"])
        g = rounds.setdefault(key, {"iteration": key[0], "round": key[1], "proposed": 0,
                                    "train_pass": 0, "to_val": 0, "accepted": 0,
                                    "categories": Counter()})
        g["proposed"] += 1
        g["train_pass"] += r["outcome"] in TRAIN_PASS
        g["to_val"] += "val/acc" in r
        g["accepted"] += r["outcome"] == "accepted"
        g["categories"][r.get("category", "other")] += 1
    out = []
    for g in rounds.values():
        g["distinct_categories"] = len(g["categories"])
        g["categories"] = ",".join(f"{k}:{v}" for k, v in sorted(g["categories"].items()))
        out.append(g)
    return sorted(out, key=lambda g: (g["iteration"], g["round"]))


def frontier(run_dir: Path) -> dict[str, Any]:
    p = Path(run_dir) / "candidates" / "frontier.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
