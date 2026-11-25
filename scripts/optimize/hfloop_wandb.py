"""Mirror an hfloop run dir (src/mho/hfloop.py) into W&B: the val frontier over wall-clock time, every proposal.

Read-only on the run dir (it only writes <run>/aux/wandb.json, its own resume state). What it logs, on the x-axis
`time/hours` (hours since the loop started: the val run of candidate_0 was requested):

  frontier/val1000           the accepted reference's full val pass rate (%), a stair: at an accept it logs the old
                             value then the new one at the same hour (line chart = steps, never a line through dots)
  throughput/proposals_raced cumulative proposals the gate decided;  throughput/accepts, round
  proposals (Table)          one row per raced proposal: round, tag, hours, estimated val %, d / se (points), t,
                             verdict, look, status (accepted / passed, not chosen / rejected), the proposer's title
  charts/frontier (Html)     the run's figure, interactive (Vega-Lite): round bands, each proposal at the time its
                             race decided it (shape = look, colour = status, +-1 SE, hover = title and stats), the
                             frontier stair

A proposal's estimate is the reference's full val rate + its paired difference at the look where its race stopped;
an accepted harness is plotted at its own full val rate when its accept (top-up / fresh rerun) landed.

    python scripts/optimize/hfloop_wandb.py --run-dir /groups/.../hfloop-toy-r1            # backfill, then exit
    python scripts/optimize/hfloop_wandb.py --run-dir ... --follow --job 3047078.pbs1     # live until the job ends

W&B credentials: WANDB_API_KEY / WANDB_ENTITY from the environment, else from --env-file (the training runs' .env;
only those two names are read, values are never printed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ENV_FILE = "~/mho-model-harness-optimization/mho-platoon-port/.env"
START = "candidate_0"


# ----------------------------------------------------------------------------------------------- reading the run
def _json(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def val_rate(records: Path) -> float | None:
    """Mean over tasks of the per-task success rate (reward > 0), in %."""
    per: dict[str, list[float]] = {}
    try:
        lines = records.read_text().splitlines()
    except OSError:
        return None
    for ln in lines:
        if ln.strip():
            r = json.loads(ln)
            per.setdefault(r["task"], []).append(1.0 if (r.get("reward") or 0) > 0 else 0.0)
    return 100 * sum(sum(v) / len(v) for v in per.values()) / len(per) if per else None


def title(cand_dir: Path) -> str:
    try:
        first = (cand_dir / "CANDIDATE.md").read_text().splitlines()[0]
    except (OSError, IndexError):
        return ""
    return re.sub(r"^#+\s*", "", first).strip()


@dataclass
class Snapshot:
    t0: float                                        # loop start (unix)
    frontier: list[tuple[float, float, str]]         # (hours, val %, tag): candidate_0 then each accept
    proposals: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)   # {round, start, end, accepted, done}
    superseded: int = 0

    def hours(self, t: float) -> float:
        return round((t - self.t0) / 3600.0, 4)


def collect(run: Path) -> Snapshot | None:
    """Everything the chart needs, from files the loop already writes. None until candidate_0's val run is done."""
    req = run / "evals" / f"val-{START}.request.json"
    done = run / "ref" / START / "status.json"
    if not req.exists() or not done.exists():
        return None
    snap = Snapshot(t0=req.stat().st_mtime, frontier=[])
    rate: dict[str, float | None] = {}

    def full(tag: str) -> float | None:
        if tag not in rate:
            rate[tag] = val_rate(run / "ref" / tag / "records.jsonl")
        return rate[tag]

    snap.frontier.append((snap.hours(done.stat().st_mtime), full(START) or 0.0, START))
    state = _json(run / "hfloop.json", {}) or {}
    accepted = [h["accepted"] for h in state.get("history", [])]
    for tag in accepted:
        st = run / "ref" / tag / "status.json"
        if st.exists() and full(tag) is not None:
            snap.frontier.append((snap.hours(st.stat().st_mtime), full(tag), tag))
    acc_time = {tag: h for h, _, tag in snap.frontier}
    prev_end = snap.frontier[0][0]
    on_disk = {int(p.name) for p in (run / "rounds").glob("*") if p.name.isdigit()}
    started = {int(k) for k, v in (state.get("rounds") or {}).items() if str(k).isdigit() and v.get("propose_started_at")}
    for n in sorted(on_disk | started):              # a round shows once its proposers start (overlap: before its gate)
        rdir = run / "rounds" / str(n)
        rs = (state.get("rounds") or {}).get(str(n), {})
        snap.superseded += sum(len(s.get("tags") or []) for s in rs.get("superseded") or [])
        race = _json(rdir / "race.json")
        start = snap.hours(rs["propose_started_at"]) if rs.get("propose_started_at") else prev_end
        band = {"round": n, "start": start, "end": None, "accepted": rs.get("accepted"), "done": False}
        snap.rounds.append(band)
        if not race:
            continue
        dec = race.get("decisions") or {}
        ref = race.get("reference")
        base = full(ref) if ref else None
        decided = race.get("decided_at_s") or {}
        t_race = float(race.get("started_at") or 0)
        for c, s in (dec.get("states") or {}).items():
            if s.get("verdict") not in ("pass", "fail") or c not in decided:
                continue
            acc = s.get("acc") or {}
            status = "accepted" if c in accepted else ("passed, not chosen" if s["verdict"] == "pass" else "rejected")
            row = {"round": n, "tag": c, "hours": snap.hours(t_race + float(decided[c])),
                   "est": None if base is None else round(base + 100 * float(acc.get("d") or 0), 3),
                   "d": round(100 * float(acc.get("d") or 0), 3), "se": round(100 * float(acc.get("se") or 0), 3),
                   "t": acc.get("t"), "verdict": s["verdict"], "look": s.get("look"), "status": status,
                   "reference": ref, "title": title(run / "candidates" / c)}
            if c in acc_time:                      # plotted at its own full val rate, when the accept landed
                row["accept_hours"], row["accept_val"] = acc_time[c], full(c)
            snap.proposals.append(row)
        if dec.get("done") or race.get("elapsed_s"):
            end = snap.hours(t_race + float(race.get("elapsed_s") or 0))
            w = dec.get("winner")
            if w and w in acc_time:
                end = acc_time[w]
            if not (w and w not in acc_time):      # an accept still topping up keeps the band open
                band.update(end=end, done=True)
                prev_end = end
    snap.proposals.sort(key=lambda r: r["hours"])
    return snap


# ----------------------------------------------------------------------------------------------- what to log
def events(snap: Snapshot) -> list[dict[str, Any]]:
    """History rows in time order, each with a stable `key` (so a follower logs each one once)."""
    out: list[dict[str, Any]] = []
    for i, (h, v, tag) in enumerate(snap.frontier):
        out.append({"key": f"accept:{tag}", "hours": h, "kind": "accept", "tag": tag, "val": v,
                    "old": snap.frontier[i - 1][1] if i else None})
    for p in snap.proposals:
        out.append({"key": f"proposal:{p['tag']}", "hours": p["hours"], "kind": "proposal", "tag": p["tag"]})
    out.sort(key=lambda e: (e["hours"], e["kind"] != "proposal"))
    return out


def frontier_at(snap: Snapshot, h: float) -> float:
    v = snap.frontier[0][1]
    for fh, fv, _ in snap.frontier:
        if fh <= h:
            v = fv
    return v


def vega_html(snap: Snapshot, name: str, ceiling: float | None = None) -> str:
    last = max([snap.frontier[-1][0]] + [p["hours"] for p in snap.proposals] + [r["end"] or 0 for r in snap.rounds])
    stair = [{"hours": h, "val": v, "tag": t} for h, v, t in snap.frontier] + [
        {"hours": last, "val": snap.frontier[-1][1], "tag": snap.frontier[-1][2]}]
    pts = []
    for p in snap.proposals:
        if p["status"] == "accepted" and p.get("accept_hours") is not None:
            pts.append({**p, "x": p["accept_hours"], "y": p["accept_val"], "lo": None, "hi": None})
        elif p["est"] is not None:
            pts.append({**p, "x": p["hours"], "y": p["est"], "lo": p["est"] - p["se"], "hi": p["est"] + p["se"]})
    bands = [{"round": f"R{r['round']}", "x0": r["start"], "x1": r["end"] if r["end"] is not None else last,
              "odd": r["round"] % 2} for r in snap.rounds]
    ys = [p["y"] for p in pts] + [s["val"] for s in stair] + ([ceiling] if ceiling else [])
    y0, y1 = (min(ys) - 3, max(ys) + 4) if ys else (0, 100)
    look_shape = {"domain": [1, 2, 3, 4], "range": ["cross", "triangle-up", "square", "diamond"]}
    status_color = {"domain": ["accepted", "passed, not chosen", "rejected"], "range": ["#2563eb", "#2563eb", "#9aa0a6"]}
    x = {"field": "x", "type": "quantitative", "title": "Hours since the loop started", "scale": {"domain": [0, last * 1.02]}}
    y = {"type": "quantitative", "title": "val pass rate (%)", "scale": {"domain": [y0, y1]}}
    layers: list[dict[str, Any]] = [
        {"data": {"values": bands}, "mark": {"type": "rect", "opacity": 0.5},
         "encoding": {"x": {"field": "x0", "type": "quantitative"}, "x2": {"field": "x1"},
                      "color": {"field": "odd", "type": "nominal", "legend": None,
                                "scale": {"domain": [0, 1], "range": ["#f6f8fa", "#e9edf2"]}}}},
        {"data": {"values": bands}, "mark": {"type": "text", "dy": -4, "color": "#6e7781", "fontWeight": "bold"},
         "encoding": {"x": {"field": "x0", "type": "quantitative"}, "x2": {"field": "x1"},
                      "text": {"field": "round"}, "y": {"value": 10}}},
        {"data": {"values": stair}, "mark": {"type": "line", "interpolate": "step-after", "color": "#2563eb",
                                             "strokeWidth": 2.5},
         "encoding": {"x": {**x, "field": "hours"}, "y": {**y, "field": "val"}}},
        {"data": {"values": [p for p in pts if p["lo"] is not None]}, "mark": {"type": "rule", "opacity": 0.45},
         "encoding": {"x": x, "y": {**y, "field": "lo"}, "y2": {"field": "hi"},
                      "color": {"field": "status", "type": "nominal", "scale": status_color, "legend": None}}},
        {"data": {"values": pts},
         "mark": {"type": "point", "size": 90, "strokeWidth": 2, "filled": False},
         "encoding": {"x": x, "y": {**y, "field": "y"},
                      "shape": {"field": "look", "type": "ordinal", "scale": look_shape,
                                "title": "decided at look"},
                      "color": {"field": "status", "type": "nominal", "scale": status_color, "title": "status"},
                      "fill": {"condition": {"test": "datum.status === 'accepted'", "value": "#2563eb"},
                               "value": "white"},
                      "tooltip": [{"field": "tag"}, {"field": "title"}, {"field": "status"},
                                  {"field": "look", "type": "ordinal"}, {"field": "round", "type": "ordinal"},
                                  {"field": "y", "title": "val % (est.)", "format": ".1f"},
                                  {"field": "d", "title": "d (pts)", "format": "+.2f"},
                                  {"field": "se", "title": "SE (pts)", "format": ".2f"},
                                  {"field": "t", "format": ".2f"}, {"field": "reference"}]}},
    ]
    if ceiling is not None:
        layers.append({"data": {"values": [{"c": ceiling}]},
                       "mark": {"type": "rule", "strokeDash": [4, 3], "color": "#c2410c"},
                       "encoding": {"y": {"field": "c", "type": "quantitative"}}})
    best = snap.frontier[-1]
    spec = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container", "height": 460,
            "title": {"text": f"{name}: {snap.frontier[0][1]:.1f}% -> {best[1]:.1f}% ({best[2]}), "
                              f"{len(snap.proposals)} proposals raced, {len(snap.frontier) - 1} accepts",
                      "anchor": "start"},
            "layer": layers, "resolve": {"scale": {"color": "independent", "shape": "independent"}}}
    return ("<html><head><meta charset='utf-8'>"
            "<script src='https://cdn.jsdelivr.net/npm/vega@5.30.0'></script>"
            "<script src='https://cdn.jsdelivr.net/npm/vega-lite@5.21.0'></script>"
            "<script src='https://cdn.jsdelivr.net/npm/vega-embed@6.26.0'></script></head>"
            "<body style='margin:0'><div id='v' style='width:100%'></div><script>"
            f"vegaEmbed('#v', {json.dumps(spec)}, {{actions: false}});</script></body></html>")


# ----------------------------------------------------------------------------------------------- W&B
def wandb_env(env_file: str) -> dict[str, str]:
    out = {k: os.environ[k] for k in ("WANDB_API_KEY", "WANDB_ENTITY") if os.environ.get(k)}
    p = Path(env_file).expanduser()
    if "WANDB_API_KEY" not in out and p.exists():
        for ln in p.read_text().splitlines():
            m = re.match(r"^\s*(?:export\s+)?(WANDB_API_KEY|WANDB_ENTITY)\s*=\s*(.*)$", ln)
            if m and m.group(1) not in out:
                out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


def loop_config_summary(path: str | None) -> dict[str, Any]:
    st = _json(Path(path)) if path else None
    keys = ("trials", "proposals", "rounds", "alpha", "fresh", "fresh_looks", "overlap", "seed", "evidence_tasks", "policy")
    return {k: st[k] for k in keys if k in st} if isinstance(st, dict) else {}


def run_id_for(run: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", f"hfloop-{run.name}")[:64]


class Mirror:
    def __init__(self, run: Path, project: str, name: str | None, ceiling: float | None, tags: list[str],
                 loop_config: str | None = None) -> None:
        import wandb
        self.wandb, self.run, self.ceiling = wandb, run, ceiling
        self.state_path = run / "aux" / "wandb.json"
        self.state = _json(self.state_path, {}) or {}
        self.logged: set[str] = set(self.state.get("logged", []))
        rid = self.state.get("run_id") or run_id_for(run)
        self.name = name or run.name
        self.wb = wandb.init(project=project, id=rid, name=self.name, resume="allow", job_type="hfloop",
                             tags=tags, config={"run_dir": str(run), **loop_config_summary(loop_config)})
        self.state["run_id"] = rid
        for k in ("frontier/*", "throughput/*", "round"):
            wandb.define_metric(k, step_metric="time/hours")

    def save(self) -> None:
        self.state["logged"] = sorted(self.logged)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name("wandb.json.tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def sync(self) -> bool:
        snap = collect(self.run)
        if snap is None:
            return False
        new = [e for e in events(snap) if e["key"] not in self.logged]
        raced = sum(1 for k in self.logged if k.startswith("proposal:"))
        accepts = sum(1 for k in self.logged if k.startswith("accept:")) - (1 if f"accept:{START}" in self.logged else 0)
        for e in new:
            h = e["hours"]
            rnd = next((p["round"] for p in snap.proposals if p["tag"] == e["tag"]), None)
            if e["kind"] == "accept":
                if e["old"] is not None:           # the stair's riser: old value, then the new one, same hour
                    self.wb.log({"time/hours": h, "frontier/val1000": e["old"]})
                    accepts += 1
                self.wb.log({"time/hours": h, "frontier/val1000": e["val"], "throughput/accepts": accepts})
            else:
                raced += 1
                self.wb.log({"time/hours": h, "frontier/val1000": frontier_at(snap, h),
                             "throughput/proposals_raced": raced, "throughput/accepts": accepts,
                             **({"round": rnd} if rnd is not None else {})})
            self.logged.add(e["key"])
        if new or not self.state.get("charted"):
            cols = ["round", "tag", "hours", "est", "d", "se", "t", "verdict", "look", "status", "reference", "title"]
            table = self.wandb.Table(columns=cols, data=[[p.get(c) for c in cols] for p in snap.proposals])
            self.wb.log({"proposals": table, "charts/frontier": self.wandb.Html(vega_html(snap, self.name, self.ceiling))},
                        commit=False)
            self.wb.summary.update({"best/val1000": snap.frontier[-1][1], "best/candidate": snap.frontier[-1][2],
                                    "start/val1000": snap.frontier[0][1], "proposals_raced": len(snap.proposals),
                                    "accepts": len(snap.frontier) - 1, "superseded": snap.superseded,
                                    "rounds_started": len(snap.rounds)})
            self.state["charted"] = True
        self.save()
        if new:
            print(f"[hfloop_wandb] {time.strftime('%H:%M:%S')} logged {len(new)} event(s); "
                  f"{len(snap.proposals)} proposals, best {snap.frontier[-1][1]:.1f}% ({snap.frontier[-1][2]})", flush=True)
        return True

    def finish(self) -> None:
        self.wb.finish()


def job_running(job: str) -> bool:
    p = subprocess.run(["qstat", job], capture_output=True, text=True)
    return p.returncode == 0 and " F " not in p.stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--project", default="mho-harness")
    ap.add_argument("--name", help="W&B run name (default: the run dir's name)")
    ap.add_argument("--tags", nargs="*", default=["hfloop"])
    ap.add_argument("--ceiling", type=float, help="draw a reference line at this val % (e.g. the defect-free harness)")
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    ap.add_argument("--follow", action="store_true", help="keep syncing every --poll seconds")
    ap.add_argument("--job", help="with --follow: stop (after a last sync) once this PBS job has ended")
    ap.add_argument("--poll", type=float, default=300.0)
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--loop-config", help="the loop's config JSON (its settings go into the W&B run config)")
    ap.add_argument("--dry", metavar="HTML", help="no W&B: print what would be logged, write the chart to HTML")
    a = ap.parse_args(argv)
    if a.dry:
        snap = collect(Path(a.run_dir))
        if snap is None:
            print("[hfloop_wandb] nothing to log yet (candidate_0 val not done)")
            return 1
        for e in events(snap):
            print(f"  {e['hours']:7.3f} h  {e['key']}" + (f"  {e['val']:.2f}%" if e["kind"] == "accept" else ""))
        print(f"rounds: {[(r['round'], r['start'], r['end']) for r in snap.rounds]}")
        Path(a.dry).write_text(vega_html(snap, a.name or Path(a.run_dir).name, a.ceiling))
        return 0
    env = wandb_env(a.env_file)
    if "WANDB_API_KEY" not in env:
        print("[hfloop_wandb] no WANDB_API_KEY (environment or --env-file)", file=sys.stderr)
        return 2
    os.environ.update(env)
    m = Mirror(Path(a.run_dir), a.project, a.name, a.ceiling, a.tags, a.loop_config)
    deadline = time.time() + a.max_hours * 3600
    try:
        m.sync()
        while a.follow and time.time() < deadline:
            time.sleep(a.poll)
            m.sync()
            if a.job and not job_running(a.job):
                m.sync()
                break
    finally:
        m.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
