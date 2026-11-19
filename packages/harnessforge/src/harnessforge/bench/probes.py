"""Behavioural probes: does a harness directory have each of the ten features?

`probe(harness_dir)` runs one scenario per feature, each in a fresh `python -I` subprocess started
in an empty temp dir (`_probe_runner.py`), so every candidate's `micro_swe` package is imported in
isolation. The harness runs end to end through its own CLI with a scripted fake LLM; only the
recorded requests (what the model was shown, with which sampling kwargs) and the trajectory are
judged. Judges look at behaviour, never at code, so a re-implementation in different code counts.

Probes need the harness's runtime deps (jinja2, pydantic, pyyaml, tenacity; litellm is stubbed)
in the probing interpreter: `pip install harnessforge[bench]`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from harnessforge.bench.features import FEATURE_NAMES

RUNNER = Path(__file__).with_name("_probe_runner.py")


@dataclass(frozen=True)
class ProbeResult:
    feature: str
    present: bool
    evidence: str
    error: str | None = None      # the run itself failed (harness crashed, timeout, ...)


# ----------------------------------------------------------------------------- helpers
def _content(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):    # typed content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return "" if c is None else str(c)


def _after_last_assistant(messages: list[dict]) -> list[dict]:
    idx = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    return messages[idx + 1:]


def _observation(calls: list[dict], i: int) -> str:
    """What the model was shown after its i-th response: the tail of request i+1."""
    if i + 1 >= len(calls):
        return ""
    return "\n".join(_content(m) for m in _after_last_assistant(calls[i + 1]["messages"]))


def _sampling(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in ("max_tokens", "max_completion_tokens", "extra_body")}


def _exit_status(run: dict) -> str:
    traj = run.get("trajectory") or {}
    return (traj.get("info") or {}).get("exit_status", "")


# ----------------------------------------------------------------------------- judges
# Each judge gets the runner output ({"calls": [...], "trajectory": ..., "error": ...}) and
# returns (present, evidence).
def _judge_output_truncation(run):
    obs = _observation(run["calls"], 0)
    ok = 0 < len(obs) <= 20000 and "HEADMARK" in obs and "TAILMARK" in obs
    return ok, f"observation after a ~60k-char output: {len(obs)} chars, head marker " \
               f"{'HEADMARK' in obs}, tail marker {'TAILMARK' in obs}"


_TIMEOUT_RE = re.compile(r"tim(e|ed)[\s_-]*out|killed", re.I)


def _judge_timeout_guidance(run):
    normal, timed = _observation(run["calls"], 0), _observation(run["calls"], 1)
    says = bool(_TIMEOUT_RE.search(timed))
    keeps = "PARTIALMARK" in timed
    clean = not _TIMEOUT_RE.search(normal)
    return says and keeps and clean, (f"timed-out command's observation mentions timeout: {says}, keeps partial "
                                      f"output: {keeps}; normal command's observation free of it: {clean}")


def _judge_persistent_cwd(run):
    obs = _observation(run["calls"], 1)
    return "probe_sub" in obs, f"`pwd` after `cd probe_sub` showed: {obs[:200]!r}"


_LENGTH_RE = re.compile(r"(token|length|output)[\s_-]*limit|cut[\s-]*off|truncat|max_tokens|finish_reason", re.I)


def _judge_length_recovery(run):
    calls = run["calls"]
    if len(calls) < 3:
        return False, f"only {len(calls)} model calls"

    def budget(c):
        return int(c["kwargs"].get("max_tokens") or c["kwargs"].get("max_completion_tokens") or 0)

    base = budget(calls[1])                       # the request that was cut off
    raised = [budget(c) for c in calls[2:4]]
    bigger = any(r > base for r in raised)
    told = bool(_LENGTH_RE.search(_observation(calls, 1)))
    return bigger or told, (f"first max_tokens={base}, next requests' max_tokens={raised}; feedback names the "
                            f"length cut-off: {told}")


def _judge_loop_warning(run):
    calls = run["calls"]
    obs = [_observation(calls, i) for i in range(5)]
    differs = any(o != obs[0] for o in obs[2:5] if o)
    resampled = any(_sampling(c["kwargs"]) != _sampling(calls[0]["kwargs"]) for c in calls[1:7])
    return differs or resampled, (f"observations of 5 identical runs identical to the first: "
                                  f"{[o == obs[0] for o in obs]}; sampling kwargs changed: {resampled}")


def _judge_format_streak_trim(run):
    calls = run["calls"]
    if len(calls) < 4:
        return False, f"only {len(calls)} model calls"
    n2, n3 = len(calls[2]["messages"]), len(calls[3]["messages"])
    return n3 <= n2, f"prompt length after 1 format error: {n2} messages; after 2 in a row: {n3}"


def _judge_missing_command_repair(run):
    calls = run["calls"]
    if len(calls) < 4:
        return False, f"only {len(calls)} model calls"
    base = len(calls[1]["messages"])            # the request that produced the `{}` call
    repair_added = calls[2]["messages"][base:]
    guided = any("command" in _content(m).lower() for m in repair_added)
    grew = len(calls[3]["messages"]) - base     # one good turn = assistant + observation = 2
    return guided and grew == 2, (f"repair request adds guidance about the command: {guided}; the next normal "
                                  f"request grew by {grew} messages (2 = no trace of the bad call)")


def _judge_submit_confirmation(run):
    n = len(run["calls"])
    status = _exit_status(run)
    return n >= 3 and status == "Submitted", f"model calls: {n} (2 = ended at the first submit); exit {status!r}"


def _judge_context_trim(run):
    calls = run["calls"]
    if len(calls) < 31:
        return False, f"only {len(calls)} model calls"
    last = calls[30]["messages"]
    size = len(json.dumps(last, ensure_ascii=False))
    newest = _observation(calls, 29)
    intact = "OBS30" in newest and newest.count("z") >= 4000
    produced = 30 * 4006
    return size < produced / 2 and intact, (f"prompt after 30 x 4k-char outputs: {size} chars "
                                            f"(outputs total {produced}); newest output intact: {intact}")


def _judge_raw_observation(run):
    obs = _observation(run["calls"], 0)
    raw = 'q"uote\\back\nline2' in obs
    return raw, f"observation shows the quote/backslash/newline unescaped: {raw}; got {obs[:160]!r}"


JUDGES: dict[str, Callable[[dict], tuple[bool, str]]] = {
    "output_truncation": _judge_output_truncation,
    "timeout_guidance": _judge_timeout_guidance,
    "persistent_cwd": _judge_persistent_cwd,
    "length_recovery": _judge_length_recovery,
    "loop_warning": _judge_loop_warning,
    "format_streak_trim": _judge_format_streak_trim,
    "missing_command_repair": _judge_missing_command_repair,
    "submit_confirmation": _judge_submit_confirmation,
    "context_trim": _judge_context_trim,
    "raw_observation": _judge_raw_observation,
}
assert set(JUDGES) == set(FEATURE_NAMES)


# ----------------------------------------------------------------------------- running
def run_scenario(harness_dir: str | Path, scenario: str, *, python: str | None = None,
                 timeout: float = 180.0, env: dict[str, str] | None = None) -> dict:
    """Run one scenario in a fresh subprocess; returns the runner's JSON output."""
    harness_dir = Path(harness_dir).resolve()
    with tempfile.TemporaryDirectory(prefix=f"hf-probe-{scenario}-") as tmp:
        out = Path(tmp) / "result.json"
        work = Path(tmp) / "work"
        work.mkdir()
        proc_env = {k: v for k, v in os.environ.items() if not k.startswith(("MICRO_SWE_", "MSWEA_"))}
        proc_env.update(env or {})
        try:
            proc = subprocess.run([python or sys.executable, "-I", str(RUNNER), "--harness", str(harness_dir),
                                   "--scenario", scenario, "--out", str(out)],
                                  cwd=work, env=proc_env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"calls": [], "trajectory": None, "error": f"probe timed out after {timeout}s"}
        if not out.exists():
            return {"calls": [], "trajectory": None,
                    "error": f"runner exited {proc.returncode}: {proc.stderr[-2000:]}"}
        return json.loads(out.read_text())


def probe_feature(harness_dir: str | Path, feature: str, **kw) -> ProbeResult:
    run = run_scenario(harness_dir, feature, **kw)
    try:
        present, evidence = JUDGES[feature](run)
    except Exception as e:  # noqa: BLE001  (a malformed run is evidence of absence, not a crash)
        present, evidence = False, f"judge failed: {type(e).__name__}: {e}"
    err = run.get("error")
    if err and "KeyboardInterrupt" not in err and "_ProbeAbort" not in err:
        evidence += f" [run error: {err.splitlines()[0]}]"
    return ProbeResult(feature, bool(present), evidence, err)


def probe(harness_dir: str | Path, features: Iterable[str] | None = None, *, python: str | None = None,
          workers: int = 5, env: dict[str, str] | None = None) -> dict[str, ProbeResult]:
    """{feature: ProbeResult} for `harness_dir` (all ten features by default)."""
    names = list(features or FEATURE_NAMES)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda f: probe_feature(harness_dir, f, python=python, env=env), names))
    return {r.feature: r for r in results}


def present_set(results: dict[str, ProbeResult]) -> frozenset[str]:
    return frozenset(f for f, r in results.items() if r.present)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m harnessforge.bench.probes",
                                 description="Probe a harness dir for the ten gold features.")
    ap.add_argument("harness_dir")
    ap.add_argument("--feature", action="append", help="probe only this feature (repeatable)")
    ap.add_argument("--python", default=None, help="interpreter with the harness's deps (default: this one)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = probe(args.harness_dir, args.feature, python=args.python)
    if args.json:
        print(json.dumps({f: asdict(r) for f, r in res.items()}, indent=2))
    else:
        for f, r in res.items():
            print(f"{'PRESENT' if r.present else 'absent '}  {f:24s} {r.evidence}")
        print(f"{len(present_set(res))}/{len(res)} present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
