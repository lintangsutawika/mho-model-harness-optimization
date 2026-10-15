"""Propose job: evolve the micro-swe-agent scaffold with an OpenHands agent.

Run as its own batch job by ``optimize_loop`` (via ``MHO_PROPOSE_JOB``). Contract:

  in  (env):
    MHO_CANDIDATE_DIR  destination for the new scaffold snapshot (required)
    MHO_PARENT_SCAFFOLD  scaffold to BUILD ON (accumulation) -- the loop sets this to the
                       frontier-best candidate dir; empty/unset -> the original base (iter 1)
    MICRO_SCAFFOLD_BASE  the original/immutable base scaffold (for diff_from_base)
    MHO_DIFF_DIR       where to write the two provenance patches (default: <candidate>_diffs)
    MHO_FRONTIER       path to frontier.json (optional)
    MHO_SUMMARY        path to evolution_summary.jsonl (optional)
    MHO_TRIALS_DIR     recent harbor trials dir, for a trajectory/metrics digest (optional)
    OPTIMIZER_MODEL / OPTIMIZER_LLM_API_KEY / OPTIMIZER_LLM_BASE_URL   the proposer LLM
  out:
    a VALID installable micro-swe-agent snapshot at MHO_CANDIDATE_DIR (pyproject.toml present),
    plus two git-format patches in MHO_DIFF_DIR: diff_from_base.patch (cumulative, base->cand)
    and diff_from_parent.patch (incremental, parent->cand). exit 0 on success; nonzero if invalid.

Flow: materialize the PARENT scaffold (frontier-best, so changes ACCUMULATE) into
MHO_CANDIDATE_DIR, drive an OpenHands Conversation (FileEditor + Terminal, workspace = the
snapshot) with the adapted meta-harness instructions (instructions/propose_harness.md) + a digest of prior
results to make ONE targeted general-purpose change, then emit the two provenance diffs. Full
code edits are allowed; the only invariant is the snapshot stays an installable package. The
OpenHands import is lazy so this module can be imported/validated without the SDK present.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from harness import scaffold  # DEFAULT_BASE + _IGNORE (single source of truth for the copy)

SKILL_DIR = Path(__file__).parent / "propose-harness"   # AgentSkills dir: must contain SKILL.md
_MAX_SUMMARY_LINES = 40
_MAX_TRIAL_DIRS = 20   # bounded: stage at most this many raw trial dirs into eval_prior/


def _attempts_table() -> str:
    """A ranked 'what was tried -> result' view from evolution_summary.jsonl, so the proposer
    LEARNS from history: build on what improved, avoid repeating what regressed. Each eval row is
    expected to carry {candidate, score, improved, hypothesis} (the loop records hypothesis from
    the candidate's CANDIDATE.md)."""
    summary = os.environ.get("MHO_SUMMARY", "")
    if not (summary and Path(summary).exists()):
        return ""
    attempts = []
    for line in Path(summary).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase") != "eval" or not r.get("score"):
            continue
        p1 = r["score"].get("solved_pass@1")
        attempts.append((r.get("candidate", "?"), p1, r.get("improved"), (r.get("hypothesis") or "").strip()))
    if not attempts:
        return ""
    best = max((a[1] for a in attempts if a[1] is not None), default=None)
    rows = []
    for cand, p1, improved, hyp in attempts:
        tag = "BEST" if (p1 is not None and p1 == best) else ("REGRESSED" if improved is False else "improved" if improved else "")
        delta = f" (Δ{p1 - best:+.3f})" if (p1 is not None and best is not None) else ""
        hyp1 = (hyp.splitlines()[0][:160] if hyp else "(no hypothesis recorded)")
        rows.append(f"- {cand}: pass@1={p1}{delta} [{tag}] — {hyp1}")
    return ("### What has been tried (learn from this: build on the BEST, do NOT repeat a "
            "REGRESSED idea)\n" + "\n".join(rows))


def _digest_prior_results() -> str:
    """Bounded text digest: attempts+outcomes table, frontier, recent summary, last trials."""
    parts: list[str] = []
    table = _attempts_table()
    if table:
        parts.append(table)
    frontier = os.environ.get("MHO_FRONTIER", "")
    if frontier and Path(frontier).exists():
        parts.append("### frontier.json\n```json\n" + Path(frontier).read_text().strip() + "\n```")
    summary = os.environ.get("MHO_SUMMARY", "")
    if summary and Path(summary).exists():
        lines = Path(summary).read_text().splitlines()[-_MAX_SUMMARY_LINES:]
        if lines:
            parts.append("### evolution_summary.jsonl (recent)\n```\n" + "\n".join(lines) + "\n```")
    trials = os.environ.get("MHO_TRIALS_DIR", "")
    agg = Path(trials) / "aggregated.jsonl" if trials else None
    if agg and agg.exists():
        rows = agg.read_text().splitlines()[:60]
        parts.append("### recent trials digest (aggregated.jsonl sample)\n```\n" + "\n".join(rows) + "\n```")
    return "\n\n".join(parts) if parts else "(no prior results — treat this as the first iteration)"


def _original_base() -> Path:
    """The immutable original scaffold, for the cumulative diff_from_base."""
    return Path(os.environ.get("MICRO_SCAFFOLD_BASE") or scaffold.DEFAULT_BASE)


def _parent_scaffold() -> Path:
    """The scaffold this iteration BUILDS ON (accumulation). The loop sets MHO_PARENT_SCAFFOLD
    to the frontier-best candidate dir; empty/unset -> the original base (first iteration)."""
    p = os.environ.get("MHO_PARENT_SCAFFOLD", "").strip()
    return Path(p) if p else _original_base()


def _materialize_from(src: Path, dst: Path) -> None:
    """Copy the parent scaffold `src` -> candidate dir `dst`. Changes accumulate because `src`
    is the frontier-best (not the fixed base)."""
    if not (src / "pyproject.toml").exists():
        sys.exit(f"[propose] parent scaffold {src} is not an installable package; set MHO_PARENT_SCAFFOLD/MICRO_SCAFFOLD_BASE")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=scaffold._IGNORE)  # noqa: SLF001 (same-repo reuse)
    print(f"[propose] materialized parent scaffold {src} -> {dst}")


# Path fragments that are noise in a scaffold diff (the base checkout has .git, but candidates
# are copied with scaffold._IGNORE which strips these -- otherwise every .git/* file shows as a
# spurious deletion, burying the real code change).
_DIFF_EXCLUDE = ("/.git/", "/__pycache__/", "/.venv/", "/venv/", "/.pytest_cache/",
                 "/.ruff_cache/", "/.mypy_cache/", ".egg-info/", "/eval_prior/")


def _stage_prior_results(dst: Path) -> None:
    """Copy prior eval results INTO the workspace (eval_prior/) so the proposer agent can
    open them as files (not only as inline prompt text). Stages: frontier.json,
    evolution_summary.jsonl, aggregated.jsonl, and a bounded sample of the raw trial dirs
    (result.json + the readable + token-sparse trajectory transcripts). No-op when the
    sources are absent (first iteration). Excluded from the candidate diff by /eval_prior/."""
    # VALIDATION ISOLATION: the optimizer only ever sees the TRAIN-SAMPLE eval (the proposer's
    # improvement signal). The val-split eval (eval-val/) is the held-out accept/reject gate and
    # MUST NEVER be staged/read by the proposer -- doing so would contaminate the accept decision
    # and overfit the val set. MHO_TRIALS_DIR is set by the loop ONLY from the train-sample eval,
    # so staging from it is safe; refuse any path that looks like a val eval just in case.
    prior = dst / "eval_prior"
    trials = os.environ.get("MHO_TRIALS_DIR", "")
    if "evaluation" in Path(trials).parts or "eval-val" in Path(trials).parts:
        raise RuntimeError(
            "refusing to stage val-split trials into the proposer workspace: "
            f"{trials} (validation set must stay hidden from the optimizer)"
        )
    srcs = {
        "frontier.json": os.environ.get("MHO_FRONTIER", ""),
        "evolution_summary.jsonl": os.environ.get("MHO_SUMMARY", ""),
        "aggregated.jsonl": str(Path(trials) / "aggregated.jsonl") if trials else "",
    }
    if not any(v and Path(v).exists() for v in srcs.values()) and not (trials and Path(trials).is_dir()):
        return
    prior.mkdir(parents=True, exist_ok=True)
    for name, path in srcs.items():
        p = Path(path) if path else None
        if p and p.is_file():
            shutil.copy2(p, prior / name)
    if trials and Path(trials).is_dir():
        traj = prior / "trials"
        for i, trial_dir in enumerate(sorted(Path(trials).glob("*"))):
            if i >= _MAX_TRIAL_DIRS:
                break
            if not trial_dir.is_dir():
                continue
            tgt = traj / trial_dir.name
            tgt.mkdir(parents=True, exist_ok=True)
            for rel in ("result.json", "agent/mini-swe-agent.txt", "agent/mini-swe-agent.trajectory.json"):
                f = trial_dir / rel
                if f.is_file():
                    tf = tgt / rel
                    tf.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, tf)
    print(f"[propose] staged prior eval results in {prior}")


def _filter_diff(diff_text: str) -> str:
    """Drop per-file sections whose path hits an excluded fragment. git-diff output is a sequence
    of sections each starting with 'diff --git '; skip a whole section when its header matches."""
    out_lines: list[str] = []
    skip = False
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            skip = any(frag in line for frag in _DIFF_EXCLUDE)
        if not skip:
            out_lines.append(line)
    return "".join(out_lines)


def _diff_only_vcs(a: Path, b: Path) -> bool:
    """True if a and b differ ONLY in excluded (VCS/cache) paths -- i.e. no real code change.
    Uses the same filter as the emitted diffs."""
    r = subprocess.run(["git", "diff", "--no-index", "--no-color", str(a), str(b)],
                       capture_output=True, text=True)
    return _filter_diff(r.stdout).strip() == ""


def _write_diff(a: Path, b: Path, out: Path) -> None:
    """Write a git-format patch of a->b (two dirs) via `git diff --no-index` (no repo needed),
    with VCS/cache noise filtered out. Exit code 1 just means 'differences found'; >1 is a real
    error."""
    r = subprocess.run(
        ["git", "diff", "--no-index", "--no-color", str(a), str(b)],
        capture_output=True, text=True,
    )
    if r.returncode > 1:
        print(f"[propose] WARN: git diff {a} {b} failed rc={r.returncode}: {r.stderr[:200]}", file=sys.stderr)
    filtered = _filter_diff(r.stdout)
    out.write_text(filtered)
    print(f"[propose] wrote {out} ({len(filtered.splitlines())} lines of real changes)")


def _emit_diffs(dst: Path) -> None:
    """Two provenance patches per iteration: cumulative (base->candidate) + incremental
    (parent->candidate). Written OUTSIDE dst so they don't pollute the installable package or
    each other's diff. Loc: MHO_DIFF_DIR, else a `<candidate>_diffs` sibling."""
    diff_dir = Path(os.environ.get("MHO_DIFF_DIR") or (dst.parent / f"{dst.name}_diffs"))
    diff_dir.mkdir(parents=True, exist_ok=True)
    _write_diff(_original_base(), dst, diff_dir / "diff_from_base.patch")
    _write_diff(_parent_scaffold(), dst, diff_dir / "diff_from_parent.patch")


def _build_instructions(dst: Path) -> str:
    # The full scaffold-evolution workflow lives in the registered `propose-harness` skill
    # (propose-harness/SKILL.md), exposed to the agent via AgentContext. This message carries
    # the per-run context (workspace, constraints, prior results) + a pointer to follow the skill.
    return (
        f"Follow the **propose-harness** skill (available in <available_skills>) for the "
        f"scaffold-evolution workflow: analyze, state ONE falsifiable hypothesis, implement a "
        f"single minimal additive change, validate the package, and write CANDIDATE.md.\n\n"
        f"## Your workspace\n"
        f"The micro-swe-agent scaffold snapshot is your working directory -- it already contains "
        f"the accumulated changes from the best iteration so far (you are building ON TOP of it, "
        f"not from scratch). Edit files here in place to add ONE more improvement; keep it an "
        f"installable package (pyproject.toml + src/).\n\n"
        f"## HARD CONSTRAINTS (a candidate that violates these is REJECTED)\n"
        f"- Make MINIMAL, mostly-ADDITIVE edits for your ONE mechanism. Change only the files that "
        f"mechanism needs.\n"
        f"- NEVER delete existing files, NEVER empty or rewrite the src/ tree, NEVER recreate the "
        f"package from scratch. The snapshot must remain the FULL working scaffold PLUS your change.\n"
        f"- Keep it an installable package that still imports. Do not run destructive shell commands "
        f"(rm -rf, git clean, moving the whole tree, etc.).\n\n"
        f"## Prior results\n{_digest_prior_results()}\n\n"
        f"The same prior results are ALSO mounted as files under `eval_prior/` in your "
        f"workspace -- open them directly with the file editor for full detail: "
        f"`eval_prior/frontier.json`, `eval_prior/evolution_summary.jsonl`, "
        f"`eval_prior/aggregated.jsonl`, and `eval_prior/trials/` (one dir per trial, with "
        f"`result.json` + `agent/mini-swe-agent.txt` + `agent/mini-swe-agent.trajectory.json`). "
        f"NOTE: you see only the training-sample eval (eval_prior/trials). The validation set "
        f"is deliberately NOT shown to you -- it is the held-out accept/reject gate. Do NOT "
        f"attempt to reconstruct or infer val results.\n"
        f"Do NOT edit anything under `eval_prior/` -- it is read-only reference material.\n\n"
        f"Now: analyze the above, state ONE falsifiable hypothesis, make the single targeted "
        f"general-purpose change (minimal + additive), verify the package still imports, and write "
        f"CANDIDATE.md (hypothesis / changed files / expected effect)."
    )


def _run_openhands(workspace: Path, instructions: str) -> None:
    # Lazy import: only the actual proposal needs the SDK (env must provide it).
    from openhands.sdk import LLM, Agent, AgentContext, Conversation, LocalWorkspace, Tool
    from openhands.sdk.skills.skill import Skill
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    model = os.environ.get("OPTIMIZER_MODEL", None)
    api_key = os.environ.get("OPTIMIZER_LLM_API_KEY", None)
    base_url = os.environ.get("OPTIMIZER_LLM_BASE_URL", None)
    llm = LLM(model=model, api_key=api_key, base_url=base_url)
    # Load the propose-harness skill (AgentSkills standard) so the SDK exposes it in
    # <available_skills> and auto-injects its content when triggered. `Skill.load` directly
    # on SKILL.md is used because load_skills_from_dir's dir discovery did not pick it up.
    skill = Skill.load(SKILL_DIR / "SKILL.md")
    agent_context = AgentContext(
        skills=[skill],
        load_public_skills=False,
    )
    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ],
        agent_context=agent_context,
    )
    # Operate IN-PLACE on the candidate dir. Passing a bare string coerces OpenHands into a
    # default/isolated workspace, so the agent's edits never touch our dir (candidate == parent).
    # LocalWorkspace(working_dir=...) roots the agent's Terminal/FileEditor tools at our snapshot.
    conversation = Conversation(agent=agent, workspace=LocalWorkspace(working_dir=str(workspace)))
    conversation.send_message(instructions)
    conversation.run()


def _py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts]


def _validate_candidate(dst: Path, parent: Path) -> tuple[bool, str]:
    """Reject a broken/gutted candidate. `pyproject.toml` alone is NOT enough -- a proposer can
    delete the whole src tree and still leave an 'installable' shell. Require: pyproject present,
    a non-trivial number of .py files vs the parent (not gutted), and every .py parses."""
    if not (dst / "pyproject.toml").exists():
        return False, "lost pyproject.toml"
    cand_py, parent_py = _py_files(dst), _py_files(parent)
    if not cand_py:
        return False, "no .py files remain (scaffold gutted)"
    # Guard against wholesale deletion: keep at least 70% of the parent's source files.
    floor = max(1, int(len(parent_py) * 0.7))
    if len(cand_py) < floor:
        return False, f"only {len(cand_py)}/{len(parent_py)} .py files remain (< {floor}; likely deleted the scaffold)"
    import ast as _ast
    for f in cand_py:
        try:
            _ast.parse(f.read_text())
        except (SyntaxError, UnicodeDecodeError) as e:
            return False, f"{f.relative_to(dst)} does not parse: {e}"
    return True, f"{len(cand_py)} .py files, all parse"


def main() -> int:
    cand = os.environ.get("MHO_CANDIDATE_DIR")
    if not cand:
        sys.exit("[propose] MHO_CANDIDATE_DIR not set")
    dst = Path(cand)
    parent = _parent_scaffold()

    _materialize_from(parent, dst)   # accumulate: build on the frontier-best
    _stage_prior_results(dst)        # mount prior eval results (scores + trajectories) into eval_prior/
    _run_openhands(dst, _build_instructions(dst))

    ok, reason = _validate_candidate(dst, parent)
    _emit_diffs(dst)  # always emit diffs -- useful to inspect even a rejected candidate
    if not ok:
        print(f"[propose] ERROR: candidate invalid -- {reason} (diffs in MHO_DIFF_DIR)", file=sys.stderr)
        return 1
    # No-op guard: a candidate byte-identical to the parent (agent changed nothing) is useless --
    # the loop would eval a duplicate. Fail so the loop notes it rather than wasting an eval.
    if _diff_only_vcs(parent, dst):
        print("[propose] ERROR: candidate is identical to the parent (agent made no code change)",
              file=sys.stderr)
        return 2
    rationale = dst / "CANDIDATE.md"
    print(f"[propose] candidate ready at {dst} ({reason})"
          + (f"; rationale: {rationale}" if rationale.exists() else "; no CANDIDATE.md written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
