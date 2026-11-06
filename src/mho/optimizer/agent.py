"""Propose job: evolve the harness OR propose what the model should learn, via an OpenHands agent.

Run as its own batch job by ``optimize_loop`` (via ``MHO_PROPOSE_JOB``). Each proposal emits
EXACTLY ONE kind of output, selected by MHO_PROPOSE_KIND:
  harness  -> edits candidates/<tag>/harness (installable scaffold) + emits two diffs.
  model    -> writes candidates/<tag>/model_specs/ (what-the-model-should-learn specs, S_k).
A joint model-harness run is orchestrated by the LOOP scheduling harness- and model-steps
across iterations; each proposer still emits one output.

  in  (env):
    MHO_RUN_DIR        the experiment root (runs_output/optimize/<run>); EVERYTHING derives
                       from it: the candidate dir <RUN>/candidates/<tag>/, frontier.json,
                       evolution_summary.jsonl, the parent scaffold (frontier-best candidate's
                       harness), the diff dir, and the train-sample eval trials (see _derive_paths).
    MHO_TAG            this iteration's tag (e.g. candidate_1_0_1); the loop picks it
                       (encodes iteration/version/try).
    MHO_PROPOSE_KIND   "harness" (default) or "model".
    MICRO_SCAFFOLD_BASE  the original/immutable base scaffold (for diff_from_base); default
                       scaffold.DEFAULT_BASE when unset.
    OPTIMIZER_MODEL / OPTIMIZER_LLM_API_KEY / OPTIMIZER_LLM_BASE_URL   the proposer LLM
  out (harness):
    a VALID installable micro-swe-agent snapshot at <RUN>/candidates/<tag>/harness,
    plus two git-format patches in <candidate>/../diff: diff_from_base.patch + diff_from_parent.patch.
  out (model):
    one or more model spec files under <RUN>/candidates/<tag>/model_specs/ (e.g. spec_1.md).
  exit 0 on success; nonzero if invalid/missing output.

Flow: materialize the PARENT scaffold (frontier-best, so changes ACCUMULATE) into
In the derived candidate dir, drive an OpenHands Conversation (FileEditor + Terminal, workspace = the
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

_OPT_DIR = Path(__file__).parent
# Each proposer emits EXACTLY ONE kind of output:
#   harness   -> propose-harness/  : edits candidates/<tag>/harness (the installable scaffold)
#   model     -> propose-model/    : writes candidates/<tag>/model_specs/ (what the model should learn)
# A "model-harness" RUN is orchestrated by the LOOP scheduling harness-steps and model-steps
# across iterations (each proposer still emitting one output), not by a combined proposer.
PROPOSE_KINDS = ("harness", "model")
_MAX_TRIAL_DIRS = 20   # bounded: stage at most this many raw trial dirs into eval_prior/
# Diff sections whose path hits one of these fragments are excluded from the candidate
# patch (VCS/cache/venv noise only). Restored from commit 9503596 (dropped in refactor).
_DIFF_EXCLUDE = ("/.git/", "/__pycache__/", "/.venv/", "/venv/", "/.pytest_cache/",
                 "/.ruff_cache/", "/.mypy_cache/", ".egg-info/")


def _skill_dir(kind: str) -> Path:
    return _OPT_DIR / f"propose-{kind}"


def _instructions_md(kind: str) -> Path:
    return _skill_dir(kind) / "instructions.md"


def _original_base() -> Path:
    """The immutable original scaffold, for the cumulative diff_from_base. Defaults to
    scaffold.DEFAULT_BASE; override with MICRO_SCAFFOLD_BASE."""
    return Path(os.environ.get("MICRO_SCAFFOLD_BASE") or scaffold.DEFAULT_BASE)


def _materialize_from(src: Path, dst: Path) -> None:
    """Copy the parent scaffold `src` -> candidate harness dir `dst`. Changes accumulate because
    `src` is the frontier-best (not the fixed base). `scaffold._IGNORE` strips VCS/cache noise."""
    if not (src / "pyproject.toml").exists():
        sys.exit(f"[propose] parent scaffold {src} is not an installable package; "
                 f"set MICRO_SCAFFOLD_BASE / a valid MHO_RUN_DIR layout")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=scaffold._IGNORE)
    print(f"[propose] materialized parent scaffold {src} -> {dst}")


def _derive_paths(candidate_dir: Path) -> dict[str, Path]:
    """Derive every proposer input path from the run layout + the target candidate dir,
    instead of ~6 individual env vars. Requires MHO_RUN_DIR (the experiment root); the
    candidate's harness dir is derived as <RUN>/candidates/<tag>/harness:
      frontier  = <RUN>/frontier.json
      summary   = <RUN>/evolution_summary.jsonl
      diff_dir  = <candidate>/../diff                 (sibling of harness under the candidate)
      parent    = <RUN>/candidates/<best>/harness     (frontier-best; else the base scaffold)
      trials    = <RUN>/candidates/<best>/eval        (the frontier-best's train-sample eval --
                    the eval the proposer analyzes). `best` is read from frontier _best.candidate.
    """
    run = Path(os.environ.get("MHO_RUN_DIR") or "").expanduser()
    if not run.is_dir():
        sys.exit(f"[propose] MHO_RUN_DIR not set or missing: {run}")
    # Relocated into candidates/ (the optimizer's single accessible root).
    frontier = run / "candidates" / "frontier.json"
    summary = run / "candidates" / "summary.json"
    diff_dir = candidate_dir.parent / "diff"
    best_tag = ""
    if frontier.is_file():
        try:
            best_tag = str(json.loads(frontier.read_text()).get("_best", {}).get("candidate", ""))
        except json.JSONDecodeError:
            pass
    # Parent = the frontier-best candidate's harness dir. The baseline (iter 0) never
    # materializes a harness (it is only an eval), so fall back to the base scaffold when the
    # best candidate has no harness yet.
    parent = (run / "candidates" / best_tag / "harness") if best_tag else Path("")
    if not parent or not (parent / "pyproject.toml").exists():
        parent = _original_base()
    cand_trials: list[Path] = []
    _tags = sorted((d for d in (run / "candidates").glob("candidate_*") if d.is_dir()),
                   key=lambda d: d.name, reverse=True)
    _ordered = [run / "candidates" / best_tag] if best_tag else []
    _ordered += [t for t in _tags if t.name != best_tag]
    for _c in _ordered:
        if len(cand_trials) >= _MAX_TRIAL_DIRS:
            break
        _src = _c / "eval_after" if (_c / "eval_after").is_dir() else _c / "eval_before"
        if _src.is_dir() and any(_src.glob("*/result.json")):
            cand_trials.append(_src)
    trials = cand_trials if cand_trials else Path("")
    return {"frontier": frontier, "summary": summary, "diff_dir": diff_dir,
            "parent": parent, "trials": trials}


def _stage_prior_results(dst: Path, paths: dict[str, Path]) -> None:
    """Copy prior eval results INTO the workspace (eval_prior/) so the proposer agent can
    open them as files (not only as inline prompt text). Stages: frontier.json,
    evolution_summary.jsonl, aggregated.jsonl, and a bounded sample of the raw trial dirs
    (result.json + the readable + token-sparse trajectory transcripts). No-op when the
    sources are absent (first iteration). Excluded from the candidate diff by /eval_prior/."""
    # VALIDATION ISOLATION: the optimizer only ever sees the TRAIN-SAMPLE eval (the proposer's
    # improvement signal). The val-split eval (eval-val/) is the held-out accept/reject gate and
    # MUST NEVER be staged/read by the proposer -- doing so would contaminate the accept decision
    # and overfit the val set. The trials are derived (paths["trials"]) ONLY from the train-sample eval,
    # so staging from it is safe; refuse any path that looks like a val eval just in case.
    prior = dst / "eval_prior"
    trials = paths["trials"]
    trial_srcs = trials if isinstance(trials, list) else ([trials] if trials and Path(trials).is_dir() else [])
    for _t in trial_srcs:
        if "evaluation" in _t.parts or "eval-val" in _t.parts:
            raise RuntimeError(
                "refusing to stage val-split trials into the proposer workspace: "
                f"{_t} (validation set must stay hidden from the optimizer)"
            )
    # NOTE: frontier.json + evolution_summary.jsonl are intentionally NOT staged into
    # eval_prior/ -- they live at the run root (<RUN>/frontier.json, <RUN>/evolution_summary.jsonl)
    # and the proposer workspace is a subdir of the run, so the agent reads them at ../../.
    prior.mkdir(parents=True, exist_ok=True)
    if trial_srcs:
        traj = prior / "trials"
        _n = 0
        for _src in trial_srcs:
            _cand = _src.parent.name
            for trial_dir in sorted(_src.glob("*")):
                if _n >= _MAX_TRIAL_DIRS:
                    break
                if not trial_dir.is_dir():
                    continue
                tgt = traj / f"{_cand}__{trial_dir.name}"
                tgt.mkdir(parents=True, exist_ok=True)
                for rel in ("result.json", "agent/mini-swe-agent.txt", "agent/mini-swe-agent.trajectory.json"):
                    f = trial_dir / rel
                    if f.is_file():
                        tf = tgt / rel
                        tf.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, tf)
                _n += 1
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


def _emit_diffs(dst: Path, paths: dict[str, Path]) -> None:
    """Two provenance patches per iteration: cumulative (base->candidate) + incremental
    (parent->candidate). Written OUTSIDE dst so they don't pollute the installable package or
    each other's diff. Loc: <candidate>/../diff (sibling of the harness dir)."""
    diff_dir = paths["diff_dir"]
    diff_dir.mkdir(parents=True, exist_ok=True)
    _write_diff(_original_base(), dst, diff_dir / "diff_from_base.patch")
    _write_diff(paths["parent"], dst, diff_dir / "diff_from_parent.patch")


def _build_instructions(instructions_md: Path, sibling_note: str = "") -> str:
    """The full proposer message, read verbatim from the kind's instructions.md. Prior results
    are MOUNTED under eval_prior/ in the workspace (see _stage_prior_results), so the message
    only points the agent at them -- nothing is inlined here."""
    text = instructions_md.read_text() if instructions_md.exists() else ""
    return text + sibling_note


# One direction per sibling (the loop runs --proposals siblings from the same evidence; without
# this they tend to make the same edit). Cycled when there are more siblings than directions.
SIBLING_FOCUS = (
    "the single most frequent failure mode in the trials",
    "a failure mode OTHER than the most frequent one (the obvious fix is covered by a sibling)",
    "efficiency: the same accuracy with fewer wasted turns / output tokens (e.g. stop earlier "
    "when the answer is settled, avoid repeated or redundant steps)",
    "a larger change to the scaffold's strategy (how it plans, checks its work, or decides "
    "when it is done) rather than a local fix",
)


def sibling_note(tag: str, sibling: str) -> str:
    """Extra instructions for sibling k of M (MHO_PROPOSE_SIBLING="k/M"); "" when M == 1."""
    try:
        k, m = (int(x) for x in sibling.split("/"))
    except ValueError:
        return ""
    if m <= 1:
        return ""
    focus = SIBLING_FOCUS[(k - 1) % len(SIBLING_FOCUS)]
    return (f"\n\n## Parallel proposals\nThis is proposal {k} of {m} written at the same time "
            f"from the same evidence; the best one that improves on the train sample is kept. "
            f"Make yours DIFFERENT from what the others are likely to try. Your focus: {focus}.\n"
            f"The sibling candidates (same tag with another letter) are being written right now: "
            f"do not read or copy them. Write only under {tag}/.\n")


def _run_openhands(workspace: Path, instructions: str, skill_dir: Path) -> None:
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
    skill = Skill.load(skill_dir / "SKILL.md")
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
    run = Path(os.environ.get("MHO_RUN_DIR") or "")
    tag = os.environ.get("MHO_TAG", "")
    kind = os.environ.get("MHO_PROPOSE_KIND", "harness")
    if not run.is_dir() or not tag:
        sys.exit("[propose] MHO_RUN_DIR (experiment root) and MHO_TAG are required")
    if kind not in PROPOSE_KINDS:
        sys.exit(f"[propose] unknown MHO_PROPOSE_KIND {kind!r} (choose from {PROPOSE_KINDS})")

    cand_root = run / "candidates"                # optimizer's single accessible root
    cand_unit = cand_root / tag                   # one dir per candidate: harness/ + model_specs/
    harness_dst = cand_unit / "harness"           # installable scaffold (kind == harness)
    spec_dst = cand_unit / "model_specs"          # what-the-model-should-learn specs (kind == model)
    paths = _derive_paths(harness_dst)
    parent = paths["parent"]

    # Workspace = candidates/ (the whole candidate archive). The optimizer thus sees EVERYTHING:
    # frontier.json, summary.json, every candidate's harness/, status.json (accepted/rejected),
    # eval_before/ + eval_after/ train-sample traces -- no eval_prior/ staging needed. The agent
    # edits <tag>/harness (harness) OR writes <tag>/model_specs (model) -- the only writable
    # output; it should treat other candidates + summary/frontier as read-only reference
    # (enforced soft by instructions; the proposal diff only counts harness/ or model_specs/).
    cand_root.mkdir(parents=True, exist_ok=True)
    cand_unit.mkdir(parents=True, exist_ok=True)
    if kind == "harness":
        _materialize_from(parent, harness_dst)    # accumulate: build on the frontier-best
    note = sibling_note(tag, os.environ.get("MHO_PROPOSE_SIBLING", "1/1"))
    if note:
        print(f"[propose] sibling {os.environ.get('MHO_PROPOSE_SIBLING')}: "
              f"{note.split('Your focus: ')[1].splitlines()[0]}", flush=True)
    _run_openhands(cand_root, _build_instructions(_instructions_md(kind), note), _skill_dir(kind))

    if kind == "model":
        from mho.optimizer import model_spec
        specs = sorted(p for p in spec_dst.rglob("*.md") if "eval_prior" not in str(p)) if spec_dst.is_dir() else []
        if not specs:
            print("[propose] ERROR: model proposer wrote no specs under " + str(spec_dst), file=sys.stderr)
            return 3
        bad: list[str] = []
        for f in specs:
            try:
                model_spec.validate_file(f)
            except (ValueError, OSError) as e:
                bad.append(str(e))
        if bad:
            print("[propose] ERROR: invalid model spec(s):\n  " + "\n  ".join(bad), file=sys.stderr)
            return 3
        print(f"[propose] model proposal: {len(specs)} valid model spec(s) -> {spec_dst}")
        return 0

    # kind == harness
    ok, reason = _validate_candidate(harness_dst, parent)
    _emit_diffs(harness_dst, paths)   # always emit diffs -- useful to inspect even a rejected candidate
    if not ok:
        print(f"[propose] ERROR: candidate invalid -- {reason} (diffs in the candidate diff dir)", file=sys.stderr)
        return 1
    if _diff_only_vcs(parent, harness_dst):
        print("[propose] ERROR: candidate is identical to the parent (agent made no code change)", file=sys.stderr)
        return 2
    rationale = harness_dst / "CANDIDATE.md"
    print(f"[propose] candidate ready at {harness_dst} ({reason})"
          + (f"; rationale: {rationale}" if rationale.exists() else "; no CANDIDATE.md written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
