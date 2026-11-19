"""The ten features of the gold harness: what they do, where they hook in, and where they come from.

`kind` says what the feature is expected to buy: "accuracy" (more tasks solved) or "efficiency"
(fewer tokens / less wall time at equal accuracy). `source` is "generic" (a standard agent
scaffold feature) or the tag of the candidate our optimizer accepted on tmax (Qwen3.5-4B,
harness-only search) that the feature is modelled on.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    title: str
    hook: str            # file(s)/function(s) of micro-swe-agent the code lives in
    behaviour: str
    why: str
    probe: str           # what the behavioural probe checks
    source: str
    kind: str            # "accuracy" | "efficiency"
    family: str          # the task family built to exercise it


FEATURES: tuple[Feature, ...] = (
    Feature(
        "output_truncation", "Long outputs shown as head + tail",
        "tools/bash.py: format_toolcall_observation_messages",
        "A command output longer than 10,000 chars is shown as its first and last 5,000 chars "
        "with a note saying how much was elided and how to see the rest.",
        "One huge log otherwise fills (or overflows) the context window; a context overflow aborts "
        "the run. Build and test logs keep the useful parts at the start and the end.",
        "Run a command printing ~60k chars with markers at the start and end; the next observation "
        "the model sees is <= 20k chars and contains both markers.",
        "generic (micro-swe-agent's own 10k head/tail template)", "accuracy", "huge_log"),
    Feature(
        "timeout_guidance", "Explained command timeouts",
        "environments/local.py: LocalEnvironment.execute",
        "When a command hits the environment timeout the model sees its partial output plus a note "
        "that it timed out and was killed, and how to background long-running commands.",
        "Without it a timed-out command looks like a bare failure (returncode -1); small models "
        "re-run the same foreground server or build until the wall clock runs out.",
        "Run `echo MARK; sleep 20` with a 2 s timeout; the next observation keeps MARK and says the "
        "command timed out / was killed (and a normal command's observation does not).",
        "generic (mini-swe-agent's timeout_template)", "accuracy", "background_service"),
    Feature(
        "persistent_cwd", "Working directory carries over",
        "environments/local.py: LocalEnvironment.execute (+ the prompt's subshell rule in "
        "config/mini.yaml)",
        "The environment remembers the shell's final directory and starts the next command there; "
        "the prompt says so.",
        "Models habitually `cd` once and then use relative paths; in a fresh subshell per command "
        "those paths silently point elsewhere.",
        "Run `mkdir -p probe_sub && cd probe_sub`, then `pwd`; the second observation contains "
        "probe_sub.",
        "generic", "accuracy", "deep_dirs"),
    Feature(
        "length_recovery", "Re-query when cut off at max_tokens",
        "models/litellm.py: Model.query",
        "A response with finish_reason=length and no tool call is re-queried once with a doubled "
        "max_tokens (bounded) instead of becoming a format error.",
        "Truncated responses were the main format-error type on tmax (16% of trials); three in a "
        "row kill the run with RepeatedFormatError.",
        "The scripted model is cut off (finish_reason=length, no tool call) at max_tokens=1000; "
        "the harness re-queries with a larger max_tokens or tells the model it hit the length "
        "limit.",
        "candidate_1_0_1b (length-truncation recovery)", "efficiency", "bulk_write"),
    Feature(
        "loop_warning", "Repeated-command warning",
        "agents/default.py: Agent.execute_actions",
        "After the same command with the same output 3 times in a row, the observation carries a "
        "note that repeating it gives no new information and suggests changing approach.",
        "Byte-identical re-issue loops were 19% of tmax trials and 59% of output tokens, all at 0 "
        "reward.",
        "Run the same command 5 times; a later repetition's observation differs from the first "
        "one's (or the sampling parameters change).",
        "candidate_4_1_5a / 7_3_1b / 18_8_5c (loop escapes)", "efficiency", "stuck_poll"),
    Feature(
        "format_streak_trim", "Only the latest format error in context",
        "agents/default.py: Agent.query (messages sent to the model)",
        "During consecutive format errors, only the most recent rejection message is sent to the "
        "model; earlier ones are dropped from the prompt (kept in the trajectory).",
        "A wall of identical rejections makes a collapsed small model repeat the collapse.",
        "Two malformed tool calls in a row, then a good one: the prompt after the second error is "
        "no longer than the prompt after the first.",
        "candidate_28_12_5a (format-streak context trim)", "accuracy", "json_escape"),
    Feature(
        "missing_command_repair", "Repair re-query for a call without a command",
        "models/litellm.py: Model.query",
        "A bash tool call whose `command` is missing or empty is re-asked once with an explicit "
        "instruction (not persisted); if the repaired call is valid the turn proceeds with no "
        "format error in the history.",
        "The `{}` empty-arguments collapse was the most frequent RepeatedFormatError cause on tmax "
        "(14.5% of trials).",
        "One `{}` call followed by a good one: the repair query carries an instruction about the "
        "command, and no trace of the bad call is in the following prompt.",
        "candidate_22_10_2a (missing-command repair re-query)", "accuracy", "json_escape"),
    Feature(
        "submit_confirmation", "Check-before-submit",
        "agents/default.py: Agent.execute_actions",
        "The first submit command is held back once with a short checklist (deliverables exist, "
        "result was run after the last edit, services still up); the second submit ends the run.",
        "Submitted-but-wrong was the largest 0-reward bucket on tmax (decide-when-done procedure).",
        "Submit once; the run continues, and a second submit ends it with exit status Submitted.",
        "candidate_6_2_3d (decide-when-done working procedure)", "accuracy", "verify_first"),
    Feature(
        "context_trim", "Old long observations stubbed",
        "agents/default.py: Agent.query (messages sent to the model)",
        "Command results older than the 6 most recent ones and longer than 1,200 chars are "
        "shortened to their first 400 chars plus a note before each query.",
        "Long sessions otherwise grow the prompt until it no longer fits.",
        "30 commands printing 4k chars each: the final prompt is under half the total output size "
        "and the newest observation is intact.",
        "generic (stale-observation trimming)", "efficiency", "many_files"),
    Feature(
        "raw_observation", "Plain-text tool results",
        "config/mini.yaml: model.observation_template",
        "Command output is shown verbatim inside <returncode>/<output> tags instead of as a "
        "JSON-escaped string.",
        "The 4B model misread the JSON wrapper (\\n, \\\", \\u003c escapes) as file content and "
        "copied escapes into deliverables (~79 tmax runs, 30% vs 40% solve rate).",
        "Print a line with a quote, a backslash and a newline; the observation contains those "
        "characters unescaped.",
        "candidate_30_14_1b (tool-result wrapper annotation)", "accuracy", "exact_text"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}


def feature_table_markdown() -> str:
    """The feature table as Markdown (used by docs/bench.md)."""
    rows = ["| # | feature | hook | behaviour | why it should help | probe | source | kind |",
            "|---|---|---|---|---|---|---|---|"]
    for i, f in enumerate(FEATURES, 1):
        rows.append(f"| {i} | `{f.name}` | {f.hook} | {f.behaviour} | {f.why} | {f.probe} | "
                    f"{f.source} | {f.kind} |")
    return "\n".join(rows)
