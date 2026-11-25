import difflib
import re

import pytest
import yaml

from harnessforge.bench import FEATURE_NAMES, builder


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return builder.build_all(tmp_path_factory.mktemp("variants"))


def _files(root):
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def test_strip_resolves_blocks_else_and_nesting():
    text = ("a\n# >>> feature:loop_warning\nb\n  # >>> feature:context_trim\n  c\n  # <<< feature:context_trim\n"
            "# --- else\nd\n# <<< feature:loop_warning\n# >>> feature:*\ne\n# <<< feature:*\n"
            "# >>> feature:raw_observation,context_trim\nf\n# <<< feature:raw_observation,context_trim\n")
    assert builder.strip(text, FEATURE_NAMES) == "a\nb\n  c\ne\nf\n"
    assert builder.strip(text, {"loop_warning"}) == "a\nb\ne\n"
    assert builder.strip(text, {"context_trim"}) == "a\nd\ne\nf\n"
    assert builder.strip(text, ()) == "a\nd\n"


@pytest.mark.parametrize("bad", ["# >>> feature:loop_warning\nx\n",
                                 "x\n# <<< feature:loop_warning\n",
                                 "# >>> feature:loop_warning\n# <<< feature:context_trim\n",
                                 "# >>> feature:nope\n# <<< feature:nope\n"])
def test_strip_rejects_bad_markers(bad):
    with pytest.raises(builder.MarkerError):
        builder.strip(bad, FEATURE_NAMES)


def test_every_feature_has_code_blocks():
    named = set().union(*(b.names for b in builder.source_blocks()))
    assert set(FEATURE_NAMES) <= named


def test_variants_are_clean_and_valid(built):
    assert set(built) == {"gold", "start", *(f"ablate_{f}" for f in FEATURE_NAMES)}
    for name, root in built.items():
        files = _files(root)
        assert "LICENSE" in files and "src/micro_swe/agents/default.py" in files
        for rel, text in files.items():
            assert not re.search(r"# (>>>|<<<|---) |@feature-file", text), (name, rel)
            if rel.endswith(".py"):
                compile(text, rel, "exec")
        cfg = yaml.safe_load(files["src/micro_swe/config/mini.yaml"])
        assert {"system_template", "instance_template"} <= set(cfg["agent"])


def test_start_has_no_trace_of_any_feature(built):
    files = _files(built["start"])
    assert "src/micro_swe/features.py" not in files
    blob = "\n".join(files.values())
    assert not re.search(r"\bfeatures?\b\.(on|flags|configure)|import features|^features:", blob, re.M)
    for name in FEATURE_NAMES:
        assert name not in blob, name
    for feature, sigs in SIGNATURES.items():
        for sig in sigs:
            assert sig not in blob, (feature, sig)


# Distinctive strings of each feature's code/messages: in gold, never in start.
SIGNATURES = {
    "output_truncation": ["MAX_OUTPUT_CHARS", "characters elided", "_truncate_output"],
    "timeout_guidance": ["TIMEOUT_NOTE", "timed out after", "nohup"],
    "persistent_cwd": ["_CWD_MARK", "trap ", "working directory carries over"],
    "length_recovery": ["length_recovery_factor", "_recover_from_length_cutoff", "finish_reason != \"length\""],
    "loop_warning": ["LOOP_WARNING", "_repeat_count", "same output each time"],
    "format_streak_trim": ["_drop_stale_format_errors", "latest rejection"],
    "missing_command_repair": ["COMMAND_REPAIR_PROMPT", "_repair_missing_command", "non-empty command"],
    "submit_confirmation": ["SUBMIT_CHECKLIST", "_submit_confirmed", "issue the submit command again"],
    "context_trim": ["_stub_old_observations", "context_keep_recent", "older output shortened"],
    "raw_observation": ["features.raw_observation", "<output>\n    {{ output.output }}"],
}


def test_signatures_are_in_gold(built):
    blob = "\n".join(_files(built["gold"]).values())
    for feature, sigs in SIGNATURES.items():
        for sig in sigs:
            assert sig in blob, (feature, sig)


@pytest.mark.parametrize("feature", FEATURE_NAMES)
def test_ablation_differs_from_gold_only_in_that_feature(built, feature):
    gold, abl = _files(built["gold"]), _files(built[f"ablate_{feature}"])
    allowed = set()
    for block in builder.source_blocks():
        if feature in block.names:
            allowed |= {l.strip() for l in block.on_lines + block.off_lines}
    changed = 0
    for rel in sorted(set(gold) | set(abl)):
        a, b = gold.get(rel, "").splitlines(), abl.get(rel, "").splitlines()
        for d in difflib.unified_diff(a, b, lineterm="", n=0):
            if d.startswith(("---", "+++", "@@")):
                continue
            changed += 1
            assert d[1:].strip() in allowed, (feature, rel, d)
    assert changed > 0
