"""DAPO math verifier (standalone tests/ form).

Mirrors the grading harness.math_verifier applies in-process: read the agent's
final "Answer: $A" line from the task output and compare it (normalized) to the
expected ground-truth answer. In the real pipeline this verifier is run by
harness.math_verifier directly from the trajectory; this file / test.sh supply
a self-contained answer-string check wherever the tasks/ are executed.
"""

import re
import sys
from pathlib import Path

EXPECTED_ANSWER = {answer}

# DAPO answer protocol marker: "Answer: $Answer" on the last substantive line.
ANSWER_RE = re.compile(r"(?:^|\n)\s*Answer:\s*(.+?)\s*$", re.MULTILINE)


def _normalize(text: str) -> str:
    """Normalize an answer string for exact comparison."""
    return " ".join(str(text).strip().lower().split())


def _extract_answer(trajectory: str) -> str:
    """Find the final 'Answer: $A' value in the agent output."""
    matches = ANSWER_RE.findall(trajectory)
    if not matches:
        return ""
    return matches[-1].lstrip("$").strip()


def main() -> int:
    # The agent's trajectory is placed at /workspace/output.txt by the harness;
    # fall back to any *.txt in /workspace when it is absent.
    candidates = [
        Path("/workspace/output.txt"),
        Path("/workspace/answer.txt"),
    ]
    trajectory = ""
    for path in candidates:
        if path.is_file():
            trajectory = path.read_text()
            break

    got = _extract_answer(trajectory)
    if not got:
        print("X no Answer: $A line found in agent output")
        return 1

    expected = _normalize(EXPECTED_ANSWER or "")
    ok = _normalize(got) == expected
    print(
        ("OK" if ok else "X")
        + f" expected={EXPECTED_ANSWER!r} got={got!r}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())