"""HMMT math verifier (standalone tests/ form).

HMMT answers are exact expressions (fractions like ``\\frac{1}{576}`` or
radicals like ``8\\sqrt{10}``), so this grades with ``math_verify`` (the same
lib MathArena uses), falling back to sympy structural equality and then to a
normalized string compare, so ``1/576``, ``\\frac{1}{576}``, ``\\dfrac{1}{576}``
all match. The expected answer is injected by the adapter; the candidate answer
is parsed from the agent's final ``Answer: $A`` line.
"""

import re
import sys
from pathlib import Path

EXPECTED_ANSWER = {answer}

# DAPO answer protocol marker: "Answer: $Answer" on the last substantive line.
ANSWER_RE = re.compile(r"(?:^|\n)\s*Answer:\s*(.+?)\s*$", re.MULTILINE)


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _extract_answer(trajectory: str) -> str:
    matches = ANSWER_RE.findall(trajectory)
    if not matches:
        return ""
    return matches[-1].lstrip("$").strip()


def _grade(cand: str, exp: str) -> bool:
    # 1) math_verify expression equivalence (LaTeX/plain fraction/radical).
    try:
        from math_verify import parse, verify
        g = parse(exp)
        c = parse(cand)
        if verify(g, c):
            return True
    except Exception:
        pass
    # 2) sympy structural equality (fractions/radicals if math_verify missing).
    try:
        import sympy as sp
        from sympy.parsing.latex import parse_latex
        gg = parse_latex(exp)
        cc = parse_latex(cand)
        if sp.simplify(gg - cc) == 0:
            return True
    except Exception:
        pass
    # 3) normalized string compare (last resort).
    return _norm(cand) == _norm(exp)


def main() -> int:
    candidates = [
        Path("/workspace/output.txt"),
        Path("/workspace/answer.txt"),
    ]
    trajectory = ""
    for path in candidates:
        if path.is_file():
            trajectory = path.read_text()
            break

    cand = _extract_answer(trajectory)
    if not cand:
        print("X no Answer: $A line found in agent output")
        return 1

    ok = _grade(cand, EXPECTED_ANSWER or "")
    print(
        ("OK" if ok else "X")
        + f" expected={EXPECTED_ANSWER!r} got={cand[:120]!r}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())