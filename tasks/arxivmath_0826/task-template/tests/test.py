"""ArXivMath math verifier (standalone tests/ form).

ArXivMath answers are exact expressions (fractions like ``\\frac{1}{576}`` or
radicals like ``8\\sqrt{10}``), so this grades with ``math_verify`` (the same
lib MathArena uses), falling back to sympy structural equality and then to a
normalized string compare, so ``1/576``, ``\\frac{1}{576}``, ``\\dfrac{1}{576}``
all match. The expected answer is injected by the adapter; the candidate answer
is read directly from /workspace/output.txt (the agent writes it there).
"""

import sys
from pathlib import Path

EXPECTED_ANSWER = {answer}

def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


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
    # The agent is instructed to write its final answer to /workspace/output.txt.
    out = Path("/workspace/output.txt")
    if not out.is_file():
        print("X no /workspace/output.txt written by agent")
        return 1
    cand = out.read_text().strip()   # agent writes the answer directly here
    if not cand:
        print("X /workspace/output.txt is empty")
        return 1

    ok = _grade(cand, EXPECTED_ANSWER or "")
    print(
        ("OK" if ok else "X")
        + f" expected={EXPECTED_ANSWER!r} got={cand[:120]!r}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
