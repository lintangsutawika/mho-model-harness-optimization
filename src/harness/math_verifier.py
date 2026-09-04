"""Harbor verifier for DAPO-Math math tasks: sympy-normalized exact-answer match.

Grades a trial by extracting the agent's final ``Answer: <answer>`` line (the
DAPO instruction mandates that format) from the agent's trajectory, then
normalizing both the candidate and the task's ground truth with sympy and
returning reward 1.0 iff they are mathematically equal, else 0.0.

Because every DAPO-Math-17k ground truth is a signed integer regex ``-?\d+``,
in practice this is integer equality after normalizing lexical variants
(``+5``, ``5.0``, `` 5 `` -> 5). Sympy parsing also tolerates fractions and
radical forms if a future dataset uses them.

Attached to a trial via ``verifier.import_path`` in the harbor trial config
(see harbor_trial_config/default.yaml). Subclasses harbor's
:class:`~harbor.verifier.base.BaseVerifier`; runs in the harbor process, not
the sandbox.
"""
from __future__ import annotations

import json
import re
from typing import Any, override

from loguru import logger

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

# DAPO's answer protocol: the final line of the response is "Answer: $Answer".
# We take the LAST line whose payload begins with an optional "Answer:" label,
# so a long CoT message is graded on its final declared answer. The label may
# be markdown-ish ({\it Answer:}); the value may be wrapped in $..$ / $$..$$.
_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:\\?\(?\\?\{?)?(?:Answer|answer|ANSWER)\s*:\s*\$?(.+?)\$?\s*$"
)
# Characters that appear in MATH_v2 answer values: digits, sign, punctuation,
# fraction/radical operators, latin letters.
_ANSWER_CHARS = set("0123456789+-/.,(){}[]^\\%$* ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _strip_math_wrappers(payload: str) -> str:
    """Remove surrounding $ / $$ / \[ \] wrappers."""
    payload = payload.strip()
    while len(payload) >= 2 and (
        (payload[0] == "$" and payload[-1] == "$")
        or (payload[0] == "\\" and payload[1] == "[")
    ):
        if payload[0] == "$":
            payload = payload[1:-1].strip()
        else:
            payload = payload[2:-2].strip()
    return payload


def _extract_answer(text: str) -> str | None:
    """Return the last ``Answer: <...>`` payload in ``text`` (or None).

    Line-oriented: keeps the payload of the last line that declares an answer.
    Falls back to a bare terse trailing value line if the model omitted an
    explicit ``Answer:`` label.
    """
    best: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _ANSWER_LINE_RE.match(line)
        if m:
            payload = _strip_math_wrappers(m.group(1).strip())
            if payload:
                best = payload
            continue
        # Fallback: a terse trailing line that looks like a bare answer value
        # (short, mostly math characters) — remember it only until a labelled
        # answer appears.
        stripped = line.rstrip(".")
        if best is None and len(stripped) <= 64 and stripped and all(
            c in _ANSWER_CHARS for c in stripped
        ):
            best = _strip_math_wrappers(stripped)
    return best


def _parse_number(text: str) -> Any | None:
    """Parse ``text`` into a sympy expression, or None if unparseable.

    Handles plain arithmetic (``+5``, ``5.0``, ``3/2``) and the two LaTeX forms
    seen in MATH answers: ``\\frac{a}{b}`` and ``\\sqrt{n}``. DAPO-Math-17k
    answers are signed integers, so this is defensive breadth, not the hot path.
    """
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
    text = text.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\,", "")
    if not text:
        return None
    try:
        import sympy as sp

        return sp.sympify(text, evaluate=False)
    except Exception:
        return None


def _answers_equal(candidate: str, ground_truth: str) -> bool:
    """Sympy-normalized exact match between two answer strings."""
    cand = _parse_number(candidate)
    truth = _parse_number(ground_truth)
    if cand is None or truth is None:
        # Fall back to trimmed string equality if either side is unparseable.
        return candidate.strip() == ground_truth.strip()
    try:
        import sympy as sp

        return sp.simplify(cand - truth) == 0
    except Exception:
        return candidate.strip() == ground_truth.strip()


def final_agent_message(trajectory_path) -> str | None:
    """The last agent-authored text message from an ATIF trajectory file."""
    try:
        data = json.loads(trajectory_path.read_text())
    except Exception:
        return None
    # ATIF: data["steps"] -> each has source/message. Prefer the final agent
    # message; fall back to the last textual message of any kind.
    steps = data.get("steps") or []
    for step in reversed(steps):
        src = step.get("source") or ""
        msg = step.get("message")
        if "agent" in str(src).lower() and isinstance(msg, str):
            return msg
    for step in reversed(steps):
        msg = step.get("message")
        if isinstance(msg, str):
            return msg
    return None


class MathVerifier(BaseVerifier):
    """Grade a DAPO math trial by sympy-normalized answer equality."""

    #: filenames of the ATIF trajectory written by mini-swe-agent under the
    #: trial's agent-artifacts dir.
    TRAJECTORY_FILENAMES = ("trajectory.json", "agent_trajectory.json")

    @override
    async def verify(self) -> VerifierResult:
        ground_truth = self._ground_truth()
        candidate = self._extract_candidate()
        if ground_truth is None:
            self.logger.error(f"no ground truth in task metadata: {self.task.paths.task_dir}")
            return VerifierResult(rewards={"reward": 0.0})
        if candidate is None:
            self.logger.warning("no 'Answer:' line found in agent trajectory -> reward 0")
            return VerifierResult(rewards={"reward": 0.0})
        correct = _answers_equal(candidate, ground_truth)
        self.logger.info(
            f"math verifier: candidate={candidate!r} truth={ground_truth!r} correct={correct}"
        )
        return VerifierResult(rewards={"reward": 1.0 if correct else 0.0})

    def _ground_truth(self) -> str | None:
        """Ground-truth answer from the task's ``[metadata]`` (task.toml)."""
        md = self.task.config.metadata or {}
        gt = md.get("answer") or md.get("ground_truth")
        return None if gt is None else str(gt)

    def _extract_candidate(self) -> str | None:
        """Pull the agent's final message from the trial's ATIF trajectory."""
        for fname in self.TRAJECTORY_FILENAMES:
            traj = self.trial_paths.agent_dir / fname
            if not traj.exists():
                continue
            msg = final_agent_message(traj)
            if msg:
                return _extract_answer(msg)
        self.logger.warning(
            f"no trajectory found under {self.trial_paths.agent_dir} "
            f"(tried {self.TRAJECTORY_FILENAMES})"
        )
        return None