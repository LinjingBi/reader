"""Shared validation utilities for report generation judges."""

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

# Question starters (case-insensitive)
QUESTION_STARTERS = frozenset(
    w.lower()
    for w in (
        "What",
        "Which",
        "How",
        "Why",
        "When",
        "Where",
        "Is",
        "Are",
        "Do",
        "Does",
        "Can",
        "Should",
    )
)


@dataclass(frozen=True)
class ValidationReport:
    """Validation result with score and reasons"""

    score: float
    reasons: List[Tuple[str, str]]  # [(rule_declaration, received_fact), ...]


T = TypeVar("T")


def word_count(s: str) -> int:
    """Count words in a string"""
    return len(_WORD_RE.findall((s or "").strip()))


def run_checks(output: T, checks: Sequence[Callable[[T], Tuple[bool, Optional[Tuple[str, str]]]]]) -> ValidationReport:
    """Run a sequence of checks and return ValidationReport"""
    passed = 0
    reason_tuples: List[Tuple[str, str]] = []
    for fn in checks:
        ok, pair = fn(output)
        if ok:
            passed += 1
        elif pair is not None:
            reason_tuples.append(pair)
    score = passed / max(1, len(checks))
    return ValidationReport(score=score, reasons=reason_tuples)


def is_question(s: str) -> bool:
    """Check if string is phrased as a question"""
    t = (s or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    first_word = t.split()[0].lower() if t.split() else ""
    return first_word in QUESTION_STARTERS
