"""Judge modules for evaluating LLM outputs

Shared utilities used by hard_rules.py and soft_rules.py
"""

import re
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from schemas.cluster_response import ClusterReport

# Regex pattern for word counting
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


@dataclass(frozen=True)
class ValidationReport:
    """Validation result with score and reasons"""
    score: float
    reasons: str


CheckFn = Callable[[ClusterReport], Tuple[bool, str]]  # (pass?, reason-if-fail)


def word_count(s: str) -> int:
    """Count words in a string"""
    return len(_WORD_RE.findall((s or "").strip()))


def run_checks(report: ClusterReport, checks: Sequence[CheckFn]) -> ValidationReport:
    """Run a sequence of checks and return ValidationReport"""
    passed = 0
    lines: list[str] = []
    for fn in checks:
        ok, msg = fn(report)
        if ok:
            passed += 1
            lines.append(f"{fn.__name__}: pass")
        else:
            lines.append(f"{fn.__name__}: error: {msg}")
    score = passed / max(1, len(checks))
    return ValidationReport(score=score, reasons="\n".join(lines))
