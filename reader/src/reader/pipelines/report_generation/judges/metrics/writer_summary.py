"""Validation rules for ReportWriterFrontMatterOutput (D-H / D-S)."""

import re
from typing import Optional, Tuple

from reader.pipelines.report_generation.judges.metrics.common import (
    ValidationReport,
    run_checks,
    word_count,
    sentence_count,
)
from reader.pipelines.report_generation.report import ReportWriterFrontMatterOutput


# ============================================================================
# Constants
# ============================================================================

BANNED_TITLES = frozenset(
    {"technical report", "research summary", "weekly report", "overview", "report"}
)
VAGUE_WORDS = frozenset(
    {"some", "various", "misc", "general", "thoughts", "notes"}
)
GENERIC_PREFIXES = ("a study of", "an overview of")
HYPE_PHRASES = (
    "state-of-the-art",
    "breakthrough",
    "proves",
    "guarantees",
    "solves",
)
GENERIC_KEYWORDS = frozenset(
    {"ai", "ml", "deep learning", "paper", "survey", "method"}
)

# Bounds
TITLE_MIN_LEN = 5
TITLE_MAX_LEN = 120
SUMMARY_MIN_LEN = 40
SUMMARY_MAX_LEN = 1200
KEYWORDS_MIN_COUNT = 5
KEYWORDS_MAX_COUNT = 12
KEYWORD_MAX_LEN = 40
TITLE_MIN_WORDS = 4
TITLE_MAX_WORDS = 12
SUMMARY_MIN_SENTENCES = 3
SUMMARY_MAX_SENTENCES = 8
MULTIWORD_KEYWORDS_MIN = 2

# Alphanumeric check for keywords
_ALPHANUM_RE = re.compile(r"[a-zA-Z0-9]")


# ============================================================================
# Hard Rules (D-H3, D-H4, D-H6)
# ============================================================================


def _check_keywords_count(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-H3: Keywords count must be 5–12."""
    n = len(output.keywords or [])
    if not (KEYWORDS_MIN_COUNT <= n <= KEYWORDS_MAX_COUNT):
        return False, (
            "Keywords count must be 5–12",
            f"got {n} keywords (expected 5–12)",
        )
    return True, None


def _check_keywords_case_insensitive_unique(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-H4: Keywords must be case-insensitively unique."""
    kw = output.keywords or []
    seen: set[str] = set()
    for k in kw:
        if k is None:
            continue
        k_lower = k.strip().lower()
        if k_lower in seen:
            return False, (
                "Keywords must be case-insensitively unique",
                f"got duplicate: {k!r}",
            )
        seen.add(k_lower)
    return True, None


def _check_title_not_banned(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-H6: Title must not be in banned set."""
    t = (output.title or "").strip().lower()
    if t in BANNED_TITLES:
        return False, (
            "Title must not be in banned set (technical report, research summary, etc.)",
            f"got banned title: {output.title!r}",
        )
    return True, None


HARD_CHECKS = (
    _check_keywords_count,
    _check_keywords_case_insensitive_unique,
    _check_title_not_banned,
)


def hard_validate_writer_summary(
    output: ReportWriterFrontMatterOutput,
) -> ValidationReport:
    """Hard validation: score is 1 if all pass, else 0."""
    r = run_checks(output, HARD_CHECKS)
    return ValidationReport(score=1.0 if r.score == 1.0 else 0.0, reasons=r.reasons)


# ============================================================================
# Soft Rules (D-H1, D-H2, D-H5, D-S1–D-S5)
# ============================================================================


def _check_title_length(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-H1: Title must be 5–120 chars, non-empty after strip."""
    t = (output.title or "").strip()
    if not t:
        return False, (
            "Title must be 5–120 chars, non-empty after strip",
            "got empty",
        )
    n = len(t)
    if not (TITLE_MIN_LEN <= n <= TITLE_MAX_LEN):
        return False, (
            "Title must be 5–120 chars, non-empty after strip",
            f"got {n} chars (min {TITLE_MIN_LEN}, max {TITLE_MAX_LEN})",
        )
    return True, None


def _check_summary_length(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-H2: Summary must be 40–1200 chars, non-empty after strip."""
    s = (output.summary or "").strip()
    if not s:
        return False, (
            "Summary must be 40–1200 chars, non-empty after strip",
            "got empty",
        )
    n = len(s)
    if not (SUMMARY_MIN_LEN <= n <= SUMMARY_MAX_LEN):
        return False, (
            "Summary must be 40–1200 chars, non-empty after strip",
            f"got {n} chars (min {SUMMARY_MIN_LEN}, max {SUMMARY_MAX_LEN})",
        )
    return True, None


def _check_keyword_sanity(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-H5: Each keyword: non-empty, ≤40 chars, ≥1 alphanumeric."""
    kw = output.keywords or []
    for i, k in enumerate(kw):
        if k is None:
            continue
        k_stripped = (k or "").strip()
        if not k_stripped:
            return False, (
                "Each keyword: non-empty, ≤40 chars, ≥1 alphanumeric",
                f"got invalid at keywords[{i}]: empty",
            )
        if len(k_stripped) > KEYWORD_MAX_LEN:
            return False, (
                "Each keyword: non-empty, ≤40 chars, ≥1 alphanumeric",
                f"got invalid at keywords[{i}]: len {len(k_stripped)} > {KEYWORD_MAX_LEN}",
            )
        if not _ALPHANUM_RE.search(k_stripped):
            return False, (
                "Each keyword: non-empty, ≤40 chars, ≥1 alphanumeric",
                f"got invalid at keywords[{i}]: {k!r} (no alphanumeric)",
            )
    return True, None


def _check_title_specific(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-S1: Title must be specific; avoid vague words and generic prefixes."""
    t = (output.title or "").strip().lower()
    if not t:
        return True, None
    words = set(t.split())
    for vw in VAGUE_WORDS:
        if vw in words:
            return False, (
                "Title must be specific; avoid vague words and generic prefixes",
                f"contains vague word {vw!r}",
            )
    for prefix in GENERIC_PREFIXES:
        if t.startswith(prefix):
            return False, (
                "Title must be specific; avoid vague words and generic prefixes",
                f"generic prefix {prefix!r}",
            )
    return True, None


def _check_title_word_count(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-S2: Title word count prefer 4–12."""
    wc = word_count(output.title or "")
    if not (TITLE_MIN_WORDS <= wc <= TITLE_MAX_WORDS):
        return False, (
            "Title word count prefer 4–12",
            f"got {wc} words (prefer 4–12)",
        )
    return True, None


def _check_summary_sentence_count(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-S3: Summary prefer 3–8 sentences."""
    sc = sentence_count(output.summary or "")
    if not (SUMMARY_MIN_SENTENCES <= sc <= SUMMARY_MAX_SENTENCES):
        return False, (
            "Summary prefer 3–8 sentences",
            f"got {sc} sentences (prefer 3–8)",
        )
    return True, None


def _check_summary_no_hype(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-S4: Summary must avoid overclaiming/hype."""
    s = (output.summary or "").lower()
    for phrase in HYPE_PHRASES:
        if phrase in s:
            idx = s.find(phrase)
            snippet = (output.summary or "")[
                max(0, idx - 20) : idx + len(phrase) + 20
            ]
            return False, (
                "Summary must avoid overclaiming/hype",
                f"found hype phrase: {snippet!r}",
            )
    return True, None


def _check_keyword_quality(
    output: ReportWriterFrontMatterOutput,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """D-S5: Keywords: avoid generic; prefer ≥2 multiword."""
    kw = output.keywords or []
    generic_found: list[str] = []
    multiword_count = 0
    for k in kw:
        if k is None:
            continue
        k_lower = (k or "").strip().lower()
        if k_lower in GENERIC_KEYWORDS:
            generic_found.append(k)
        if " " in k or "-" in k:
            multiword_count += 1
    reasons: list[str] = []
    if generic_found:
        reasons.append(f"generic keyword(s): {generic_found[:3]}{'...' if len(generic_found) > 3 else ''}")
    if multiword_count < MULTIWORD_KEYWORDS_MIN:
        reasons.append(f"only {multiword_count} multiword (prefer ≥{MULTIWORD_KEYWORDS_MIN})")
    if reasons:
        return False, (
            "Keywords: avoid generic; prefer ≥2 multiword",
            "; ".join(reasons),
        )
    return True, None


SOFT_CHECKS = (
    _check_title_length,
    _check_summary_length,
    _check_keyword_sanity,
    _check_title_specific,
    _check_title_word_count,
    _check_summary_sentence_count,
    _check_summary_no_hype,
    _check_keyword_quality,
)


def soft_validate_writer_summary(
    output: ReportWriterFrontMatterOutput,
) -> ValidationReport:
    """Soft validation: runs all soft checks and returns average score."""
    return run_checks(output, SOFT_CHECKS)
