"""Validation rules for ReportWriterSectionOutput (W2)."""

import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from reader.pipelines.report_generation.judges.metrics.common import (
    ValidationReport,
    run_checks,
    word_count,
)
from reader.pipelines.report_generation.report import ReportWriterSectionOutput

SECTION_TEXT_MIN_CHARS = 80
VALID_CONFIDENCE = frozenset({"high", "medium", "low"})

# Citation format: [ ... ] spans
_CITATION_RE = re.compile(r"\[[^\]]+\]")

# Raw ID patterns to reject outside citation tokens (W2-H4)
_RAW_ID_PATTERNS = [
    re.compile(r"paper_id\s*=", re.I),
    re.compile(r"report_id\s*=", re.I),
    re.compile(r"\bP\d+\b"),
    re.compile(r"\bR\d+\b"),
]

# Over-claiming phrases (W2-S2)
OVER_CLAIMING_PHRASES = (
    "sota",
    "state-of-the-art",
    "state of the art",
    "beats",
    "outperforms",
    "breakthrough",
    "proves",
    "guarantees",
)


class WriterWritingJudgeInput(BaseModel):
    """Judge input bundling output and validation context."""

    output: ReportWriterSectionOutput
    outline_item: str = ""
    allowed_citations: List[str] = Field(default_factory=list)


def _get_output(j: WriterWritingJudgeInput) -> ReportWriterSectionOutput:
    return j.output


def _allowed_citations_set(j: WriterWritingJudgeInput) -> frozenset[str]:
    return frozenset(j.allowed_citations)


def _extract_citation_tokens(text: str) -> List[str]:
    """Extract [ ... ] spans from text."""
    return _CITATION_RE.findall(text)


def _section_name_aligns_with_outline(section_name: str, outline_item: str) -> bool:
    """Check if section_name aligns with outline_item (substring, token overlap, or equality)."""
    sn = (section_name or "").strip().lower()
    oi = (outline_item or "").strip().lower()
    if not sn or not oi:
        return True  # Skip if either empty
    if sn == oi:
        return True
    if sn in oi or oi in sn:
        return True
    # Token overlap: meaningful tokens (len > 1, not just punctuation)
    sn_tokens = {t for t in re.findall(r"\b[\w'-]+\b", sn) if len(t) > 1}
    oi_tokens = {t for t in re.findall(r"\b[\w'-]+\b", oi) if len(t) > 1}
    overlap = sn_tokens & oi_tokens
    return len(overlap) >= 1


# ============================================================================
# Hard Rules (W2-H1 through W2-H5)
# ============================================================================


def _check_section_name_aligns(j: WriterWritingJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W2-H1: section_name aligns with outline_item."""
    out = j.output
    if not _section_name_aligns_with_outline(out.section_name, j.outline_item):
        return False, (
            "section_name must align with outline_item (substring/token overlap/equality)",
            f"section_name {out.section_name!r} does not align with outline_item {j.outline_item!r}",
        )
    return True, None


def _check_section_text_non_empty(j: WriterWritingJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W2-H2: section_text > 80 chars after strip."""
    text = (j.output.section_text or "").strip()
    if len(text) <= SECTION_TEXT_MIN_CHARS:
        return False, (
            "section_text must be non-empty (> 80 chars after strip)",
            f"got {len(text)} chars (min 80)",
        )
    return True, None


def _check_citation_allowlist(j: WriterWritingJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W2-H3: Every citation token must be in allowed_citations."""
    text = j.output.section_text or ""
    tokens = _extract_citation_tokens(text)
    allowed = _allowed_citations_set(j)
    for token in tokens:
        if token not in allowed:
            return False, (
                "Every citation token in section_text must be in allowed_citations",
                f"citation {token!r} not in allowed_citations",
            )
    return True, None


def _check_no_raw_unknown_ids(j: WriterWritingJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W2-H4: No raw unknown IDs (paper_id=, report_id=, P123, R7) outside citation tokens."""
    text = j.output.section_text or ""
    citation_spans = [(m.start(), m.end()) for m in _CITATION_RE.finditer(text)]

    def _inside_citation(pos: int) -> bool:
        for start, end in citation_spans:
            if start <= pos < end:
                return True
        return False

    for pattern in _RAW_ID_PATTERNS:
        for m in pattern.finditer(text):
            if not _inside_citation(m.start()):
                return False, (
                    "section_text must not contain raw unknown IDs (paper_id=, report_id=, P123, R7) outside citation tokens",
                    f"found raw ID pattern at position {m.start()}",
                )
    return True, None


def _check_confidence_valid(j: WriterWritingJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W2-H5: confidence non-empty, all in {high,medium,low}, recommended len==1."""
    conf = j.output.confidence or []
    if not conf:
        return False, (
            "confidence must be non-empty, all entries in {high,medium,low}, recommended len==1",
            "got empty confidence list",
        )
    for c in conf:
        val = (c or "").strip().lower()
        if val not in VALID_CONFIDENCE:
            return False, (
                "confidence must be non-empty, all entries in {high,medium,low}, recommended len==1",
                f"got {c!r}",
            )
    if len(conf) != 1:
        return False, (
            "confidence must be non-empty, all entries in {high,medium,low}, recommended len==1",
            f"got len {len(conf)} (recommend 1)",
        )
    return True, None


HARD_CHECKS = (
    _check_section_name_aligns,
    _check_section_text_non_empty,
    _check_citation_allowlist,
    _check_no_raw_unknown_ids,
    _check_confidence_valid,
)


def hard_validate_writer_writing(j: WriterWritingJudgeInput) -> ValidationReport:
    """Hard validation: score is 1 if all pass, else 0."""
    r = run_checks(j, HARD_CHECKS)
    return ValidationReport(score=1.0 if r.score == 1.0 else 0.0, reasons=r.reasons)


# ============================================================================
# Soft Rules (W2-S1, W2-S2, W2-S5)
# ============================================================================


def _check_citation_density(j: WriterWritingJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W2-S1: Soft target ≥1 allowed citation per paragraph."""
    text = j.output.section_text or ""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return True, None
    tokens = _extract_citation_tokens(text)
    allowed = _allowed_citations_set(j)
    valid_citations = [t for t in tokens if t in allowed]
    n_para = len(paragraphs)
    n_cite = len(valid_citations)
    if n_cite < n_para:
        return False, (
            "Soft target: ≥1 allowed citation per paragraph",
            f"got {n_cite} citations in {n_para} paragraphs",
        )
    return True, None


def _check_no_over_claiming(j: WriterWritingJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W2-S2: Penalize over-claiming language."""
    text = (j.output.section_text or "").lower()
    for phrase in OVER_CLAIMING_PHRASES:
        if phrase in text:
            idx = text.find(phrase)
            snippet = (j.output.section_text or "")[max(0, idx - 20) : idx + len(phrase) + 20]
            return False, (
                "Penalize over-claiming language (SOTA, beats, outperforms, etc.)",
                f"found over-claiming at: {snippet!r}",
            )
    return True, None


def _check_word_count_soft(j: WriterWritingJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W2-S5: Word count soft - warn if exceeding by small margin. Use 500 as soft limit."""
    text = (j.output.section_text or "").strip()
    wc = word_count(text)
    soft_limit = 500
    if wc > soft_limit:
        return False, (
            "Word count soft: warn if exceeding by small margin",
            f"got {wc} words (soft limit {soft_limit})",
        )
    return True, None


SOFT_CHECKS = (
    _check_citation_density,
    _check_no_over_claiming,
    _check_word_count_soft,
)


def soft_validate_writer_writing(j: WriterWritingJudgeInput) -> ValidationReport:
    """Soft validation: runs all soft checks and returns average score."""
    return run_checks(j, SOFT_CHECKS)
