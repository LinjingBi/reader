"""Hard validation rules for cluster reports

Hard rules are must-pass checks that result in score 0.0 if any fail.
"""

import re
from typing import Optional, Set, Tuple

from pydantic import ValidationError

from schemas.cluster_response import (
    ClusterReport,
    TITLE_MAX_WORDS,
    ABOUT_MIN_WORDS,
    ABOUT_MAX_WORDS,
    WHY_MIN_WORDS,
    WHY_MAX_WORDS,
    KEYWORDS_MIN_ITEMS,
    KEYWORDS_MAX_ITEMS,
    KEYWORD_MIN_WORDS,
    KEYWORD_MAX_WORDS,
)
from judges import ValidationReport, CheckFn, word_count, run_checks

# Regex pattern for citation matching
_CITATION_RE = re.compile(r"\[[^\[\]]+\]")  # basic [paper_id] matcher


def tag_word_count(tag: str) -> int:
    """Count words separated by whitespace (after lowering/stripping)"""
    return len([w for w in (tag or "").strip().split() if w])


def is_title_case_no_colon(title: str) -> bool:
    """Check if title is Title Case and contains no colon"""
    t = (title or "").strip()
    if ":" in t:
        return False
    return t == t.title()


def has_inline_citation(s: str) -> bool:
    """Check if string contains inline citation like [paper_id]"""
    return bool(_CITATION_RE.search(s or ""))


# ----------------------------
# Parsing helpers
# ----------------------------

def parse_cluster_report(json_text: str) -> ClusterReport:
    """Parse LLM JSON into a typed ClusterReport. Raises ValidationError if shape/type mismatches."""
    return ClusterReport.model_validate_json(json_text)


def try_parse_cluster_report(json_text: str) -> Tuple[ClusterReport | None, ValidationError | None]:
    """Best-effort parse: returns (report, error)."""
    try:
        return parse_cluster_report(json_text), None
    except ValidationError as e:
        return None, e


# ----------------------------
# Hard validation checks
# ----------------------------

def check_title(report: ClusterReport) -> Tuple[bool, str]:
    """Check title word count and format"""
    v = (report.title or "").strip()
    wc = word_count(v)
    if wc < 1 or wc > TITLE_MAX_WORDS:
        return False, f"title must be 1–{TITLE_MAX_WORDS} words, got {wc}"
    if not is_title_case_no_colon(v):
        return False, "title must be Title Case and contain no colon"
    return True, ""


def check_about(report: ClusterReport) -> Tuple[bool, str]:
    """Check what_this_cluster_is_about word count, citations, and terminology"""
    v = (report.what_this_cluster_is_about or "").strip()
    wc = word_count(v)
    if not (ABOUT_MIN_WORDS <= wc <= ABOUT_MAX_WORDS):
        return False, f"what_this_cluster_is_about must be {ABOUT_MIN_WORDS}–{ABOUT_MAX_WORDS} words, got {wc}"
    if not has_inline_citation(v):
        return False, "what_this_cluster_is_about must include at least one inline citation like [paper_id]"
    if "cluster" in v.lower():
        return False, 'what_this_cluster_is_about should say "topic" not "cluster"'
    return True, ""


def check_why_it_matters(report: ClusterReport) -> Tuple[bool, str]:
    """Check why_it_matters word count"""
    v = (report.why_it_matters or "").strip()
    wc = word_count(v)
    if not (WHY_MIN_WORDS <= wc <= WHY_MAX_WORDS):
        return False, f"why_it_matters must be {WHY_MIN_WORDS}–{WHY_MAX_WORDS} words, got {wc}"
    return True, ""


def check_keyword_list(report: ClusterReport) -> Tuple[bool, str]:
    """Check keyword_list format, word counts, and deduplication"""
    raw = report.keyword_list
    cleaned = [k.strip() for k in raw if k and k.strip()]
    if len(cleaned) != len(raw):
        return False, "keyword_list contains empty/whitespace-only items"

    # lowercase + validate
    lowered = [k.lower() for k in cleaned]
    for k in lowered:
        wc = tag_word_count(k)
        if wc < KEYWORD_MIN_WORDS or wc > KEYWORD_MAX_WORDS:
            return False, f"keyword must be {KEYWORD_MIN_WORDS}–{KEYWORD_MAX_WORDS} words: {k!r}"
        if "#" in k:
            return False, f"keyword must not include hashtags: {k!r}"

    # dedupe
    seen = set()
    deduped = []
    for k in lowered:
        if k not in seen:
            seen.add(k)
            deduped.append(k)

    if not (KEYWORDS_MIN_ITEMS <= len(deduped) <= KEYWORDS_MAX_ITEMS):
        return False, f"keyword_list must be {KEYWORDS_MIN_ITEMS}–{KEYWORDS_MAX_ITEMS} unique items after dedupe; got {len(deduped)}"

    return True, ""


HARD_CHECKS: tuple[CheckFn, ...] = (
    check_title,
    check_about,
    check_why_it_matters,
    check_keyword_list,
)


def _hard_report_field_validation(report: ClusterReport) -> ValidationReport:
    """Hard validation: score is 1 if all pass, else 0."""
    r = run_checks(report, HARD_CHECKS)
    return ValidationReport(score=1.0 if r.score == 1.0 else 0.0, reasons=r.reasons)


# ----------------------------
# Citation validation
# ----------------------------

def _extract_input_paper_ids(input_data: dict) -> Set[str]:
    """Extract all paper_ids from input papers"""
    paper_ids = set()
    papers = input_data.get("papers", [])
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def _extract_output_paper_ids_from_cluster_report(cluster_report: ClusterReport) -> Set[str]:
    """Extract all paper_ids from ClusterReport (representative_papers and reading_order)"""
    paper_ids = set()

    # From representative_papers
    for paper in cluster_report.representative_papers:
        if paper.paper_id:
            paper_ids.add(paper.paper_id)

    # From reading_order
    for item in cluster_report.reading_order:
        if item.paper_id:
            paper_ids.add(item.paper_id)

    return paper_ids


def _validate_citations(cluster_report: Optional[ClusterReport], input_data: dict) -> tuple[bool, str]:
    """
    Validate that all cited paper_ids exist in input data.

    Returns:
        Tuple of (passed: bool, invalid_citations: List[str], reason: str)
    """
    if cluster_report is None:
        return False, [], "Skipped (JSON parsing failed)"

    input_paper_ids = _extract_input_paper_ids(input_data)
    output_paper_ids = _extract_output_paper_ids_from_cluster_report(cluster_report)

    invalid_citations = []
    for paper_id in output_paper_ids:
        if paper_id not in input_paper_ids:
            invalid_citations.append(paper_id)

    passed = len(invalid_citations) == 0
    if passed:
        reason = ""
    else:
        reason = f"Invalid citations: {', '.join(invalid_citations[:5])}" + (f" (and {len(invalid_citations) - 5} more)" if len(invalid_citations) > 5 else "")

    return passed, reason


# ----------------------------
# Main hard validation function
# ----------------------------

def hard_validate_cluster_report(cluster_report: Optional[ClusterReport], input_data: dict) -> ValidationReport:
    """
    Hard validation: runs parsing, field validation, and citation checks.
    
    Args:
        cluster_report: ClusterReport instance
        input_data: Original input data (for citation validation)
    
    Returns:
        ValidationReport with score (1.0 if all pass, 0.0 else) and concatenated reasons string
    """
    if not cluster_report:
        return ValidationReport(score=0.0, reasons="Skipped (cluster_report is None)")
    
    reasons_lines: list[str] = []
    all_passed = True
    
    
    # Step 1: Field validation
    field_result = _hard_report_field_validation(cluster_report)
    reasons_lines.append(field_result.reasons)
    if field_result.score != 1.0:
        all_passed = False
    
    # Step 2: Citation check
    citations_passed, citations_reason = _validate_citations(cluster_report, input_data)
    if not citations_passed:
        reasons_lines.append(f"_validate_citations: error: {citations_reason}")
        all_passed = False
    
    score = 1.0 if all_passed else 0.0
    return ValidationReport(score=score, reasons="\n".join(reasons_lines))

