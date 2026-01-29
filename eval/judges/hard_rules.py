"""Hard validation rules for cluster reports

Hard rules are must-pass checks that result in score 0.0 if any fail.
"""

import re
from typing import Optional, Set, Tuple

from pydantic import ValidationError

from schemas.cluster_response import ClusterReport
from judges import ValidationReport, CheckFn, run_checks

# Regex pattern for citation matching
_CITATION_RE = re.compile(r"\[[^\[\]]+\]")  # basic [paper_id] matcher


def is_title_case_no_colon(title: str) -> bool:
    """Check if title is Title Case and contains no colon"""
    t = (title or "").strip()
    if ":" in t:
        return False
    
    # Standard title case: capitalize first word and major words
    # Lowercase articles, conjunctions, and short prepositions
    words = t.split()
    if not words:
        return False
    
    # First word must be capitalized
    if not words[0][0].isupper():
        return False
    
    # Small words that should be lowercase (unless first word)
    small_words = {"a", "an", "and", "as", "at", "but", "by", "for", "from", 
                   "in", "into", "of", "on", "or", "the", "to", "with"}
    
    for word in words[1:]:  # Skip first word
        # If it's a small word, it should be lowercase
        if word.lower() in small_words:
            if word[0].isupper():
                return False
        else:
            # Major words should be capitalized
            if not word[0].isupper():
                return False
    
    return True


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

def check_title_format(report: ClusterReport) -> Tuple[bool, str]:
    """Check title format (Title Case and no colon)"""
    v = (report.title or "").strip()
    if not is_title_case_no_colon(v):
        return False, "title must be Title Case and contain no colon"
    return True, ""


def check_about_citations(report: ClusterReport) -> Tuple[bool, str]:
    """Check what_this_cluster_is_about citations requirement"""
    v = (report.what_this_cluster_is_about or "").strip()
    if not has_inline_citation(v):
        return False, "what_this_cluster_is_about must include at least one inline citation like [paper_id]"
    if "cluster" in v.lower():
        return False, 'what_this_cluster_is_about should say "topic" not "cluster"'
    return True, ""


def check_keyword_list_format(report: ClusterReport) -> Tuple[bool, str]:
    """Check keyword_list format (no empty items, no hashtags)"""
    raw = report.keyword_list
    cleaned = [k.strip() for k in raw if k and k.strip()]
    if len(cleaned) != len(raw):
        return False, "keyword_list contains empty/whitespace-only items"

    # Check for hashtags
    for k in cleaned:
        if "#" in k:
            return False, f"keyword must not include hashtags: {k!r}"

    return True, ""


HARD_CHECKS: tuple[CheckFn, ...] = (
    check_title_format,
    check_about_citations,
    check_keyword_list_format,
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
    if field_result.reasons:  # Only append if there are failure messages
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

