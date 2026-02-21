"""Metrics and validation for cluster reports

This module consolidates all judge code for validating LLM-generated cluster reports.
It includes common functions, hard rules, soft rules, and the main judge_output function.
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Set, Tuple


from reader.pipelines.hf_data.report import (
    ClusterReport,
    TITLE_MAX_WORDS,
    ONE_LINER_MAX_WORDS,
    ABOUT_MIN_WORDS,
    ABOUT_MAX_WORDS,
    WHY_MIN_WORDS,
    WHY_MAX_WORDS,
    KEYWORDS_MIN_ITEMS,
    KEYWORDS_MAX_ITEMS,
    KEYWORD_MIN_WORDS,
    KEYWORD_MAX_WORDS,
    CONF_RATIONALE_MAX_WORDS_PER_ITEM,
    SEARCH_QUERY_MIN_TERMS,
    SEARCH_QUERY_MAX_TERMS,
    NOTES_MAX_WORDS_PER_ITEM,
    READING_ORDER_MAX_WORDS_PER_ITEM_REASON,
)


# ============================================================================
# Common Functions
# ============================================================================

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


def tag_word_count(tag: str) -> int:
    """Count words separated by whitespace (after lowering/stripping)"""
    return len([w for w in (tag or "").strip().split() if w])


def run_checks(report: ClusterReport, checks: Sequence[CheckFn]) -> ValidationReport:
    """Run a sequence of checks and return ValidationReport"""
    passed = 0
    lines: list[str] = []
    for fn in checks:
        ok, msg = fn(report)
        if ok:
            passed += 1
            # Only record failure messages, not pass messages
        else:
            lines.append(f"{fn.__name__}: error: {msg}")
    score = passed / max(1, len(checks))
    return ValidationReport(score=score, reasons="\n".join(lines))


# ============================================================================
# Hard Rules
# ============================================================================

# Regex pattern for citation matching
_CITATION_RE = re.compile(r"\[[^\[\]]+\]")  # basic [paper_id] matcher


# current rules too rigid, muted.
def is_title_case(title: str) -> bool:
    """Check if title is Title Case"""

    t = (title or "").strip()

    
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


# ---------------------------------
# Hard validation checks per field
# ---------------------------------

def check_title_format(report: ClusterReport) -> Tuple[bool, str]:
    """Check title format (no colon)"""
    v = (report.title or "").strip()
    if ":" in v:
        return False, "title must contain no colon"
    return True, ""


def check_what_this_topic_is_about(report: ClusterReport) -> Tuple[bool, str]:
    """Check what_this_topic_is_about requirement"""
    v = (report.what_this_topic_is_about or "").strip()
    if not has_inline_citation(v):
        return False, "what_this_topic_is_about must include at least one inline citation like [paper_id]"
    if "cluster" in v.lower():
        return False, 'what_this_topic_is_about should say "topic" not "cluster"'
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
    check_what_this_topic_is_about,
    check_keyword_list_format,
)


def _hard_report_field_validation(report: ClusterReport) -> ValidationReport:
    """Hard validation: score is 1 if all pass, else 0."""
    r = run_checks(report, HARD_CHECKS)
    return ValidationReport(score=1.0 if r.score == 1.0 else 0.0, reasons=r.reasons)


# -------------------------------------
# Hard validation checks cross fields
# -------------------------------------

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

# Citation validation for representative_papers and reading_order
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
    
    # Step 2: Citation check for representative_papers and reading_order
    citations_passed, citations_reason = _validate_citations(cluster_report, input_data)
    if not citations_passed:
        reasons_lines.append(f"_validate_citations: error: {citations_reason}")
        all_passed = False
    
    score = 1.0 if all_passed else 0.0
    return ValidationReport(score=score, reasons="\n".join(reasons_lines))


# ============================================================================
# Soft Rules
# ============================================================================

# ----------------------------
# Soft validation checks
# ----------------------------

def check_title_word_count(report: ClusterReport) -> Tuple[bool, str]:
    """Check title word count"""
    v = (report.title or "").strip()
    wc = word_count(v)
    if wc < 1 or wc > TITLE_MAX_WORDS:
        return False, f"title must be 1–{TITLE_MAX_WORDS} words, got {wc}"
    return True, ""


def check_one_liner(report: ClusterReport) -> Tuple[bool, str]:
    """Check one_liner word count"""
    v = (report.one_liner or "").strip()
    wc = word_count(v)
    if wc < 1 or wc > ONE_LINER_MAX_WORDS:
        return False, f"one_liner must be 1–{ONE_LINER_MAX_WORDS} words, got {wc}"
    return True, ""


def check_about_word_count(report: ClusterReport) -> Tuple[bool, str]:
    """Check what_this_topic_is_about word count"""
    v = (report.what_this_topic_is_about or "").strip()
    wc = word_count(v)
    if not (ABOUT_MIN_WORDS <= wc <= ABOUT_MAX_WORDS):
        return False, f"what_this_topic_is_about must be {ABOUT_MIN_WORDS}–{ABOUT_MAX_WORDS} words, got {wc}"
    return True, ""


def check_why_it_matters_word_count(report: ClusterReport) -> Tuple[bool, str]:
    """Check why_it_matters word count"""
    v = (report.why_it_matters or "").strip()
    wc = word_count(v)
    if not (WHY_MIN_WORDS <= wc <= WHY_MAX_WORDS):
        return False, f"why_it_matters must be {WHY_MIN_WORDS}–{WHY_MAX_WORDS} words, got {wc}"
    return True, ""


def check_confidence_rationale(report: ClusterReport) -> Tuple[bool, str]:
    """Check confidence_rationale word counts per item"""
    bullets = report.confidence_rationale
    for i, b in enumerate(bullets):
        wc = word_count(b)
        if wc > CONF_RATIONALE_MAX_WORDS_PER_ITEM:
            return False, f"confidence_rationale[{i}] must be <= {CONF_RATIONALE_MAX_WORDS_PER_ITEM} words, got {wc}"
    return True, ""


def check_search_query_seed(report: ClusterReport) -> Tuple[bool, str]:
    """Check search_query_seed format and term count"""
    v = (report.search_query_seed or "").strip()
    if "\n" in v:
        return False, "search_query_seed must be one line"
    terms = [t for t in re.split(r"\s+", v) if t]
    if not (SEARCH_QUERY_MIN_TERMS <= len(terms) <= SEARCH_QUERY_MAX_TERMS):
        return False, f"search_query_seed must have {SEARCH_QUERY_MIN_TERMS}–{SEARCH_QUERY_MAX_TERMS} terms, got {len(terms)}"
    return True, ""


def check_notes(report: ClusterReport) -> Tuple[bool, str]:
    """Check notes word counts per item"""
    for i, b in enumerate(report.notes):
        wc = word_count(b)
        if wc > NOTES_MAX_WORDS_PER_ITEM:
            return False, f"notes[{i}] must be <= {NOTES_MAX_WORDS_PER_ITEM} words, got {wc}"
    return True, ""


def check_reading_order_item_reasons(report: ClusterReport) -> Tuple[bool, str]:
    """Check reading_order item reason word counts"""
    for i, item in enumerate(report.reading_order):
        wc = word_count(item.why_read_now)
        if wc > READING_ORDER_MAX_WORDS_PER_ITEM_REASON:
            return False, f"reading_order[{i}].why_read_now must be <= {READING_ORDER_MAX_WORDS_PER_ITEM_REASON} words, got {wc}"
    return True, ""


def check_keyword_word_counts(report: ClusterReport) -> Tuple[bool, str]:
    """Check keyword_list word counts per item"""
    raw = report.keyword_list
    cleaned = [k.strip() for k in raw if k and k.strip()]
    
    for i, k in enumerate(cleaned):
        wc = tag_word_count(k)
        if wc < KEYWORD_MIN_WORDS or wc > KEYWORD_MAX_WORDS:
            return False, f"keyword_list[{i}] must be {KEYWORD_MIN_WORDS}–{KEYWORD_MAX_WORDS} words: {k!r}"
    return True, ""


def check_keyword_list_count(report: ClusterReport) -> Tuple[bool, str]:
    """Check keyword_list count after deduplication"""
    raw = report.keyword_list
    cleaned = [k.strip() for k in raw if k and k.strip()]
    
    # Dedupe (case-insensitive)
    seen = set()
    deduped = []
    for k in cleaned:
        k_lower = k.lower()
        if k_lower not in seen:
            seen.add(k_lower)
            deduped.append(k)
    
    if not (KEYWORDS_MIN_ITEMS <= len(deduped) <= KEYWORDS_MAX_ITEMS):
        return False, f"keyword_list must be {KEYWORDS_MIN_ITEMS}–{KEYWORDS_MAX_ITEMS} unique items after dedupe; got {len(deduped)}"
    return True, ""


def check_name_generic(report: ClusterReport) -> Tuple[bool, str]:
    """
    Check for generic topic names.
    
    Generic names: "AI", "LLM", "Vision", "Machine Learning", etc.
    Returns (True, "") if no penalty, (False, reason) if penalty exists.
    """
    title = report.title
    generic_terms = {
        "ai", "llm", "vision", "machine learning", "deep learning",
        "neural network", "nlp", "computer vision", "ml", "dl"
    }

    title_lower = (title or "").lower().strip()

    # Check if title is exactly a generic term or starts with it
    if not title_lower or title_lower in generic_terms:
        return False, "Generic title detected (too generic)"
    
    return True, ""


SOFT_CHECKS: tuple[CheckFn, ...] = (
    check_title_word_count,
    check_one_liner,
    check_about_word_count,
    check_why_it_matters_word_count,
    check_confidence_rationale,
    check_search_query_seed,
    check_notes,
    check_reading_order_item_reasons,
    check_keyword_word_counts,
    check_keyword_list_count,
    check_name_generic,
)


def soft_validate_cluster_report(cluster_report: Optional[ClusterReport]) -> ValidationReport:
    """
    Soft validation: runs all soft checks and returns average score.
    
    Args:
        cluster_report: Parsed ClusterReport to validate
    
    Returns:
        ValidationReport with:
        - score: average of all rule scores (0.0 to 1.0)
        - reasons: concatenated string of all check results
    """
    if not cluster_report:
        return ValidationReport(score=0.0, reasons="Skipped (cluster_report is None)")
    return run_checks(cluster_report, SOFT_CHECKS)


# ============================================================================
# Judge Output
# ============================================================================

@dataclass
class JudgeOutput:
    """Output from judge_output function matching HeuristicResult format"""
    sub_scores: Dict[str, float]  # All rule scores (0.0 or 1.0 for bools)
    overall: float  # 0.0 if any must-pass fails, else 1.0 + soft_schema_valid.score
    reasons: Dict[str, str]  # Human-readable reasons for each rule


def judge_output(cluster_report: ClusterReport, input_data: dict) -> JudgeOutput:
    """
    Judge LLM output using Pydantic validation and heuristic checks.
    
    Args:
        cluster_report: Parsed ClusterReport instance from LLM
        input_data: Original input data (for citation validation)
    
    Returns:
        JudgeOutput: Contains sub_scores, overall, and reasons
    """
    sub_scores: Dict[str, float] = {}
    reasons: Dict[str, str] = {}
    
    # 1. Hard validation (includes field validation and citation checks)
    hard_result = hard_validate_cluster_report(cluster_report, input_data)
    sub_scores["hard_schema_valid"] = hard_result.score
    # Only include reasons if there are failure messages (not empty string)
    if hard_result.reasons:
        reasons["hard_schema_valid"] = hard_result.reasons
    
    # 2. Soft validation (includes name_generic check)
    soft_result = soft_validate_cluster_report(cluster_report)
    sub_scores["soft_schema_valid"] = soft_result.score
    # Only include reasons if there are failure messages (not empty string)
    if soft_result.reasons:
        reasons["soft_schema_valid"] = soft_result.reasons

    
    # Compute overall score
    # Must-pass rules: hard_schema_valid
    must_pass_rules = ["hard_schema_valid"]
    must_pass_failed = any(sub_scores.get(rule, 0.0) == 0.0 for rule in must_pass_rules)
    
    if must_pass_failed:
        overall = 0.0
    else:
        # If all must-pass rules pass, overall is 1 + soft_schema_valid.score
        overall = 1.0 + sub_scores.get("soft_schema_valid", 0.0)
    
    return JudgeOutput(
        sub_scores=sub_scores,
        overall=overall,
        reasons=reasons
    )

