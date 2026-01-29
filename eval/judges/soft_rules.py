"""Soft validation rules for cluster reports

Soft rules are scored checks that contribute to an average score.
"""

import re
from typing import Optional, Tuple

from schemas.cluster_response import (
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
from judges import ValidationReport, CheckFn, word_count, tag_word_count, run_checks


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
    """Check what_this_cluster_is_about word count"""
    v = (report.what_this_cluster_is_about or "").strip()
    wc = word_count(v)
    if not (ABOUT_MIN_WORDS <= wc <= ABOUT_MAX_WORDS):
        return False, f"what_this_cluster_is_about must be {ABOUT_MIN_WORDS}–{ABOUT_MAX_WORDS} words, got {wc}"
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

