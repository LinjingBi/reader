"""Pydantic models for per-cluster LLM response.

Design goal:
- Pydantic models define *structure/types* only (so JSON parsing is strict & typed).
- All semantic constraints (word limits, item counts, formatting rules, etc.) are defined
  as global constants and validated in the judges module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, conlist

# ----------------------------
# Global constraints (single source of truth)
# ----------------------------

TITLE_MAX_WORDS = 12
ONE_LINER_MAX_WORDS = 25

ABOUT_MIN_WORDS = 80
ABOUT_MAX_WORDS = 140

WHY_MIN_WORDS = 60
WHY_MAX_WORDS = 120

CONF_RATIONALE_MIN_ITEMS = 2
CONF_RATIONALE_MAX_ITEMS = 4
CONF_RATIONALE_MAX_WORDS_PER_ITEM = 18

REP_PAPERS_MIN_ITEMS = 2
REP_PAPERS_MAX_ITEMS = 5

READING_ORDER_MIN_ITEMS = 3
READING_ORDER_MAX_ITEMS = 7
READING_ORDER_MAX_WORDS_PER_ITEM_REASON = 12

SEARCH_QUERY_MIN_TERMS = 2
SEARCH_QUERY_MAX_TERMS = 5

NOTES_MAX_ITEMS = 5
NOTES_MAX_WORDS_PER_ITEM = 20

KEYWORDS_MIN_ITEMS = 5
KEYWORDS_MAX_ITEMS = 12
KEYWORD_MIN_WORDS = 1
KEYWORD_MAX_WORDS = 3


# ----------------------------
# Pydantic models (structure only)
# ----------------------------

class RepresentativePaper(BaseModel):
    """Representative paper in this topic."""
    paper_id: str = Field(..., description="paper_id, referenced like [paper_id] in the report")
    title: str = Field(..., description="paper title")


class ReadingOrderItem(BaseModel):
    """One item in the suggested reading order."""
    paper_id: str = Field(..., description="paper_id")
    why_read_now: str = Field(
        ...,
        description=(
            "Short reason for this placement in the reading order. "
            f"Target <= {READING_ORDER_MAX_WORDS_PER_ITEM_REASON} words."
        ),
    )


class ClusterReport(BaseModel):
    """Cluster/topic report returned by the LLM (JSON)."""

    title: str = Field(
        ...,
        description=f"Title Case, no colon. Target <= {TITLE_MAX_WORDS} words.",
    )
    one_liner: str = Field(
        ...,
        description=f"Plain-English summary. Target <= {ONE_LINER_MAX_WORDS} words.",
    )
    what_this_cluster_is_about: str = Field(
        ...,
        description=(
            "Describe the shared theme using only provided information. Explain how multiple papers relate. "
            f"Target {ABOUT_MIN_WORDS}–{ABOUT_MAX_WORDS} words. Include inline citations [paper_id]. "
            "Use the word “topic”, not “cluster”."
        ),
    )
    why_it_matters: str = Field(
        ...,
        description=(
            "Practical and research significance. No hype or speculation. "
            f"Target {WHY_MIN_WORDS}–{WHY_MAX_WORDS} words. Use hedged verbs if unclear."
        ),
    )

    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="Self-rated confidence in this topic summary given the provided paper summaries/keywords.",
    )

    confidence_rationale: conlist(str, min_length=CONF_RATIONALE_MIN_ITEMS, max_length=CONF_RATIONALE_MAX_ITEMS) = Field(
        ...,
        description=(
            "Bullet list justifying confidence using cluster size, cohesion, and evidence quality. "
            f"{CONF_RATIONALE_MIN_ITEMS}–{CONF_RATIONALE_MAX_ITEMS} items, each <= {CONF_RATIONALE_MAX_WORDS_PER_ITEM} words."
        ),
    )

    representative_papers: conlist(RepresentativePaper, min_length=REP_PAPERS_MIN_ITEMS, max_length=REP_PAPERS_MAX_ITEMS) = Field(
        ...,
        description=f"{REP_PAPERS_MIN_ITEMS}–{REP_PAPERS_MAX_ITEMS} representative papers.",
    )

    reading_order: conlist(ReadingOrderItem, min_length=READING_ORDER_MIN_ITEMS, max_length=READING_ORDER_MAX_ITEMS) = Field(
        ...,
        description=(
            f"{READING_ORDER_MIN_ITEMS}–{READING_ORDER_MAX_ITEMS} items. "
            "Order from most central/accessible to more detailed papers."
        ),
    )

    search_query_seed: str = Field(
        ...,
        description=f"One line, {SEARCH_QUERY_MIN_TERMS}–{SEARCH_QUERY_MAX_TERMS} key terms.",
    )

    notes: conlist(str, max_length=NOTES_MAX_ITEMS) = Field(
        ...,
        description=(
            f"Up to {NOTES_MAX_ITEMS} bullets. Each <= {NOTES_MAX_WORDS_PER_ITEM} words. "
            "Include warnings about mixed themes, missing information, or ambiguity when applicable."
        ),
    )

    keyword_list: conlist(str, min_length=KEYWORDS_MIN_ITEMS, max_length=KEYWORDS_MAX_ITEMS) = Field(
        ...,
        description=(
            "Keywords extracted from provided paper keywords + the topic theme. "
            f"{KEYWORDS_MIN_ITEMS}–{KEYWORDS_MAX_ITEMS} items, lowercase, deduped; "
            f"each item {KEYWORD_MIN_WORDS}–{KEYWORD_MAX_WORDS} words; no hashtags."
        ),
    )
