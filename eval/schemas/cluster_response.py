"""Pydantic models for per-cluster LLM response"""

from __future__ import annotations

import re
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _word_count(s: str) -> int:
    """Count words in a string"""
    return len(_WORD_RE.findall(s))


def _is_lowercase_tag(tag: str) -> bool:
    """Check if tag is lowercase (allow digits/hyphens/spaces, but require letters to be lowercase)"""
    return tag == tag.lower()


def _tag_word_count(tag: str) -> int:
    """Count words separated by whitespace"""
    return len([w for w in tag.strip().split() if w])


class RepresentativePaper(BaseModel):
    """Representative paper in cluster"""
    paper_id: str = Field(..., description="paper_id, referenced like [paper_id] in the report")
    title: str = Field(..., description="paper title")


class ReadingOrderItem(BaseModel):
    """Reading order item with reason"""
    paper_id: str = Field(..., description="paper_id")
    why_read_now: str = Field(..., description="Short reason, <= 12 words (best-effort)")


class ClusterCardSectionA(BaseModel):
    """SECTION A — Human-readable report fields"""
    title: str
    one_liner: str
    what_this_cluster_is_about: str
    why_it_matters: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    confidence_rationale: List[str] = Field(..., min_length=2, max_length=4)
    representative_papers: List[RepresentativePaper] = Field(..., min_length=2, max_length=5)
    reading_order: List[ReadingOrderItem] = Field(..., min_length=3, max_length=7)
    search_query_seed: str
    notes: List[str] = Field(..., max_length=5)


class ClusterIndexSectionB(BaseModel):
    """
    SECTION B — JSON index for database storage
    
    Keys:
      - title: must equal SECTION A title exactly (cannot be enforced without Section A passed in)
      - summary: 60–110 words
      - keyword_list: 5–12 items; automatically lowercased; deduped; each tag 1–3 words
    """
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    keyword_list: List[str] = Field(..., min_length=5, max_length=12)

    @field_validator("summary")
    @classmethod
    def validate_summary_word_count(cls, v: str) -> str:
        """Validate summary is 60–110 words"""
        wc = _word_count(v)
        if wc < 60 or wc > 110:
            raise ValueError(f"summary must be 60–110 words; got {wc}")
        return v.strip()

    @field_validator("keyword_list")
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Validate keywords: auto-convert to lowercase, 1-3 words each, no hashtags, deduped"""
        # strip + basic checks
        cleaned = [k.strip() for k in v if k and k.strip()]
        if len(cleaned) != len(v):
            raise ValueError("keyword_list contains empty/whitespace-only items")

        # Convert to lowercase automatically
        lowercased = [k.lower() for k in cleaned]

        for k in lowercased:
            wc = _tag_word_count(k)
            if wc < 1 or wc > 3:
                raise ValueError(f"keyword must be 1–3 words: {k!r}")
            if "#" in k:
                raise ValueError(f"keyword must not include hashtags: {k!r}")

        # dedupe while preserving order
        seen = set()
        deduped = []
        for k in lowercased:
            if k not in seen:
                seen.add(k)
                deduped.append(k)

        if len(deduped) < 5 or len(deduped) > 12:
            raise ValueError(
                f"keyword_list must be 5–12 unique items after dedupe; got {len(deduped)}"
            )
        return deduped

    @model_validator(mode="after")
    def validate_title_nonempty(self) -> "ClusterIndexSectionB":
        """Ensure title is non-empty after stripping"""
        self.title = self.title.strip()
        if not self.title:
            raise ValueError("title must be non-empty")
        return self


class ClusterReport(BaseModel):
    """
    One response object that contains:
      - section_a: structured fields (you can render these into Markdown cards yourself)
      - section_b: JSON index for DB storage
    """
    section_a: ClusterCardSectionA
    section_b: ClusterIndexSectionB

    @model_validator(mode="after")
    def validate_title_match(self) -> "ClusterReport":
        """Ensure section_b.title matches section_a.title exactly"""
        if self.section_b.title != self.section_a.title:
            raise ValueError(
                f"SECTION B title must equal SECTION A title exactly. "
                f"SECTION A: {self.section_a.title!r}, SECTION B: {self.section_b.title!r}"
            )
        return self

