from __future__ import annotations

import re
from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _word_count(s: str) -> int:
    return len(_WORD_RE.findall(s))


def _is_lowercase_tag(tag: str) -> bool:
    # allow digits/hyphens/spaces, but require letters to be lowercase
    return tag == tag.lower()


def _tag_word_count(tag: str) -> int:
    # count "words" separated by whitespace
    return len([w for w in tag.strip().split() if w])


class ClusterIndexSectionB(BaseModel):
    """
    SECTION B — JSON index

    Keys:
      - title: must equal SECTION A title exactly (cannot be enforced without Section A passed in)
      - summary: 60–110 words
      - keyword_list: 5–12 items; lowercase; deduped; each tag 1–3 words
    """

    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    keyword_list: List[str] = Field(..., min_length=5, max_length=12)

    @fieator("summary")
    @classmethod
    def validate_summary_word_count(cls, v: str) -> str:
        wc = _word_count(v)
        if wc < 60 or wc > 110:
            raise ValueError(f"summary must be 60–110 words; got {wc}")
        return v.strip()

    @field_validator("keyword_list")
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        # strip + basic checks
        cleaned = [k.strip() for k in v if k and k.strip()]
        if len(cleaned) != len(v):
            raise ValueError("keyword_list contains empty/whitespace-only items")

        for k in cleaned:
            if not _is_lowercase_tag(k):
                raise ValueError(f"keyword must be lowercase: {k!r}")
            wc = _tag_word_count(k)
            if wc < 1 or wc > 3:
                raise ValueError(f"keyword must be 1–3 words: {k!r}")
            if "#" in k:
                raise ValueError(f"keyword must not include hashtags: {k!r}")

        # dedupe while preserving order
        seen = set()
       uped = []
        for k in cleaned:
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
        self.title = self.title.strip()
        if not self.title:
            raise ValueError("title must be non-empty")
        return self


# Optional helper if you want to enforce: "title must equal SECTION A title exactly"
def validate_section_b_title_matches(section_a_title: str, section_b: ClusterIndexSectionB) -> None:
    if section_b.title != section_a_title:
        raise ValueError(
            f"SECTION B title must equal SECTION A title exactly. "
            f"SECTION A: {section_a_title!r}, SECTION B: {section_b.title!r}"
        )

