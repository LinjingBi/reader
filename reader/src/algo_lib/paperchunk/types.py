from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

# -------------------------
# Identity / domain aliases
# -------------------------
PaperId = str
Url = str
SelectorId = str   # e.g. "summary", "method", "conclusion"
TextId = str       # e.g. f"{paper_id}:{source}:{block_index}"

SourceType = Literal["html", "pdf"]

# -------------------------
# Fetch layer
# -------------------------
@dataclass(frozen=True)
class FetchResult:
    paper_id: PaperId
    url: Url

    ok: bool
    status_code: Optional[int] = None
    error: Optional[str] = None

    html: Optional[str] = None
    pdf_bytes: Optional[bytes] = None

    fetched_html: bool = False
    fetched_pdf: bool = False

# -------------------------
# Parse layer
# -------------------------
@dataclass(frozen=True)
class HeadingBlock:
    """One extracted section-like unit from a paper source (html/pdf)."""

    paper_id: PaperId
    source: SourceType

    block_index: int
    heading_raw: str
    heading_key: str

    text_raw: str

    heading_path: Tuple[str, ...] = ()
    char_start: Optional[int] = None
    char_end: Optional[int] = None

@dataclass(frozen=True)
class ParseResult:
    paper_id: PaperId
    url: Url

    ok: bool
    source_used: Optional[SourceType] = None
    error: Optional[str] = None

    blocks: List[HeadingBlock] = None  # empty list if ok but no blocks

    used_html: bool = False
    used_pdf: bool = False

# -------------------------
# Match layer (kept for future use)
# -------------------------
@dataclass(frozen=True)
class AliasHit:
    selector_id: SelectorId
    alias: str

@dataclass(frozen=True)
class HeadingMatch:
    paper_id: PaperId
    source: SourceType
    block_index: int

    heading_raw: str
    heading_key: str

    matched_selectors: Tuple[SelectorId, ...]
    matched_alias_hits: Tuple[AliasHit, ...]

    is_combined_heading: bool = False
    join_token_used: Optional[str] = None

# -------------------------
# Debug event (both modes)
# -------------------------
@dataclass(frozen=True)
class DebugHeadingEvent:
    paper_id: PaperId
    url: Url

    source_used: Optional[SourceType]
    block_index: Optional[int]

    heading_raw: Optional[str]
    heading_key: Optional[str]

    matched_selectors: Tuple[SelectorId, ...] = ()
    matched_aliases: Tuple[str, ...] = ()

    status: Literal[
        "fetch_fail",
        "parse_fail",
        "no_blocks",
        "unmapped_heading",
        "mapped_heading",
        "mapped_combined_heading",
    ] = "mapped_heading"

    note: Optional[str] = None

# -------------------------
# Scoring output
# -------------------------
TextTable = Dict[TextId, str]

@dataclass(frozen=True)
class ScoreRow:
    """
    Represents a selector -> texts scoring relationship, not text -> selectors scoring.
    """
    paper_id: PaperId
    selector_id: SelectorId
    text_id: TextId
    score: float

@dataclass(frozen=True)
class ScoreSummary:
    total_papers: int
    fetched_ok: int
    parsed_ok: int
    scored_ok: int
    used_html: int
    used_pdf: int
    ok_papers: int
    partial_papers: int
    fail_papers: int
    required_selectors: Tuple[SelectorId, ...] = ()

@dataclass(frozen=True)
class ScoreOutput:
    summary: ScoreSummary
    debug_heading_events: List[DebugHeadingEvent]
    text_table: TextTable
    score_table: List[ScoreRow]

# -------------------------
# Training output
# -------------------------
@dataclass(frozen=True)
class TrainSummary:
    total_papers: int
    fetched_ok: int
    parsed_ok: int
    used_html: int
    used_pdf: int
    required_selectors: Tuple[SelectorId, ...] = ()
    papers_with_required_all: int = 0
    papers_with_required_partial: int = 0
    papers_failed: int = 0

@dataclass(frozen=True)
class TrainOutput:
    summary: TrainSummary
    debug_heading_events: List[DebugHeadingEvent]
    unmapped_candidates: Dict[str, object]
    proposals: Dict[str, object]
