from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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

class PaperStatus(str, Enum):
    """Status of a paper in the scoring pipeline."""
    ok = "ok"  # Paper has all required selectors found
    partial = "partial"  # Paper has some mappings but missing required selectors
    error = "error"  # Paper failed at any stage (fetch, parse, no blocks, or no mappings)

PapersStatus = Dict[PaperId, PaperStatus]

@dataclass(frozen=True)
class ScoreRow:
    paper_id: PaperId
    selector_id: SelectorId
    text_id: TextId
    score: float

@dataclass(frozen=True)
class ScoreSummary:
    """
    Summary statistics for paper scoring pipeline execution.
    
    Fields track papers through three stages: fetch -> parse -> score.
    Relationships: fetched_ok >= parsed_ok >= scored_ok (cascading success).
    """
    total_papers: int  # Total number of papers processed
    fetched_ok: int  # Papers successfully fetched from URLs
    parsed_ok: int  # Papers successfully parsed into heading blocks (subset of fetched_ok)
    scored_ok: int  # Papers with at least one mapped heading (subset of parsed_ok)
    used_html: int  # Papers parsed using HTML source (sum with used_pdf <= parsed_ok)
    used_pdf: int  # Papers parsed using PDF source (sum with used_html <= parsed_ok)
    ok_papers: int  # Papers with all required selectors found (subset of scored_ok)
    partial_papers: int  # Papers with some mappings but missing required selectors (subset of scored_ok)
    fail_papers: int  # Papers that failed at any stage (fetch, parse, no blocks, or no mappings)
    required_selectors: Tuple[SelectorId, ...] = ()  # Selectors that must be found for a paper to be "ok"

@dataclass(frozen=True)
class RulesMeta:
    """Rules metadata containing version information."""
    version: int
    compiled_regex_version: int

@dataclass(frozen=True)
class ScoreOutput:
    summary: ScoreSummary
    debug_heading_events: List[DebugHeadingEvent]
    text_table: TextTable
    # Represents a selector -> texts scoring relationship, not text -> selectors scoring.
    sel2texts_score_table: List[ScoreRow]
    papers_status: PapersStatus  # Mapping of paper_id to its status (ok/partial/error)
    rules_meta: RulesMeta  # Rules metadata (version and compiled_regex_version)

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
