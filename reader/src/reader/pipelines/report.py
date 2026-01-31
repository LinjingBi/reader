"""Pydantic models for fresh paper payload"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, computed_field, conlist


# Pydantic models for fresh paper payload
class ClusterMemberInput(BaseModel):
    """Cluster member entry with paper ID, rank, and similarity to centroid."""
    paper_id: str = Field(..., description="Formatted paper ID (e.g., 'hf:2501.12948')")
    rank_in_cluster: int = Field(..., description="Rank of paper within cluster")
    sim_to_centroid: float = Field(..., description="Similarity to cluster centroid")


class PaperInput(BaseModel):
    """Paper entry in the payload."""
    raw_paper_id: str = Field(..., exclude=True, description="Raw paper ID (excluded from serialization)")
    title: str = Field(..., description="Paper title")
    summary: str = Field(..., description="Paper summary/abstract")
    keywords: List[str] = Field(..., description="List of keywords")
    url: str = Field(default="", description="Paper URL")
    published_at: str = Field(default="", description="Publication date")
    
    def __init__(self, raw_paper_id: str, **kwargs):
        """Initialize PaperInput with raw paper ID."""
        super().__init__(raw_paper_id=raw_paper_id, **kwargs)
    
    @computed_field
    @property
    def paper_id(self) -> str:
        """Format paper ID with 'hf:' prefix."""
        return f"hf:{self.raw_paper_id}"


class EmbedConfigPayload(BaseModel):
    """Embedding configuration payload."""
    model_name: str = Field(..., description="Embedding model name")
    mode: str = Field(..., description="Embedding mode")
    top_n_keywords: int = Field(..., description="Number of top keywords used")


class EmbedConfig(BaseModel):
    """Embedding configuration with dynamic config ID."""
    json_payload: EmbedConfigPayload = Field(..., description="Embedding configuration payload")
    
    @computed_field
    @property
    def embed_config_id(self) -> str:
        """Get embed_config_id from algo_lib.embedding version."""
        try:
            from algo_lib.embedding import __version__ as embed_version
            return f"algo_lib.embedding|{embed_version}"
        except ImportError:
            raise ValueError("algo_lib.embedding is not versioned")


class ClusterConfigPayload(BaseModel):
    """Clustering configuration payload."""
    k: int = Field(..., description="Number of clusters")
    seed: int = Field(..., description="Random seed")
    algorithm: str = Field(default="kmeans", description="Clustering algorithm")


class ClusterConfig(BaseModel):
    """Clustering configuration with dynamic config ID."""
    json_payload: ClusterConfigPayload = Field(..., description="Clustering configuration payload")
    
    @computed_field
    @property
    def cluster_config_id(self) -> str:
        """Get cluster_config_id from algo_lib.clustering version."""
        try:
            from algo_lib.clustering import __version__ as cluster_version
            return f"algo_lib.clustering|{cluster_version}"
        except ImportError:
            raise ValueError("algo_lib.clustering is not versioned")


class ClusterInput(BaseModel):
    """Cluster entry in the payload."""
    cluster_index: int = Field(..., description="Cluster index")
    size: int = Field(..., description="Number of papers in cluster")
    cohesion: float = Field(..., description="Cluster cohesion score")
    members: List[ClusterMemberInput] = Field(..., description="List of cluster members")


class FreshPaperPayload(BaseModel):
    """Fresh paper payload report class."""
    source: str = Field(default="hf_monthly", description="Data source")
    period_start: str = Field(..., description="Period start date (YYYY-MM-DD)")
    period_end: str = Field(..., description="Period end date (YYYY-MM-DD)")
    raw_json: str = Field(default="", description="Optional raw JSON string")
    embed_config: EmbedConfig = Field(..., description="Embedding configuration")
    cluster_config: ClusterConfig = Field(..., description="Clustering configuration")
    papers: List[PaperInput] = Field(..., description="List of papers")
    clusters: List[ClusterInput] = Field(..., description="List of clusters")


# ----------------------------
# Cluster Report Models
# ----------------------------
# Global constraints (single source of truth)

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
            "Use the word \"topic\", not \"cluster\"."
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

