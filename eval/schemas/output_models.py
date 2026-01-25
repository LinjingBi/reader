"""Pydantic models for final output schema validation"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class RepresentativePaperOutput(BaseModel):
    """Representative paper in output format"""
    paper_id: str
    reason_representative: str = Field(..., min_length=1, max_length=300)


class ReadingOrderItemOutput(BaseModel):
    """Reading order item in output format"""
    paper_id: str
    url: str
    why_read_next: str = Field(..., min_length=1, max_length=220)


class ClusterCard(BaseModel):
    """Cluster card matching p0_output_json_schema.json"""
    cluster_key: str = Field(..., description="Stable UI key; recommended format: 'cluster_index:<n>'")
    topic_name: str = Field(..., min_length=1, max_length=100)
    one_liner: str = Field(..., min_length=1, max_length=220)
    tags: List[str] = Field(..., min_length=3, max_length=7)
    what_this_cluster_is_about: str = Field(..., min_length=1, max_length=1200)
    why_it_matters: str = Field(..., min_length=1, max_length=1200)
    confidence: Literal["high", "medium", "low"]
    confidence_rationale: str = Field(..., min_length=1, max_length=300)
    representative_papers: List[RepresentativePaperOutput] = Field(..., min_length=1, max_length=5)
    reading_order: List[ReadingOrderItemOutput] = Field(..., min_length=1, max_length=10)
    search_query_seed: str = Field(..., min_length=1, max_length=200)
    notes: str = Field(..., max_length=600)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate tags: each tag 1-40 chars"""
        for i, tag in enumerate(v):
            if len(tag) < 1:
                raise ValueError(f"tags[{i}]: must be at least 1 char")
            if len(tag) > 40:
                raise ValueError(f"tags[{i}]: {len(tag)} chars (max 40)")
        return v


class OutputSchema(BaseModel):
    """Final output schema matching p0_output_json_schema.json"""
    snapshot_id: str
    cluster_run_id: str
    period_start: str
    period_end: str
    cluster_cards: List[ClusterCard] = Field(..., min_length=1)

