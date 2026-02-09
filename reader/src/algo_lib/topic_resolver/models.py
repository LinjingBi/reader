"""
Pydantic models for topic resolver inputs and outputs.
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class TopicResolveAction(str, Enum):
    """Action type for topic resolution."""
    CREATE = "create"
    MERGE = "merge"


class TopicInput(BaseModel):
    """Input model for a topic."""
    id: str = Field(..., description="Topic ID")
    centroid_b64: str = Field(..., description="Topic centroid as base64-encoded float32 bytes")
    centroid_weight: float = Field(..., gt=0, description="Topic centroid weight (must be positive)")


class ClusterInput(BaseModel):
    """Input model for a cluster."""
    id: str = Field(..., description="Cluster ID (cluster pk hash)")
    centroid_b64: str = Field(..., description="Cluster centroid as base64-encoded float32 bytes")
    centroid_weight: float = Field(..., gt=0, description="Cluster centroid weight (must be positive)")


class TopicResolveOutput(BaseModel):
    """Output model for topic resolution result."""
    action: TopicResolveAction = Field(
        ...,
        description=f"Action taken: {', '.join([action.value.upper() for action in TopicResolveAction])}"
    )
    merge_to_topic: str | None = Field(
        None,
        description=f"Topic ID if action is {TopicResolveAction.MERGE.value.upper()}, None if {TopicResolveAction.CREATE.value.upper()}"
    )
    new_topic_centroid_b64: str = Field(..., description="New topic centroid as base64-encoded float32 bytes (normalized)")
    new_topic_weight: float = Field(..., gt=0, description="New topic weight (combined weight if merge, cluster weight if create)")
    score: float = Field(..., ge=0.0, le=1.0, description="Best similarity score. If merge: top 1 sim, if create: 1.0")

