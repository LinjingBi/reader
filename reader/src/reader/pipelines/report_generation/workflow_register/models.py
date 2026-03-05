"""Data models for workflow register."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel


class StepRunStatus(str, Enum):
    """Status for step nodes."""

    not_run = "not_run"
    done = "done"
    error = "error"


class LoopRunStatus(str, Enum):
    """Status for loop nodes."""

    not_run = "not_run"
    complete = "complete"
    partial = "partial"
    error = "error"


class StepNodeRecord(BaseModel):
    """Record for a step node."""

    node_id: str
    node_type: Literal["step"] = "step"
    status: StepRunStatus = StepRunStatus.not_run
    output: Optional[dict] = None  # Serialized output; None if not_run or error with no output


class LoopNodeRecord(BaseModel):
    """Record for a loop node."""

    node_id: str
    node_type: Literal["loop"] = "loop"
    status: LoopRunStatus = LoopRunStatus.not_run
    output: Optional[dict] = None  # Serialized output


class WorkflowNodeDef(BaseModel):
    """Workflow graph node definition."""

    id: str
    kind: Literal["step", "loop"]
    display_name: str
    contains: Optional[list[str]] = None  # IDs of child loop nodes (for steps that wrap loops)


class WorkflowTraceNode(BaseModel):
    """Node entry in the trace report output."""

    id: str
    kind: Literal["step", "loop"]
    display_name: Optional[str] = None
    contains: Optional[list[str]] = None
    status: str
    output: Optional[dict] = None


class WorkflowTraceReport(BaseModel):
    """Full trace report for workflow rerun support."""

    workflow_id: str
    cluster_pk_hash: str
    timestamp: str
    config: dict  # Serialized ReportGenerationConfig (model_dump(mode="json"))
    nodes: list[WorkflowTraceNode]
