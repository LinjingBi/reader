"""Pydantic models for pipeline outputs/reports"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Dict, Any, Optional

from pydantic import BaseModel, Field, computed_field, conlist

# ----------------------------
# llm report generation models
# ----------------------------

# ---------- Enums ----------
class LLMReportPlannerDepthMode(str, Enum):
    Onboard = "Onboard"
    Continue = "Continue"
    Deepen = "Deepen"
    Restructure = "Restructure"


class LLMReportPlannerDeclaredLevel(str, Enum):
    intro = "intro"
    intermediate = "intermediate"
    deep_dive = "deep-dive"


class LLMReportPlannerSufficiency(str, Enum):
    sufficient = "sufficient"
    borderline = "borderline"
    insufficient = "insufficient"


# Subset of LLMReportPlannerSufficiency: sufficient and borderline allow evidence collection termination.
EvidenceCollectionTerminationSufficiency = [
    LLMReportPlannerSufficiency.sufficient,
    LLMReportPlannerSufficiency.borderline,
]


# ---------- Output Models ----------
class LLMReportPlannerSubthread(BaseModel):
    name: str = Field(description="A thematic bucket name grounded in evidence keywords/themes.")
    paper_ids: List[str] = Field(default_factory=list, description="Paper ids included in this subthread, if available.")


# do not add description for each field, they are defined in the "spec.py"
class LLMReportPlannerPlan(BaseModel):
    depth_mode_final: LLMReportPlannerDepthMode
    declared_level_final: LLMReportPlannerDeclaredLevel

    subthreads_final: List[LLMReportPlannerSubthread] = Field(min_length=2, max_length=4)

    next_targets: List[str] = Field(min_length=3, max_length=8)
    outline: List[str] = Field(min_length=6, max_length=12)
    skip_or_defer: List[str] = Field(default_factory=list, max_length=5)

    sufficiency: LLMReportPlannerSufficiency

# detail-level selectors, no summary level selectors
PaperSelector = Literal[
    "introduction",
    "related_work",
    "method",
    "experiment",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    # "appendix",
    # "full_text",
]
# detail-level selectors, no summary level selectors
HistoryReportSelector = Literal[
    "covered_bullets",
    "next_targets",
    "subthreads",
    "outline",
    "evidence_gaps",
    "sufficiency",
    # "plan",
    # "full_text",  # unsupported for supplement lookup in v0 (no report table column; memo cmd rejects)
]


SupportField = Literal[
    # Plan fields:
    "depth_mode_final",
    "declared_level_final",
    "subthreads_final",
    "outline",
    "next_targets",
    "skip_or_defer",
    "sufficiency",
    "evidence_gaps",
    # # (Optional) if you later add writer-facing fields:
    # "writer_section",
    # "cross_paper_comparison",
]


class NextStepInput(BaseModel):

    support_field: SupportField = Field(..., description="Which plan field / writing purpose this input supports.")
    # Retrieval target (exactly one should be set)
    paper_id: Optional[str] = Field(None, description="Paper to fetch from. Omit for history-only needs.")
    history_report_id: Optional[str] = Field(
        None,
        description="History report identifier to fetch from (if you have ids). Omit for paper-only needs.",
    )
    # Retrieval selectors (use the one that matches the target)
    paper_selectors: List[PaperSelector] = Field(default_factory=list, description="Which paper part(s) to extract.")
    history_selectors: List[HistoryReportSelector] = Field(default_factory=list, description="Which history fields to extract.")

    # intentionally a bit ambiguous — lets the model express uncertainty plainly. this can be used to calibrate the workflow understanding vs the model's understanding.
    why: str = Field(
        ...,
        description=(
            "A simple question that this input is trying to answer. "
            "Write it as a direct question (e.g., 'What is the evaluation protocol and baseline set?')."
        ),
    )
    @property
    def target_kind(self) -> Literal["paper", "history"]:
        if self.paper_id:
            return "paper"
        if self.history_report_id:
            return "history"
        raise ValueError("No target kind found for next step input")

    @property
    def has_valid_selectors(self) -> bool:
        if self.target_kind == "paper":
            return len(self.paper_selectors) > 0 and len(self.history_selectors) == 0
        if self.target_kind == "history":
            return len(self.history_selectors) > 0 and len(self.paper_selectors) == 0
        return False


class EvidenceGap(BaseModel):
    # intentionally a bit ambiguous — lets the model express uncertainty plainly. this can be used to calibrate the workflow understanding vs the model's understanding.
    why: str = Field(..., description="Why this gap matters (what it blocks or could cause hallucination).")

    blocked_fields: List[str]
    paper_id: Optional[str] = Field(None, description="Paper to fetch from. Omit for history-only needs.")
    history_report_id: Optional[str] = Field(
        None,
        description="History report identifier to fetch from (if you have ids). Omit for paper-only needs.",
    )
    paper_selectors: List[PaperSelector] = Field(default_factory=list, description="Which paper part(s) to extract.")
    history_selectors: List[HistoryReportSelector] = Field(default_factory=list, description="Which history fields to extract.")
    priority: Literal[1, 2, 3] = Field(..., description="1=highest, 3=lowest urgency.")

    @property
    def target_kind(self) -> Literal["paper", "history"]:
        if self.paper_id:
            return "paper"
        if self.history_report_id:
            return "history"
        raise ValueError("No target kind found for evidence gap")

    @property
    def has_valid_selectors(self) -> bool:
        if self.target_kind == "paper":
            return len(self.paper_selectors) > 0 and len(self.history_selectors) == 0
        if self.target_kind == "history":
            return len(self.history_selectors) > 0 and len(self.paper_selectors) == 0
        return False

class LLMReportPlannerOutput(BaseModel):
    plan: LLMReportPlannerPlan
    # next_step_inputs: List[NextStepInput]
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list)
