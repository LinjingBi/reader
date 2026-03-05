"""Pydantic models for pipeline outputs/reports"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Dict, Any, Optional, Set

from pydantic import BaseModel, Field, computed_field, model_validator

# ----------------------------
# llm report planner models
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



# ----------------------------
# llm report writer models
# ----------------------------
class WriterSupplementRequest(BaseModel):
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
            "A simple question that this request is trying to answer. "
            "Write it as a direct question (e.g., 'What is the evaluation protocol and baseline set?')."
        ),
    )

    @property
    def target_kind(self) -> Literal["paper", "history"]:
        if self.paper_id:
            return "paper"
        if self.history_report_id:
            return "history"
        raise ValueError("No target kind found for supplement request")

    @property
    def has_valid_selectors(self) -> bool:
        if self.target_kind == "paper":
            return len(self.paper_selectors) > 0 and len(self.history_selectors) == 0
        if self.target_kind == "history":
            return len(self.history_selectors) > 0 and len(self.paper_selectors) == 0
        return False



class ReportWriterSupplementOutput(BaseModel):
    supplements_requests: List[WriterSupplementRequest] = Field(
        default_factory=list,
        description="Supplements needed for writing. Empty list allowed.",
        max_length=10,
    )


# Input models for writer steps (used to build prompts)
class ReportWriterSupplementInput(BaseModel):
    """Input for supply step: decide what supplements to request for one outline item."""
    materials: Any = Field(..., description="GetReportGenerationMetadataResponse as dict")
    plan: Any = Field(..., description="LLMReportPlannerPlan as dict")
    target_outline: str = Field(..., description="Current outline item to write")
    available_ids: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Whitelist: paper_id list, report_id list",
    )


class ReportWriterSectionInput(BaseModel):
    """Input for write step: write one section with fetched supplements."""
    materials: Any = Field(..., description="GetReportGenerationMetadataResponse as dict")
    plan: Any = Field(..., description="LLMReportPlannerPlan as dict")
    target_section: str = Field(..., description="Current outline item to write")
    written_sections: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Draft so far: [{\"title\": ..., \"body\": ...}]",
    )
    allowed_citations: List[str] = Field(
        default_factory=list,
        description="Allowed citation tokens: [paper id: xxx], [report id: xxx], [section name: xxx]",
    )
    supplements: Any = Field(
        ...,
        description="GetReportGenerationSupplyResponse (paper_supplements, report_supplements)",
    )


class ReportWriterSectionOutput(BaseModel):
    section_name: str = Field(..., description="Section name (usually derived from outline item).")
    section_text: str = Field(..., description="The written section content.")
    confidence: List[Literal["high", "medium", "low"]] = Field(
        ...,
        description="confidence in writting materials and supplements support for this section.",
    )

class ReportWriterFrontMatterOutput(BaseModel):
    title: str = Field(
        ...,
        min_length=5,
        max_length=120,
        description="Reflects the dominant theme(concise, specific).",
    )
    summary: str = Field(
        ...,
        min_length=40,
        max_length=1200,
        description="Decision-oriented and faithful to REPORT_BODY. Generic; no citations needed.",
    )
    keywords: Set[str] = Field(
        ...,
        min_length=5,
        max_length=12,
        description="Noun phrases for the report.",
    )

    @model_validator(mode="after")
    def _keywords_case_insensitive_unique(self) -> "ReportWriterFrontMatterOutput":
        """Enforce case-insensitive uniqueness (W4-H1)."""
        if self.keywords and len(self.keywords) != len({k.lower() for k in self.keywords}):
            raise ValueError("keywords must be case-insensitively unique")
        return self


class ObservationReport(BaseModel):
    """Report body and front matter for serialization to local FS."""

    body: List[ReportWriterSectionOutput] = Field(..., description="Report sections.")
    front_matter: ReportWriterFrontMatterOutput = Field(..., description="Title, summary, keywords.")


class SaveReportToFsOutput(BaseModel):
    """Output of save_report_to_fs step."""

    report_path: str = Field(..., description="Full path to the written report JSON file.")
    signature: str = Field(..., description="SHA256 hex digest of the written file.")


# ----------------------------
# Topic resolver config (EmbedConfig-style for new_memory)
# ----------------------------

class TopicResolverConfigPayload(BaseModel):
    """Topic resolver configuration payload."""
    topic_resolver_threshold: float = Field(..., description="Similarity threshold (0-1) for topic resolution")


class TopicResolverConfig(BaseModel):
    """Topic resolver configuration with dynamic config ID."""
    json_payload: TopicResolverConfigPayload = Field(..., description="Topic resolver configuration payload")

    @computed_field
    @property
    def topic_resolver_config_id(self) -> str:
        """Get topic_resolver_config_id from algo_lib.topic_resolver version."""
        try:
            from algo_lib.topic_resolver import __version__ as topic_resolver_version
            return f"algo_lib.topic_resolver|{topic_resolver_version}"
        except ImportError:
            raise ValueError("algo_lib.topic_resolver is not versioned")
