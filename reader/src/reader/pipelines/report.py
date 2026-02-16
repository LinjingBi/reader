"""Pydantic models for pipeline outputs/reports"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Dict, Any, Optional

from pydantic import BaseModel, Field, computed_field, conlist

# ----------------------------
# Cluster Report Models(llm enriched report)
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
    what_this_topic_is_about: str = Field(
        ...,
        description=(
            "Describe the shared theme using only provided information. Explain how multiple papers relate. "
            f"Target {ABOUT_MIN_WORDS}–{ABOUT_MAX_WORDS} words. Include inline citations [paper_id]. "

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
            "Bullet list justifying confidence using the group size, cohesion, and evidence quality. "
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

# ----------------------------
# cluster semantic models from llm summarization(for memo cluster injection)
# ----------------------------


class LLMConfigInput(BaseModel):
    """LLM config input matching the llm_config table structure."""
    llm_config_id: str = Field(..., description="LLM config ID in format: model|prompt_template")
    json_payload: Dict[str, Any] = Field(..., description="LLM config JSON payload with provider, model, temperature, max_tokens, endpoint")


class ClusterObservation(BaseModel):
    """Observation data for a single cluster."""
    llm_config: LLMConfigInput = Field(..., description="LLM configuration used")
    payload_json: Dict[str, Any] = Field(..., description="Cluster observation payload JSON")
    summary: str = Field(..., description="Cluster summary")
    title: str = Field(..., description="Cluster title")
    keywords_json: List[str] = Field(..., description="Keywords as JSON list")


# Type alias for inject-clusters-observation input (map of pk_hash -> ClusterObservation)
InjectClustersObservationInput = Dict[str, ClusterObservation]



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

# TODO: generate from memo dynamically
PaperSelector = Literal[
    "summary",
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

HistoryReportSelector = Literal[
    "summary",
    "covered_bullets",
    "next_targets",
    "subthreads",
    "outline",
    "evidence_gaps",
    "plan",
    "full_json",
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
            return len(self.selectors) > 0 and len(self.history_selectors) == 0
        if self.target_kind == "history":
            return len(self.history_selectors) > 0 and len(self.selectors) == 0
        return False

class LLMReportPlannerOutput(BaseModel):
    plan: LLMReportPlannerPlan
    next_step_inputs: List[NextStepInput]
    evidence_gaps: List[EvidenceGap]
