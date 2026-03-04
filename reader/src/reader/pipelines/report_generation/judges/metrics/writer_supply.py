"""Validation rules for ReportWriterSupplementOutput (W1)."""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from reader.pipelines.report_generation.judges.metrics.common import (
    ValidationReport,
    run_checks,
    word_count,
    is_question,
)
from reader.pipelines.report_generation.report import (
    ReportWriterSupplementOutput,
    WriterSupplementRequest,
)

# Word count limits from report_writer_fast_rules.md W1-S2
WHY_MIN_WORDS = 6
WHY_MAX_WORDS = 25

class WriterSupplyJudgeInput(BaseModel):
    """Judge input bundling output and validation context."""

    output: ReportWriterSupplementOutput
    available_paper_ids: List[str] = Field(default_factory=list)
    available_history_report_ids: List[str] = Field(default_factory=list)


def _get_output(j: WriterSupplyJudgeInput) -> ReportWriterSupplementOutput:
    return j.output


def _paper_ids(j: WriterSupplyJudgeInput) -> frozenset[str]:
    return frozenset(j.available_paper_ids)


def _history_report_ids(j: WriterSupplyJudgeInput) -> frozenset[str]:
    return frozenset(j.available_history_report_ids)


# ============================================================================
# Hard Rules (W1-H1 through W1-H5)
# ============================================================================


def _check_target_id_exclusive(j: WriterSupplyJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W1-H1: Exactly one target id per request."""
    for i, req in enumerate(j.output.supplements_requests):
        has_paper = req.paper_id is not None and (req.paper_id or "").strip() != ""
        has_history = req.history_report_id is not None and (req.history_report_id or "").strip() != ""
        if has_paper and has_history:
            return False, (
                "WriterSupplementRequest must have exactly one of paper_id or history_report_id set",
                f"got both set at index {i}",
            )
        if not has_paper and not has_history:
            return False, (
                "WriterSupplementRequest must have exactly one of paper_id or history_report_id set",
                f"got neither set at index {i}",
            )
    return True, None


def _check_selector_exclusivity(j: WriterSupplyJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W1-H2: Selector exclusivity matches target kind."""
    for i, req in enumerate(j.output.supplements_requests):
        try:
            if not req.has_valid_selectors:
                return False, (
                    "WriterSupplementRequest: if paper_id set then paper_selectors non-empty and history_selectors empty; if history_report_id set then opposite",
                    f"got invalid selector combination at index {i}",
                )
        except ValueError:
            return False, (
                "WriterSupplementRequest: if paper_id set then paper_selectors non-empty and history_selectors empty; if history_report_id set then opposite",
                f"got invalid selector combination at index {i}",
            )
    return True, None


def _check_id_whitelist(j: WriterSupplyJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W1-H3: paper_id and history_report_id must be in whitelist."""
    paper_ids = _paper_ids(j)
    history_ids = _history_report_ids(j)
    for i, req in enumerate(j.output.supplements_requests):
        if req.paper_id and (req.paper_id or "").strip():
            pid = (req.paper_id or "").strip()
            if pid not in paper_ids:
                return False, (
                    "paper_id must be in available_paper_ids; history_report_id must be in available_history_report_ids",
                    f"got invalid paper_id {pid!r} at index {i}",
                )
        if req.history_report_id and (req.history_report_id or "").strip():
            hid = (req.history_report_id or "").strip()
            if hid not in history_ids:
                return False, (
                    "paper_id must be in available_paper_ids; history_report_id must be in available_history_report_ids",
                    f"got invalid history_report_id {hid!r} at index {i}",
                )
    return True, None


def _check_no_empty_strings(j: WriterSupplyJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W1-H4: why non-empty; no empty selector strings."""
    for i, req in enumerate(j.output.supplements_requests):
        if not (req.why and req.why.strip()):
            return False, (
                "WriterSupplementRequest.why must not be empty or whitespace-only; no empty selector strings",
                f"got empty at supplements_requests[{i}].why",
            )
        for sel in req.paper_selectors or []:
            if not (sel and str(sel).strip()):
                return False, (
                    "WriterSupplementRequest.why must not be empty or whitespace-only; no empty selector strings",
                    f"got empty selector at supplements_requests[{i}]",
                )
        for sel in req.history_selectors or []:
            if not (sel and str(sel).strip()):
                return False, (
                    "WriterSupplementRequest.why must not be empty or whitespace-only; no empty selector strings",
                    f"got empty selector at supplements_requests[{i}]",
                )
    return True, None


HARD_CHECKS = (
    _check_target_id_exclusive,
    _check_selector_exclusivity,
    _check_id_whitelist,
    _check_no_empty_strings,
)


def hard_validate_writer_supply(j: WriterSupplyJudgeInput) -> ValidationReport:
    """Hard validation: score is 1 if all pass, else 0."""
    r = run_checks(j, HARD_CHECKS)
    return ValidationReport(score=1.0 if r.score == 1.0 else 0.0, reasons=r.reasons)


# ============================================================================
# Soft Rules (W1-S1, W1-S2, W1-S3, W1-S5)
# ============================================================================


def _check_why_is_question(j: WriterSupplyJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W1-S1: why should be phrased as a question."""
    for i, req in enumerate(j.output.supplements_requests):
        if not is_question(req.why or ""):
            return False, (
                "supplements_requests[].why should be phrased as a question (ends with ? or question starter)",
                f"got non-question phrasing at supplements_requests[{i}].why",
            )
    return True, None


def _check_why_word_count(j: WriterSupplyJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W1-S2: why 6-25 words."""
    for i, req in enumerate(j.output.supplements_requests):
        wc = word_count(req.why or "")
        if not (WHY_MIN_WORDS <= wc <= WHY_MAX_WORDS):
            return False, (
                "supplements_requests[].why should be 6–25 words",
                f"got {wc} words at supplements_requests[{i}].why",
            )
    return True, None


def _check_minimality(j: WriterSupplyJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W1-S3: Prefer 0-3 requests; penalize >3, strongly >6."""
    n = len(j.output.supplements_requests)
    if n > 6:
        return False, (
            "Prefer 0–3 supplement requests; penalize >3, strongly >6",
            f"got {n} requests (prefer ≤3)",
        )
    if n > 3:
        return False, (
            "Prefer 0–3 supplement requests; penalize >3, strongly >6",
            f"got {n} requests (prefer ≤3)",
        )
    return True, None


def _check_no_duplicates(j: WriterSupplyJudgeInput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """W1-S5: Avoid duplicate (target_id, selectors) requests."""
    seen: set[Tuple[str, Tuple[str, ...]]] = set()
    duplicates: List[int] = []
    for i, req in enumerate(j.output.supplements_requests):
        target_id = (req.paper_id or req.history_report_id or "").strip()
        selectors = tuple(
            sorted(
                list(req.paper_selectors or []) + list(req.history_selectors or [])
            )
        )
        key = (target_id, selectors)
        if key in seen:
            duplicates.append(i)
        seen.add(key)
    if duplicates:
        return False, (
            "Avoid duplicate (target_id, selectors) requests",
            f"got duplicate at indices {duplicates}",
        )
    return True, None


SOFT_CHECKS = (
    _check_why_is_question,
    _check_why_word_count,
    _check_minimality,
    _check_no_duplicates,
)


def soft_validate_writer_supply(j: WriterSupplyJudgeInput) -> ValidationReport:
    """Soft validation: runs all soft checks and returns average score."""
    return run_checks(j, SOFT_CHECKS)
