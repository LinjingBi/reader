"""Report generation pipeline metrics.

Validation rules for LLMReportPlannerOutput. Rules are organized per field.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, get_args

from reader.pipelines.report_generation.judges.metrics.common import (
    ValidationReport,
    run_checks,
    word_count,
)
from reader.pipelines.report_generation.report import (
    LLMReportPlannerOutput,
    SupportField,
    LLMReportPlannerSufficiency,
)


# ============================================================================
# Constants
# ============================================================================

KNOWN_SUPPORT_FIELDS = frozenset(get_args(SupportField))

# Word count limits from call1_validation_rules.md
EVIDENCE_GAP_WHY_MIN_WORDS = 6
EVIDENCE_GAP_WHY_MAX_WORDS = 35
OUTLINE_MAX_WORDS_PER_ITEM = 18
NEXT_TARGETS_MIN_WORDS = 3
NEXT_TARGETS_MAX_WORDS = 16

# Generic subthread names to reject
GENERIC_SUBTHREAD_NAMES = frozenset(
    w.lower()
    for w in ("misc", "others", "general", "various", "stuff", "tbd", "unknown", "ai", "ml", "paper")
)

# Narrative phrasing to avoid in next_targets
NARRATIVE_PHRASE_PREFIX = "in this report we will"


CheckFn = Callable[[LLMReportPlannerOutput], Tuple[bool, Optional[Tuple[str, str]]]]


# ============================================================================
# Hard Rules (per field)
# ============================================================================


def check_evidence_gap_target_id(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """EvidenceGap must have exactly one of paper_id or history_report_id set"""
    for i, eg in enumerate(output.evidence_gaps):
        has_paper = eg.paper_id is not None and eg.paper_id.strip() != ""
        has_history = eg.history_report_id is not None and eg.history_report_id.strip() != ""
        if has_paper and has_history:
            return False, (
                "EvidenceGap must not have both paper_id and history_report_id set",
                f"got both set at index {i}",
            )
        elif not has_paper and not has_history:
            return False, (
                "EvidenceGap must have at least one of paper_id or history_report_id set",
                f"got neither set at index {i}",
            )
    return True, None


def check_evidence_gap_selector_exclusivity(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """EvidenceGap: if paper_id set then paper_selectors non-empty and history_selectors empty; if history_report_id set then opposite"""
    for i, eg in enumerate(output.evidence_gaps):
        try:
            if not eg.has_valid_selectors:
                return False, (
                    "EvidenceGap: if paper_id set then paper_selectors non-empty and history_selectors empty; if history_report_id set then opposite",
                    f"got invalid selector combination at index {i}",
                )
        except ValueError:
            return False, (
                "EvidenceGap: if paper_id set then paper_selectors non-empty and history_selectors empty; if history_report_id set then opposite",
                f"got invalid selector combination at index {i}",
            )
    return True, None


def check_subthread_name_non_empty(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """LLMReportPlannerSubthread.name must not be empty or whitespace-only"""
    for i, st in enumerate(output.plan.subthreads_final):
        if not (st.name and st.name.strip()):
            return False, (
                "LLMReportPlannerSubthread.name must not be empty or whitespace-only",
                f"got empty at subthreads_final[{i}].name",
            )
    return True, None


def check_next_targets_non_empty(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.next_targets items must not be empty or whitespace-only"""
    for i, item in enumerate(output.plan.next_targets):
        if not (item and str(item).strip()):
            return False, (
                "plan.next_targets items must not be empty or whitespace-only",
                f"got empty at next_targets[{i}]",
            )
    return True, None


def check_outline_non_empty(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.outline items must not be empty or whitespace-only"""
    for i, item in enumerate(output.plan.outline):
        if not (item and str(item).strip()):
            return False, (
                "plan.outline items must not be empty or whitespace-only",
                f"got empty at outline[{i}]",
            )
    return True, None


def check_skip_or_defer_non_empty(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.skip_or_defer items must not be empty or whitespace-only"""
    for i, item in enumerate(output.plan.skip_or_defer):
        if not (item and str(item).strip()):
            return False, (
                "plan.skip_or_defer items must not be empty or whitespace-only",
                f"got empty at skip_or_defer[{i}]",
            )
    return True, None


def check_evidence_gap_why_non_empty(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """EvidenceGap.why must not be empty or whitespace-only"""
    for i, eg in enumerate(output.evidence_gaps):
        if not (eg.why and eg.why.strip()):
            return False, (
                "EvidenceGap.why must not be empty or whitespace-only",
                f"got empty at evidence_gaps[{i}].why",
            )
    return True, None


def check_evidence_gap_blocked_fields_non_empty(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """EvidenceGap.blocked_fields items must not be empty or whitespace-only"""
    for i, eg in enumerate(output.evidence_gaps):
        for j, bf in enumerate(eg.blocked_fields):
            if not (bf and str(bf).strip()):
                return False, (
                    "EvidenceGap.blocked_fields items must not be empty or whitespace-only",
                    f"got empty at evidence_gaps[{i}].blocked_fields[{j}]",
                )
    return True, None


def check_subthread_names_unique(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """LLMReportPlannerSubthread.name must be unique (case-insensitive)"""
    seen: set[str] = set()
    for i, st in enumerate(output.plan.subthreads_final):
        name_lower = (st.name or "").strip().lower()
        if name_lower in seen:
            return False, (
                "LLMReportPlannerSubthread.name must be unique (case-insensitive)",
                f"got duplicate subthread name: {st.name!r}",
            )
        seen.add(name_lower)
    return True, None


def check_outline_unique(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.outline items must be unique"""
    seen: set[str] = set()
    for i, item in enumerate(output.plan.outline):
        val = (item or "").strip()
        if val in seen:
            return False, (
                "plan.outline items must be unique",
                f"got duplicate outline item: {item!r}",
            )
        seen.add(val)
    return True, None


def check_next_targets_unique(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.next_targets items must be unique"""
    seen: set[str] = set()
    for i, item in enumerate(output.plan.next_targets):
        val = (item or "").strip()
        if val in seen:
            return False, (
                "plan.next_targets items must be unique",
                f"got duplicate next_targets item: {item!r}",
            )
        seen.add(val)
    return True, None


HARD_CHECKS: tuple[CheckFn, ...] = (
    check_evidence_gap_target_id,
    check_evidence_gap_selector_exclusivity,
    check_subthread_name_non_empty,
    check_next_targets_non_empty,
    check_outline_non_empty,
    check_skip_or_defer_non_empty,
    check_evidence_gap_why_non_empty,
    check_evidence_gap_blocked_fields_non_empty,
    check_subthread_names_unique,
    check_outline_unique,
    check_next_targets_unique,
)


def hard_validate_planner_output(output: LLMReportPlannerOutput) -> ValidationReport:
    """Hard validation: score is 1 if all pass, else 0."""
    r = run_checks(output, HARD_CHECKS)
    return ValidationReport(score=1.0 if r.score == 1.0 else 0.0, reasons=r.reasons)


# ============================================================================
# Soft Rules (per field)
# ============================================================================


def check_evidence_gap_why_word_count(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """EvidenceGap.why must be 6-35 words"""
    for i, eg in enumerate(output.evidence_gaps):
        wc = word_count(eg.why or "")
        if not (EVIDENCE_GAP_WHY_MIN_WORDS <= wc <= EVIDENCE_GAP_WHY_MAX_WORDS):
            return False, (
                "EvidenceGap.why must be 6-35 words",
                f"got {wc} words at evidence_gaps[{i}].why",
            )
    return True, None


def check_outline_headings_word_count(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.outline items must be ≤18 words"""
    for i, item in enumerate(output.plan.outline):
        t = (item or "").strip()
        wc = word_count(t)
        if wc > OUTLINE_MAX_WORDS_PER_ITEM:
            return False, (
                "plan.outline items must be ≤18 words",
                f"got {wc} words at outline[{i}]",
            )
    return True, None


def check_outline_headings_single_line(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.outline items must be single line"""
    for i, item in enumerate(output.plan.outline):
        t = (item or "").strip()
        if "\n" in t:
            return False, (
                "plan.outline items must be single line",
                f"got multiline at outline[{i}]",
            )
    return True, None


def check_outline_headings_no_trailing_period(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.outline items must not end with trailing period"""
    for i, item in enumerate(output.plan.outline):
        t = (item or "").strip()
        if t.endswith("."):
            return False, (
                "plan.outline items must not end with trailing period",
                f"got trailing period at outline[{i}]",
            )
    return True, None


def check_next_targets_actionable_word_count(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.next_targets items must be 3-16 words"""
    for i, item in enumerate(output.plan.next_targets):
        t = (item or "").strip()
        wc = word_count(t)
        if not (NEXT_TARGETS_MIN_WORDS <= wc <= NEXT_TARGETS_MAX_WORDS):
            snippet = t[:60] + ("..." if len(t) > 60 else "")
            return False, (
                "plan.next_targets items must be 3-16 words",
                f"got {wc} words at next_targets[{i}]: {snippet!r}",
            )
    return True, None


def check_next_targets_actionable_single_line(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.next_targets items must be single line"""
    for i, item in enumerate(output.plan.next_targets):
        t = (item or "").strip()
        if "\n" in t:
            return False, (
                "plan.next_targets items must be single line",
                f"got multiline at next_targets[{i}]",
            )
    return True, None


def check_next_targets_actionable_no_trailing_period(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.next_targets items must not end with trailing period"""
    for i, item in enumerate(output.plan.next_targets):
        t = (item or "").strip()
        if t.endswith("."):
            return False, (
                "plan.next_targets items must not end with trailing period",
                f"got trailing period at next_targets[{i}]",
            )
    return True, None


def check_next_targets_actionable_no_narrative_phrasing(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.next_targets items must not use narrative phrasing like 'In this report we will...'"""
    for i, item in enumerate(output.plan.next_targets):
        t = (item or "").strip()
        if t.lower().startswith(NARRATIVE_PHRASE_PREFIX):
            return False, (
                "plan.next_targets items must not use narrative phrasing like 'In this report we will...'",
                f"got narrative phrasing at next_targets[{i}]: {t[:50]!r}...",
            )
    return True, None


def check_skip_or_defer_no_overlap(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.skip_or_defer must not overlap plan.next_targets (case-insensitive)"""
    next_lower = {s.strip().lower() for s in output.plan.next_targets if s and s.strip()}
    skip_lower = {s.strip().lower() for s in output.plan.skip_or_defer if s and s.strip()}
    overlap = next_lower & skip_lower
    if overlap:
        return False, (
            "plan.skip_or_defer must not overlap plan.next_targets (case-insensitive)",
            f"got overlap: {list(overlap)[:5]}{'...' if len(overlap) > 5 else ''}",
        )
    return True, None


def check_subthread_name_specific(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """LLMReportPlannerSubthread.name must be specific, not a placeholder"""
    for i, st in enumerate(output.plan.subthreads_final):
        name_lower = (st.name or "").strip().lower()
        if name_lower in GENERIC_SUBTHREAD_NAMES:
            return False, (
                "LLMReportPlannerSubthread.name must be specific, not a placeholder",
                f"got generic name at subthreads_final[{i}].name: {st.name!r}",
            )
    return True, None


def check_evidence_gap_target_id_unique(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """Each paper_id or history_report_id must appear in evidence_gaps at most once"""
    id_to_indices: Dict[str, List[int]] = {}
    for i, eg in enumerate(output.evidence_gaps):
        target_id = (eg.paper_id or eg.history_report_id or "").strip()
        if target_id:
            id_to_indices.setdefault(target_id, []).append(i)
    duplicates = {tid: idxs for tid, idxs in id_to_indices.items() if len(idxs) > 1}
    if not duplicates:
        return True, None
    dup_ids = sorted(duplicates.keys())
    return False, (
        "Each paper_id or history_report_id must appear in evidence_gaps at most once",
        f"got duplicate: {dup_ids}",
    )


def check_subthread_paper_id_duplication(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """LLMReportPlannerSubthread.paper_ids must not have excessive reuse across subthreads"""
    subthreads = output.plan.subthreads_final
    if len(subthreads) < 2:
        return True, None
    # Count paper_id occurrences across all subthreads
    paper_to_subthreads: Dict[str, List[int]] = {}
    for i, st in enumerate(subthreads):
        for pid in st.paper_ids:
            if pid and pid.strip():
                paper_to_subthreads.setdefault(pid, []).append(i)
    # Penalize if any paper_id appears in more than half of subthreads
    threshold = max(1, len(subthreads) // 2)
    overused = [pid for pid, indices in paper_to_subthreads.items() if len(indices) > threshold]
    if overused:
        return False, (
            "LLMReportPlannerSubthread.paper_ids must not have excessive reuse across subthreads",
            f"got {len(overused)} paper_ids shared across >{threshold} subthreads",
        )
    return True, None


def check_sufficiency_sufficient_evidence_gaps_minimal(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.sufficiency: if sufficient then evidence gaps must be minimal"""
    suf = output.plan.sufficiency
    if suf != LLMReportPlannerSufficiency.sufficient:
        return True, None
    val = suf.value if hasattr(suf, "value") else str(suf)
    if len(output.evidence_gaps) > 3:  # "minimal" = allow a few
        return False, (
            "plan.sufficiency: if sufficient then evidence gaps must be minimal",
            f"got sufficiency={val} but {len(output.evidence_gaps)} evidence_gaps (expected minimal)",
        )
    return True, None


def check_sufficiency_insufficient_non_empty(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.sufficiency: if insufficient then evidence_gaps must be non-empty"""
    suf = output.plan.sufficiency
    if suf != LLMReportPlannerSufficiency.insufficient:
        return True, None
    val = suf.value if hasattr(suf, "value") else str(suf)
    if not output.evidence_gaps:
        return False, (
            "plan.sufficiency: if insufficient then evidence_gaps must be non-empty",
            f"got sufficiency={val} but empty evidence_gaps",
        )
    return True, None


def check_evidence_gap_blocked_fields_known(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """EvidenceGap.blocked_fields must reference known plan/support fields"""
    for i, eg in enumerate(output.evidence_gaps):
        for bf in eg.blocked_fields:
            bf_stripped = (bf or "").strip()
            if bf_stripped and bf_stripped not in KNOWN_SUPPORT_FIELDS:
                return False, (
                    "EvidenceGap.blocked_fields must reference known plan/support fields",
                    f"got unknown blocked_field at evidence_gaps[{i}]: {bf!r}",
                )
    return True, None


def check_outline_distribution_sanity(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.outline must not have repetitive phrasing or identical structural artifacts"""
    items = output.plan.outline
    if len(items) < 2:
        return True, None
    first_words = []
    for item in items:
        t = (item or "").strip()
        words = t.split()
        first_words.append(words[0].lower() if words else "")
    if len(set(first_words)) == 1 and first_words[0]:
        return False, (
            "plan.outline must not have repetitive phrasing or identical structural artifacts",
            "got repetitive pattern: all outline items start with same prefix",
        )
    return True, None


def check_next_targets_distribution_sanity(output: LLMReportPlannerOutput) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """plan.next_targets must not have repetitive phrasing or identical structural artifacts"""
    items = output.plan.next_targets
    if len(items) < 2:
        return True, None
    first_words = []
    for item in items:
        t = (item or "").strip()
        words = t.split()
        first_words.append(words[0].lower() if words else "")
    if len(set(first_words)) == 1 and first_words[0]:
        return False, (
            "plan.next_targets must not have repetitive phrasing or identical structural artifacts",
            "got repetitive pattern: all next_targets items start with same prefix",
        )
    return True, None


SOFT_CHECKS: tuple[CheckFn, ...] = (
    check_evidence_gap_why_word_count,
    check_outline_headings_word_count,
    check_outline_headings_single_line,
    check_outline_headings_no_trailing_period,
    check_next_targets_actionable_word_count,
    check_next_targets_actionable_single_line,
    check_next_targets_actionable_no_trailing_period,
    check_next_targets_actionable_no_narrative_phrasing,
    check_skip_or_defer_no_overlap,
    check_subthread_name_specific,
    check_subthread_paper_id_duplication,
    check_sufficiency_sufficient_evidence_gaps_minimal,
    check_sufficiency_insufficient_non_empty,
    check_evidence_gap_blocked_fields_known,
    check_evidence_gap_target_id_unique,
    check_outline_distribution_sanity,
    check_next_targets_distribution_sanity,
)


def soft_validate_planner_output(output: LLMReportPlannerOutput) -> ValidationReport:
    """Soft validation: runs all soft checks and returns average score."""
    return run_checks(output, SOFT_CHECKS)
