"""Pipeline building blocks"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from algo_lib.topic_resolver.models import TopicInput, ClusterInput as TopicResolverClusterInput, TopicResolveOutput, TopicResolveAction
from algo_lib.topic_resolver.resolver import resolve_topic
from algo_lib.topic_resolver.errors import TopicResolverError

from reader.pipelines.report_generation.config.config import ReportGenerationConfig, LLMGeminiConfig
from reader.adapters import memo
from reader.pipelines.report_generation.db.store import init_report_job, ReportJobStore, ReportJobStatus
from reader.adapters.memo import (
    ClusterObservationData,
    FrontMatterInput,
    GetReportGenerationMetadataResponse,
    GetReportGenerationSupplyRequest,
    GetReportGenerationSupplyResponse,
    MemoGetReportGenerationSupplyError,
    MemoNewMemoryError,
    NewMemoryRequest,
    NewMemoryResponse,
    PaperSupplementRequest,
    ReportSupplementRequest,
    ResolvedTopicInput,
    SaveMemoryInput,
    TopicResolverConfigInput,
)
from reader.adapters.llm import (
    LLMClient,
    TokenBucket,
    LLMGenerationError,
)
from reader.pipelines.report_generation.judges.protocol import JudgeLoopExitCondition
from reader.pipelines.report_generation.report import (
    LLMClientWrapper,
    LLMReportPlannerOutput,
    EvidenceCollectionTerminationSufficiency,
    EvidenceGap,
    ObservationReport,
    ReportJobAction,
    ReportWriterFrontMatterOutput,
    ReportWriterSectionInput,
    ReportWriterSectionOutput,
    ReportWriterSupplementInput,
    ReportWriterSupplementOutput,
    SaveReportToFsOutput,
    TopicResolverConfig,
    TopicResolverConfigPayload,
    WriterSupplementRequest,
)
from reader.pipelines.report_generation.judges import (
    LLMJudge,
    front_matter_judge,
    planner_judge,
    WriterSupplyJudgeWrapper,
    WriterWritingJudgeWrapper,
)
from reader.logging.logging_setup import get_logger
from reader.pipelines.report_generation.prompts.planner.build import UserIntent, get_planner_intent_spec, build_plan_guidance, build_planner_prompt
from reader.pipelines.report_generation.prompts.writer.build import (
    build_evidence_requests_prompt,
    build_section_writing_prompt,
    build_summary_writing_prompt,
)
from reader.tui.clusters_observation import display_clusters_observation
from reader.pipelines.report_generation.workflow_register import (
    LoopRunStatus,
    record_loop,
    record_step,
    with_workflow_register,
)

logger = get_logger()

class ReportGenerationRuntimeError(Exception):
    """Exception for report generation runtime errors."""
    pass


# ---------- Judge retry helper ----------


def _should_exit_judge_loop(
    judge_result,
    attempt: int,
    max_retries: int,
    retry_threshold: float,
) -> Optional[JudgeLoopExitCondition]:
    """Decide whether the judge retry loop should conclude."""
    if judge_result.overall > retry_threshold:
        return JudgeLoopExitCondition.JUDGE_ACCEPTED
    if attempt >= max_retries:
        return JudgeLoopExitCondition.RETRIES_EXHAUSTED
    return None


async def _call_llm_with_judge_retry(
    llm_client: LLMClient,
    prompt: str,
    response_model,
    temperature: float,
    max_tokens: int,
    judge,
    item_pk: str,
    max_retries: int,
    retry_threshold: float,
    log_path: Optional[str] = None,
):
    """
    Call LLM with structured output and judge retry logic.
    Retries until judge accepts (overall > retry_threshold) or max_retries exhausted.
    Returns (best_output, status).
    """
    loop_prefix = f"[judge {judge.name}] - [cluster {item_pk}]"
    logger.info(f"{loop_prefix} - start")

    prompt_base = prompt
    best_output = None
    best_score = float("-inf")
    exit_reason: Optional[JudgeLoopExitCondition] = None

    for attempt in range(max_retries + 1):
        try:
            output = await llm_client.call_structured_async(
                prompt=prompt,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            judge_result = judge.judge(output)
            if log_path and item_pk:
                judge.log_to_jsonl(log_path, item_pk, output, judge_result)

            if judge_result.overall > best_score:
                best_score = judge_result.overall
                best_output = output

            exit_reason = _should_exit_judge_loop(
                judge_result, attempt, max_retries, retry_threshold
            )
            if exit_reason is not None:
                if exit_reason == JudgeLoopExitCondition.JUDGE_ACCEPTED:
                    logger.info(
                        f"{loop_prefix} - attempt {attempt + 1}/{max_retries + 1}: "
                        f"overall score {best_score:.2f} > {retry_threshold}, accepted"
                    )
                elif exit_reason == JudgeLoopExitCondition.RETRIES_EXHAUSTED:
                    logger.warning(
                        f"{loop_prefix} - attempt {attempt + 1}/{max_retries + 1}: "
                        f"judge retries exhausted, returning best output (overall score: {best_score:.2f})"
                    )
                break

            if attempt < max_retries:
                prompt, num_warnings = judge.inject_warnings_into_prompt(
                    prompt_base, judge_result
                )
                logger.warning(
                    f"{loop_prefix} - attempt {attempt + 1}/{max_retries + 1}: "
                    f"overall score {judge_result.overall:.2f} <= {retry_threshold}, "
                    f"injecting {num_warnings} warning(s), retrying"
                )
            else:
                num_warnings = judge.count_warnings(judge_result)
                logger.warning(
                    f"{loop_prefix} - attempt {attempt + 1}/{max_retries + 1}: "
                    f"overall score {judge_result.overall:.2f} <= {retry_threshold}, "
                    f"{num_warnings} warning(s) (last judge retry, returning best)"
                )

        except LLMGenerationError as e:
            logger.error(
                f"{loop_prefix} - attempt {attempt + 1}/{max_retries + 1}: llm call failed, breaking retry: {e}"
            )
            exit_reason = JudgeLoopExitCondition.LLM_ERROR
            break
        except Exception as e:
            logger.error(
                f"{loop_prefix} - attempt {attempt + 1}/{max_retries + 1}: unexpected error, breaking retry: {e}",
                exc_info=True,
            )
            exit_reason = JudgeLoopExitCondition.ERROR
            break

    none_output = best_output is None
    if not none_output and exit_reason == JudgeLoopExitCondition.JUDGE_ACCEPTED:
        status = LoopRunStatus.complete
    elif not none_output and exit_reason == JudgeLoopExitCondition.RETRIES_EXHAUSTED:
        status = LoopRunStatus.partial
    elif not none_output and exit_reason in (
        JudgeLoopExitCondition.LLM_ERROR,
        JudgeLoopExitCondition.ERROR,
    ):
        status = LoopRunStatus.partial
        logger.warning(
            f"{loop_prefix} - terminated due to an error. The returned result is from the best overall score llm call."
        )
    elif none_output and exit_reason in (
        JudgeLoopExitCondition.LLM_ERROR,
        JudgeLoopExitCondition.ERROR,
    ):
        status = LoopRunStatus.error
    else:
        error_msg = (
            f"{loop_prefix} - undefined exit reason({exit_reason}) and best_output(none:{none_output}) "
            "condition for exit status resolution."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"{loop_prefix} - finished, reasons: {exit_reason}, status: {status.value}, empty result: {none_output}"
    )
    return (best_output, status)


@record_loop("planner_judge_loop")
async def _call_planner_judge_retry(
    llm_client: LLMClient,
    prompt: str,
    response_model,
    temperature: float,
    max_tokens: int,
    judge,
    item_pk: str,
    max_retries: int,
    retry_threshold: float,
    log_path: Optional[str] = None,
):
    """Wrapper for planner judge retry; records as planner_judge_loop."""
    return await _call_llm_with_judge_retry(
        llm_client,
        prompt=prompt,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        item_pk=item_pk,
        max_retries=max_retries,
        retry_threshold=retry_threshold,
        log_path=log_path,
    )


@record_loop("writer_supply_judge_loop")
async def _call_writer_supply_judge_retry(
    llm_client: LLMClient,
    prompt: str,
    response_model,
    temperature: float,
    max_tokens: int,
    judge,
    item_pk: str,
    max_retries: int,
    retry_threshold: float,
    log_path: Optional[str] = None,
):
    """Wrapper for writer supply judge retry; records as writer_supply_judge_loop."""
    return await _call_llm_with_judge_retry(
        llm_client,
        prompt=prompt,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        item_pk=item_pk,
        max_retries=max_retries,
        retry_threshold=retry_threshold,
        log_path=log_path,
    )


@record_loop("writer_writing_judge_loop")
async def _call_writer_writing_judge_retry(
    llm_client: LLMClient,
    prompt: str,
    response_model,
    temperature: float,
    max_tokens: int,
    judge,
    item_pk: str,
    max_retries: int,
    retry_threshold: float,
    log_path: Optional[str] = None,
):
    """Wrapper for writer writing judge retry; records as writer_writing_judge_loop."""
    return await _call_llm_with_judge_retry(
        llm_client,
        prompt=prompt,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        item_pk=item_pk,
        max_retries=max_retries,
        retry_threshold=retry_threshold,
        log_path=log_path,
    )


@record_loop("front_matter_judge_loop")
async def _call_front_matter_judge_retry(
    llm_client: LLMClient,
    prompt: str,
    response_model,
    temperature: float,
    max_tokens: int,
    judge,
    item_pk: str,
    max_retries: int,
    retry_threshold: float,
    log_path: Optional[str] = None,
):
    """Wrapper for front matter judge retry; records as front_matter_judge_loop."""
    return await _call_llm_with_judge_retry(
        llm_client,
        prompt=prompt,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        item_pk=item_pk,
        max_retries=max_retries,
        retry_threshold=retry_threshold,
        log_path=log_path,
    )


# ============================================================================
# Cluster and intent selection (TUI)
# ============================================================================


async def select_cluster_and_intent(
    clusters_observation: Dict[str, ClusterObservationData],
) -> Tuple[str, UserIntent]:
    """
    Display clusters in TUI for user to select a cluster and intent, then convert
    the selected intent string to UserIntent enum.

    Args:
        clusters_observation: Dict mapping pk_hash to ClusterObservationData

    Returns:
        Tuple of (selected_pk_hash, selected_intent_enum)

    Raises:
        ValueError: If the user intent string returned by display_clusters_observation
            TUI is not a valid UserIntent display string
    """
    user_intent_options = UserIntent.get_all_display_strings()
    selected_pk_hash, selected_intent = await display_clusters_observation(
        clusters_observation, user_intent_options
    )
    if selected_intent is None:
        raise ValueError(
            "No user intent was selected (user quit) in display_clusters_observation TUI"
        )
    try:
        selected_intent_enum = UserIntent.from_display_string(selected_intent)
    except ValueError as e:
        raise ValueError(
            f"Invalid user_intent '{selected_intent}' returned by display_clusters_observation TUI"
        ) from e
    return selected_pk_hash, selected_intent_enum

# ============================================================================
# LLM client initialization (used by report generation)
# ============================================================================

def _initialize_llm_client(llm_config: LLMGeminiConfig) -> LLMClient:
    """
    Initialize LLM client with rate limiting buckets.

    Args:
        llm_config: LLMGeminiConfig instance (model is read from config)

    Returns:
        Initialized LLMClient instance

    Raises:
        ValueError: If API key is not found in environment variable
    """
    # Get API key from environment variable
    api_key = os.getenv(llm_config.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {llm_config.api_key_env}")

    # Initialize TokenBucket instances for rate limiting
    rpm_bucket = TokenBucket(
        capacity=llm_config.gemini_rpm_limit,
        refill_rate=llm_config.gemini_rpm_limit,
        name="gemini_rpm"
    )

    tpm_bucket = TokenBucket(
        capacity=llm_config.gemini_tpm_limit,
        refill_rate=llm_config.gemini_tpm_limit,
        name="gemini_tpm"
    )

    # Create LLMClient instance (executor enables non-blocking async calls)
    executor = getattr(llm_config, 'gemini_call_executor', None)
    max_retry = llm_config.max_retry
    llm_client = LLMClient(
        model=llm_config.model,
        api_key=api_key,
        rpm_bucket=rpm_bucket,
        tpm_bucket=tpm_bucket,
        executor=executor,
        max_retry=max_retry,
    )

    logger.info(f"Initialized LLM client with model: {llm_config.model}")
    return llm_client


# ============================================================================
# Report generation blocks
# ============================================================================

# ---------- Evidence collection loop enums ----------


class EvidenceLoopExitCondition(str, Enum):
    """Reason the report planner evidence collection (call 1) concluded."""

    SUFFICIENCY_TERMINAL = "sufficiency_terminal"
    EVIDENCE_GAPS_BELOW_THRESHOLD = "evidence_gaps_below_threshold"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    ERROR = "error"


class StepTerminationStatus(str, Enum):
    """Termination status for step-like operations (e.g. report plan generation), steps have to return binary status(done or error)"""

    done = "done"
    error = "error"


class StepWritingExitCondition(str, Enum):
    """Reason the per-outline writing loop concluded."""

    COMPLETE = "complete"
    SUPPLY_LLM_ERROR = "supply_llm_error"
    SUPPLY_FETCH_ERROR = "supply_fetch_error"
    WRITING_LLM_ERROR = "writing_llm_error"
    UNKNOWN_ERROR = "unknown_error"


def _deduplicate_evidence_gap_requests(
    evidence_gaps: List[EvidenceGap],
) -> Tuple[List[PaperSupplementRequest], List[ReportSupplementRequest]]:
    """
    Merge evidence gaps into unique (id, selectors) requests.
    Selector deduplication is case-insensitive exact match.
    Returns (paper_requests, report_requests).
    """
    paper_merge: Dict[str, set] = {}  # paper_id -> set of selector (lowercased for dedupe)
    report_merge: Dict[int, set] = {}  # report_id -> set of selector (lowercased for dedupe)
    total_gaps = len(evidence_gaps)

    for gap in evidence_gaps:
        if not gap.has_valid_selectors:
            continue
        if gap.target_kind == "paper" and gap.paper_id:
            key = gap.paper_id
            paper_merge.setdefault(key, set()).update(s.lower() for s in (gap.paper_selectors or []))
        elif gap.target_kind == "history" and gap.history_report_id:
            try:
                report_id = int(gap.history_report_id)
            except (ValueError, TypeError):
                continue
            report_merge.setdefault(report_id, set()).update(
                s.lower() for s in (gap.history_selectors or [])
            )

    paper_requests = [
        PaperSupplementRequest(paper_id=pid, selectors=sorted(sels))
        for pid, sels in paper_merge.items()
    ]
    report_requests = [
        ReportSupplementRequest(report_id=rid, selectors=sorted(sels))
        for rid, sels in report_merge.items()
    ]
    merged_count = len(paper_requests) + len(report_requests)

    logger.info(
        f"Report planner evidence gap deduplication: {total_gaps} gaps -> "
        f"deduplicated to {merged_count}({len(paper_requests)} paper requests, {len(report_requests)} report requests)"
    )
    return paper_requests, report_requests


def _derive_available_ids(
    materials: GetReportGenerationMetadataResponse,
) -> Dict[str, List[str]]:
    """Derive paper_id and report_id whitelists from materials."""
    paper_ids = [
        p.paper_id
        for p in (materials.new_observation_key_paper_details or [])
    ]
    report_ids = [
        str(r.report_id)
        for r in (materials.history_reports or [])
    ]
    return {"paper_id": paper_ids, "report_id": report_ids}


def _writer_requests_to_supply_request(
    supplements_requests: List[WriterSupplementRequest],
) -> GetReportGenerationSupplyRequest:
    """Convert WriterSupplementRequest list to GetReportGenerationSupplyRequest (deduplicated)."""
    # Before deduplication: count requests and selectors
    before_paper_reqs = sum(
        1 for r in supplements_requests
        if r.has_valid_selectors and r.target_kind == "paper" and r.paper_id
    )
    before_report_reqs = sum(
        1 for r in supplements_requests
        if r.has_valid_selectors and r.target_kind == "history" and r.history_report_id
    )
    before_paper_selectors = sum(
        len(r.paper_selectors or [])
        for r in supplements_requests
        if r.has_valid_selectors and r.target_kind == "paper"
    )
    before_report_selectors = sum(
        len(r.history_selectors or [])
        for r in supplements_requests
        if r.has_valid_selectors and r.target_kind == "history"
    )

    paper_merge: Dict[str, set] = {}
    report_merge: Dict[int, set] = {}
    for req in supplements_requests:
        if not req.has_valid_selectors:
            continue
        if req.target_kind == "paper" and req.paper_id:
            paper_merge.setdefault(req.paper_id, set()).update(
                req.paper_selectors or []
            )
        elif req.target_kind == "history" and req.history_report_id:
            try:
                report_id = int(req.history_report_id)
            except (ValueError, TypeError):
                continue
            report_merge.setdefault(report_id, set()).update(
                req.history_selectors or []
            )
    paper_requests = [
        PaperSupplementRequest(paper_id=pid, selectors=sorted(sels))
        for pid, sels in paper_merge.items()
    ]
    report_requests = [
        ReportSupplementRequest(report_id=rid, selectors=sorted(sels))
        for rid, sels in report_merge.items()
    ]

    after_paper_reqs = len(paper_requests)
    after_report_reqs = len(report_requests)
    after_paper_selectors = sum(len(sels) for sels in paper_merge.values())
    after_report_selectors = sum(len(sels) for sels in report_merge.values())
    logger.info(
        "Writer supply deduplication: paper_requests %d->%d, report_requests %d->%d, "
        "paper_selectors %d->%d, report_selectors %d->%d",
        before_paper_reqs, after_paper_reqs,
        before_report_reqs, after_report_reqs,
        before_paper_selectors, after_paper_selectors,
        before_report_selectors, after_report_selectors,
    )
    return GetReportGenerationSupplyRequest(
        paper_requests=paper_requests,
        report_requests=report_requests,
    )


def _build_allowed_citations(
    materials: GetReportGenerationMetadataResponse,
    written_sections: List[ReportWriterSectionOutput],
) -> List[str]:
    """Build allowed citation tokens: [paper id: xxx], [report id: xxx], [section name: xxx]."""
    tokens: List[str] = []
    for p in (materials.new_observation_key_paper_details or []):
        tokens.append(f"[paper id: {p.paper_id}]")
    for r in (materials.history_reports or []):
        tokens.append(f"[report id: {r.report_id}]")
    for s in written_sections:
        if s.section_name:
            tokens.append(f"[section name: {s.section_name}]")
    return tokens


# Statuses that force evidence loop to exit with ERROR (extensible)
_EVIDENCE_LOOP_JUDGE_ERROR_STATUSES: Tuple[LoopRunStatus, ...] = (LoopRunStatus.error,)

# TODO: add unit test to enforce output must be all None or all not None at the same time
def _should_exit_evidence_loop(
    planner_output: Optional[LLMReportPlannerOutput],
    loop_turn: int,
    cfg: ReportGenerationConfig,
    judge_status: LoopRunStatus,
    last_planner_output: Optional[LLMReportPlannerOutput],
) -> Tuple[Optional[EvidenceLoopExitCondition], Optional[LLMReportPlannerOutput]]:
    """
    Decide whether the evidence collection loop should conclude.

    Layer 1 (planner output sanity): check judge_status and planner_output.
    Layer 2 (evidence level): (a) sufficiency terminal, (b) evidence gaps below threshold,
    (c) max iterations reached.

    Returns:
        (exit_status, exit_planner_output). When exit_status is set, exit_planner_output
        is the planner output to use: last_planner_output for layer 1 exit, planner_output for layer 2.
        When no exit, returns (None, None).
    """
    # Layer 1: planner output sanity check
    if judge_status in _EVIDENCE_LOOP_JUDGE_ERROR_STATUSES:
        if planner_output is not None:
            return (EvidenceLoopExitCondition.ERROR, planner_output)
        else:
            return (EvidenceLoopExitCondition.ERROR, last_planner_output)
    if planner_output is None:
        raise ValueError(
            f"planner_output should not be None when judge_status ({judge_status}) "
            f"not in _EVIDENCE_LOOP_JUDGE_ERROR_STATUSES ({list(_EVIDENCE_LOOP_JUDGE_ERROR_STATUSES)})"
        )
    # Layer 2: evidence level check
    # a: plan.sufficiency in termination set
    if planner_output.plan.sufficiency in EvidenceCollectionTerminationSufficiency:
        return (EvidenceLoopExitCondition.SUFFICIENCY_TERMINAL, planner_output)
    # b: evidence_gaps count below threshold
    threshold = cfg.report_generation.max_evidence_gaps_threshold
    if len(planner_output.evidence_gaps) < threshold:
        return (EvidenceLoopExitCondition.EVIDENCE_GAPS_BELOW_THRESHOLD, planner_output)
    # c: max iterations reached
    max_iter = cfg.report_generation.max_evidence_loop_iterations
    if loop_turn >= max_iter:
        return (EvidenceLoopExitCondition.MAX_ITERATIONS_REACHED, planner_output)
    return (None, None)


def _filter_already_provided_selectors(
    paper_reqs: List[PaperSupplementRequest],
    report_reqs: List[ReportSupplementRequest],
    phase2_supplement: Dict,
) -> Tuple[List[PaperSupplementRequest], List[ReportSupplementRequest]]:
    """
    Filter out selectors already in phase2_supplement.
    Selector comparison is case-insensitive exact match.
    Returns same format.
    """
    before_paper = len(paper_reqs)
    before_report = len(report_reqs)

    uncached_paper = []
    for pr in paper_reqs:
        provided = phase2_supplement["paper_supplements"].get(pr.paper_id, {})
        provided_lower = {k.lower() for k in provided}
        new_selectors = [s for s in pr.selectors if s.lower() not in provided_lower]
        if new_selectors:
            uncached_paper.append(PaperSupplementRequest(paper_id=pr.paper_id, selectors=new_selectors))

    uncached_report = []
    for rr in report_reqs:
        provided = phase2_supplement["report_supplements"].get(str(rr.report_id), {})
        provided_lower = {k.lower() for k in provided}
        new_selectors = [s for s in rr.selectors if s.lower() not in provided_lower]
        if new_selectors:
            uncached_report.append(ReportSupplementRequest(report_id=rr.report_id, selectors=new_selectors))

    cached_paper = before_paper - len(uncached_paper)
    cached_report = before_report - len(uncached_report)
    logger.info(
        f"Report planner evidence cache filter(request/already-provided): {len(uncached_paper)}/{cached_paper} paper, "
        f"{len(uncached_report)}/{cached_report} report"
    )
    return (uncached_paper, uncached_report)


@record_loop("run_evidence_completion_loop")
async def _run_evidence_completion_loop(
    cluster_pk_hash: str,
    phase1_metadata: GetReportGenerationMetadataResponse,
    plan_guidance: str,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
    planner_judge: LLMJudge,
) -> Tuple[Optional[LLMReportPlannerOutput], LoopRunStatus]:
    """
    Run the report planner evidence collection (call 1) until a conclusion condition is met.

    Args:
        cluster_pk_hash: Cluster pk_hash
        phase1_metadata: Summary-level metadata (fetched once before calling this loop)
        plan_guidance: Pre-built plan guidance string
        llm_client: LLM client
        cfg: Report generation config

    Returns:
        (last_planner_output, status). last_planner_output may be None only when status is error
        and no successful planner call occurred.
    """
    last_planner_output: Optional[LLMReportPlannerOutput] = None
    exit_condition: Optional[EvidenceLoopExitCondition] = None
    loop_turn = 1
    phase2_supplement: Dict = {"paper_supplements": {}, "report_supplements": {}}
    loop_prefix = f"[report planner evidence collection (call 1)] - [cluster {cluster_pk_hash}]"
    logger.info(f"{loop_prefix} - start")
    try:
        while True:
            if loop_turn != 1:
                # Turn 2+: deduplicate, filter already-provided, call memo, merge into phase2_supplement
                paper_reqs, report_reqs = _deduplicate_evidence_gap_requests(
                    last_planner_output.evidence_gaps
                )
                uncached_paper, uncached_report = _filter_already_provided_selectors(
                    paper_reqs, report_reqs, phase2_supplement
                )

                if uncached_paper or uncached_report:
                    req = GetReportGenerationSupplyRequest(
                        paper_requests=uncached_paper,
                        report_requests=uncached_report,
                    )
                    resp = await memo.get_report_generation_supply(req, cfg.memo)
                    for paper_id, selector_map in resp.paper_supplements.items():
                        phase2_supplement["paper_supplements"].setdefault(paper_id, {}).update(selector_map)
                    for report_id, field_map in resp.report_supplements.items():
                        phase2_supplement["report_supplements"].setdefault(report_id, {}).update(field_map)
                elif not paper_reqs and not report_reqs:
                    logger.warning(
                        f"{loop_prefix} - No new supplement added after dedup and cache filter. keep using the previous supplement."
                    )

            has_supplement = phase2_supplement["paper_supplements"] or phase2_supplement["report_supplements"]
            planner_prompt = build_planner_prompt(
                phase1_metadata=phase1_metadata,
                plan_guidance=plan_guidance,
                phase2_supplement=phase2_supplement if has_supplement else None,
            )
            planner_output, judge_status_raw = await _call_planner_judge_retry(
                llm_client,
                prompt=planner_prompt,
                response_model=LLMReportPlannerOutput,
                temperature=cfg.report_generation.llm_gemini.temperature,
                max_tokens=cfg.report_generation.llm_gemini.max_tokens,
                judge=planner_judge,
                item_pk=cluster_pk_hash,
                max_retries=cfg.report_generation.max_planner_judge_retries,
                retry_threshold=cfg.report_generation.planner_judge_retry_threshold,
                log_path=cfg.report_generation.planner_output_log_path,
            )
            judge_status = judge_status_raw

            exit_condition, exit_planner_output = _should_exit_evidence_loop(
                planner_output, loop_turn, cfg, judge_status, last_planner_output
            )
            if exit_condition is not None:
                last_planner_output = exit_planner_output
                break

            last_planner_output = planner_output
            loop_turn += 1

    except Exception as e:
        logger.error(
            f"{loop_prefix} - terminated due to an error: {e}",
            exc_info=True,
        )
        exit_condition = EvidenceLoopExitCondition.ERROR

    # Map exit condition to status and log
    empty_plan = last_planner_output is None
    if not empty_plan and exit_condition == EvidenceLoopExitCondition.SUFFICIENCY_TERMINAL:
        status = LoopRunStatus.complete
        logger.info(
            f"{loop_prefix} - concluded: plan sufficiency is sufficient or borderline — evidence collection complete."
        )
    elif not empty_plan and exit_condition == EvidenceLoopExitCondition.EVIDENCE_GAPS_BELOW_THRESHOLD:
        status = LoopRunStatus.complete
        count = len(last_planner_output.evidence_gaps) if last_planner_output else 0
        threshold = cfg.report_generation.max_evidence_gaps_threshold
        logger.info(
            f"{loop_prefix} - concluded: evidence gaps remaining ({count}) below threshold ({threshold}) — evidence collection complete."
        )
    elif not empty_plan and exit_condition == EvidenceLoopExitCondition.MAX_ITERATIONS_REACHED:
        status = LoopRunStatus.partial
        max_iter = cfg.report_generation.max_evidence_loop_iterations
        logger.info(
            f"{loop_prefix} - concluded: reached maximum iterations ({max_iter}) — evidence collection partial."
        )
    elif not empty_plan and exit_condition in (EvidenceLoopExitCondition.ERROR):
        status = LoopRunStatus.partial
        logger.warning(
            f"{loop_prefix} - terminated due to an error. The returned plan is from the last successful planner call."
        )
    elif empty_plan and exit_condition in (EvidenceLoopExitCondition.ERROR):
        status = LoopRunStatus.error
        logger.warning(
            f"{loop_prefix} - call failed (no successful pass)."
        )
    else:
        error_msg = f"{loop_prefix} - undefined exit condition({exit_condition}) and empty_plan({empty_plan}) condition for exit status resolution."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"{loop_prefix} - finished, reasons: {exit_condition}, status: {status.value}, empty planner output: {empty_plan}")
    return last_planner_output, status


@record_step("generate_report_plan", contains=["run_evidence_completion_loop"])
async def _generate_report_plan(
    cluster_pk_hash: str,
    user_intent: UserIntent,
    new_topic_metadata: GetReportGenerationMetadataResponse,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
    planner_judge: LLMJudge,
) -> Tuple[Optional[LLMReportPlannerOutput], StepTerminationStatus]:
    """
    Generate a report plan via evidence collection loop (planner calls until exit condition).

    Returns:
        (plan, step_status). plan is None on error; step_status is done or error.
    """
    step_prefix = f"[report plan generation step] - [cluster {cluster_pk_hash}]"
    intent_spec = get_planner_intent_spec(user_intent)
    plan_guidance = build_plan_guidance(intent_spec)

    try:
        logger.info(f"{step_prefix} - start refining report plan with supplement collection loop")
        plan, status = await _run_evidence_completion_loop(
            cluster_pk_hash, new_topic_metadata, plan_guidance, llm_client, cfg, planner_judge
        )
    except Exception as e:
        logger.error(
            f"{step_prefix} - supplement collection loop failed: {e}",
            exc_info=True,
        )
        return (None, StepTerminationStatus.error)

    # Handle (plan, status)
    if status in (LoopRunStatus.complete, LoopRunStatus.partial):
        if status == LoopRunStatus.partial:
            logger.warning(
                f"{step_prefix} - report plan completion status is partial even it is not empty."
            )
        result_plan = plan
        result_status = StepTerminationStatus.done
    elif status == LoopRunStatus.error:
        logger.error(f"{step_prefix} - failed to generate report plan due to error")
        result_plan = None
        result_status = StepTerminationStatus.error
    else:
        raise ValueError(f"{step_prefix} - unexpected status: {status}")

    logger.info(
        f"{step_prefix} - finished, plan={'empty' if result_plan is None else 'present'}, "
        f"status: {result_status.value}"
    )
    return (result_plan, result_status)


@record_loop("run_writing_loop")
async def _run_writing_loop(
    cluster_pk_hash: str,
    plan: LLMReportPlannerOutput,
    materials: GetReportGenerationMetadataResponse,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[List[ReportWriterSectionOutput]], LoopRunStatus]:
    """
    Run the per-outline writing loop: supply -> fetch -> write for each outline item.

    Returns:
        (sections, status). sections may be None only when status is error and no sections were written.
    """
    available_ids = _derive_available_ids(materials)
    sections: List[ReportWriterSectionOutput] = []
    exit_condition: Optional[StepWritingExitCondition] = None
    loop_prefix = f"[report writing loop] - [cluster {cluster_pk_hash}]"
    wcfg = cfg.report_generation
    llm_cfg = cfg.report_generation.llm_gemini

    _WRITING_LOOP_ERROR_CONDITIONS: Tuple[StepWritingExitCondition, ...] = (
        StepWritingExitCondition.SUPPLY_LLM_ERROR,
        StepWritingExitCondition.SUPPLY_FETCH_ERROR,
        StepWritingExitCondition.WRITING_LLM_ERROR,
    )

    logger.info(f"{loop_prefix} - start")
    try:
        for target_outline in plan.plan.outline:
            # Supply: LLM decides what supplements to request
            supplement_input = ReportWriterSupplementInput(
                materials=materials.model_dump(),
                plan=plan.plan.model_dump(),
                target_outline=target_outline,
                available_ids=available_ids,
            )
            supply_prompt = build_evidence_requests_prompt(
                supplement_input,
                template_name=wcfg.writer_supplement_prompt_template,
            )
            supply_judge = WriterSupplyJudgeWrapper(
                available_paper_ids=available_ids["paper_id"],
                available_history_report_ids=available_ids["report_id"],
            )
            supplement_output, supply_status = await _call_writer_supply_judge_retry(
                llm_client,
                prompt=supply_prompt,
                response_model=ReportWriterSupplementOutput,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
                judge=supply_judge,
                item_pk=cluster_pk_hash,
                max_retries=wcfg.max_writer_supplement_judge_retries,
                retry_threshold=wcfg.writer_supplement_judge_retry_threshold,
                log_path=wcfg.writer_supplement_output_log_path,
            )
            if supply_status == LoopRunStatus.error or supplement_output is None:
                exit_condition = StepWritingExitCondition.SUPPLY_LLM_ERROR
                break

            # Fetch: memo get-report-generation-supply (skip if supply returned empty)
            if not supplement_output.supplements_requests:
                logger.warning(
                    f"{loop_prefix} - supply returned empty request list for outline {target_outline!r}, skipping fetch"
                )
                supply_resp = GetReportGenerationSupplyResponse(
                    paper_supplements={},
                    report_supplements={},
                )
            else:
                try:
                    fetch_req = _writer_requests_to_supply_request(
                        supplement_output.supplements_requests
                    )
                    supply_resp = await memo.get_report_generation_supply(
                        fetch_req, cfg.memo
                    )
                except MemoGetReportGenerationSupplyError as e:
                    logger.warning(
                        f"{loop_prefix} - supply fetch failed for outline {target_outline!r}: {e}"
                    )
                    exit_condition = StepWritingExitCondition.SUPPLY_FETCH_ERROR
                    break

            # Write: LLM writes section with supplements
            allowed_citations = _build_allowed_citations(materials, sections)
            section_input = ReportWriterSectionInput(
                materials=materials.model_dump(),
                plan=plan.plan.model_dump(),
                target_section=target_outline,
                written_sections=[{"title": s.section_name, "body": s.section_text} for s in sections],
                allowed_citations=allowed_citations,
                supplements=supply_resp,
            )
            write_prompt = build_section_writing_prompt(
                section_input,
                template_name=wcfg.writer_section_writing_prompt_template,
            )
            writing_judge = WriterWritingJudgeWrapper(
                outline_item=target_outline,
                allowed_citations=allowed_citations,
            )
            section_output, write_status = await _call_writer_writing_judge_retry(
                llm_client,
                prompt=write_prompt,
                response_model=ReportWriterSectionOutput,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
                judge=writing_judge,
                item_pk=cluster_pk_hash,
                max_retries=wcfg.max_writer_writing_judge_retries,
                retry_threshold=wcfg.writer_writing_judge_retry_threshold,
                log_path=wcfg.writer_writing_output_log_path,
            )
            if write_status == LoopRunStatus.error or section_output is None:
                exit_condition = StepWritingExitCondition.WRITING_LLM_ERROR
                break

            sections.append(section_output)
        else:
            exit_condition = StepWritingExitCondition.COMPLETE
    except Exception as e:
        logger.error(
            f"{loop_prefix} - terminated due to an error: {e}",
            exc_info=True,
        )
        exit_condition = StepWritingExitCondition.UNKNOWN_ERROR

    # Map exit condition to LoopRunStatus
    empty_sections = len(sections) == 0
    if exit_condition == StepWritingExitCondition.COMPLETE:
        status = LoopRunStatus.complete
        if empty_sections:
            logger.warning(
                f"{loop_prefix} - concluded: all outline items processed but sections is empty."
            )
        else:
            logger.info(
                f"{loop_prefix} - concluded: all outline items processed."
            )
    elif exit_condition in _WRITING_LOOP_ERROR_CONDITIONS:
        status = LoopRunStatus.partial
        logger.warning(
            f"{loop_prefix} - incomplete writing due to exit condition {exit_condition.value}, "
            f"sections: {len(sections)} — writing partial."
        )
    elif exit_condition == StepWritingExitCondition.UNKNOWN_ERROR:
        status = LoopRunStatus.error
        if empty_sections:
            logger.warning(
                f"{loop_prefix} - terminated due to unknown error, no sections written."
            )
    else:
        raise ValueError(f"{loop_prefix} - unexpected exit condition: {exit_condition}")

    result_sections = None if (status == LoopRunStatus.error and empty_sections) else sections
    logger.info(
        f"{loop_prefix} - finished, reasons: {exit_condition}, status: {status.value}, empty sections: {empty_sections}"
    )
    return result_sections, status


@record_step("generate_report_body", contains=["run_writing_loop"])
async def _generate_report_body(
    cluster_pk_hash: str,
    plan: LLMReportPlannerOutput,
    materials: GetReportGenerationMetadataResponse,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[List[ReportWriterSectionOutput]], StepTerminationStatus]:
    """
    Generate report body via per-outline writing loop (supply -> fetch -> write for each outline item).

    Returns:
        (sections, step_status). sections is None on error; step_status is done or error.
    """
    step_prefix = f"[report body generation step] - [cluster {cluster_pk_hash}]"

    try:
        logger.info(f"{step_prefix} - start writing report body with per-outline loop")
        sections, status = await _run_writing_loop(
            cluster_pk_hash, plan, materials, llm_client, cfg
        )
    except Exception as e:
        logger.error(
            f"{step_prefix} - per-outline writing loop failed: {e}",
            exc_info=True,
        )
        return (None, StepTerminationStatus.error)


    if status == LoopRunStatus.complete:
        result_sections = sections
        result_status = StepTerminationStatus.done
    elif status in (LoopRunStatus.partial, LoopRunStatus.error):
        logger.warning(
            f"{step_prefix} - report body generation ended with status {status.value}, returning {len(sections)} sections."
        )
        result_sections = sections
        result_status = StepTerminationStatus.error
    else:
        raise ValueError(f"{step_prefix} - unexpected status: {status}")

    logger.info(
        f"{step_prefix} - finished, sections={'empty' if result_sections is None else 'present'}, "
        f"status: {result_status.value}"
    )
    return (result_sections, result_status)


@record_step("generate_report_front_matter")
async def _generate_report_front_matter(
    cluster_pk_hash: str,
    sections_result: List[ReportWriterSectionOutput],
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[ReportWriterFrontMatterOutput], StepTerminationStatus]:
    """
    Generate report front matter (title, summary, keywords) from report body sections.

    Returns:
        (front_matter, step_status). front_matter is None on error; step_status is done or error.
    """
    step_prefix = f"[report front matter generation step] - [cluster {cluster_pk_hash}]"
    wcfg = cfg.report_generation
    llm_cfg = wcfg.llm_gemini

    logger.info(f"{step_prefix} - start generating report front matter (title, summary, keywords)")
    prompt = build_summary_writing_prompt(
        sections_result,
        template_name=wcfg.writer_summary_prompt_template,
    )
    front_matter_output, judge_status = await _call_front_matter_judge_retry(
        llm_client,
        prompt=prompt,
        response_model=ReportWriterFrontMatterOutput,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        judge=front_matter_judge,
        item_pk=cluster_pk_hash,
        max_retries=wcfg.max_writer_summary_judge_retries,
        retry_threshold=wcfg.writer_summary_judge_retry_threshold,
        log_path=wcfg.writer_summary_output_log_path,
    )

    if judge_status in (LoopRunStatus.complete, LoopRunStatus.partial):
        result_status = StepTerminationStatus.done
        result_output = front_matter_output
        if judge_status == LoopRunStatus.partial:
            logger.warning(
                f"{step_prefix} - front matter completion status is partial, marking as done."
            )
    elif judge_status == LoopRunStatus.error:
        result_status = StepTerminationStatus.error
        result_output = None
    else:
        raise ValueError(
            f"{step_prefix} - unexpected judge status: {judge_status}"
        )

    logger.info(
        f"{step_prefix} - finished, front_matter={'present' if result_output else 'empty'}, "
        f"status: {result_status.value}"
    )
    return (result_output, result_status)


# Union of forbidden filename chars: Windows + macOS + Linux
_FORBIDDEN_FILENAME_CHARS = re.compile(r'[/\\:*?"<>|\x00\s]+')


def _sanitize_filename(title: str) -> str:
    """Sanitize title for use as filename. Replaces forbidden chars on Windows, macOS, Linux."""
    s = _FORBIDDEN_FILENAME_CHARS.sub('_', title)
    s = s.strip('_')
    s = re.sub(r'_+', '_', s)
    return s or "report"


@record_step("save_report_to_fs")
async def _save_report_to_fs(
    cluster_pk_hash: str,
    sections_result: List[ReportWriterSectionOutput],
    front_matter: ReportWriterFrontMatterOutput,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[SaveReportToFsOutput], StepTerminationStatus]:
    """
    Save report (body + front matter) to local FS as JSON under history_reports.

    Returns:
        (SaveReportToFsOutput, step_status). output is None on error; step_status is done or error.
    """
    step_prefix = f"[save report to FS step] - [cluster {cluster_pk_hash}]"
    logger.info(f"{step_prefix} - start saving report to local FS")
    try:
        observation_report = ObservationReport(
            cluster_pk_hash=cluster_pk_hash,
            body=sections_result,
            front_matter=front_matter
        )
        history_dir = cfg.cache.history_reports
        history_dir.mkdir(parents=True, exist_ok=True)
        sanitized_title = _sanitize_filename(front_matter.title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sanitized_title}_{timestamp}.json"
        file_path = history_dir / filename

        json_bytes = json.dumps(
            observation_report.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        file_path.write_bytes(json_bytes)

        signature = hashlib.sha256(json_bytes).hexdigest()

        output = SaveReportToFsOutput(
            report_path=str(file_path),
            signature=signature,
        )
        logger.info(
            f"{step_prefix} - finished, report_path={output.report_path}, "
            f"status: {StepTerminationStatus.done.value}"
        )
        return (output, StepTerminationStatus.done)
    except Exception as e:
        logger.error(
            f"{step_prefix} - save report to FS failed: {e}",
            exc_info=True,
        )
        logger.info(f"{step_prefix} - finished, status: {StepTerminationStatus.error.value}")
        return (None, StepTerminationStatus.error)


def _user_intent_to_intent_mode(user_intent: UserIntent) -> str:
    """Map UserIntent enum to DB intent_mode string."""
    return user_intent.name.lower()


@record_step("save_report_to_db")
async def _save_report_to_db(
    cluster_pk_hash: str,
    user_intent: UserIntent,
    resolved_topic: TopicResolveOutput,
    plan: LLMReportPlannerOutput,
    front_matter: ReportWriterFrontMatterOutput,
    save_output: SaveReportToFsOutput,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[NewMemoryResponse], StepTerminationStatus]:
    """
    Persist report generation results to the database via memo new-memory command.

    Returns:
        (NewMemoryResponse, step_status). output is None on error; step_status is done or error.
    """
    step_prefix = f"[save report to DB step] - [cluster {cluster_pk_hash}]"
    logger.info(f"{step_prefix} - start persisting report to database")
    try:
        topic_resolver_config = TopicResolverConfig(
            json_payload=TopicResolverConfigPayload(
                topic_resolver_threshold=cfg.report_generation.topic_resolver_threshold
            )
        )
        payload = NewMemoryRequest(
            cluster_pk_hash=cluster_pk_hash,
            intent_mode=_user_intent_to_intent_mode(user_intent),
            resolved_topic=ResolvedTopicInput(
                action=resolved_topic.action.value,
                merge_to_topic=resolved_topic.merge_to_topic,
                new_topic_centroid_b64=resolved_topic.new_topic_centroid_b64,
                new_topic_weight=resolved_topic.new_topic_weight,
                score=resolved_topic.score,
            ),
            plan=plan.plan.model_dump(mode="json"),
            front_matter=FrontMatterInput(
                title=front_matter.title,
                summary=front_matter.summary,
                keywords=list(front_matter.keywords),
            ),
            save_output=SaveMemoryInput(
                report_path=save_output.report_path,
                signature=save_output.signature,
            ),
            topic_resolver_config=TopicResolverConfigInput(
                topic_resolver_config_id=topic_resolver_config.topic_resolver_config_id,
                json_payload={"topic_resolver_threshold": cfg.report_generation.topic_resolver_threshold},
            ),
        )
        response = await memo.new_memory(payload, cfg.memo)
        logger.info(f"{step_prefix} - finished, report_id={response.report_id}, status: {StepTerminationStatus.done.value}")
        return (response, StepTerminationStatus.done)
    except MemoNewMemoryError as e:
        logger.error(f"{step_prefix} - memo new-memory failed: {e}", exc_info=True)
        logger.info(f"{step_prefix} - finished, status: {StepTerminationStatus.error.value}")
        return (None, StepTerminationStatus.error)
    except Exception as e:
        logger.error(f"{step_prefix} - save report to DB failed: {e}", exc_info=True)
        logger.info(f"{step_prefix} - finished, status: {StepTerminationStatus.error.value}")
        return (None, StepTerminationStatus.error)


@record_step("resolve_report_topic")
async def _resolve_report_topic(
    cluster_pk_hash: str, cfg: ReportGenerationConfig
) -> Tuple[Optional[TopicResolveOutput], StepTerminationStatus]:
    """
    Resolve a cluster to a topic using the topic resolver.

    This step handles the topic resolution logic:
    - Fetches topic resolver metadata from memo
    - Converts metadata to TopicInput and ClusterInput formats
    - Calls resolve_topic with the threshold from config
    - Logs the resolution result

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        cfg: ReportGenerationConfig instance

    Returns:
        (resolved_topic, step_status). resolved_topic is None on error; step_status is done or error.
    """
    step_prefix = f"[resolve report topic step] - [cluster {cluster_pk_hash}]"
    logger.info(f"{step_prefix} - start resolving cluster to topic")
    try:
        # Get topic resolver metadata from memo
        metadata = await memo.get_topic_resolver_metadata(cluster_pk_hash, cfg.memo)

        # Convert TopicCentroid list to TopicInput list
        topics = [
            TopicInput(
                id=topic.id,
                centroid_b64=topic.centroid_b64,
                centroid_weight=topic.centroid_weight,
            )
            for topic in metadata.topics
        ]

        # Convert ClusterMetadata to ClusterInput
        cluster = TopicResolverClusterInput(
            id=cluster_pk_hash,
            centroid_b64=metadata.cluster.centroid,
            centroid_weight=metadata.cluster.centroid_weight,
        )

        # Get threshold from config
        resolve_threshold = cfg.report_generation.topic_resolver_threshold

        # Resolve topic
        resolved_topic = resolve_topic(topics, cluster, resolve_threshold)

        if resolved_topic.action == TopicResolveAction.MERGE:
            logger.info(
                f"{step_prefix} - Topic resolution for cluster {cluster_pk_hash}: MERGE to topic {resolved_topic.merge_to_topic} "
                f"(similarity score: {resolved_topic.score:.4f}, new weight: {resolved_topic.new_topic_weight:.2f})"
            )
        elif resolved_topic.action == TopicResolveAction.CREATE:
            logger.info(
                f"{step_prefix} - Topic resolution for cluster {cluster_pk_hash}: CREATE new topic "
                f"(new weight: {resolved_topic.new_topic_weight:.2f})"
            )
        else:
            raise ValueError(f"{step_prefix} - unexpected topic resolution action: {resolved_topic.action}")

        logger.info(f"{step_prefix} - finished, action={resolved_topic.action.value}, status: {StepTerminationStatus.done.value}")
        return (resolved_topic, StepTerminationStatus.done)

    except TopicResolverError as e:
        logger.error(f"Topic resolver error for cluster {cluster_pk_hash}: {e}", exc_info=True)
        logger.info(f"{step_prefix} - finished, status: {StepTerminationStatus.error.value}")
        return (None, StepTerminationStatus.error)
    except Exception as e:
        logger.error(f"Unexpected error during topic resolution for cluster {cluster_pk_hash}: {e}", exc_info=True)
        logger.info(f"{step_prefix} - finished, status: {StepTerminationStatus.error.value}")
        return (None, StepTerminationStatus.error)


@record_step("fetch_report_generation_metadata")
async def _fetch_report_generation_metadata(
    cluster_pk_hash: str,
    resolved_topic: TopicResolveOutput,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[GetReportGenerationMetadataResponse], StepTerminationStatus]:
    """
    Fetch report generation metadata from memo.

    Returns:
        (metadata, step_status). metadata is None on error; step_status is done or error.
    """
    step_prefix = f"[fetch report generation metadata step] - [cluster {cluster_pk_hash}]"
    logger.info(f"{step_prefix} - start fetching report generation metadata")
    try:
        new_topic_metadata = await memo.get_report_generation_metadata(
            cluster_pk_hash=cluster_pk_hash,
            config=cfg.memo,
            topic_id=resolved_topic.merge_to_topic,
            add_top_papers=True,
        )
    except Exception as e:
        logger.error(
            f"Report generation metadata fetch failed for cluster {cluster_pk_hash}: {e}",
            exc_info=True,
        )
        logger.info(f"{step_prefix} - finished, metadata=none, status: {StepTerminationStatus.error.value}")
        return (None, StepTerminationStatus.error)

    logger.info(f"{step_prefix} - finished, metadata=present, status: {StepTerminationStatus.done.value}")
    return (new_topic_metadata, StepTerminationStatus.done)


@record_step("initialize_report_generation_llm_client")
async def _initialize_report_generation_llm_client(
    cluster_pk_hash: str,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[LLMClientWrapper], StepTerminationStatus]:
    """
    Initialize LLM client for report generation.

    Returns:
        (llm_client_wrapper, step_status). llm_client_wrapper is None on error; step_status is done or error.
    """
    step_prefix = f"[initialize report generation LLM client step] - [cluster {cluster_pk_hash}]"
    logger.info(f"{step_prefix} - start initializing LLM client")
    try:
        llm_cfg = cfg.report_generation.llm_gemini
        llm_client = _initialize_llm_client(llm_cfg)
        llm_client_wrapper = LLMClientWrapper(llm_config=llm_cfg, llm_client=llm_client)
    except Exception as e:
        logger.error(
            f"LLM client initialization failed for cluster {cluster_pk_hash}: {e}",
            exc_info=True,
        )
        logger.info(f"{step_prefix} - finished, llm_client=none, status: {StepTerminationStatus.error.value}")
        return (None, StepTerminationStatus.error)

    logger.info(f"{step_prefix} - finished, llm_client=initialized, status: {StepTerminationStatus.done.value}")
    return (llm_client_wrapper, StepTerminationStatus.done)


@with_workflow_register("kick_off_report_job")
async def _kick_off_report_job(cluster_pk_hash: str, user_intent: UserIntent, cfg: ReportGenerationConfig) -> None:
    """
    Kick off a report generation job by resolving the cluster to a topic.
    Uses async LLM call (non-blocking) when executor is configured.

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        user_intent: User intent enum
        cfg: ReportGenerationConfig instance
    """
    # step: resolve report topic
    resolved_topic, step_status = await _resolve_report_topic(cluster_pk_hash, cfg)
    if step_status == StepTerminationStatus.error or resolved_topic is None:
        raise ReportGenerationRuntimeError(f"Report topic resolution failed for cluster {cluster_pk_hash}")

    # step: fetch report generation metadata
    new_topic_metadata, step_status = await _fetch_report_generation_metadata(cluster_pk_hash, resolved_topic, cfg)
    if step_status == StepTerminationStatus.error or new_topic_metadata is None:
        raise ReportGenerationRuntimeError(f"Report generation metadata fetch failed for cluster {cluster_pk_hash}")

    # step: initialize report generation LLM client
    llm_client_wrapper, step_status = await _initialize_report_generation_llm_client(cluster_pk_hash, cfg)
    if step_status == StepTerminationStatus.error or llm_client_wrapper is None:
        raise ReportGenerationRuntimeError(f"Report generation LLM client initialization failed for cluster {cluster_pk_hash}")

    # step:  generate report plan
    plan, step_status = await _generate_report_plan(
        cluster_pk_hash, user_intent, new_topic_metadata, llm_client_wrapper.llm_client, cfg, planner_judge
    )
    if step_status == StepTerminationStatus.error or plan is None:
        raise ReportGenerationRuntimeError(f"Report planner call failed for cluster {cluster_pk_hash}")

    # step: generate report body
    sections_result, step_status = await _generate_report_body(
        cluster_pk_hash, plan, new_topic_metadata, llm_client_wrapper.llm_client, cfg
    )
    if step_status == StepTerminationStatus.error or sections_result is None:
        raise ReportGenerationRuntimeError(f"Report body writing failed for cluster {cluster_pk_hash}")

    # step: generate report front matter
    front_matter, step_status = await _generate_report_front_matter(
        cluster_pk_hash, sections_result, llm_client_wrapper.llm_client, cfg
    )
    if step_status == StepTerminationStatus.error or front_matter is None:
        raise ReportGenerationRuntimeError(f"Report front matter generation failed for cluster {cluster_pk_hash}")

    # step: save report to local fs
    save_output, step_status = await _save_report_to_fs(
        cluster_pk_hash, sections_result, front_matter, cfg
    )
    if step_status == StepTerminationStatus.error or save_output is None:
        raise ReportGenerationRuntimeError(f"Save report to local FS failed for cluster {cluster_pk_hash}")

    # step: write to db
    _, step_status = await _save_report_to_db(
        cluster_pk_hash, user_intent, resolved_topic, plan, front_matter, save_output, cfg
    )
    if step_status == StepTerminationStatus.error:
        raise ReportGenerationRuntimeError(f"Save report to DB failed for cluster {cluster_pk_hash}")


async def start_generation(cluster_pk_hash: str, user_intent: UserIntent, cfg: ReportGenerationConfig) -> None:
    """
    Start report generation for a new job.
    
    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        user_intent: User intent enum
        cfg: ReportGenerationConfig instance
        
    Raises:
        Exception: If report generation fails
    """
    log_prefix = f"[start generation] - [cluster {cluster_pk_hash}]"
    logger.info(f"{log_prefix} - start generating report")
    try:
        await _kick_off_report_job(cluster_pk_hash, user_intent, cfg)
        # Update job status to done
        db_store = ReportJobStore(cfg.cache.report_generation_db_path, cfg.cache.report_generation_db_migrations_path)
        try:
            now_str = datetime.now(timezone.utc).isoformat()
            db_store.update_report_job_status(cluster_pk_hash, ReportJobStatus.DONE, now_str)
        finally:
            db_store.close()
        logger.info(f"{log_prefix} - finished.")
    except Exception as e:
        # Update job status to error
        db_store = ReportJobStore(cfg.cache.report_generation_db_path, cfg.cache.report_generation_db_migrations_path)
        try:
            now_str = datetime.now(timezone.utc).isoformat()
            db_store.update_report_job_status(cluster_pk_hash, ReportJobStatus.ERROR, now_str)
        finally:
            db_store.close()
        logger.error(f"{log_prefix} - failed: {e}", exc_info=True)
        raise
