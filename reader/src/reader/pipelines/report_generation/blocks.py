"""Pipeline building blocks"""

from __future__ import annotations

import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

from algo_lib.topic_resolver.models import TopicInput, ClusterInput as TopicResolverClusterInput, TopicResolveOutput
from algo_lib.topic_resolver.resolver import resolve_topic
from algo_lib.topic_resolver.errors import TopicResolverError

from reader.pipelines.report_generation.config.config import ReportGenerationConfig, LLMGeminiConfig
from reader.adapters import memo
from reader.adapters.memo import (
    ClusterObservationData,
    GetReportGenerationMetadataResponse,
    GetReportGenerationSupplyRequest,
    GetReportGenerationSupplyResponse,
    MemoGetReportGenerationSupplyError,
    PaperSupplementRequest,
    ReportSupplementRequest,
)
from reader.adapters.llm import (
    LLMClient,
    TokenBucket,
    LLMGenerationError,
    JudgeLoopTerminationStatus,
)
from reader.pipelines.report_generation.report import (
    LLMReportPlannerOutput,
    EvidenceCollectionTerminationSufficiency,
    EvidenceGap,
    ReportWriterFrontMatterOutput,
    ReportWriterSectionInput,
    ReportWriterSectionOutput,
    ReportWriterSupplementInput,
    ReportWriterSupplementOutput,
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

logger = get_logger()

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

def _initialize_llm_client(llm_config: LLMGeminiConfig, model: str) -> LLMClient:
    """
    Initialize LLM client with rate limiting buckets.

    Args:
        llm_config: LLMGeminiConfig instance
        model: Model name to use for the LLM client

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
        model=model,
        api_key=api_key,
        rpm_bucket=rpm_bucket,
        tpm_bucket=tpm_bucket,
        executor=executor,
        max_retry=max_retry,
    )

    logger.info(f"Initialized LLM client with model: {model}")
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


class LoopTerminationStatus(str, Enum):
    """Enum for loop-like operations termination status (used by evidence collection and judge retry loops)."""

    complete = "complete"
    partial = "partial"
    error = "error"


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
    Returns (paper_requests, report_requests, duplicate_count).
    """
    paper_merge: Dict[str, set] = {}  # paper_id -> set of selector
    report_merge: Dict[int, set] = {}  # report_id -> set of selector
    total_gaps = len(evidence_gaps)

    for gap in evidence_gaps:
        if not gap.has_valid_selectors:
            continue
        if gap.target_kind == "paper" and gap.paper_id:
            key = gap.paper_id
            paper_merge.setdefault(key, set()).update(gap.paper_selectors)
        elif gap.target_kind == "history" and gap.history_report_id:
            try:
                report_id = int(gap.history_report_id)
            except (ValueError, TypeError):
                continue
            report_merge.setdefault(report_id, set()).update(gap.history_selectors)

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
_EVIDENCE_LOOP_JUDGE_ERROR_STATUSES: Tuple[LoopTerminationStatus, ...] = (LoopTerminationStatus.error,)

# TODO: add unit test to enforce output must be all None or all not None at the same time
def _should_exit_evidence_loop(
    planner_output: Optional[LLMReportPlannerOutput],
    loop_turn: int,
    cfg: ReportGenerationConfig,
    judge_status: LoopTerminationStatus,
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
    """Filter out selectors already in phase2_supplement. Returns same format."""
    before_paper = len(paper_reqs)
    before_report = len(report_reqs)

    uncached_paper = []
    for pr in paper_reqs:
        provided = phase2_supplement["paper_supplements"].get(pr.paper_id, {})
        new_selectors = [s for s in pr.selectors if s not in provided]
        if new_selectors:
            uncached_paper.append(PaperSupplementRequest(paper_id=pr.paper_id, selectors=new_selectors))

    uncached_report = []
    for rr in report_reqs:
        provided = phase2_supplement["report_supplements"].get(str(rr.report_id), {})
        new_selectors = [s for s in rr.selectors if s not in provided]
        if new_selectors:
            uncached_report.append(ReportSupplementRequest(report_id=rr.report_id, selectors=new_selectors))

    cached_paper = before_paper - len(uncached_paper)
    cached_report = before_report - len(uncached_report)
    logger.info(
        f"Report planner evidence cache filter(request/already-provided): {len(uncached_paper)}/{cached_paper} paper, "
        f"{len(uncached_report)}/{cached_report} report"
    )
    return (uncached_paper, uncached_report)


async def _run_evidence_completion_loop(
    cluster_pk_hash: str,
    phase1_metadata: GetReportGenerationMetadataResponse,
    plan_guidance: str,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
    planner_judge: LLMJudge,
) -> Tuple[Optional[LLMReportPlannerOutput], LoopTerminationStatus]:
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
            planner_output, judge_status_raw = await llm_client.call_structured_with_judge_retry(
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
            judge_status = LoopTerminationStatus(judge_status_raw.value)

            exit_condition, exit_planner_output = _should_exit_evidence_loop(
                planner_output, loop_turn, cfg, judge_status, last_planner_output
            )
            if exit_condition is not None:
                last_planner_output = exit_planner_output
                break

            last_planner_output = planner_output
            loop_turn += 1

    except Exception as e:
        logger.warning(
            f"{loop_prefix} - terminated due to an error: {e}",
            exc_info=True,
        )
        exit_condition = EvidenceLoopExitCondition.ERROR

    # Map exit condition to status and log
    empty_plan = last_planner_output is None
    if not empty_plan and exit_condition == EvidenceLoopExitCondition.SUFFICIENCY_TERMINAL:
        status = LoopTerminationStatus.complete
        logger.info(
            f"{loop_prefix} - concluded: plan sufficiency is sufficient or borderline — evidence collection complete."
        )
    elif not empty_plan and exit_condition == EvidenceLoopExitCondition.EVIDENCE_GAPS_BELOW_THRESHOLD:
        status = LoopTerminationStatus.complete
        count = len(last_planner_output.evidence_gaps) if last_planner_output else 0
        threshold = cfg.report_generation.max_evidence_gaps_threshold
        logger.info(
            f"{loop_prefix} - concluded: evidence gaps remaining ({count}) below threshold ({threshold}) — evidence collection complete."
        )
    elif not empty_plan and exit_condition == EvidenceLoopExitCondition.MAX_ITERATIONS_REACHED:
        status = LoopTerminationStatus.partial
        max_iter = cfg.report_generation.max_evidence_loop_iterations
        logger.info(
            f"{loop_prefix} - concluded: reached maximum iterations ({max_iter}) — evidence collection partial."
        )
    elif not empty_plan and exit_condition in (EvidenceLoopExitCondition.ERROR):
        status = LoopTerminationStatus.partial
        logger.warning(
            f"{loop_prefix} - terminated due to an error. The returned plan is from the last successful planner call."
        )
    elif empty_plan and exit_condition in (EvidenceLoopExitCondition.ERROR):
        status = LoopTerminationStatus.error
        logger.warning(
            f"{loop_prefix} - call failed (no successful pass)."
        )
    else:
        error_msg = f"{loop_prefix} - undefined exit condition({exit_condition}) and empty_plan({empty_plan}) condition for exit status resolution."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"{loop_prefix} - finished, reasons: {exit_condition}, status: {status.value}, empty planner output: {empty_plan}")
    return last_planner_output, status


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
        logger.warning(
            f"{step_prefix} - supplement collection loop failed: {e}",
            exc_info=True,
        )
        return (None, StepTerminationStatus.error)

    # Handle (plan, status)
    if status in (LoopTerminationStatus.complete, LoopTerminationStatus.partial):
        if status == LoopTerminationStatus.partial:
            logger.warning(
                f"{step_prefix} - report plan completion status is partial even it is not empty."
            )
        result_plan = plan
        result_status = StepTerminationStatus.done
    elif status == LoopTerminationStatus.error:
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


async def _run_writing_loop(
    cluster_pk_hash: str,
    plan: LLMReportPlannerOutput,
    materials: GetReportGenerationMetadataResponse,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[List[ReportWriterSectionOutput]], LoopTerminationStatus]:
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
            supplement_output, supply_status = await llm_client.call_structured_with_judge_retry(
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
            if supply_status == JudgeLoopTerminationStatus.error or supplement_output is None:
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
            section_output, write_status = await llm_client.call_structured_with_judge_retry(
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
            if write_status == JudgeLoopTerminationStatus.error or section_output is None:
                exit_condition = StepWritingExitCondition.WRITING_LLM_ERROR
                break

            sections.append(section_output)
        else:
            exit_condition = StepWritingExitCondition.COMPLETE
    except Exception as e:
        logger.warning(
            f"{loop_prefix} - terminated due to an error: {e}",
            exc_info=True,
        )
        exit_condition = StepWritingExitCondition.UNKNOWN_ERROR

    # Map exit condition to LoopTerminationStatus
    empty_sections = len(sections) == 0
    if exit_condition == StepWritingExitCondition.COMPLETE:
        status = LoopTerminationStatus.complete
        if empty_sections:
            logger.warning(
                f"{loop_prefix} - concluded: all outline items processed but sections is empty."
            )
        else:
            logger.info(
                f"{loop_prefix} - concluded: all outline items processed."
            )
    elif exit_condition in _WRITING_LOOP_ERROR_CONDITIONS:
        status = LoopTerminationStatus.partial
        logger.warning(
            f"{loop_prefix} - incomplete writing due to exit condition {exit_condition.value}, "
            f"sections: {len(sections)} — writing partial."
        )
    elif exit_condition == StepWritingExitCondition.UNKNOWN_ERROR:
        status = LoopTerminationStatus.error
        if empty_sections:
            logger.warning(
                f"{loop_prefix} - terminated due to unknown error, no sections written."
            )
    else:
        raise ValueError(f"{loop_prefix} - unexpected exit condition: {exit_condition}")

    result_sections = None if (status == LoopTerminationStatus.error and empty_sections) else sections
    logger.info(
        f"{loop_prefix} - finished, reasons: {exit_condition}, status: {status.value}, empty sections: {empty_sections}"
    )
    return result_sections, status


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
        logger.warning(
            f"{step_prefix} - per-outline writing loop failed: {e}",
            exc_info=True,
        )
        return (None, StepTerminationStatus.error)


    if status == LoopTerminationStatus.complete:
        result_sections = sections
        result_status = StepTerminationStatus.done
    elif status in (LoopTerminationStatus.partial, LoopTerminationStatus.error):
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

    prompt = build_summary_writing_prompt(
        sections_result,
        template_name=wcfg.writer_summary_prompt_template,
    )
    front_matter_output, judge_status = await llm_client.call_structured_with_judge_retry(
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

    if judge_status in (JudgeLoopTerminationStatus.complete, JudgeLoopTerminationStatus.partial):
        result_status = StepTerminationStatus.done
        result_output = front_matter_output
        if judge_status == JudgeLoopTerminationStatus.partial:
            logger.warning(
                f"{step_prefix} - front matter completion status is partial, marking as done."
            )
    elif judge_status == JudgeLoopTerminationStatus.error:
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


async def _resolve_report_job_topic(cluster_pk_hash: str, cfg: ReportGenerationConfig) -> TopicResolveOutput:
    """
    Resolve a cluster to a topic using the topic resolver.

    This helper function handles the topic resolution logic:
    - Fetches topic resolver metadata from memo
    - Converts metadata to TopicInput and ClusterInput formats
    - Calls resolve_topic with the threshold from config
    - Logs the resolution result

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        cfg: ReportGenerationConfig instance

    Raises:
        TopicResolverError: If topic resolution fails
        Exception: For any other unexpected errors
    """
    # Get topic resolver metadata from memo
    metadata = await memo.get_topic_resolver_metadata(cluster_pk_hash, cfg.memo)

    try:
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
        return resolve_topic(topics, cluster, resolve_threshold)

    except TopicResolverError as e:
        logger.error(f"Topic resolver error for cluster {cluster_pk_hash}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during topic resolution for cluster {cluster_pk_hash}: {e}", exc_info=True)
        raise


async def _kick_off_report_job(cluster_pk_hash: str, user_intent: UserIntent, cfg: ReportGenerationConfig) -> None:
    """
    Kick off a report generation job by resolving the cluster to a topic.
    Uses async LLM call (non-blocking) when executor is configured.

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        user_intent: User intent enum
        cfg: ReportGenerationConfig instance
    """
    # new topic resolution
    resolved_topic = await _resolve_report_job_topic(cluster_pk_hash, cfg)
    if resolved_topic.action.value == "merge":
        logger.info(
            f"Topic resolution for cluster {cluster_pk_hash}: MERGE to topic {resolved_topic.merge_to_topic} "
            f"(similarity score: {resolved_topic.score:.4f}, new weight: {resolved_topic.new_topic_weight:.2f})"
        )
    else:
        logger.info(
            f"Topic resolution for cluster {cluster_pk_hash}: CREATE new topic "
            f"(new weight: {resolved_topic.new_topic_weight:.2f})"
        )
    # fetch new topic metadata
    add_top_papers = user_intent != UserIntent.QUICK_BACKGROUND
    try:
        new_topic_metadata = await memo.get_report_generation_metadata(
            cluster_pk_hash=cluster_pk_hash,
            config=cfg.memo,
            topic_id=resolved_topic.merge_to_topic,
            add_top_papers=add_top_papers,
        )
    except Exception as e:
        logger.warning(
            f"New topic metadata fetch failed for cluster {cluster_pk_hash}: {e}",
            exc_info=True,
        )
        # write to memo with error metadata
        # need to update the report job status to error
        return

    # Initialize LLM client
    llm_cfg = cfg.report_generation.llm_gemini
    llm_client = _initialize_llm_client(llm_cfg, llm_cfg.model)

    # call 1 to llm report planner
    plan, step_status = await _generate_report_plan(
        cluster_pk_hash, user_intent, new_topic_metadata, llm_client, cfg, planner_judge
    )
    if step_status == StepTerminationStatus.error or plan is None:
        logger.warning(f"Report planner call failed for cluster {cluster_pk_hash}")
        # write to memo with error metadata
        # need to update the report job status to error
        raise

    sections_result, step_status = await _generate_report_body(
        cluster_pk_hash, plan, new_topic_metadata, llm_client, cfg
    )
    if step_status == StepTerminationStatus.error or sections_result is None:
        logger.warning(f"Report body writing failed for cluster {cluster_pk_hash}")
        # write to memo with error metadata
        # need to update the report job status to error
        # !!!! if sections_result is not none, save a writing checkpoint to local for manual retry
        raise

    # generate report front matter
    front_matter, step_status = await _generate_report_front_matter(
        cluster_pk_hash, sections_result, llm_client, cfg
    )
    if step_status == StepTerminationStatus.error or front_matter is None:
        logger.warning(f"Report front matter generation failed for cluster {cluster_pk_hash}")
        # write to memo with error metadata
        # need to update the report job status to error
        raise


async def create_report_job(cluster_pk_hash: str, user_intent: UserIntent, cfg: ReportGenerationConfig) -> Optional[Tuple[str, str]]:
    """
    Create a report generation job as the first step for any report generation request.
    Non-blocking: uses async LLM calls when executor is configured.

    Calls memo.start_report_job and handles/logs the response. Processes different response cases:
    - Case 1: Report already done (status='done') -> returns ('fetch_report_and_print_report_url', message)
    - Case 2: Recent error occurred (status='error') -> returns ('wait_for_report_job_to_finish', message)
    - Case 3: New job started (status='running', new_job=True, message doesn't contain 'errored expired') -> runs kick_off, returns ('kick_off_report_job', message)
    - Case 4: Existing job already running (status='running', new_job=False) -> returns ('wait_for_report_job_to_finish', message)
    - Case 5: Previous error expired, new job started (status='running', new_job=True, message contains 'errored expired') -> runs kick_off, returns ('kick_off_report_job', message)
    - Unexpected status -> raises exception

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        user_intent: User intent enum
        cfg: ReportGenerationConfig instance

    Returns:
        Tuple of (function_name, descriptive_message), or None if memo is disabled
    """
    start_report_job_response = await memo.start_report_job(cluster_pk_hash, cfg.memo)

    # kick off a new report generation job
    if start_report_job_response.status == 'running' and start_report_job_response.new_job:
        # Check if it's error_expired or running_new by message content
        if 'errored expired' in start_report_job_response.message.lower():
            # Case 5: Error expired, job is running now
            logger.info(f"Memo start-report-job: Previous error expired, new job started. message={start_report_job_response.message}")
            await _kick_off_report_job(cluster_pk_hash, user_intent, cfg)
            return ('kick_off_report_job', f"Previous error expired, new job started. {start_report_job_response.message}")
        else:
            # Case 3: New job is running
            logger.info(f"Memo start-report-job: New job started. message={start_report_job_response.message}")
            await _kick_off_report_job(cluster_pk_hash, user_intent, cfg)
            return ('kick_off_report_job', f"New job started. {start_report_job_response.message}")
    # wait for the existing job to finish
    elif start_report_job_response.status == 'running' and not start_report_job_response.new_job:
        # Case 4: Existing job already running
        logger.info(f"Memo start-report-job: Existing job already running. message={start_report_job_response.message}")
        return ('wait_for_report_job_to_finish', f"Existing job already running. {start_report_job_response.message}")
    # report generation failed
    elif start_report_job_response.status == 'error':
        # Case 2: Recent error, need to wait
        logger.warning(f"Memo start-report-job: Recent error occurred. message={start_report_job_response.message}")
        return ('wait_for_report_job_to_finish', f"Recent error occurred, need to wait. {start_report_job_response.message}")
    # move to fetch the report and print the report url
    elif start_report_job_response.status == 'done':
        # Case 1: Report already done
        logger.info(f"Memo start-report-job: Report already generated. report_id={start_report_job_response.report_id}, message={start_report_job_response.message}")
        return ('fetch_report_and_print_report_url', f"Report already generated. report_id={start_report_job_response.report_id}. {start_report_job_response.message}")

    else:
        # Unexpected status
        logger.warning(f"Memo start-report-job: Unexpected response. status={start_report_job_response.status}, new_job={start_report_job_response.new_job}, message={start_report_job_response.message}")
        raise ValueError(f"Unexpected response. status={start_report_job_response.status}, new_job={start_report_job_response.new_job}, message={start_report_job_response.message}")


# -------------------------

# -------------------------
