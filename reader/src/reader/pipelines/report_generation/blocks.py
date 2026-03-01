"""Pipeline building blocks"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from algo_lib.topic_resolver.models import TopicInput, ClusterInput as TopicResolverClusterInput, TopicResolveOutput
from algo_lib.topic_resolver.resolver import resolve_topic
from algo_lib.topic_resolver.errors import TopicResolverError

from reader.pipelines.report_generation.config.config import ReportGenerationConfig, LLMGeminiConfig
from reader.adapters import memo
from reader.adapters.memo import (
    ClusterObservationData,
    PaperCard,
    StartReportJobResponse,
    GetReportPlannerMetadataResponse,
    GetReportPlannerSupplementRequest,
    PaperSupplementRequest,
    ReportSupplementRequest,
)
from reader.adapters.llm import LLMClient, TokenBucket, LLMGenerationError
from pydantic import ValidationError
from reader.pipelines.report_generation.report import (
    LLMReportPlannerOutput,
    EvidenceGap,
    EvidenceCollectionTerminationSufficiency,
)
from reader.pipelines.report_generation.metrics import (
    judge_output,
    JudgeOutput,
    count_judge_warnings,
    inject_judge_warnings_into_prompt,
)
from reader.logging.logging_setup import get_logger
from reader.pipelines.report_generation.prompts.planner.build import UserIntent, get_planner_intent_spec, build_plan_guidance, build_planner_prompt
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
    llm_client = LLMClient(
        model=model,
        api_key=api_key,
        rpm_bucket=rpm_bucket,
        tpm_bucket=tpm_bucket,
        executor=executor
    )

    logger.info(f"Initialized LLM client with model: {model}")
    return llm_client


# ============================================================================
# Report generation blocks
# ============================================================================


def _append_planner_output_to_jsonl(
    log_path: Optional[str],
    cluster_pk_hash: str,
    planner_output: Optional[LLMReportPlannerOutput],
    judge_output_result: JudgeOutput,
) -> None:
    """Append (planner_output, judge_output) to JSONL file. No-op if log_path is None or empty."""
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cluster_pk_hash": cluster_pk_hash,
        "date": datetime.now(timezone.utc).isoformat(),
        "planner_output": planner_output.model_dump() if planner_output is not None else None,
        "judge_output": asdict(judge_output_result),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Successfully appended report planner output for cluster {cluster_pk_hash} to {log_path}")


async def _call_planner_with_judge_retry(
    cluster_pk_hash: str,
    planner_prompt: str,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
) -> Optional[LLMReportPlannerOutput]:
    """
    Call report planner LLM with judge retry logic.

    Returns:
        planner_output, or None on LLM/validation failure.
    """
    prompt_base = planner_prompt
    max_retries = cfg.report_generation.max_judge_retries
    retry_threshold = cfg.report_generation.judge_retry_threshold
    llm_cfg = cfg.report_generation.llm_gemini

    best_planner_output: Optional[LLMReportPlannerOutput] = None
    best_judge_result: Optional[JudgeOutput] = None
    best_score = float("-inf")

    for attempt in range(max_retries + 1):
        try:
            planner_output = await llm_client.call_structured_raw_async(
                prompt=planner_prompt,
                response_model=LLMReportPlannerOutput,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
            )
            judge_result = judge_output(planner_output)

            if judge_result.overall > best_score:
                best_score = judge_result.overall
                best_planner_output = planner_output
                best_judge_result = judge_result

            if judge_result.overall > retry_threshold:
                logger.info(
                    f"cluster {cluster_pk_hash} attempt {attempt + 1}/{max_retries + 1}: "
                    f"planner overall score {best_score:.2f} > {retry_threshold}, accepted"
                )
                _append_planner_output_to_jsonl(
                    cfg.report_generation.planner_output_log_path,
                    cluster_pk_hash,
                    best_planner_output,
                    best_judge_result,
                )
                return best_planner_output

            if attempt < max_retries:
                planner_prompt, num_warnings = inject_judge_warnings_into_prompt(prompt_base, judge_result)
                logger.warning(
                    f"cluster {cluster_pk_hash} attempt {attempt + 1}/{max_retries + 1}: "
                    f"planner overall score {judge_result.overall:.2f} <= {retry_threshold}, "
                    f"injecting {num_warnings} warning(s), retrying"
                )
            else:
                num_warnings = count_judge_warnings(judge_result)
                logger.warning(
                    f"cluster {cluster_pk_hash} attempt {attempt + 1}/{max_retries + 1}: "
                    f"planner overall score {judge_result.overall:.2f} <= {retry_threshold}, "
                    f"{num_warnings} warning(s) (last judge retry, returning best)"
                )

        except (LLMGenerationError, ValidationError) as e:
            logger.warning(f"cluster {cluster_pk_hash} attempt {attempt + 1}/{max_retries + 1}: llm call failed: {e}")
            if best_planner_output is None:
                if isinstance(e, LLMGenerationError):
                    reasons = {"llm_generation_error": [("llm api call must succeed to return planner output", str(e))]}
                else:
                    reasons = {
                        "validation_error": [
                            ("llm structured output must be valid json matching LLMReportPlannerOutput schema", str(e))
                        ]
                    }
                failed_judge = JudgeOutput(sub_scores={}, overall=0.0, reasons=reasons)
                best_judge_result = failed_judge
            continue
        except Exception as e:
            logger.warning(
                f"cluster {cluster_pk_hash} attempt {attempt + 1}/{max_retries + 1}: unexpected error, breaking retry: {e}",
                exc_info=True,
            )
            if best_planner_output is None:
                reasons = {"internal_error": [("unexpected error during planner call", str(e))]}
                best_judge_result = JudgeOutput(sub_scores={}, overall=0.0, reasons=reasons)
            _append_planner_output_to_jsonl(
                cfg.report_generation.planner_output_log_path,
                cluster_pk_hash,
                best_planner_output,
                best_judge_result,
            )
            return best_planner_output

    logger.warning(
        f"cluster {cluster_pk_hash} attempt {max_retries + 1}/{max_retries + 1}: "
        f"judge retries exhausted, returning best planner_output (overall score: {best_score:.2f})"
    )
    _append_planner_output_to_jsonl(
        cfg.report_generation.planner_output_log_path,
        cluster_pk_hash,
        best_planner_output,
        best_judge_result,
    )
    return best_planner_output


# ---------- Evidence collection loop enums ----------


class EvidenceLoopExitCondition(str, Enum):
    """Reason the report planner evidence collection (call 1) concluded."""

    SUFFICIENCY_TERMINAL = "sufficiency_terminal"
    EVIDENCE_GAPS_BELOW_THRESHOLD = "evidence_gaps_below_threshold"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    ERROR = "error"


class EvidenceCollectionStatus(str, Enum):
    """Final status of the report planner evidence collection process."""

    complete = "complete"
    partial = "partial"
    error = "error"


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


def _should_exit_evidence_loop(
    planner_output: Optional[LLMReportPlannerOutput],
    loop_turn: int,
    cfg: ReportGenerationConfig,
) -> Optional[EvidenceLoopExitCondition]:
    """
    Decide whether the evidence collection loop should conclude.

    Priority order: (a) sufficiency terminal, (b) evidence gaps below threshold,
    (c) max iterations reached. Returns None to continue, otherwise the exit condition.
    """
    if planner_output is None:
        return None
    # 1.a: plan.sufficiency in termination set
    if planner_output.plan.sufficiency in EvidenceCollectionTerminationSufficiency:
        return EvidenceLoopExitCondition.SUFFICIENCY_TERMINAL
    # 1.b: evidence_gaps count below threshold
    threshold = cfg.report_generation.max_evidence_gaps_threshold
    if len(planner_output.evidence_gaps) < threshold:
        return EvidenceLoopExitCondition.EVIDENCE_GAPS_BELOW_THRESHOLD
    # 1.c: max iterations reached
    max_iter = cfg.report_generation.max_evidence_loop_iterations
    if loop_turn >= max_iter:
        return EvidenceLoopExitCondition.MAX_ITERATIONS_REACHED
    return None


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
    phase1_metadata: GetReportPlannerMetadataResponse,
    plan_guidance: str,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[LLMReportPlannerOutput], EvidenceCollectionStatus]:
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
                    req = GetReportPlannerSupplementRequest(
                        paper_requests=uncached_paper,
                        report_requests=uncached_report,
                    )
                    resp = await memo.get_report_planner_supplement(req, cfg.memo)
                    for paper_id, selector_map in resp.paper_supplements.items():
                        phase2_supplement["paper_supplements"].setdefault(paper_id, {}).update(selector_map)
                    for report_id, field_map in resp.report_supplements.items():
                        phase2_supplement["report_supplements"].setdefault(report_id, {}).update(field_map)
                elif not paper_reqs and not report_reqs:
                    logger.warning(
                        f"No new supplement added after dedup and cache filter for cluster {cluster_pk_hash}. keep using the previous supplement."
                    )

            has_supplement = phase2_supplement["paper_supplements"] or phase2_supplement["report_supplements"]
            planner_prompt = build_planner_prompt(
                phase1_metadata=phase1_metadata,
                plan_guidance=plan_guidance,
                phase2_supplement=phase2_supplement if has_supplement else None,
            )
            planner_output = await _call_planner_with_judge_retry(
                cluster_pk_hash, planner_prompt, llm_client, cfg
            )

            if planner_output is None:
                exit_condition = EvidenceLoopExitCondition.ERROR
                break

            last_planner_output = planner_output
            exit_condition = _should_exit_evidence_loop(planner_output, loop_turn, cfg)
            if exit_condition is not None:
                break

            loop_turn += 1

    except Exception as e:
        logger.warning(
            f"Report planner evidence collection (call 1) terminated due to an error for cluster {cluster_pk_hash}: {e}",
            exc_info=True,
        )
        exit_condition = EvidenceLoopExitCondition.ERROR

    # Map exit condition to status and log
    if exit_condition == EvidenceLoopExitCondition.SUFFICIENCY_TERMINAL:
        status = EvidenceCollectionStatus.complete
        logger.info(
            f"Report planner evidence collection (call 1) concluded for cluster {cluster_pk_hash}: "
            f"plan sufficiency is sufficient or borderline — evidence collection complete."
        )
    elif exit_condition == EvidenceLoopExitCondition.EVIDENCE_GAPS_BELOW_THRESHOLD:
        status = EvidenceCollectionStatus.complete
        count = len(last_planner_output.evidence_gaps) if last_planner_output else 0
        threshold = cfg.report_generation.max_evidence_gaps_threshold
        logger.info(
            f"Report planner evidence collection (call 1) concluded for cluster {cluster_pk_hash}: "
            f"evidence gaps remaining ({count}) below threshold ({threshold}) — evidence collection complete."
        )
    elif exit_condition == EvidenceLoopExitCondition.MAX_ITERATIONS_REACHED:
        status = EvidenceCollectionStatus.partial
        max_iter = cfg.report_generation.max_evidence_loop_iterations
        logger.info(
            f"Report planner evidence collection (call 1) concluded for cluster {cluster_pk_hash}: "
            f"reached maximum iterations ({max_iter}) — evidence collection partial."
        )
    else:
        status = EvidenceCollectionStatus.error
        if last_planner_output is not None:
            logger.warning(
                f"Report planner evidence collection (call 1) terminated due to an error for cluster {cluster_pk_hash}. "
                f"The returned plan is from the last successful planner call, not a final result."
            )
        else:
            logger.warning(
                f"Report planner call failed for cluster {cluster_pk_hash} (no successful pass)."
            )

    return (last_planner_output, status)


async def _generate_report_plan(
    cluster_pk_hash: str,
    user_intent: UserIntent,
    resolved_topic: TopicResolveOutput,
    llm_client: LLMClient,
    cfg: ReportGenerationConfig,
) -> Tuple[Optional[LLMReportPlannerOutput], EvidenceCollectionStatus]:
    """
    Generate a report plan via two-phase evidence collection.

    Phase 1: Fetch summary-level metadata and build plan guidance (once).
    Phase 2: Run evidence completion loop (planner calls until exit condition).

    Returns:
        (last_planner_output, evidence_status)
    """
    # Phase 1: fetch summary-level metadata once
    intent_spec = get_planner_intent_spec(user_intent)
    add_top_papers = user_intent != UserIntent.QUICK_BACKGROUND
    phase1_metadata = await memo.get_report_planner_metadata(
        cluster_pk_hash=cluster_pk_hash,
        config=cfg.memo,
        topic_id=resolved_topic.merge_to_topic,
        add_top_papers=add_top_papers,
    )
    plan_guidance = build_plan_guidance(intent_spec)

    # Phase 2: evidence completion loop
    return await _run_evidence_completion_loop(
        cluster_pk_hash, phase1_metadata, plan_guidance, llm_client, cfg
    )


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

    # Initialize LLM client
    llm_cfg = cfg.report_generation.llm_gemini
    llm_client = _initialize_llm_client(llm_cfg, llm_cfg.model)

    # call 1 to llm report planner
    last_planner_output, evidence_status = await _generate_report_plan(
        cluster_pk_hash, user_intent, resolved_topic, llm_client, cfg
    )
    # !!!! placeholder[exit point]
    if last_planner_output is None:
        logger.warning(f"Report planner call failed for cluster {cluster_pk_hash}")
        # write to memo with error metadata
        # need to update the report job status to error
    # call 1.5 to llm report writer
    # call 2 to llm report writer


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
