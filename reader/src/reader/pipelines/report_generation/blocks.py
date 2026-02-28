"""Pipeline building blocks"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from algo_lib.topic_resolver.models import TopicInput, ClusterInput as TopicResolverClusterInput
from algo_lib.topic_resolver.resolver import resolve_topic
from algo_lib.topic_resolver.errors import TopicResolverError

from reader.pipelines.report_generation.config.config import ReportGenerationConfig, LLMGeminiConfig
from reader.adapters import memo
from reader.adapters.memo import PaperCard, StartReportJobResponse, GetReportPlannerMetadataResponse
from reader.adapters.llm import LLMClient, TokenBucket, LLMGenerationError
from pydantic import ValidationError
from reader.pipelines.report_generation.report import LLMReportPlannerOutput
from reader.pipelines.report_generation.metrics import (
    judge_output,
    JudgeOutput,
    count_judge_warnings,
    inject_judge_warnings_into_prompt,
)
from reader.logging.logging_setup import get_logger
from reader.pipelines.report_generation.prompts.planner.build import UserIntent, get_planner_intent_spec, build_planner_prompt

logger = get_logger()

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
) -> Tuple[Optional[LLMReportPlannerOutput], JudgeOutput]:
    """
    Call report planner LLM with judge retry logic.

    Returns:
        Tuple of (planner_output, judge_output). planner_output may be None on LLM/validation failure.
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
                return best_planner_output, best_judge_result

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
            return best_planner_output, best_judge_result

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
    return best_planner_output, best_judge_result


async def _resolve_report_job_topic(cluster_pk_hash: str, cfg: ReportGenerationConfig) -> None:
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


async def _kick_off_report_job(cluster_pk_hash: str, user_intent: str, cfg: ReportGenerationConfig) -> None:
    """
    Kick off a report generation job by resolving the cluster to a topic.
    Uses async LLM call (non-blocking) when executor is configured.

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        user_intent: User intent string
        cfg: ReportGenerationConfig instance
    """
    # call 1 to llm report planner
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
    planner_prompt = await _generate_planner_prompt(user_intent, cluster_pk_hash, cfg, resolved_topic.merge_to_topic)

    # Initialize LLM client 
    llm_cfg = cfg.report_generation.llm_gemini
    llm_client = _initialize_llm_client(llm_cfg, llm_cfg.model)
    # call 1 to llm report planner with judge retry
    planner_output, judge_result = await _call_planner_with_judge_retry(
        cluster_pk_hash, planner_prompt, llm_client, cfg
    )
    if planner_output is not None:
        logger.info(f"Successfully called report planner for cluster {cluster_pk_hash}, overall score: {judge_result.overall:.2f}")
        # call 2 to llm report writter
        # organize metadata for db updates and save report to local fs
    else:
        logger.warning(f"Report planner call failed for cluster {cluster_pk_hash}")
        # write to memo with error metadata
        # need to update the report job status to error





    


async def create_report_job(cluster_pk_hash: str, user_intent: str, cfg: ReportGenerationConfig) -> Optional[Tuple[str, str]]:
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
        user_intent: User intent string
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


# forplanner prompt generation
async def _generate_planner_prompt(
    user_intent: str,
    cluster_pk_hash: str,
    config: ReportGenerationConfig,
    topic_id: Optional[str] = None,
) -> str:
    """
    Generate planner prompt by fetching cluster metadata and building the prompt.

    Args:
        user_intent: User intent as string (e.g., "Quick Background (5-10 min overview)")
        cluster_pk_hash: Cluster primary key hash
        config: ReportGenerationConfig instance
        topic_id: Optional topic ID (as string) to include top ≤3 reports for that topic

    Returns:
        Final prompt string ready to be sent to the LLM

    Raises:
        ValueError: If user_intent is invalid, memo is disabled, or other errors occur
    """
    # Convert user_intent string to UserIntent enum
    try:
        intent_enum = UserIntent.from_display_string(user_intent)
    except ValueError as e:
        raise ValueError(f"Invalid user_intent: {user_intent}") from e

    # Get IntentSpec for the user intent (production: evidence gaps only)
    intent_spec = get_planner_intent_spec(intent_enum)

    # Determine add_top_papers: False only for QUICK_BACKGROUND, True for all others
    add_top_papers = intent_enum != UserIntent.QUICK_BACKGROUND

    # Call memo.get_report_planner_metadata
    cluster_metadata = await memo.get_report_planner_metadata(
        cluster_pk_hash=cluster_pk_hash,
        config=config.memo,
        topic_id=topic_id,
        add_top_papers=add_top_papers,
    )

    # Call build_planner_prompt with intent_spec and cluster_metadata
    prompt = build_planner_prompt(intent_spec=intent_spec, cluster_metadata=cluster_metadata)

    return prompt


# -------------------------

# -------------------------
