"""Pipeline building blocks"""

import json
import os
from typing import Optional, Tuple

from algo_lib.topic_resolver.models import TopicInput, ClusterInput as TopicResolverClusterInput
from algo_lib.topic_resolver.resolver import resolve_topic
from algo_lib.topic_resolver.errors import TopicResolverError

from reader.config import ReaderConfig
from reader.adapters import memo
from reader.adapters.memo import PaperCard, StartReportJobResponse, GetReportPlannerMetadataResponse
from reader.adapters.llm import LLMClient, TokenBucket, LLMGenerationError
from pydantic import ValidationError
from reader.pipelines.report import (
    LLMConfigInput,
    LLMReportPlannerOutput,
)
from reader.logging.logging_setup import get_logger
from reader.prompts.report_planner.build import UserIntent, get_intent_spec
from reader.prompts.report_planner.build import build_baseline_planner_prompt

logger = get_logger()

# ============================================================================
# LLM client initialization (used by report generation)
# ============================================================================

def _initialize_llm_client(cfg: ReaderConfig, model: str) -> LLMClient:
    """
    Initialize LLM client with rate limiting buckets.
    
    Args:
        cfg: ReaderConfig instance
        model: Model name to use for the LLM client
    
    Returns:
        Initialized LLMClient instance
    
    Raises:
        ValueError: If API key is not found in environment variable
    """
    # Get API key from environment variable
    api_key = os.getenv(cfg.llm_gemini.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {cfg.llm_gemini.api_key_env}")
    
    # Initialize TokenBucket instances for rate limiting
    rpm_bucket = TokenBucket(
        capacity=cfg.llm_gemini.gemini_rpm_limit,
        refill_rate=cfg.llm_gemini.gemini_rpm_limit,
        name="gemini_rpm"
    )
    
    tpm_bucket = TokenBucket(
        capacity=cfg.llm_gemini.gemini_tpm_limit,
        refill_rate=cfg.llm_gemini.gemini_tpm_limit,
        name="gemini_tpm"
    )
    
    # Create LLMClient instance
    llm_client = LLMClient(
        model=model,
        api_key=api_key,
        rpm_bucket=rpm_bucket,
        tpm_bucket=tpm_bucket
    )
    
    logger.info(f"Initialized LLM client with model: {model}")
    return llm_client


# ============================================================================
# Report generation blocks
# ============================================================================

def _resolve_report_job_topic(cluster_pk_hash: str, cfg: ReaderConfig) -> None:
    """
    Resolve a cluster to a topic using the topic resolver.
    
    This helper function handles the topic resolution logic:
    - Fetches topic resolver metadata from memo
    - Converts metadata to TopicInput and ClusterInput formats
    - Calls resolve_topic with the threshold from config
    - Logs the resolution result
    
    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        cfg: ReaderConfig instance
    
    Raises:
        TopicResolverError: If topic resolution fails
        Exception: For any other unexpected errors
    """
    # Get topic resolver metadata from memo
    metadata = memo.get_topic_resolver_metadata(cluster_pk_hash, cfg.memo)
    
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


def create_report_job(cluster_pk_hash: str, user_intent: str, cfg: ReaderConfig) -> Optional[Tuple[str, str]]:
    """
    Create a report generation job as the first step for any report generation request.
    
    Calls memo.start_report_job and handles/logs the response. Processes different response cases:
    - Case 1: Report already done (status='done') -> returns ('fetch_report_and_print_report_url', message)
    - Case 2: Recent error occurred (status='error') -> returns ('wait_for_report_job_to_finish', message)
    - Case 3: New job started (status='running', new_job=True, message doesn't contain 'errored expired') -> returns ('kick_off_report_job', message)
    - Case 4: Existing job already running (status='running', new_job=False) -> returns ('wait_for_report_job_to_finish', message)
    - Case 5: Previous error expired, new job started (status='running', new_job=True, message contains 'errored expired') -> returns ('kick_off_report_job', message)
    - Unexpected status -> raises exception
    
    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        cfg: ReaderConfig instance
    
    Returns:
        Tuple of (function_name, descriptive_message), or None if memo is disabled
    """
    def _kick_off_report_job(cluster_pk_hash: str, user_intent: str, cfg: ReaderConfig):
        """
        Kick off a report generation job by resolving the cluster to a topic.
        
        Args:
            cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
            cfg: ReaderConfig instance
        """
        # call 1 to llm report planner
        resolved_topic =_resolve_report_job_topic(cluster_pk_hash, cfg)
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
        planner_prompt = _generate_planner_prompt(user_intent, cluster_pk_hash, cfg, resolved_topic.merge_to_topic)
        
        # Initialize LLM client and call report planner
        llm_client = _initialize_llm_client(cfg, cfg.llm_gemini.models[0])
        
        try:
            planner_output = llm_client.call_structured_raw(
                prompt=planner_prompt,
                response_model=LLMReportPlannerOutput,
                temperature=cfg.llm_gemini.temperature,
                max_tokens=cfg.llm_gemini.max_tokens
            )
            logger.info(f"Successfully generated report planner output for cluster {cluster_pk_hash}")
        except LLMGenerationError as e:
            logger.error(f"LLM report planner call error for cluster {cluster_pk_hash}: {str(e)}", exc_info=True)
            raise
        except ValidationError as e:
            logger.error(f"Validation error in report planner call for cluster {cluster_pk_hash} (after retries exhausted): {str(e)}", exc_info=True)
            raise
        # call 2 to llm report planner
        # organize metadata for db updates and save report to local fs
        




        
        


    def _wait_for_report_job_to_finish(cluster_pk_hash: str, cfg: ReaderConfig):
        pass


    def _fetch_report_and_print_report_url(cluster_pk_hash: str, cfg: ReaderConfig):
        pass
    
    start_report_job_response = memo.start_report_job(cluster_pk_hash, cfg.memo)
    
    # kick off a new report generation job
    if start_report_job_response.status == 'running' and start_report_job_response.new_job:
        # Check if it's error_expired or running_new by message content
        if 'errored expired' in start_report_job_response.message.lower():
            # Case 5: Error expired, job is running now
            logger.info(f"Memo start-report-job: Previous error expired, new job started. message={start_report_job_response.message}")
            return ('kick_off_report_job', f"Previous error expired, new job started. {start_report_job_response.message}")
        else:
            # Case 3: New job is running
            logger.info(f"Memo start-report-job: New job started. message={start_report_job_response.message}")
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
def _generate_planner_prompt(
    user_intent: str,
    cluster_pk_hash: str,
    config: ReaderConfig,
    topic_id: Optional[str] = None,
) -> str:
    """
    Generate planner prompt by fetching cluster metadata and building the prompt.
    
    Args:
        user_intent: User intent as string (e.g., "Quick Background (5-10 min overview)")
        cluster_pk_hash: Cluster primary key hash
        config: ReaderConfig instance
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
    
    # Get IntentSpec for the user intent
    intent_spec = get_intent_spec(intent_enum)
    
    # Determine add_top_papers: False only for QUICK_BACKGROUND, True for all others
    add_top_papers = intent_enum != UserIntent.QUICK_BACKGROUND
    
    # Call memo.get_report_planner_metadata
    cluster_metadata = memo.get_report_planner_metadata(
        cluster_pk_hash=cluster_pk_hash,
        config=config.memo,
        topic_id=topic_id,
        add_top_papers=add_top_papers,
    )
    
    # Call build_baseline_planner_prompt with intent_spec and cluster_metadata
    prompt = build_baseline_planner_prompt(intent_spec=intent_spec, cluster_metadata=cluster_metadata)
    
    return prompt


# -------------------------

# -------------------------
