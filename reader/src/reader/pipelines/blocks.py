"""Pipeline building blocks"""

import json
import os
import calendar
import hashlib
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any, Tuple
import numpy as np

from algo_lib.clustering import get_best_clustering
from algo_lib.typing import PaperLike
from algo_lib.topic_resolver.models import TopicInput, ClusterInput as TopicResolverClusterInput
from algo_lib.topic_resolver.resolver import resolve_topic
from algo_lib.topic_resolver.errors import TopicResolverError

from reader.config import ReaderConfig, render_best_cluster_text_report_path, render_best_cluster_report_path
from reader.adapters.hf import get_monthly_report, parse_papers, save_papers_to_file
from reader.adapters import memo
from reader.adapters.memo import GetBestRunResponse, ClusterCard, PaperCard, StartReportJobResponse, GetReportPlannerMetadataResponse
from reader.adapters.llm import LLMClient, TokenBucket, LLMGenerationError
from pydantic import ValidationError
from reader.pipelines.report import (
    FreshPaperPayload,
    PaperInput,
    ClusterMemberInput,
    ClusterInput,
    EmbedConfig,
    EmbedConfigPayload,
    ClusterConfig,
    ClusterConfigPayload,
    ClusterReport,
    LLMConfigInput,
    ClusterObservation,
    InjectClustersObservationInput,
    LLMReportPlannerOutput,
)
from reader.pipelines.metrics import judge_output, JudgeOutput
from reader.logging.logging_setup import get_logger
from reader.prompts.report_planner.spec import UserIntent, get_intent_spec
from reader.prompts.report_planner.build import build_planner_prompt

logger = get_logger()

# ============================================================================
# Fetch metadata blocks
# ============================================================================
def _extract_period_dates(month_key: str) -> tuple[str, str]:
    """
    Extract period_start and period_end from month key.
    
    Args:
        month_key: Format "month=YYYY-MM" (e.g., "month=2025-01")
    
    Returns:
        Tuple of (period_start, period_end) in YYYY-MM-DD format
    """
    # Parse "month=2025-01" to get year and month
    parts = month_key.split('=')
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Invalid month key format: {month_key}")
    
    year_month = parts[1]
    year, month = map(int, year_month.split('-'))
    
    # First day of month
    period_start = f"{year:04d}-{month:02d}-01"
    
    # Last day of month
    last_day = calendar.monthrange(year, month)[1]
    period_end = f"{year:04d}-{month:02d}-{last_day:02d}"
    
    return period_start, period_end


def get_hf_paper_metadata(cfg: ReaderConfig) -> tuple[list, str, str]:
    """
    Get paper metadata from HF API or cached file.
    
    Args:
        cfg: ReaderConfig instance
    
    Returns:
        Tuple of (papers, period_start, period_end)
    """
    import asyncio
    
    month_key = cfg.run.month_key
    papers_report_file = cfg.sources.hf.output_json
    
    # Check if papers_report.json exists, generate if missing
    if not Path(papers_report_file).exists():
        logger.info(f"{papers_report_file} not found, generating from HF API...")
        results = asyncio.run(get_monthly_report(cfg))
        save_papers_to_file(results, cfg)
        logger.info(f"Generated {papers_report_file}")
    
    # Load papers_report_file
    with open(papers_report_file, "r") as f:
        data = json.load(f)
    papers_data = data['papers']
    
    # Process single month
    if month_key not in papers_data:
        raise ValueError(f"Month {month_key} not found in papers_data. Available months: {list(papers_data.keys())}")
    
    papers_list = papers_data[month_key]
    logger.info(f"Processing {month_key}")
    
    # Extract period dates from month key
    period_start, period_end = _extract_period_dates(month_key)
    logger.info(f"Period: {period_start} to {period_end}")
    
    # Create Paper objects from JSON data
    papers = parse_papers(papers_list, cfg)
    
    return papers, period_start, period_end

# ============================================================================
# Embedding + clustering blocks
# ============================================================================
def write_best_clustering_text_report(
    papers: Sequence[PaperLike],
    cluster_members_ordered: Dict[int, List[int]],
    header: str = "",
    max_summary_chars: int = 350,
    report_dir: str = 'best_clustering_reports.md',
) -> None:
    """
    Write a human-readable text report for a chosen clustering:
    - cluster sizes 
    - optional TF-IDF keyword hints
    - each paper: title + (truncated) summary + url
    
    Args:
        papers: Sequence of paper-like objects
        cluster_members_ordered: Dictionary mapping cluster_id -> list of paper indices sorted by similarity
        header: Optional header string to write at the top
        max_summary_chars: Maximum characters for summary truncation
        report_dir: Path to the report file
    """
    clusters = cluster_members_ordered
    
    with open(report_dir, 'a+') as f:
        if header:
            f.write("\n" + "=" * 90 + '\n')
            f.write(header + '\n')

        # sort by cluster size desc
        cluster_order = sorted(clusters.keys(), key=lambda c: len(clusters[c]), reverse=True)

        for c in cluster_order:
            idxs = clusters[c]
            f.write("\n" + "-" * 90 + "\n")
            f.write(f"Cluster {c} | size={len(idxs)}\n")

            for i in idxs:
                p = papers[i]
                summ = p.summary.strip() if p.summary else ""
                if max_summary_chars and len(summ) > max_summary_chars:
                    summ = summ[:max_summary_chars] + "…"
                f.write(f"\n[{p.pid}] {p.title}\n")
                if p.url:
                    f.write(f"URL: {p.url}\n")
                if summ:
                    f.write(f"Summary: {summ}\n")


def generate_fresh_paper_payload(
    papers: Sequence[PaperLike],
    member_similarities: Dict[int, Dict[int, float]],
    cluster_cohesion_dict: Dict[int, float],
    cluster_centroids: Dict[int, np.ndarray],
    period_start: str,
    period_end: str,
    embed_model_name: str,
    best_mode: str,
    best_k: int,
    top_n_keywords: int,
    seed: int,
    config: ReaderConfig,
    raw_json: str = "",
    output_path: Optional[str] = None,
) -> FreshPaperPayload:
    """
    Generate fresh_paper_payload.json format report.
    
    Args:
        papers: Sequence of paper-like objects (must have published_at field)
        member_similarities: Dict[cluster_id] -> Dict[paper_idx] -> similarity to centroid
        cluster_cohesion_dict: Dict[cluster_id] -> average cohesion
        cluster_centroids: Dict[cluster_id] -> centroid vector (normalized numpy array)
        period_start: Start date in YYYY-MM-DD format
        period_end: End date in YYYY-MM-DD format
        embed_model_name: Embedding model name
        best_mode: Best embedding mode selected
        best_k: Best k value selected
        top_n_keywords: Number of top keywords used
        seed: Random seed used
        config: ReaderConfig instance
        raw_json: Optional raw JSON string
        output_path: Path for JSON output file. If None, no file will be written.
    
    Returns:
        FreshPaperPayload instance matching fresh_paper_payload.json format
    """
        
    # Build papers array
    papers_list = []
    for p in papers:
        papers_list.append(PaperInput(
            raw_paper_id=p.pid,
            title=p.title,
            summary=p.summary,
            keywords=p.keywords,
            url=p.url,
            published_at=p.published_at,
        ))
    
    # Build clusters array
    clusters_list = []
    
    for c, papers_sim in member_similarities.items():
        idxs = sorted(member_similarities[c], key=member_similarities[c].get, reverse=True)
        members_list = []
        
        for rank, paper_idx in enumerate(idxs):
            sim = papers_sim[paper_idx]
            # Get formatted paper_id from PaperInput instance
            paper_input = papers_list[paper_idx]
            members_list.append(ClusterMemberInput(
                paper_id=paper_input.paper_id,
                rank_in_cluster=rank,
                sim_to_centroid=sim,
            ))
        
        # Convert centroid to base64-encoded float32 bytes
        if c not in cluster_centroids:
            raise ValueError(f"Cluster {c} missing centroid in cluster_centroids")
        centroid = cluster_centroids[c]
        # Convert numpy array to float32 bytes (little-endian)
        # Use '<f4' to explicitly ensure little-endian byte order
        centroid_bytes = centroid.astype('<f4').tobytes()
        # Encode to base64
        centroid_b64 = base64.b64encode(centroid_bytes).decode('utf-8')
        
        clusters_list.append(ClusterInput(
            cluster_index=c,
            size=len(idxs),
            cohesion=cluster_cohesion_dict[c],
            centroid_b64=centroid_b64,
            members=members_list,
        ))
    
    # Build embed_config
    embed_config = EmbedConfig(
        json_payload=EmbedConfigPayload(
            model_name=embed_model_name,
            mode=best_mode,
            top_n_keywords=top_n_keywords,
        )
    )
    
    # Build cluster_config
    cluster_config = ClusterConfig(
        json_payload=ClusterConfigPayload(
            k=best_k,
            seed=seed,
            algorithm="kmeans",
        )
    )
    
    # Build complete payload
    payload = FreshPaperPayload(
        source="hf_monthly",
        period_start=period_start,
        period_end=period_end,
        raw_json=raw_json,
        embed_config=embed_config,
        cluster_config=cluster_config,
        papers=papers_list,
        clusters=clusters_list,
    )
    
    # Write JSON file only if output_path is provided
    if output_path is not None:
        logger.info(f"Generating {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(payload.model_dump_json(indent=2, exclude_none=False))
    
    return payload


def generate_clustering_reports(
    cfg: ReaderConfig,
    papers: list,
    period_start: str,
    period_end: str,
) -> FreshPaperPayload:
    """
    Generate clustering reports and payload.
    
    Args:
        cfg: ReaderConfig instance
        papers: List of Paper objects
        period_start: Start date in YYYY-MM-DD format
        period_end: End date in YYYY-MM-DD format
    
    Returns:
        FreshPaperPayload instance
    """
    month_key = cfg.run.month_key
    
    # Get best clustering with enhanced metadata
    result = get_best_clustering(
        papers=papers,
        embed_model_name=cfg.algos.embedding.model,
        modes=cfg.algos.embedding.modes,
        k_candidates=cfg.algos.clustering.k_candidates,
        top_n_keywords=cfg.algos.embedding.top_n_keywords,
        seed=cfg.algos.clustering.random_seed
    )
    logger.info(f"{month_key} Top choice: mode: {result.mode} k: {result.k} embed_model: {cfg.algos.embedding.model}")

    # Generate JSON payload
    cluster_json_report_path = render_best_cluster_report_path(cfg, month_key)
    fresh_paper_payload = generate_fresh_paper_payload(
        papers=papers,
        member_similarities=result.cluster_members_similarities,
        cluster_cohesion_dict=result.cluster_cohesion,
        cluster_centroids=result.cluster_centroids,
        period_start=period_start,
        period_end=period_end,
        embed_model_name=cfg.algos.embedding.model,
        best_mode=result.mode,
        best_k=result.k,
        top_n_keywords=cfg.algos.embedding.top_n_keywords,
        seed=cfg.algos.clustering.random_seed,
        config=cfg,
        raw_json="",  # Optional: can be set to actual raw JSON if available
        output_path=cluster_json_report_path,  # Will be None if not configured, which triggers default behavior
    )
    # Write text report if configured
    cluster_text_report_path = render_best_cluster_text_report_path(cfg, month_key)
    if cluster_text_report_path:
        logger.info(f"Appending {month_key} best clustering text report to {cluster_text_report_path}")
        # Remove existing report if it exists
        if os.path.exists(cluster_text_report_path):
            os.remove(cluster_text_report_path)
        
        write_best_clustering_text_report(
            papers=papers,
            cluster_members_ordered=result.cluster_members_ordered,
            header=f"# {month_key} BEST CLUSTERING (mode={result.mode}, k={result.k})",
            max_summary_chars=350,
            report_dir=cluster_text_report_path,
        )
    
    return fresh_paper_payload


# ============================================================================
# LLM clusters summarization/enrichment blocks
# ============================================================================

def load_template(template_path: Path) -> str:
    """
    Load prompt template from file.
    
    Args:
        template_path: Path to template file
    
    Returns:
        Template content as string
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def render_template_per_cluster(template_content: str, cluster_data: dict) -> str:
    """
    Render template for a single cluster by replacing {{CLUSTER_JSON}} placeholder.
    
    Args:
        template_content: Template string with {{CLUSTER_JSON}} placeholder
        cluster_data: Single cluster dict with papers array
    
    Returns:
        Rendered prompt string
    """
    rendered = template_content.replace(
        "{{CLUSTER_JSON}}",
        json.dumps(cluster_data, indent=2, ensure_ascii=False)
    )
    
    return rendered


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


def _convert_cluster_card_to_dict(cluster_card: ClusterCard) -> Dict[str, Any]:
    """
    Convert ClusterCard to dict format expected by template.
    
    Args:
        cluster_card: ClusterCard instance with PaperCard papers
    
    Returns:
        Dictionary with papers array in format expected by template
    """
    papers_list = []
    for paper_card in cluster_card.papers:
        papers_list.append({
            "paper_id": paper_card.paper_id,
            "title": paper_card.title,
            "summary": paper_card.summary,
            "keywords": paper_card.keywords,
            "url": paper_card.url,
            "rank_in_cluster": paper_card.rank_in_cluster,
            "sim_to_centroid": paper_card.sim_to_centroid,
        })
    
    result = {
        "papers": papers_list,
        "size": cluster_card.size,
    }
    if cluster_card.cohesion is not None:
        result["cohesion"] = cluster_card.cohesion
    
    return result


def summarize_clusters_parallel(
    cfg: ReaderConfig,
    best_run_response: GetBestRunResponse
) -> Dict[str, Tuple[Optional[ClusterReport], JudgeOutput]]:
    """
    Process all clusters in parallel and generate ClusterReport summaries.
    
    Args:
        cfg: ReaderConfig instance
        best_run_response: GetBestRunResponse from memo adapter
    
    Returns:
        List of tuples (Optional[ClusterReport], JudgeOutput):
        - Each tuple contains (cluster_report_or_none, judge_output_or_failed_judge)
        - One tuple per cluster in the same order as input clusters
    """
     # Validate config exists
    if not cfg.cluster_summarization or not cfg.cluster_summarization.enable:
        logger.debug("Cluster summarization is disabled, returning empty list")
        return []
    
    # Load prompt template using resolved path from config
    template_path = cfg.cluster_summarization.prompt_template_path
    template_content = load_template(template_path)
    logger.info(f"Loaded prompt template from {template_path}")
    
    # Initialize LLM client
    llm_client = _initialize_llm_client(cfg, cfg.cluster_summarization.llm_model)
    
    # Process clusters in parallel

    def process_single_cluster(cluster_card: ClusterCard, cluster_idx: int) -> Tuple[Optional[ClusterReport], JudgeOutput]:
        """
        Process a single cluster and return ClusterReport and JudgeOutput.
        
        Args:
            cluster_card: ClusterCard instance
            cluster_idx: Cluster index
        
        Returns:
            Tuple of (Optional[ClusterReport], JudgeOutput):
            - ClusterReport: Parsed cluster report if successful, None otherwise
            - JudgeOutput: Contains validation scores and reasons (or failed_judge on error)
        """

        # Convert ClusterCard to dict format
        cluster_dict = _convert_cluster_card_to_dict(cluster_card)
        
        # Render template with cluster data
        prompt = render_template_per_cluster(template_content, cluster_dict)
        
        # Call LLM with structured output (returns parsed ClusterReport)
        cluster_report = llm_client.call_structured_raw(
            prompt=prompt,
            response_model=ClusterReport,
            temperature=cfg.llm_gemini.temperature,
            max_tokens=cfg.llm_gemini.max_tokens
        )
        
        # Use judge_output to validate response
        # cluster_dict is used for citation validation
        judge_result = judge_output(cluster_report, cluster_dict)
        
        logger.info(f"Successfully processed cluster {cluster_idx} (overall score: {judge_result.overall})")
        return cluster_report, judge_result
            
    
    # Process clusters in parallel using ThreadPoolExecutor
    clusters = best_run_response.clusters
    passed_clusters = 0
    logger.info(f"Processing {len(clusters)} clusters in parallel")
    
    with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
        # Submit all tasks and create mapping from future to pk_hash
        future_to_pk_hash = {}
        for i, cluster_card in enumerate(clusters):
            cluster_index = cluster_card.cluster_index
            pk_hash = cluster_card.pk_hash
            future = executor.submit(process_single_cluster, cluster_card, cluster_index)
            future_to_pk_hash[future] = pk_hash
        
        # Collect results as they complete (more efficient than waiting in order)
        # Store results in dict to maintain order
        results_dict = {}
        for future in as_completed(future_to_pk_hash):
            cluster_report, judge_result = None, None
            pk_hash = future_to_pk_hash[future]
            try:
                cluster_report, judge_result = future.result()
                results_dict[pk_hash] = (cluster_report, judge_result)
                if cluster_report is None:
                    logger.warning(f"Cluster with pk_hash {pk_hash} returned None (processing failed, overall score: {judge_result.overall})")
                else:
                    passed_clusters += 1
            except LLMGenerationError as e:
                logger.error(f"LLM generation error collecting cluster with pk_hash {pk_hash} result: {str(e)}", exc_info=True)
                # Add a failed judge output with None for cluster_report
                failed_judge = JudgeOutput(
                    sub_scores=None,
                    overall=0.0,
                    reasons={"llm_generation_error": str(e)}
                )
                results_dict[pk_hash] = (None, failed_judge)
            except ValidationError as e:
                logger.error(f"Validation error collecting cluster with pk_hash {pk_hash} result (after retries exhausted): {str(e)}", exc_info=True)
                # Add a failed judge output with None for cluster_report
                failed_judge = JudgeOutput(
                    sub_scores=None,
                    overall=0.0,
                    reasons={"validation_error": str(e)}
                )
                results_dict[pk_hash] = (None, failed_judge)
            except Exception as e:
                logger.error(f"Error collecting cluster with pk_hash {pk_hash} result: {str(e)}", exc_info=True)
                # Add a failed judge output with None for cluster_report
                failed_judge = JudgeOutput(
                    sub_scores=None,
                    overall=0.0,
                    reasons={"internal_error": f"Exception: {str(e)}"}
                )
                results_dict[pk_hash] = (None, failed_judge)
        
        logger.info(f"Successfully processed {passed_clusters}/{len(clusters)} clusters")
    return results_dict


def serialize_cluster_reports(
    cluster_reports: Dict[str, Tuple[Optional[ClusterReport], JudgeOutput]],
    output_path: str
) -> None:
    """
    Serialize cluster reports dictionary to JSON file.
    
    Handles Pydantic objects (ClusterReport) and dataclass objects (JudgeOutput)
    by converting them to dictionaries before JSON serialization.
    
    Args:
        cluster_reports: Dictionary mapping pk_hash to tuple of 
                        (Optional[ClusterReport], JudgeOutput)
        output_path: Path to output JSON file
    """
    # Convert the dict to a JSON-serializable format
    serializable_dict = {}
    for pk_hash, (cluster_report, judge_output) in cluster_reports.items():
        # Convert ClusterReport (Pydantic) to dict if present
        cluster_report_dict = None
        if cluster_report is not None:
            cluster_report_dict = cluster_report.model_dump()
        
        # Convert JudgeOutput (dataclass) to dict
        judge_output_dict = asdict(judge_output)
        
        serializable_dict[pk_hash] = {
            "cluster_report": cluster_report_dict,
            "judge_output": judge_output_dict
        }
    
    # Write to JSON file
    logger.info(f"Serializing cluster reports to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully serialized cluster reports to {output_path}")


def convert_cluster_reports_to_memo_payload(
    cluster_reports: Dict[str, Tuple[Optional[ClusterReport], JudgeOutput]],
    cfg: ReaderConfig
) -> InjectClustersObservationInput:
    """
    Convert cluster reports from summarize_clusters_parallel to memo inject_clusters_observation payload format.
    
    Args:
        cluster_reports: Dictionary mapping pk_hash to tuple of (Optional[ClusterReport], JudgeOutput)
        cfg: ReaderConfig instance
    
    Returns:
        InjectClustersObservationInput dict mapping pk_hash to ClusterObservation
    """
    # Load template content to compute prompt_hash
    template_path = cfg.cluster_summarization.prompt_template_path
    template_content = load_template(template_path)
    
    # Compute prompt_hash: SHA256 hash of template_content
    prompt_hash_hex = hashlib.sha256(template_content.encode('utf-8')).hexdigest()
    prompt_hash = f"sha256:{prompt_hash_hex}"
    
    # Build LLM config
    llm_config_id = f"{cfg.cluster_summarization.llm_model}|{prompt_hash}"
    llm_config_json_payload = {
        "provider": "google",
        "model": cfg.cluster_summarization.llm_model,
        "temperature": cfg.llm_gemini.temperature,
        "max_tokens": cfg.llm_gemini.max_tokens,
    }
    llm_config = LLMConfigInput(
        llm_config_id=llm_config_id,
        json_payload=llm_config_json_payload
    )
    
    # Build payload
    payload: InjectClustersObservationInput = {}
    
    for pk_hash, (cluster_report, _) in cluster_reports.items():
        # Skip if cluster_report is None (failed clusters)
        if cluster_report is None:
            logger.debug(f"Skipping failed cluster with pk_hash {pk_hash}")
            continue
        
        # Use ClusterReport.model_dump() as payload_json
        payload_json = cluster_report.model_dump()
        
        # Extract fields from ClusterReport
        title = cluster_report.title
        summary = cluster_report.what_this_topic_is_about + "\n" + cluster_report.why_it_matters
        keywords_json = cluster_report.keyword_list
        
        # Create ClusterObservation
        observation = ClusterObservation(
            llm_config=llm_config,
            payload_json=payload_json,
            summary=summary,
            title=title,
            keywords_json=keywords_json
        )
        
        payload[pk_hash] = observation
    
    logger.info(f"Converted {len(payload)} cluster reports to memo payload format")
    return payload


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
    metadata = memo.get_topic_resolver_metadata(cluster_pk_hash, cfg)
    if metadata is None:
        logger.warning(f"Failed to get topic resolver metadata for cluster {cluster_pk_hash} (memo may be disabled)")
        return
    
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
    if not cfg.memo.enabled:
        return None
    
    start_report_job_response = memo.start_report_job(cluster_pk_hash, cfg)
    
    if start_report_job_response is None:
        return None
    
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
        config=config,
        topic_id=topic_id,
        add_top_papers=add_top_papers,
    )
    
    # Check if memo returned None (memo disabled)
    if cluster_metadata is None:
        raise ValueError("memo.get_report_planner_metadata returned None (memo is disabled). cluster_metadata is required.")
    
    # Call build_planner_prompt with intent_spec and cluster_metadata
    prompt = build_planner_prompt(intent_spec=intent_spec, cluster_metadata=cluster_metadata)
    
    return prompt


# -------------------------

# -------------------------
