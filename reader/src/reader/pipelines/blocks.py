"""Pipeline building blocks"""

import json
import os
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any, Tuple

from algo_lib.clustering import get_best_clustering
from algo_lib.typing import PaperLike

from reader.config import ReaderConfig, render_best_cluster_text_report_path, render_best_cluster_report_path
from reader.adapters.hf import get_monthly_report, parse_papers, save_papers_to_file
from reader.adapters.memo import GetBestRunResponse, ClusterCard, PaperCard
from reader.adapters.llm import LLMClient, TokenBucket
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
)
from reader.pipelines.metrics import judge_output, JudgeOutput
from reader.logging.logging_setup import get_logger

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
        
        clusters_list.append(ClusterInput(
            cluster_index=c,
            size=len(idxs),
            cohesion=cluster_cohesion_dict[c],
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
# LLM summarization blocks
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
    
    return {"papers": papers_list}


def summarize_clusters_parallel(
    cfg: ReaderConfig,
    best_run_response: GetBestRunResponse
) -> List[Tuple[Optional[ClusterReport], JudgeOutput]]:
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
        model=cfg.cluster_summarization.llm_model,
        api_key=api_key,
        rpm_bucket=rpm_bucket,
        tpm_bucket=tpm_bucket
    )
    
    logger.info(f"Initialized LLM client with model: {cfg.cluster_summarization.llm_model}")
    
    # Process clusters in parallel
    results = []
    
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
        try:
            # Convert ClusterCard to dict format
            cluster_dict = _convert_cluster_card_to_dict(cluster_card)
            
            # Render template with cluster data
            prompt = render_template_per_cluster(template_content, cluster_dict)
            
            # Call LLM with structured output
            raw_response = llm_client.call_structured_raw(
                prompt=prompt,
                response_model=ClusterReport,
                temperature=cfg.llm_gemini.temperature,
                max_tokens=cfg.llm_gemini.max_tokens
            )
            
            # Use judge_output to validate and parse response
            # cluster_dict is used for citation validation
            judge_result, cluster_report = judge_output(raw_response, cluster_dict)
            
            logger.info(f"Successfully processed cluster {cluster_idx} (overall score: {judge_result.overall})")
            return cluster_report, judge_result
            
        except Exception as e:
            logger.error(f"Error processing cluster {cluster_idx}: {str(e)}", exc_info=True)
            # Return None for cluster_report and a failed JudgeOutput
            failed_judge = JudgeOutput(
                sub_scores=None,
                overall=0.0,
                reasons={"error": f"Exception: {str(e)}"}
            )
            return None, failed_judge
    
    # Process clusters in parallel using ThreadPoolExecutor
    clusters = best_run_response.clusters
    logger.info(f"Processing {len(clusters)} clusters in parallel")
    
    with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
        # Submit all tasks and create mapping from future to cluster_index
        future_to_index = {}
        for i, cluster_card in enumerate(clusters):
            cluster_index = cluster_card.cluster_index
            future = executor.submit(process_single_cluster, cluster_card, cluster_index)
            future_to_index[future] = cluster_index
        
        # Collect results as they complete (more efficient than waiting in order)
        # Store results in dict to maintain order
        results_dict = {}
        for future in as_completed(future_to_index):
            cluster_index = future_to_index[future]
            try:
                cluster_report, judge_result = future.result()
                results_dict[cluster_index] = (cluster_report, judge_result)
                if cluster_report is None:
                    logger.warning(f"Cluster {cluster_index} returned None (processing failed, overall score: {judge_result.overall})")
            except Exception as e:
                logger.error(f"Error collecting cluster {cluster_index} result: {str(e)}", exc_info=True)
                # Add a failed judge output with None for cluster_report
                failed_judge = JudgeOutput(
                    sub_scores=None,
                    overall=0.0,
                    reasons={"collection_error": f"Exception: {str(e)}"}
                )
                results_dict[cluster_index] = (None, failed_judge)
        
        # Build final results list in the same order as input clusters
        for cluster_card in clusters:
            cluster_index = cluster_card.cluster_index
            if cluster_index in results_dict:
                results.append(results_dict[cluster_index])
            else:
                # This shouldn't happen, but handle it gracefully
                logger.error(f"Missing result for cluster {cluster_index}")
                failed_judge = JudgeOutput(
                    sub_scores=None,
                    overall=0.0,
                    reasons={"missing_result": "Result not found in results_dict"}
                )
                results.append((None, failed_judge))
    
    successful_count = sum(1 for cluster_report, _ in results if cluster_report is not None)
    logger.info(f"Successfully processed {successful_count} out of {len(clusters)} clusters")
    return results
