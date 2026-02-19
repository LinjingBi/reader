"""HF data pipeline building blocks"""

import os
import base64
import json
import calendar
from pathlib import Path
from typing import Dict, List, Sequence, Optional
import numpy as np

from algo_lib.clustering import get_best_clustering
from algo_lib.typing import PaperLike
from algo_lib.paperchunk.types import PaperId, Url

from reader.pipelines.hf_data.config.config import (
    HFDataPipeConfig,
    render_best_cluster_text_report_path,
    render_best_cluster_report_path,
)
from reader.pipelines.hf_data.report import (
    FreshPaperPayload,
    PaperInput,
    ClusterMemberInput,
    ClusterInput,
    EmbedConfig,
    EmbedConfigPayload,
    ClusterConfig,
    ClusterConfigPayload,
)
from reader.adapters.hf import get_monthly_report, parse_papers, save_papers_to_file
from reader.adapters.memo import FreshPaperResponseWithDetails
from reader.logging.logging_setup import get_logger

logger = get_logger()


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


def get_hf_paper_metadata(cfg: HFDataPipeConfig) -> tuple[list, str, str]:
    """
    Get paper metadata from HF API or cached file.
    
    Args:
        cfg: HFDataPipeConfig instance
    
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
    config: HFDataPipeConfig,
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
        config: HFDataPipeConfig instance
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
    cfg: HFDataPipeConfig,
    papers: list,
    period_start: str,
    period_end: str,
) -> FreshPaperPayload:
    """
    Generate clustering reports and payload.
    
    Args:
        cfg: HFDataPipeConfig instance
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


def convert_papers_for_chunking(fresh_paper_response: FreshPaperResponseWithDetails) -> Dict[PaperId, Url]:
    """
    Convert papers from fresh_paper_response to the format expected by run_scoring.
    
    Args:
        fresh_paper_response: FreshPaperResponseWithDetails instance containing paper details
    
    Returns:
        Dictionary mapping PaperId to Url
    """
    papers_dict: Dict[PaperId, Url] = {}
    if fresh_paper_response.details:
        for pk_hash, paper_list in fresh_paper_response.details.items():
            for paper_output in paper_list:
                papers_dict[paper_output.paper_id] = paper_output.paper_url
    return papers_dict

