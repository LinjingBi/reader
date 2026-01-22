"""Monthly pipeline orchestration"""

import json
import os
import calendar
from pathlib import Path

from algo_lib.clustering import get_best_clustering

from reader.config import ReaderConfig, render_best_cluster_text_report_path, render_best_cluster_report_path
from reader.adapters.hf import get_monthly_report, parse_papers, save_papers_to_file
from reader.adapters.memo import fresh_paper
from reader.pipelines.report import write_best_clustering_text_report, generate_fresh_paper_payload


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


def _get_hf_paper_metadata(cfg: ReaderConfig) -> tuple[list, str, str]:
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
        print(f"{papers_report_file} not found, generating from HF API...")
        results = asyncio.run(get_monthly_report(cfg))
        save_papers_to_file(results, cfg)
        print(f"Generated {papers_report_file}")
    
    # Load papers_report_file
    with open(papers_report_file, "r") as f:
        data = json.load(f)
    papers_data = data['papers']
    
    # Process single month
    if month_key not in papers_data:
        raise ValueError(f"Month {month_key} not found in papers_data. Available months: {list(papers_data.keys())}")
    
    papers_list = papers_data[month_key]
    print(f"Processing {month_key}")
    
    # Extract period dates from month key
    period_start, period_end = _extract_period_dates(month_key)
    print(f"Period: {period_start} to {period_end}")
    
    # Create Paper objects from JSON data
    papers = parse_papers(papers_list, cfg)
    
    return papers, period_start, period_end


def _generate_clustering_reports(
    cfg: ReaderConfig,
    papers: list,
    period_start: str,
    period_end: str,
) -> dict:
    """
    Generate clustering reports and payload.
    
    Args:
        cfg: ReaderConfig instance
        papers: List of Paper objects
        period_start: Start date in YYYY-MM-DD format
        period_end: End date in YYYY-MM-DD format
    
    Returns:
        Fresh paper payload dictionary
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
    print(f"\n{month_key} Top choice: mode: {result.mode} k: {result.k} embed_model: {cfg.algos.embedding.model}")
    # Write text report if configured
    cluster_text_report_path = render_best_cluster_text_report_path(cfg, month_key)
    if cluster_text_report_path:
        print(f"Appending {month_key} best clustering text report to {cluster_text_report_path}")
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
    
    return fresh_paper_payload


def run_monthly(cfg: ReaderConfig) -> None:
    """
    Run the monthly pipeline for the configured month.
    
    Args:
        cfg: ReaderConfig instance
    """
    # Get paper metadata from HF API or cached file
    papers, period_start, period_end = _get_hf_paper_metadata(cfg)
    
    # Generate clustering reports and payload
    fresh_paper_payload = _generate_clustering_reports(cfg, papers, period_start, period_end)

    # Optionally call memo adapter if enabled
    if cfg.memo.enabled:
        memo_result = fresh_paper(fresh_paper_payload, cfg)
        if memo_result:
            print(f"Memo ingest successful: snapshot_id={memo_result.get('snapshot_id')}, cluster_run_id={memo_result.get('cluster_run_id')}")
