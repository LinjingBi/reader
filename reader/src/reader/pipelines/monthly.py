"""Monthly pipeline orchestration"""

import json
import os
import calendar
from pathlib import Path
from typing import Dict, Sequence, Optional
import numpy as np

from algo_lib.clustering import get_best_clustering, write_best_clustering_report
from algo_lib.clustering.ordering import order_cluster_members_by_centroid_similarity
from algo_lib.typing import PaperLike

from reader.config import ReaderConfig, render_best_cluster_report_path
from reader.adapters.hf import get_monthly_report, parse_papers, save_papers_to_file
from reader.adapters.memo import fresh_paper


def extract_period_dates(month_key: str) -> tuple[str, str]:
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


def get_embed_config_id() -> str:
    """
    Get embed_config_id from algo_lib.embedding version.
    
    Returns:
        embed_config_id string in format "algo_lib.embedding|{version}"
    """
    try:
        from algo_lib.embedding import __version__ as embed_version
        return f"algo_lib.embedding|{embed_version}"
    except ImportError:
        raise ValueError("algo_lib.embedding is not versioned")


def get_cluster_config_id() -> str:
    """
    Get cluster_config_id from algo_lib.clustering version.
    
    Returns:
        cluster_config_id string in format "algo_lib.clustering|{version}"
    """
    try:
        from algo_lib.clustering import __version__ as cluster_version
        return f"algo_lib.clustering|{cluster_version}"
    except ImportError:
        raise ValueError("algo_lib.clustering is not versioned")


def format_paper_id(paper_id: str) -> str:
    """
    Format paper ID with "hf:" prefix.
    
    Args:
        paper_id: Raw paper ID (e.g., "2501.12948")
    
    Returns:
        Formatted paper ID (e.g., "hf:2501.12948")
    """
    return f"hf:{paper_id}"


def generate_fresh_paper_payload(
    papers: Sequence[PaperLike],
    labels: np.ndarray,
    embeddings: np.ndarray,
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
) -> Dict:
    """
    Generate fresh_paper_payload.json format report.
    
    Args:
        papers: Sequence of paper-like objects (must have published_at field)
        labels: Cluster labels array
        embeddings: Embeddings array
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
        output_path: Path for JSON output file (default: auto-generated)
    
    Returns:
        Dictionary matching fresh_paper_payload.json format
    """
    if output_path is None:
        output_path = f'fresh_paper_payload-{period_start}-{period_end}.json'
    
    print(f"Generating {output_path}")
    
    # Get ordered cluster members
    clusters = order_cluster_members_by_centroid_similarity(embeddings, labels)
    
    # Build papers array
    papers_array = []
    for p in papers:
        papers_array.append({
            "paper_id": format_paper_id(p.pid),
            "title": p.title,
            "summary": p.summary,
            "keywords": p.keywords,
            "url": p.url,
            "published_at": p.published_at,
        })
    
    # Build clusters array
    clusters_array = []
    cluster_order = sorted(clusters.keys())
    
    for c in cluster_order:
        idxs = clusters[c]
        members_list = []
        
        for rank, paper_idx in enumerate(idxs):
            sim = member_similarities.get(c, {}).get(paper_idx, 0.0)
            members_list.append({
                "paper_id": format_paper_id(papers[paper_idx].pid),
                "rank_in_cluster": rank,
                "sim_to_centroid": sim,
            })
        
        clusters_array.append({
            "cluster_index": c,
            "size": len(idxs),
            "cohesion": cluster_cohesion_dict.get(c, 0.0),
            "members": members_list,
        })
    
    # Build embed_config
    embed_config = {
        "embed_config_id": get_embed_config_id(),
        "json_payload": {
            "model_name": embed_model_name,
            "mode": best_mode,
            "top_n_keywords": top_n_keywords,
        }
    }
    
    # Build cluster_config
    cluster_config = {
        "cluster_config_id": get_cluster_config_id(),
        "json_payload": {
            "k": best_k,
            "seed": seed,
            "algorithm": "kmeans",
        }
    }
    
    # Build complete payload
    payload = {
        "source": "hf_monthly",
        "period_start": period_start,
        "period_end": period_end,
        "raw_json": raw_json,
        "embed_config": embed_config,
        "cluster_config": cluster_config,
        "papers": papers_array,
        "clusters": clusters_array,
    }
    
    # Write JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    # Optionally call memo adapter if enabled
    if config.memo.enabled:
        memo_result = fresh_paper(payload, config)
        if memo_result:
            print(f"Memo ingest successful: snapshot_id={memo_result.get('snapshot_id')}, cluster_run_id={memo_result.get('cluster_run_id')}")
    
    return payload


def run_monthly(cfg: ReaderConfig) -> None:
    """
    Run the monthly pipeline for the configured month.
    
    Args:
        cfg: ReaderConfig instance
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
    period_start, period_end = extract_period_dates(month_key)
    print(f"Period: {period_start} to {period_end}")
    
    # Create Paper objects from JSON data
    papers = parse_papers(papers_list, cfg)
    
    # Get best clustering with enhanced metadata
    df_best, best_labels, best_embed, cluster_cohesion_dict, member_similarities = get_best_clustering(
        papers=papers,
        embed_model_name=cfg.algos.embedding.model,
        ks=cfg.algos.clustering.k_candidates,
        top_n_keywords=cfg.algos.embedding.top_n_keywords,
        modes=cfg.algos.embedding.modes,
        seed=cfg.algos.clustering.random_seed
    )
    
    # Generate best clustering report
    print(f"\n{month_key} Top choice:", df_best)
    
    # Render report path from template
    cluster_report_dir = render_best_cluster_report_path(cfg, month_key)
    print(f"Appending {month_key} best clustering report to {cluster_report_dir}")
    
    # Remove existing report if it exists
    if os.path.exists(cluster_report_dir):
        os.remove(cluster_report_dir)
    
    # Write text report
    write_best_clustering_report(
        papers=papers,
        labels=best_labels,
        embeddings=best_embed,
        header=f"# {month_key} BEST CLUSTERING (mode={df_best['mode']}, k={df_best['k']})",
        max_summary_chars=350,
        report_dir=cluster_report_dir,
    )
    
    # Generate JSON payload
    generate_fresh_paper_payload(
        papers=papers,
        labels=best_labels,
        embeddings=best_embed,
        member_similarities=member_similarities,
        cluster_cohesion_dict=cluster_cohesion_dict,
        period_start=period_start,
        period_end=period_end,
        embed_model_name=cfg.algos.embedding.model,
        best_mode=df_best['mode'],
        best_k=df_best['k'],
        top_n_keywords=cfg.algos.embedding.top_n_keywords,
        seed=cfg.algos.clustering.random_seed,
        config=cfg,
        raw_json="",  # Optional: can be set to actual raw JSON if available
    )
