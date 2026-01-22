"""Report generation functions for clustering results"""

import json
from typing import Dict, List, Sequence, Optional

from algo_lib.typing import PaperLike
from reader.config import ReaderConfig


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
) -> Dict:
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
        Dictionary matching fresh_paper_payload.json format
    """
        
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
    
    for c, papers_sim in member_similarities.items():
        idxs = sorted(member_similarities[c], key=member_similarities[c].get, reverse=True)
        members_list = []
        
        for rank, paper_idx in enumerate(idxs):
            sim = papers_sim[paper_idx]
            members_list.append({
                "paper_id": format_paper_id(papers[paper_idx].pid),
                "rank_in_cluster": rank,
                "sim_to_centroid": sim,
            })
        
        clusters_array.append({
            "cluster_index": c,
            "size": len(idxs),
            "cohesion": cluster_cohesion_dict[c],
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
    
    # Write JSON file only if output_path is provided
    if output_path is not None:
        print(f"Generating {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    
    return payload


def format_paper_id(paper_id: str) -> str:
    """
    Format paper ID with "hf:" prefix.
    
    Args:
        paper_id: Raw paper ID (e.g., "2501.12948")
    
    Returns:
        Formatted paper ID (e.g., "hf:2501.12948")
    """
    return f"hf:{paper_id}"


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

