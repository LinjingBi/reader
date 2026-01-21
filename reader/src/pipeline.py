import json
from fetch import Paper, serialize_to_paper_objects
import os
from typing import List, Dict, Sequence, Optional
import numpy as np
import calendar

from algo_lib.clustering import get_best_clustering, write_best_clustering_report
from algo_lib.clustering.ordering import order_cluster_members_by_centroid_similarity
from algo_lib.typing import PaperLike


## configs
# fetch
"""
target_month = "month=2025-01"
fetch_url = 'https://huggingface.co/api/daily_papers'
hf_paper_url = 'https://huggingface.co/papers/'
output_json = 'papers_report.json' # for save_papers_to_file only
"""
# embed+clustering
"""
#embedding parameters
embed_model_name = "BAAI/bge-small-en-v1.5"
MODES = ["B", "C"]
top_n_keywords = 10

# kmeans clustering parameters
ks = [4, 5]
seed = 42

# output
cluster_report_dir = f'best_clustering_reports_{target_month}.md'
"""



"""
# embedding text modes

A: title + summary

B: title + summary + top N keywords

C: title + summary + all keywords
"""
MODES = ["B", "C"]
top_n_keywords = 10

# Target month to process (e.g., "month=2025-01")
target_month = "month=2025-01"
cluster_report_dir = f'best_clustering_reports_{target_month}.md'
# usually monthly report
papers_report_file = 'papers_report.json'

embed_model_name = "BAAI/bge-small-en-v1.5"

# kmeans clustering parameters
ks = [4, 5]
seed = 42


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
        raw_json: Optional raw JSON string
        output_path: Path for JSON output file (default: 'fresh_paper_payload.json')
    
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
    
    return payload


def run_pipeline():
    if os.path.exists(cluster_report_dir):
        os.remove(cluster_report_dir)
    
    
    
    # Step 0: Load papers_report_file
    with open(papers_report_file, "r") as f:
        data = json.load(f)
    papers_data = data['papers']
    
    # Step 1: Process single month
    if target_month not in papers_data:
        raise ValueError(f"Month {target_month} not found in papers_data. Available months: {list(papers_data.keys())}")
    
    papers_list = papers_data[target_month]
    print(f"Processing {target_month}")
    
    # Extract period dates from month key
    period_start, period_end = extract_period_dates(target_month)
    print(f"Period: {period_start} to {period_end}")
    
    # Create Paper objects from JSON data
    papers = serialize_to_paper_objects(papers_list)
    
    # Step 2: Get best clustering with enhanced metadata
    df_best, best_labels, best_embed, cluster_cohesion_dict, member_similarities = get_best_clustering(
        papers=papers,
        embed_model_name=embed_model_name,
        ks=ks,
        top_n_keywords=top_n_keywords,
        modes=MODES,
        seed=seed
    )
    
    # Generate best clustering report
    print(f"\n{target_month} Top choice:", df_best)
    print(f"Appending {target_month} best clustering report to {cluster_report_dir}")
    
    # Step 3: Write text report
    write_best_clustering_report(
        papers=papers,
        labels=best_labels,
        embeddings=best_embed,
        header=f"# {target_month} BEST CLUSTERING (mode={df_best['mode']}, k={df_best['k']})",
        max_summary_chars=350,
        report_dir=cluster_report_dir,
    )
    
    # Step 3: Generate JSON payload
    generate_fresh_paper_payload(
        papers=papers,
        labels=best_labels,
        embeddings=best_embed,
        member_similarities=member_similarities,
        cluster_cohesion_dict=cluster_cohesion_dict,
        period_start=period_start,
        period_end=period_end,
        embed_model_name=embed_model_name,
        best_mode=df_best['mode'],
        best_k=df_best['k'],
        top_n_keywords=top_n_keywords,
        seed=seed,
        raw_json="",  # Optional: can be set to actual raw JSON if available
    )

if __name__ == "__main__":
    run_pipeline()