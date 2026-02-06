#!/usr/bin/env python3
"""
12-Month Embedding and Clustering Script

Processes all 12 months from papers_report.json, performs embedding (mode="B")
and clustering (grid search k=[4,5]), outputs clusters in JSON format expected
by test_t_threshold.py.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Add src directory to Python path to enable algo_lib imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Try to import from algo_lib
try:
    from algo_lib.embedding.embedder import Embedder
    from algo_lib.clustering import get_best_clustering
    from algo_lib.embedding import __version__ as embed_version
except ImportError as e:
    raise ImportError(
        f"Failed to import from algo_lib: {e}. "
        f"Make sure {src_dir} exists and contains algo_lib."
    )


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Paper:
    """Paper model matching PaperLike protocol."""
    pid: str
    title: str
    summary: str
    keywords: List[str]
    url: str = ""
    published_at: str = ""


# ============================================================================
# Helper Functions
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


def parse_papers(papers_list: List[Dict], paper_page_base_url: str = "https://huggingface.co/papers/") -> List[Paper]:
    """
    Parse paper dictionaries from JSON into Paper objects.
    
    Args:
        papers_list: List of paper dictionaries from JSON
        paper_page_base_url: Base URL for paper pages
    
    Returns:
        List of Paper objects
    """
    result = []
    for paper_data in papers_list:
        paper = paper_data['paper']
        
        # Extract published_at and convert to YYYY-MM-DD format
        published_at = ""
        pub_date_str = paper.get('publishedAt', "")
        if pub_date_str:
            try:
                # Parse ISO format: "2025-01-22T15:19:35.000Z"
                dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                published_at = dt.strftime('%Y-%m-%d')
            except (ValueError, AttributeError):
                pass
        
        result.append(Paper(
            pid=paper['id'],
            title=paper['title'],
            summary=paper['summary'],
            keywords=paper.get('ai_keywords', []),
            url=f"{paper_page_base_url}{paper['id']}",
            published_at=published_at
        ))
    return result


# ============================================================================
# Processing Functions
# ============================================================================

def process_month(
    month_key: str,
    papers: List[Paper],
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    k_candidates: Sequence[int] = [4, 5],
    top_n_keywords: int = 10,
    seed: int = 42,
    print_results: bool = False,
) -> tuple[str, str, Dict[int, np.ndarray], Dict[int, int], int]:
    """
    Process a single month: embed papers and perform clustering.
    
    Args:
        month_key: Month key (e.g., "month=2025-01")
        papers: List of Paper objects
        embed_model_name: Embedding model name
        k_candidates: Sequence of k values to try
        top_n_keywords: Number of top keywords for mode "B"
        seed: Random seed for KMeans
        print_results: Whether to print grid search results
    
    Returns:
        Tuple of (period_start, period_end, cluster_centroids, cluster_sizes, embedding_dim)
    """
    # Extract period dates
    period_start, period_end = _extract_period_dates(month_key)
    
    # Get best clustering (mode="B" fixed, grid search over k)
    result = get_best_clustering(
        papers=papers,
        embed_model_name=embed_model_name,
        modes=["B"],  # Fixed mode="B"
        k_candidates=k_candidates,
        top_n_keywords=top_n_keywords,
        seed=seed,
        print_results=print_results,
    )
    
    # Get embedding dimension from centroids
    # All centroids should have the same dimension
    if result.cluster_centroids:
        first_centroid = next(iter(result.cluster_centroids.values()))
        embedding_dim = len(first_centroid)
    else:
        # Fallback: encode one paper to get dimension
        # This should rarely happen, but handle edge case
        embedder = Embedder(model_name=embed_model_name)
        if papers:
            test_embedding = embedder.encode_papers(
                papers[:1],
                mode="B",
                top_n=top_n_keywords,
            )
            embedding_dim = test_embedding.shape[1]
        else:
            # Last resort: try to get from model
            try:
                embedding_dim = embedder.model.get_sentence_embedding_dimension()
            except AttributeError:
                # Default fallback (should not happen)
                embedding_dim = 384
    
    # Extract cluster sizes
    cluster_sizes = {
        cluster_idx: len(members)
        for cluster_idx, members in result.cluster_members_ordered.items()
    }
    
    return period_start, period_end, result.cluster_centroids, cluster_sizes, embedding_dim


def format_output(
    all_clusters: List[Dict],
    embed_config_id: str,
    dim: int,
) -> Dict:
    """
    Format clusters into JSON output format expected by test_t_threshold.py.
    
    Args:
        all_clusters: List of cluster dictionaries with period_start, period_end,
                     cluster_key, centroid, size
        embed_config_id: Embedding config ID
        dim: Embedding dimension
    
    Returns:
        Dictionary with embed_config_id, dim, and clusters
    """
    return {
        "embed_config_id": embed_config_id,
        "dim": dim,
        "clusters": all_clusters,
    }


# ============================================================================
# Main Function
# ============================================================================

def main() -> None:
    """Main function to process all 12 months and generate output."""
    parser = argparse.ArgumentParser(
        description="Process 12 months of papers for embedding and clustering"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../src/papers_report.json",
        help="Path to papers_report.json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="clusters_12m.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name",
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        nargs="+",
        default=[4, 5],
        help="K values to try for clustering",
    )
    parser.add_argument(
        "--top-n-keywords",
        type=int,
        default=10,
        help="Number of top keywords for mode B",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans",
    )
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print grid search results for each month",
    )
    args = parser.parse_args()
    
    # Load papers_report.json
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading papers from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    papers_data = data.get("papers", {})
    if not papers_data:
        raise ValueError("No 'papers' key found in input JSON")
    
    # Get embed_config_id
    try:
        embed_config_id = f"algo_lib.embedding|{embed_version}"
    except NameError:
        embed_config_id = "algo_lib.embedding|unknown"
    
    # Process all months
    all_clusters = []
    embedding_dim = None
    
    # Sort month keys to process in order
    month_keys = sorted([k for k in papers_data.keys() if k.startswith("month=")])
    
    print(f"Processing {len(month_keys)} months...")
    
    for i, month_key in enumerate(month_keys, 1):
        print(f"\n[{i}/{len(month_keys)}] Processing {month_key}...")
        
        papers_list = papers_data[month_key]
        if not papers_list:
            print(f"  No papers found for {month_key}, skipping...")
            continue
        
        # Parse papers
        papers = parse_papers(papers_list)
        print(f"  Parsed {len(papers)} papers")
        
        # Process month
        period_start, period_end, cluster_centroids, cluster_sizes, dim = process_month(
            month_key=month_key,
            papers=papers,
            embed_model_name=args.embed_model,
            k_candidates=args.k_candidates,
            top_n_keywords=args.top_n_keywords,
            seed=args.seed,
            print_results=args.print_results,
        )
        
        # Store embedding dimension (should be same for all months)
        if embedding_dim is None:
            embedding_dim = dim
        elif embedding_dim != dim:
            print(f"  WARNING: Embedding dimension mismatch: {embedding_dim} vs {dim}")
        
        # Create cluster entries
        for cluster_idx, centroid in cluster_centroids.items():
            cluster_key = f"{month_key}_cluster_{cluster_idx}"
            size = cluster_sizes.get(cluster_idx, 0)
            
            all_clusters.append({
                "period_start": period_start,
                "period_end": period_end,
                "cluster_key": cluster_key,
                "centroid": centroid.tolist(),  # Convert numpy array to list
                "size": size,
            })
        
        print(f"  Created {len(cluster_centroids)} clusters")
    
    # Format and write output
    output_data = format_output(
        all_clusters=all_clusters,
        embed_config_id=embed_config_id,
        dim=embedding_dim if embedding_dim is not None else 384,  # Fallback
    )
    
    output_path = Path(args.output)
    print(f"\nWriting output to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! Generated {len(all_clusters)} clusters across {len(month_keys)} months.")
    print(f"Embedding dimension: {output_data['dim']}")
    print(f"Embed config ID: {output_data['embed_config_id']}")


if __name__ == "__main__":
    main()

