"""
Best clustering selection via grid search and scoring - refactored version.

This module separates grid search from final clustering computation.
"""

from __future__ import annotations
from typing import Sequence, Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from algo_lib.typing import PaperLike, BestClusteringResult
from algo_lib.embedding.embedder import Embedder
from algo_lib.clustering.kmeans import KmeanClusterer
from algo_lib.clustering.metrics import (
    safe_global_metrics,
    member_similarities,
    cluster_cohesion,
    keyword_entropy_per_cluster,
    cluster_members_ordered,
)


# Global score weights for clustering evaluation
SCORE_WEIGHTS: Dict[str, float] = {
    "silhouette_cosine": 1.5,
    "avg_cluster_cohesion": 2.0,
    "worst_cluster_cohesion": 1.0,
    "davies_bouldin": -0.7,
    "avg_kw_entropy": -0.3,
}


@dataclass
class BestClusteringArgs:
    """Best clustering configuration from grid search."""
    mode: str
    k: int
    seed: int


def _grid_search(
    papers: Sequence[PaperLike],
    embedder: Embedder,
    modes: Sequence[str],
    k_candidates: Sequence[int],
    top_n_keywords: int,
    seed: Optional[int] = None,
    print_results: bool = True,
) -> BestClusteringArgs:
    """
    Perform grid search over modes and k values, score each candidate,
    and return the best configuration args.
    
    Args:
        papers: Sequence of paper-like objects to cluster
        embedder: Embedder instance to use
        modes: Sequence of embedding modes to try (e.g., ["B", "C"])
        k_candidates: Sequence of k values to try (e.g., [4, 5])
        top_n_keywords: Number of top keywords for mode "B"
        seed: Random seed for KMeans
        print_results: Whether to print the results DataFrame
    
    Returns:
        BestClusteringArgs with the best configuration (mode, k, seed)
    """
    if seed is None:
        seed = 42
    
    rows = []
    
    for mode in modes:
        X = embedder.encode_texts(
            papers,
            mode=mode,
            top_n=top_n_keywords,
        )
        
        for k in k_candidates:
            kmeans = KmeanClusterer(
                papers,
                X,
                random_state=seed,
                n_clusters=k,
            )
            labels = kmeans.fit_predict()
            
            # Compute metrics
            gm = safe_global_metrics(X, labels)
            member_sims = member_similarities(X, labels)
            coh_mean = cluster_cohesion(member_sims)
            avg_coh = float(np.mean(list(coh_mean.values()))) if coh_mean else float("nan")
            min_coh = float(np.min(list(coh_mean.values()))) if coh_mean else float("nan")
            
            ent = keyword_entropy_per_cluster(papers, labels)
            avg_ent = float(np.nanmean(list(ent.values()))) if ent else float("nan")
            max_ent = float(np.nanmax(list(ent.values()))) if ent else float("nan")
            
            rows.append({
                "mode": mode,
                "k": k,
                **gm,
                "avg_cluster_cohesion": avg_coh,  # higher better
                "worst_cluster_cohesion": min_coh,  # higher better
                "avg_kw_entropy": avg_ent,  # lower better
                "max_kw_entropy": max_ent,  # lower better
            })
    
    df = pd.DataFrame(rows)
    
    # Compute composite score
    def safe(x):  # convert NaN to neutral-ish
        return np.nan_to_num(x, nan=0.0)
    
    df["score"] = (
        SCORE_WEIGHTS["silhouette_cosine"] * safe(df["silhouette_cosine"])
        + SCORE_WEIGHTS["avg_cluster_cohesion"] * safe(df["avg_cluster_cohesion"])
        + SCORE_WEIGHTS["worst_cluster_cohesion"] * safe(df["worst_cluster_cohesion"])
        + SCORE_WEIGHTS["davies_bouldin"] * safe(df["davies_bouldin"])
        + SCORE_WEIGHTS["avg_kw_entropy"] * safe(df["avg_kw_entropy"])
    )
    
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    if print_results:
        print(df.to_string(index=False))
    
    # Get best result args
    best_row = df.iloc[0]
    
    return BestClusteringArgs(
        mode=str(best_row["mode"]),
        k=int(best_row["k"]),
        seed=seed,
    )


def get_best_clustering(
    papers: Sequence[PaperLike],
    embed_model_name: str,
    modes: Sequence[str],
    k_candidates: Sequence[int],
    top_n_keywords: int,
    seed: Optional[int] = None,
    print_results: bool = True,
) -> BestClusteringResult:
    """
    Perform grid search to find best clustering configuration, then recompute
    the best clustering using those args and return the result.
    
    Args:
        papers: Sequence of paper-like objects to cluster
        embed_model_name: Name of the embedding model to use
        modes: Sequence of embedding modes to try (e.g., ["B", "C"])
        k_candidates: Sequence of k values to try (e.g., [4, 5])
        top_n_keywords: Number of top keywords for mode "B"
        seed: Random seed for KMeans
        print_results: Whether to print the results DataFrame
    
    Returns:
        BestClusteringResult with the best clustering configuration
    """
    # Create embedder once and reuse it
    embedder = Embedder(model_name=embed_model_name)
    
    # Step 1: Grid search to find best args
    best_args = _grid_search(
        papers=papers,
        embedder=embedder,
        modes=modes,
        k_candidates=k_candidates,
        top_n_keywords=top_n_keywords,
        seed=seed,
        print_results=print_results,
    )
    
    # Step 2: Recompute embeddings using best mode
    best_embeddings = embedder.encode_texts(
        papers,
        mode=best_args.mode,
        top_n=top_n_keywords,
    )
    
    # Step 3: Recompute clustering using best k and seed
    kmeans = KmeanClusterer(
        papers,
        best_embeddings,
        random_state=best_args.seed,
        n_clusters=best_args.k,
    )
    best_labels = kmeans.fit_predict()
    
    # Step 4: Compute member similarities and cluster cohesion
    member_sims = member_similarities(best_embeddings, best_labels)
    coh_mean = cluster_cohesion(member_sims)
    
    # Step 5: Order cluster members by similarity (derived from member_sims)
    cluster_members_ordered_dict = cluster_members_ordered(member_sims)
    
    return BestClusteringResult(
        mode=best_args.mode,
        k=best_args.k,
        cluster_members_ordered=cluster_members_ordered_dict,
        cluster_members_similarities=member_sims,
        cluster_cohesion=coh_mean,
    )
