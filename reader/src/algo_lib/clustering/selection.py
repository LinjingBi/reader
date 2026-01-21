"""
Best clustering selection via grid search and scoring.
"""

from __future__ import annotations
from typing import Sequence, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

from algo_lib.typing import PaperLike
from algo_lib.embedding.embedder import Embedder
from algo_lib.clustering.kmeans import KmeanClusterer
from algo_lib.clustering.metrics import (
    safe_global_metrics,
    member_similarities,
    cluster_cohesion,
    keyword_entropy_per_cluster,
)
from algo_lib.clustering.ordering import order_cluster_members_by_centroid_similarity


@dataclass
class BestClusteringResult:
    """
    Result of best clustering selection.
    """
    mode: str
    k: int
    labels: np.ndarray  # shape [n_papers]
    embeddings: np.ndarray  # shape [n_papers, dim]
    metrics: Dict[str, float]
    score: float
    cluster_members_ordered: Dict[int, List[int]]  # cluster_label -> list of paper indices


def select_best_clustering(
    papers: Sequence[PaperLike],
    embedder: Embedder,
    modes: Sequence[str],
    k_candidates: Sequence[int],
    top_n_keywords: int,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    score_weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    print_results: bool = True,
) -> BestClusteringResult:
    """
    Perform grid search over modes and k values, score each candidate,
    and return the best clustering.
    
    Args:
        papers: Sequence of paper-like objects to cluster
        embedder: Embedder instance to use
        modes: Sequence of embedding modes to try (e.g., ["B", "C"])
        k_candidates: Sequence of k values to try (e.g., [4, 5])
        top_n_keywords: Number of top keywords for mode "B"
        score_weights: Optional custom score weights. Default:
            {
                "silhouette_cosine": 1.5,
                "avg_cluster_cohesion": 2.0,
                "worst_cluster_cohesion": 1.0,
                "davies_bouldin": -0.7,
                "avg_kw_entropy": -0.3,
            }
        seed: Random seed for KMeans
        print_results: Whether to print the results DataFrame
    
    Returns:
        BestClusteringResult with the best clustering configuration
    """
    if seed is None:
        seed = 42
    
    if score_weights is None:
        score_weights = {
            "silhouette_cosine": 1.5,
            "avg_cluster_cohesion": 2.0,
            "worst_cluster_cohesion": 1.0,
            "davies_bouldin": -0.7,
            "avg_kw_entropy": -0.3,
        }
    
    rows = []
    labels_list = []
    embeddings_list = []
    
    for mode in modes:
        X = embedder.encode_texts(
            papers,
            mode=mode,
            top_n=top_n_keywords,
        )
        embeddings_list.append(X)
        
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
                "embed_model": embed_model_name,
                **gm,
                "avg_cluster_cohesion": avg_coh,  # higher better
                "worst_cluster_cohesion": min_coh,  # higher better
                "avg_kw_entropy": avg_ent,  # lower better
                "max_kw_entropy": max_ent,  # lower better
                "idx": len(rows),
                "embed_X": len(embeddings_list) - 1,
            })
            labels_list.append(labels)
    
    df = pd.DataFrame(rows)
    
    # Compute composite score
    def safe(x):  # convert NaN to neutral-ish
        return np.nan_to_num(x, nan=0.0)
    
    df["score"] = (
        score_weights["silhouette_cosine"] * safe(df["silhouette_cosine"])
        + score_weights["avg_cluster_cohesion"] * safe(df["avg_cluster_cohesion"])
        + score_weights["worst_cluster_cohesion"] * safe(df["worst_cluster_cohesion"])
        + score_weights["davies_bouldin"] * safe(df["davies_bouldin"])
        + score_weights["avg_kw_entropy"] * safe(df["avg_kw_entropy"])
    )
    
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    if print_results:
        print(df.to_string(index=False))
    
    # Get best result
    best_row = df.iloc[0]
    best_idx = int(best_row["idx"])
    best_embed_idx = int(best_row["embed_X"])
    
    best_labels = labels_list[best_idx]
    best_embeddings = embeddings_list[best_embed_idx]
    
    # Order cluster members
    cluster_members_ordered = order_cluster_members_by_centroid_similarity(
        best_embeddings,
        best_labels,
    )
    
    # Build metrics dict
    metrics = {
        "silhouette_cosine": float(best_row["silhouette_cosine"]),
        "davies_bouldin": float(best_row["davies_bouldin"]),
        "calinski_harabasz": float(best_row["calinski_harabasz"]),
        "avg_cluster_cohesion": float(best_row["avg_cluster_cohesion"]),
        "worst_cluster_cohesion": float(best_row["worst_cluster_cohesion"]),
        "avg_kw_entropy": float(best_row["avg_kw_entropy"]),
        "max_kw_entropy": float(best_row["max_kw_entropy"]),
    }
    
    return BestClusteringResult(
        mode=str(best_row["mode"]),
        k=int(best_row["k"]),
        labels=best_labels,
        embeddings=best_embeddings,
        metrics=metrics,
        score=float(best_row["score"]),
        cluster_members_ordered=cluster_members_ordered,
    )


def get_best_clustering(
    papers: Sequence[PaperLike],
    embed_model_name: str,
    ks: Sequence[int],
    top_n_keywords: int,
    modes: Sequence[str],
    seed: Optional[int] = None,
    print_results: bool = True,
) -> tuple[Dict[str, Union[float, int, str]], np.ndarray, np.ndarray, Dict[int, float], Dict[int, Dict[int, float]]]:
    """
    Get best clustering using select_best_clustering.
    Returns tuple: (best_row_dict, best_labels, best_embeddings, cluster_cohesion, member_similarities)
    
    This is a convenience wrapper that returns a dict format for backward compatibility.
    For new code, prefer using select_best_clustering directly.
    
    Args:
        papers: Sequence of paper-like objects to cluster
        embed_model_name: Name of the embedding model to use
        ks: Sequence of k values to try (e.g., [4, 5])
        top_n_keywords: Number of top keywords for mode "B"
        modes: Sequence of embedding modes to try (e.g., ["B", "C"])
        seed: Random seed for KMeans
        print_results: Whether to print the results DataFrame
    
    Returns:
        Tuple of (best_row_dict, best_labels, best_embeddings, cluster_cohesion, member_similarities)
        where best_row_dict contains mode, k, embed_model, metrics, and score
        cluster_cohesion: Dict[cluster_id] -> average cohesion (float)
        member_similarities: Dict[cluster_id] -> Dict[paper_idx] -> similarity to centroid (float)
    """
    embedder = Embedder(model_name=embed_model_name)
    
    result = select_best_clustering(
        papers=papers,
        embedder=embedder,
        modes=modes,
        k_candidates=ks,
        top_n_keywords=top_n_keywords,
        embed_model_name=embed_model_name,
        seed=seed,
        print_results=print_results,
    )
    
    # Compute member similarities to centroid for each cluster
    member_similarities_dict = member_similarities(result.embeddings, result.labels)
    
    # Compute mean cohesion per cluster
    cohesion_dict = cluster_cohesion(member_similarities_dict)
    
    # Build dict matching old format for backward compatibility
    best_row_dict = {
        "mode": result.mode,
        "k": result.k,
        "embed_model": embed_model_name,
        **result.metrics,
        "score": result.score,
    }
    
    return best_row_dict, result.labels, result.embeddings, cohesion_dict, member_similarities_dict

