"""
Clustering metrics computation.
"""

from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from algo_lib.typing import PaperLike



def safe_global_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Internal clustering metrics. Robust to edge cases.
    
    Args:
        embeddings: Embeddings array
        labels: Cluster labels
    
    Returns:
        Dictionary with metric names and values (NaN if computation fails)
    """
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        return {
            "silhouette_cosine": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    out = {}
    try:
        out["silhouette_cosine"] = float(
            silhouette_score(embeddings, labels, metric="cosine")
        )
    except Exception:
        out["silhouette_cosine"] = float("nan")
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(embeddings, labels))
    except Exception:
        out["davies_bouldin"] = float("nan")
    try:
        out["calinski_harabasz"] = float(
            calinski_harabasz_score(embeddings, labels)
        )
    except Exception:
        out["calinski_harabasz"] = float("nan")
    return out


def member_similarities(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, Dict[int, float]]:
    """
    Cosine similarity to centroid per cluster member.
    X is normalized -> cosine = dot(x, centroid_norm).
    
    Args:
        embeddings: Normalized embeddings array
        labels: Cluster labels
    
    Returns:
        Dictionary mapping cluster_id -> Dict[paper_idx] -> similarity to centroid
    """
    out: Dict[int, Dict[int, float]] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        C = embeddings[idx]
        centroid = C.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        sims = C @ centroid  # cosine similarity since embeddings normalized
        out[int(c)] = {int(idx[i]): float(sims[i]) for i in range(len(idx))}
    return out


def cluster_cohesion(
    member_similarities: Dict[int, Dict[int, float]],
) -> Dict[int, float]:
    """
    Compute mean cohesion per cluster from member similarities.
    
    Args:
        member_similarities: Dict[cluster_id] -> Dict[paper_idx] -> similarity
    
    Returns:
        Dictionary mapping cluster_id -> average cosine similarity to centroid
    """
    out: Dict[int, float] = {}
    for c, similarities in member_similarities.items():
        if similarities:
            out[int(c)] = float(np.mean(list(similarities.values())))
        else:
            out[int(c)] = 0.0
    return out


def keyword_entropy_per_cluster(
    papers: Sequence[PaperLike],
    labels: np.ndarray,
) -> Dict[int, float]:
    """
    Uses *provided keywords only* (not embedded text).
    Entropy high => mixed vocabulary, low => focused.
    
    Args:
        papers: Sequence of paper-like objects
        labels: Cluster labels
    
    Returns:
        Dictionary mapping cluster_id -> keyword entropy
    """
    ent: Dict[int, float] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        counts: Dict[str, int] = {}
        total = 0
        for i in idx:
            for kw in papers[i].keywords:
                kw_norm = kw.strip().lower()
                if not kw_norm:
                    continue
                counts[kw_norm] = counts.get(kw_norm, 0) + 1
                total += 1
        if total == 0:
            ent[int(c)] = float("nan")
            continue
        probs = np.array([v / total for v in counts.values()], dtype=np.float32)
        ent[int(c)] = float(-np.sum(probs * np.log(probs + 1e-12)))
    return ent

