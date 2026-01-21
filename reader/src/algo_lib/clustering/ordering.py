"""
Cluster member ordering utilities.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np


def order_cluster_members_by_centroid_similarity(
    embeddings: np.ndarray,  # (n, d) normalized embeddings
    labels: np.ndarray,  # (n,) cluster ids
) -> Dict[int, List[int]]:
    """
    Returns: dict[cluster_id] -> list of paper indices,
    sorted by cosine similarity to centroid (most representative first).
    
    Assumes embeddings rows are L2-normalized.
    
    Args:
        embeddings: Normalized embeddings array
        labels: Cluster labels
    
    Returns:
        Dictionary mapping cluster_id -> list of paper indices sorted by similarity
    """
    out: Dict[int, List[int]] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        C = embeddings[idx]
        centroid = C.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        sims = C @ centroid  # cosine similarity since embeddings normalized
        order = np.argsort(-sims)  # descending order

        out[int(c)] = [int(idx[i]) for i in order]
    return out

