"""
Density-based clustering
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


if TYPE_CHECKING:
    from fetch import Paper



class KmeanClusterer:
    def __init__(self, papers: List[Paper], embeddings: np.ndarray, random_state: int = 42, n_clusters: int = 5):
        """
        Initialize KMeans clusterer with papers and their embeddings.

        Args:
            papers: List of Paper objects to cluster
            embeddings: Embeddings array corresponding to papers
            random_state: Random seed for reproducibility
            n_clusters: Number of clusters for KMeans
        """
        self.papers = papers
        self.embeddings = embeddings
        self.model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        self.labels = None  # Will be set by fit_predict

    def fit_predict(self) -> np.ndarray:
        """
        Fit the model and predict cluster labels for the embeddings.
        Stores labels in self.labels and returns them.
        """
        self.labels = self.model.fit_predict(self.embeddings)
        return self.labels
    
    def safe_global_metrics(self):
        """
        Internal clustering metrics. Robust to edge cases.
        """
        unique = sorted(set(self.labels.tolist()))
        if len(unique) < 2:
            return {"silhouette_cosine": float("nan"),
                    "davies_bouldin": float("nan"),
                    "calinski_harabasz": float("nan")}

        out = {}
        try:
            out["silhouette_cosine"] = float(silhouette_score(self.embeddings, self.labels, metric="cosine"))
        except Exception:
            out["silhouette_cosine"] = float("nan")
        try:
            out["davies_bouldin"] = float(davies_bouldin_score(self.embeddings, self.labels))
        except Exception:
            out["davies_bouldin"] = float("nan")
        try:
            out["calinski_harabasz"] = float(calinski_harabasz_score(self.embeddings, self.labels))
        except Exception:
            out["calinski_harabasz"] = float("nan")
        return out
    
    def cluster_cohesion(self):
        """
        Avg cosine similarity to centroid per cluster.
        X is normalized -> cosine = dot(x, centroid_norm).
        """
        out: Dict[int, float] = {}
        for c in sorted(set(self.labels.tolist())):
            idx = np.where(self.labels == c)[0]
            if len(idx) == 0:
                continue
            C = self.embeddings[idx]
            centroid = C.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            sims = C @ centroid
            out[int(c)] = float(np.mean(sims))
        return out
    
    def keyword_entropy_per_cluster(self):
        """
        Uses *provided keywords only* (not embedded text).
        Entropy high => mixed vocabulary, low => focused.
        """
        ent: Dict[int, float] = {}
        for c in sorted(set(self.labels.tolist())):
            idx = np.where(self.labels == c)[0]
            counts: Dict[str, int] = {}
            total = 0
            for i in idx:
                for kw in self.papers[i].keywords:
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
    
