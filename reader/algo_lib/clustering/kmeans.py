"""
KMeans clustering implementation.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from sklearn.cluster import KMeans

from algo_lib.typing import PaperLike


class KmeanClusterer:
    """
    KMeans clusterer for papers using their embeddings.
    """
    
    def __init__(
        self,
        papers: Sequence[PaperLike],
        embeddings: np.ndarray,
        random_state: int = 42,
        n_clusters: int = 5,
    ):
        """
        Initialize KMeans clusterer with papers and their embeddings.

        Args:
            papers: Sequence of paper-like objects to cluster
            embeddings: Embeddings array corresponding to papers
            random_state: Random seed for reproducibility
            n_clusters: Number of clusters for KMeans
        """
        self.papers = papers
        self.embeddings = embeddings
        self.model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        self.labels: np.ndarray | None = None  # Will be set by fit_predict

    def fit_predict(self) -> np.ndarray:
        """
        Fit the model and predict cluster labels for the embeddings.
        Stores labels in self.labels and returns them.
        """
        self.labels = self.model.fit_predict(self.embeddings)
        return self.labels

