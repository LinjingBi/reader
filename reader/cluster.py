"""
Density-based clustering
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from fetch import Paper

# Optional plotting
try:
    import umap  # type: ignore
    import matplotlib.pyplot as plt
    HAS_VIZ = True
    HAS_UMAP = True
except ImportError:
    HAS_VIZ = False
    HAS_UMAP = False


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
        self.rs = random_state
        self.k = n_clusters
        self.model_name = "kmeans"
        self.model = KMeans(n_clusters=self.k, n_init="auto", random_state=self.rs)
        self.labels = None  # Will be set by fit_predict

    def fit_predict(self) -> np.ndarray:
        """
        Fit the model and predict cluster labels for the embeddings.
        Stores labels in self.labels and returns them.
        """
        self.labels = self.model.fit_predict(self.embeddings)
        return self.labels
    
    def group_by_cluster(self, items: Optional[List[str]] = None) -> Dict[int, List[str]]:
        """
        Packs items by cluster id.
        If items is None, uses paper titles from self.papers.
        """
        if self.labels is None:
            raise ValueError("Must call fit_predict() first to generate labels")

        if items is None:
            items = [p.title for p in self.papers]

        cluster_dict = defaultdict(list)
        for label, item in zip(self.labels, items):
            cluster_dict[int(label)].append(item)
        return cluster_dict
    
    def cosine_centroid_cohesion(self) -> Dict[int, float]:
        """
        For each cluster, compute avg cosine similarity to its centroid.
        Since embeddings are normalized, cosine sim = dot(x, centroid_normed).
        Higher cohesion => cluster less 'mixed'.
        """
        if self.labels is None:
            raise ValueError("Must call fit_predict() first to generate labels")

        out: Dict[int, float] = {}
        for c in sorted(set(self.labels.tolist())):
            idx = np.where(self.labels == c)[0]
            if len(idx) == 0:
                continue
            cluster = self.embeddings[idx]
            centroid = cluster.mean(axis=0)
            norm = np.linalg.norm(centroid) + 1e-12
            centroid = centroid / norm
            sims = cluster @ centroid
            out[int(c)] = float(np.mean(sims))
        return out
    
    def centroid_representatives(self, top_n: int = 5) -> Dict[int, List[Tuple[float, Paper]]]:
        """
        For each cluster, return top_n papers closest to centroid (cosine sim).
        """
        if self.labels is None:
            raise ValueError("Must call fit_predict() first to generate labels")

        reps: Dict[int, List[Tuple[float, Paper]]] = {}
        for c in sorted(set(self.labels.tolist())):
            idx = np.where(self.labels == c)[0]
            cluster = self.embeddings[idx]
            centroid = cluster.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            sims = cluster @ centroid  # cosine similarity
            order = np.argsort(-sims)[:top_n]
            reps[int(c)] = [(float(sims[i]), self.papers[int(idx[i])]) for i in order]
        return reps
    
    def cluster_keyword_hints(self, top_k: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Compute TF-IDF over the provided keywords (not full text),
        then summarize each cluster by average TF-IDF weights.
        This is great as a 'hint' for later LLM naming.
        """
        if self.labels is None:
            raise ValueError("Must call fit_predict() first to generate labels")

        # Build a document per paper from keywords only
        docs = []
        for p in self.papers:
            # keep short + consistent
            docs.append(" ".join([kw.replace(" ", "_") for kw in p.keywords]))

        # If all docs are empty, return nothing
        if all(len(d.strip()) == 0 for d in docs):
            return {int(c): [] for c in set(self.labels.tolist())}

        vec = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")
        M = vec.fit_transform(docs)  # shape [n_papers, vocab]
        vocab = np.array(vec.get_feature_names_out())

        hints: Dict[int, List[Tuple[str, float]]] = {}
        for c in sorted(set(self.labels.tolist())):
            idx = np.where(self.labels == c)[0]
            if len(idx) == 0:
                hints[int(c)] = []
                continue
            # average tf-idf across cluster
            mean_scores = np.asarray(M[idx].mean(axis=0)).ravel()
            top = np.argsort(-mean_scores)[:top_k]
            terms = [(vocab[i], float(mean_scores[i])) for i in top if mean_scores[i] > 0]
            hints[int(c)] = terms
        return hints
    
    def compute_global_metrics(self) -> Dict[str, float]:
        """
        Internal metrics (no human/LLM needed).
        Note: silhouette needs >= 2 clusters and no singleton issues; handle safely.
        """
        if self.labels is None:
            raise ValueError("Must call fit_predict() first to generate labels")

        metrics: Dict[str, float] = {}
        unique = sorted(set(self.labels.tolist()))
        if len(unique) < 2:
            metrics["silhouette"] = float("nan")
            metrics["davies_bouldin"] = float("nan")
            metrics["calinski_harabasz"] = float("nan")
            return metrics

        # Some metrics can error if a cluster has 1 item; be robust.
        try:
            metrics["silhouette"] = float(silhouette_score(self.embeddings, self.labels, metric="cosine"))
        except Exception:
            metrics["silhouette"] = float("nan")
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(self.embeddings, self.labels))
        except Exception:
            metrics["davies_bouldin"] = float("nan")
        try:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(self.embeddings, self.labels))
        except Exception:
            metrics["calinski_harabasz"] = float("nan")

        return metrics
    
    def reduce_2d(self) -> np.ndarray:
        """
        Reduce embeddings to 2D for visualization.
        UMAP generally looks better than PCA for embeddings.
        """
        if HAS_UMAP:
            reducer = umap.UMAP(
                n_neighbors=12,
                min_dist=0.05,
                n_components=2,
                random_state=self.rs,
                metric="cosine",
            )
            return reducer.fit_transform(self.embeddings)
        # fallback
        return PCA(n_components=2, random_state=self.rs).fit_transform(self.embeddings)
    
    def plot_clusters(
        self,
        X2: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        annotate_top_per_cluster: int = 2,
    ):
        """
        2D scatter with:
        - points colored by cluster
        - cluster centroids plotted as 'X'
        - annotate a few representative titles per cluster (avoid clutter)
        """
        if self.labels is None:
            raise ValueError("Must call fit_predict() first to generate labels")

        if not HAS_VIZ:
            print("matplotlib not installed. Skipping visualization.")
            return

        if X2 is None:
            X2 = self.reduce_2d()

        if title is None:
            title = f"{self.model_name} Cluster visualization ({'UMAP' if HAS_UMAP else 'PCA'} view)"

        plt.figure(figsize=(10, 7))
        plt.scatter(X2[:, 0], X2[:, 1], c=self.labels, s=30, alpha=0.85)
        plt.title(title)

        # centroids in 2D
        for c in sorted(set(self.labels.tolist())):
            idx = np.where(self.labels == c)[0]
            cxy = X2[idx].mean(axis=0)
            plt.scatter([cxy[0]], [cxy[1]], marker="X", s=250)
            plt.text(cxy[0], cxy[1], f"  C{c}", fontsize=10, va="center")

            # annotate a couple of closest-to-centroid in 2D
            d = np.linalg.norm(X2[idx] - cxy, axis=1)
            pick = idx[np.argsort(d)[:annotate_top_per_cluster]]
            for j in pick:
                t = self.papers[j].title
                t = (t[:55] + "…") if len(t) > 55 else t
                plt.text(X2[j, 0], X2[j, 1], f" {t}", fontsize=8, alpha=0.9)

        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        plt.tight_layout()
        plt.show()
    
    def plot_umap(self):
        """
        Optional 2D visualization using UMAP (legacy method name for compatibility).
        """
        self.plot_clusters(title=f"{self.model_name} Cluster visualization (UMAP)")


if __name__ == "__main__":
    # smoke test using random vectors and dummy papers
    from fetch import Paper
    papers = [
        Paper(pid=f"paper_{i}", title=f"Paper {i}", summary=f"Summary {i}", keywords=[f"kw{i}"])
        for i in range(20)
    ]
    X = np.random.randn(20, 384)  # pretend these are real embeddings

    cl = KmeanClusterer(papers, X, random_state=42, n_clusters=5)
    cl.fit_predict()
    # print cluster groups
    grouped = cl.group_by_cluster()

    for cid, items in grouped.items():
        print(f"Cluster {cid}:")
        for it in items:
            print("  -", it)
    # plot umap
    cl.plot_umap()
