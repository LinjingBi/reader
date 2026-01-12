from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Optional UMAP for nicer 2D plots
try:
    import umap  # type: ignore
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

import matplotlib.pyplot as plt


@dataclass
class Paper:
    pid: str
    title: str
    summary: str
    keywords: List[str]
    url: str = ""


def build_embedding_text(p: Paper) -> str:
    """
    Build the text we embed.
    NOTE: keywords help, but can dominate if too many / too generic.
    Keep it short + structured.
    """
    kws = ", ".join(p.keywords[:10])  # cap to avoid keyword domination
    parts = [
        f"TITLE: {p.title}",
        f"SUMMARY: {p.summary}",
        f"KEYWORDS: {kws}" if kws else "KEYWORDS:"
    ]
    return "\n".join(parts)


def embed_papers(
    papers: List[Paper],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embeds paper text with a fast, strong general embedding model.
    normalize_embeddings=True makes cosine similarity straightforward.
    """
    model = SentenceTransformer(model_name)
    texts = [build_embedding_text(p) for p in papers]
    X = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(X, dtype=np.float32)


def cosine_centroid_cohesion(X: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    """
    For each cluster, compute avg cosine similarity to its centroid.
    Since X is normalized, cosine sim = dot(x, centroid_normed).
    Higher cohesion => cluster less 'mixed'.
    """
    out: Dict[int, float] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        cluster = X[idx]
        centroid = cluster.mean(axis=0)
        norm = np.linalg.norm(centroid) + 1e-12
        centroid = centroid / norm
        sims = cluster @ centroid
        out[int(c)] = float(np.mean(sims))
    return out


def centroid_representatives(
    X: np.ndarray,
    labels: np.ndarray,
    papers: List[Paper],
    top_n: int = 5,
) -> Dict[int, List[Tuple[float, Paper]]]:
    """
    For each cluster, return top_n papers closest to centroid (cosine sim).
    """
    reps: Dict[int, List[Tuple[float, Paper]]] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        cluster = X[idx]
        centroid = cluster.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = cluster @ centroid  # cosine similarity
        order = np.argsort(-sims)[:top_n]
        reps[int(c)] = [(float(sims[i]), papers[int(idx[i])]) for i in order]
    return reps


def cluster_keyword_hints(
    papers: List[Paper],
    labels: np.ndarray,
    top_k: int = 10,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Compute TF-IDF over the provided keywords (not full text),
    then summarize each cluster by average TF-IDF weights.
    This is great as a 'hint' for later LLM naming.
    """
    # Build a document per paper from keywords only
    docs = []
    for p in papers:
        # keep short + consistent
        docs.append(" ".join([kw.replace(" ", "_") for kw in p.keywords]))

    # If all docs are empty, return nothing
    if all(len(d.strip()) == 0 for d in docs):
        return {int(c): [] for c in set(labels.tolist())}

    vec = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")
    M = vec.fit_transform(docs)  # shape [n_papers, vocab]
    vocab = np.array(vec.get_feature_names_out())

    hints: Dict[int, List[Tuple[str, float]]] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            hints[int(c)] = []
            continue
        # average tf-idf across cluster
        mean_scores = np.asarray(M[idx].mean(axis=0)).ravel()
        top = np.argsort(-mean_scores)[:top_k]
        terms = [(vocab[i], float(mean_scores[i])) for i in top if mean_scores[i] > 0]
        hints[int(c)] = terms
    return hints


def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    return km.fit_predict(X)


def compute_global_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Internal metrics (no human/LLM needed).
    Note: silhouette needs >= 2 clusters and no singleton issues; handle safely.
    """
    metrics: Dict[str, float] = {}
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        metrics["silhouette"] = float("nan")
        metrics["davies_bouldin"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
        return metrics

    # Some metrics can error if a cluster has 1 item; be robust.
    try:
        metrics["silhouette"] = float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        metrics["silhouette"] = float("nan")
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        metrics["davies_bouldin"] = float("nan")
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        metrics["calinski_harabasz"] = float("nan")

    return metrics


def reduce_2d(X: np.ndarray) -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization.
    UMAP generally looks better than PCA for embeddings.
    """
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=12,
            min_dist=0.05,
            n_components=2,
            random_state=42,
            metric="cosine",
        )
        return reducer.fit_transform(X)
    # fallback
    return PCA(n_components=2, random_state=42).fit_transform(X)


def plot_clusters(
    X2: np.ndarray,
    labels: np.ndarray,
    papers: List[Paper],
    title: str,
    annotate_top_per_cluster: int = 2,
):
    """
    2D scatter with:
    - points colored by cluster
    - cluster centroids plotted as 'X'
    - annotate a few representative titles per cluster (avoid clutter)
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=30, alpha=0.85)
    plt.title(title)

    # centroids in 2D
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        cxy = X2[idx].mean(axis=0)
        plt.scatter([cxy[0]], [cxy[1]], marker="X", s=250)
        plt.text(cxy[0], cxy[1], f"  C{c}", fontsize=10, va="center")

        # annotate a couple of closest-to-centroid in 2D (quick & dirty)
        # (for “pretty” reps, use centroid_representatives() on original X)
        d = np.linalg.norm(X2[idx] - cxy, axis=1)
        pick = idx[np.argsort(d)[:annotate_top_per_cluster]]
        for j in pick:
            t = papers[j].title
            t = (t[:55] + "…") if len(t) > 55 else t
            plt.text(X2[j, 0], X2[j, 1], f" {t}", fontsize=8, alpha=0.9)

    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    plt.show()


def main(papers: List[Paper]):
    # 1) embed
    X = embed_papers(papers, model_name="BAAI/bge-small-en-v1.5")

    # 2) reduce once for plotting (same projection reused)
    X2 = reduce_2d(X)

    # 3) run 3 clusterings: k=3..5, print metrics + cohesion + hints + reps, plot
    for k in [3, 4, 5]:
        labels = run_kmeans(X, k=k)

        metrics = compute_global_metrics(X, labels)
        cohesion = cosine_centroid_cohesion(X, labels)
        hints = cluster_keyword_hints(papers, labels, top_k=10)
        reps = centroid_representatives(X, labels, papers, top_n=5)

        print("\n" + "=" * 80)
        print(f"KMeans k={k}")
        print("Global metrics:", metrics)
        print("Cluster cohesion (avg cosine to centroid):")
        for c in sorted(cohesion.keys()):
            print(f"  Cluster {c}: {cohesion[c]:.3f}  (higher = tighter)")

        for c in sorted(set(labels.tolist())):
            members = np.where(labels == c)[0]
            print(f"\n--- Cluster {c} | size={len(members)} ---")
            if hints.get(c):
                top_terms = ", ".join([t for t, _ in hints[c][:8]])
                print(f"TF-IDF keyword hints: {top_terms}")
            else:
                print("TF-IDF keyword hints: (none)")

            print("Representatives (closest to centroid):")
            for sim, p in reps[c]:
                print(f"  [{sim:.3f}] {p.title}  ({p.pid})")

        plot_clusters(
            X2=X2,
            labels=labels,
            papers=papers,
            title=f"KMeans k={k} ({'UMAP' if HAS_UMAP else 'PCA'} view)",
            annotate_top_per_cluster=2,
        )


if __name__ == "__main__":
    # Replace this with your real loader (HF API or your file)
    demo = [
        Paper(pid="2501.00001", title="Demo Paper A", summary="Robotics grasping with diffusion.", keywords=["robotics", "diffusion", "grasping"]),
        Paper(pid="2501.00002", title="Demo Paper B", summary="LLM agents for tool use.", keywords=["LLM", "agents", "tool-use"]),
        Paper(pid="2501.00003", title="Demo Paper C", summary="Self-driving perception with transformers.", keywords=["autonomous-driving", "vision", "transformers"]),
        Paper(pid="2501.00004", title="Demo Paper D", summary="Video generation with latent diffusion.", keywords=["video", "diffusion", "generation"]),
        Paper(pid="2501.00005", title="Demo Paper E", summary="RL fine-tuning with preference models.", keywords=["RLHF", "alignment", "preference"]),
    ]
    main(demo)
