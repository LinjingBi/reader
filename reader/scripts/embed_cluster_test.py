from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import json
import os

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

"""
Text modes:

A: title + summary

B: title + summary + top N keywords

C: title + summary + all keywords

Embedding: configurable (default BAAI/bge-small-en-v1.5)

Clustering: KMeans with k ∈ {3,4,5}

Metrics:

silhouette (cosine)

Davies–Bouldin

Calinski–Harabasz

average cluster cohesion (avg cosine-to-centroid)

worst cluster cohesion (min over clusters)

cluster keyword entropy (optional, uses provided keywords)


"""

# -----------------------------
# Input structure (as you specified)
# -----------------------------
@dataclass
class Paper:
    pid: str
    title: str
    summary: str
    keywords: List[str]
    url: str = ""


# -----------------------------
# Text construction modes
# -----------------------------
def build_text(p: Paper, mode: str, top_n: int = 10) -> str:
    """
    mode:
      - "A": title + summary
      - "B": title + summary + top_n keywords
      - "C": title + summary + all keywords
    """
    base = f"TITLE: {p.title}\nSUMMARY: {p.summary}".strip()

    if mode == "A":
        return base

    if mode == "B":
        kws = p.keywords[:top_n]
        return base + "\nKEYWORDS: " + ", ".join(kws)

    if mode == "C":
        return base + "\nKEYWORDS: " + ", ".join(p.keywords)

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# Embedding
# -----------------------------
def embed_texts(
    texts: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    X = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine-friendly
    )
    return np.asarray(X, dtype=np.float32)


# -----------------------------
# Clustering + metrics
# -----------------------------
def kmeans_labels(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    return km.fit_predict(X)


def cluster_cohesion(X: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    """
    Avg cosine similarity to centroid per cluster.
    X is normalized -> cosine = dot(x, centroid_norm).
    """
    out: Dict[int, float] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        C = X[idx]
        centroid = C.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        sims = C @ centroid
        out[int(c)] = float(np.mean(sims))
    return out


def safe_global_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Internal clustering metrics. Robust to edge cases.
    """
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        return {"silhouette_cosine": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}

    out = {}
    try:
        out["silhouette_cosine"] = float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        out["silhouette_cosine"] = float("nan")
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        out["davies_bouldin"] = float("nan")
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        out["calinski_harabasz"] = float("nan")
    return out


def keyword_entropy_per_cluster(papers: List[Paper], labels: np.ndarray) -> Dict[int, float]:
    """
    Uses *provided keywords only* (not embedded text).
    Entropy high => mixed vocabulary, low => focused.
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


# -----------------------------
# Main ablation runner
# -----------------------------
def run_ablation(
    papers: List[Paper],
    model_name: str = "BAAI/bge-small-en-v1.5",
    ks: List[int] = [4, 5],
    top_n_keywords: int = 10,
    seed: int = 42,
    compute_kw_entropy: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with metrics for each (mode, k) run.
    """
    rows = []
    labels_list = []
    Xs = []

    for mode in ["B", "C"]:
        texts = [build_text(p, mode=mode, top_n=top_n_keywords) for p in papers]
        X = embed_texts(texts, model_name=model_name)
        Xs.append(X)

        for k in ks:
            labels = kmeans_labels(X, k=k, seed=seed)

            gm = safe_global_metrics(X, labels)
            coh = cluster_cohesion(X, labels)
            avg_coh = float(np.mean(list(coh.values()))) if coh else float("nan")
            min_coh = float(np.min(list(coh.values()))) if coh else float("nan")

            if compute_kw_entropy:
                ent = keyword_entropy_per_cluster(papers, labels)
                avg_ent = float(np.nanmean(list(ent.values()))) if ent else float("nan")
                max_ent = float(np.nanmax(list(ent.values()))) if ent else float("nan")
            else:
                avg_ent = float("nan")
                max_ent = float("nan")

            rows.append({
                "mode": mode,
                "k": k,
                "embed_model": model_name,
                **gm,
                "avg_cluster_cohesion": avg_coh,     # higher better
                "worst_cluster_cohesion": min_coh,   # higher better
                "avg_kw_entropy": avg_ent,           # lower better
                "max_kw_entropy": max_ent,           # lower better
                "idx": len(rows),
                "embed_X": len(Xs)-1,
            })
            labels_list.append(labels)
            # print(f"mode: {mode} k: {k} \n metrics: {rows[-1]}")

    df = pd.DataFrame(rows)

    # A simple ranking heuristic:
    # - maximize silhouette and cohesion
    # - minimize davies_bouldin and entropy
    # (weights are rough; you can tune later)
    def safe(x):  # convert NaN to neutral-ish
        return np.nan_to_num(x, nan=0.0)

    df["score"] = (
        1.5 * safe(df["silhouette_cosine"])
        + 2.0 * safe(df["avg_cluster_cohesion"])
        + 1.0 * safe(df["worst_cluster_cohesion"])
        - 0.7 * safe(df["davies_bouldin"])
        - 0.3 * safe(df["avg_kw_entropy"])
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
    return df.iloc[0].to_dict(), labels_list[int(df.iloc[0]["idx"])], Xs[int(df.iloc[0]["embed_X"])]


# -----------------------------
# Reporting helpers (minimal add-on)
# -----------------------------
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def order_cluster_members_by_centroid_similarity(
    X: np.ndarray,           # (n, d) normalized embeddings
    labels: np.ndarray,      # (n,) cluster ids
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Returns: dict[cluster_id] -> list of (paper_index, cosine_sim_to_centroid),
    sorted by cosine similarity desc (most representative first).
    Assumes X rows are L2-normalized.
    """
    out: Dict[int, List[Tuple[int, float]]] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        C = X[idx]
        centroid = C.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        sims = C @ centroid  # cosine similarity since X normalized
        order = np.argsort(-sims)

        out[int(c)] = [ int(idx[i]) for i in order]
    return out

def write_best_clustering_report(
    papers: List[Paper],
    labels: np.ndarray,
    embed,
    header: str = "",
    max_summary_chars: int = 350,
    report_dir = 'best_clustering_reports.md'
):
    """
    Print a human-readable report for a chosen clustering:
    - cluster sizes 
    - optional TF-IDF keyword hints
    - each paper: title + (truncated) summary + url
    """
    with open(report_dir, 'a+') as f:
        if header:
            f.write("\n" + "=" * 90 + '\n')
            f.write(header+'\n')

        # clusters: Dict[int, List[int]] = defaultdict(list)
        # for i, lab in enumerate(labels.tolist()):
        #     clusters[int(lab)].append(i)
        clusters = order_cluster_members_by_centroid_similarity(embed, labels)

        # sort by cluster size desc
        cluster_order = sorted(clusters.keys(), key=lambda c: len(clusters[c]), reverse=True)

        for c in cluster_order:
            idxs = clusters[c]
            f.write("\n" + "-" * 90+"\n")
            f.write(f"Cluster {c} | size={len(idxs)}\n")

            for i in idxs:
                p = papers[i]
                summ = p.summary.strip() if p.summary else ""
                if max_summary_chars and len(summ) > max_summary_chars:
                    summ = summ[:max_summary_chars] + "…"
                f.write(f"\n[{p.pid}] {p.title}\n")
                if p.url:
                    f.write(f"URL: {p.url}\n")
                if summ:
                    f.write(f"Summary: {summ}\n")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    
    # Replace with your real List[Paper]
    # demo = [
    #     Paper(pid="p1", title="Diffusion for Video", summary="We propose a diffusion model for video generation...", keywords=["diffusion", "video", "generation"]),
    #     Paper(pid="p2", title="LLM Tool Agents", summary="We study tool-using agents for LLMs...", keywords=["LLM", "agents", "tool-use"]),
    #     Paper(pid="p3", title="Robotic Grasping", summary="A new grasp planning approach with...", keywords=["robotics", "grasping", "planning"]),
    #     Paper(pid="p4", title="Autonomous Driving", summary="Perception for self-driving using transformers...", keywords=["autonomous-driving", "vision", "transformers"]),
    #     Paper(pid="p5", title="Alignment via Preferences", summary="Preference optimization for alignment...", keywords=["alignment", "RLHF", "preferences"]),
    # ]
    with open("../reader/papers_report.json", "r") as f:
        data = json.load(f)
    papers_data = data['papers']

    cluster_report_dir = 'best_clustering_reports.md'

    if os.path.exists(cluster_report_dir):
        os.remove(cluster_report_dir)

    for month, papers_list in papers_data.items():
        print("!!!!!!!!!!!!!!! LOOKING AT ", month)
        # Create Paper objects from JSON data
        papers = []
        for paper_data in papers_list:
            paper = paper_data['paper']
            paper_obj = Paper(
                pid=paper['id'],
                title=paper['title'],
                summary=paper['summary'],
                keywords=paper.get('ai_keywords', []),
                url=paper.get('projectPage', '')
            )
            papers.append(paper_obj)
        # !!! check one month first
        # break

        demo = papers
        df_best, best_labels, best_embed = run_ablation(
            papers=demo,
            model_name="BAAI/bge-small-en-v1.5",
            ks=[4, 5],
            top_n_keywords=10,
            seed=42,
            compute_kw_entropy=True,
        )

        
        print(f"\n{month} Top choice:", df_best)
        # print(f"\n{month} Top choice labels:", labels_list[df.iloc[0].to_dict()['idx']])
        print(f"Appending {month} best clustering report to {cluster_report_dir}")
        write_best_clustering_report(
            papers=demo,
            labels=best_labels,
            embed=best_embed,
            header=f"# {month} BEST CLUSTERING (mode={df_best['mode']}, k={df_best['k']})",
            max_summary_chars=350,
            report_dir=cluster_report_dir,
        )

        # break
