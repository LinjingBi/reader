import json
from fetch import Paper, serialize_to_paper_objects
from embed import Embedder
from cluster import KmeanClusterer
import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


"""
# embedding text modes

A: title + summary

B: title + summary + top N keywords

C: title + summary + all keywords
"""
MODES = ["B", "C"]
top_n_keywords = 10

cluster_report_dir = 'best_clustering_reports.md'
# usually monthly report
papers_report_file = 'papers_report.json'

embed_model_name = "BAAI/bge-small-en-v1.5"

# kmeans clustering parameters
ks = [4, 5]

seed = 42

def _order_cluster_members_by_centroid_similarity(
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

def _write_best_clustering_report(
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
        clusters = _order_cluster_members_by_centroid_similarity(embed, labels)

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


def _get_best_clusterings(
    papers: List[Paper], 
    embed_model_name: str, 
    ks: List[int], 
    top_n_keywords: int, 
    seed: int):

    rows = []
    labels_list = []
    Xs = []
    embedder = Embedder(model_name=embed_model_name)

    for mode in MODES:
        X = embedder.encode_texts(papers, mode=mode, top_n=top_n_keywords)
        Xs.append(X)

        for k in ks:
            kmeans = KmeanClusterer(papers, X, random_state=seed, n_clusters=k)
            labels = kmeans.fit_predict(X)

            gm = kmeans.safe_global_metrics(X, labels)
            coh = kmeans.cluster_cohesion(X, labels)
            avg_coh = float(np.mean(list(coh.values()))) if coh else float("nan")
            min_coh = float(np.min(list(coh.values()))) if coh else float("nan")


            ent = kmeans.keyword_entropy_per_cluster(papers, labels)
            avg_ent = float(np.nanmean(list(ent.values()))) if ent else float("nan")
            max_ent = float(np.nanmax(list(ent.values()))) if ent else float("nan")


            rows.append({
                "mode": mode,
                "k": k,
                "embed_model": embed_model_name,
                **gm,
                "avg_cluster_cohesion": avg_coh,     # higher better
                "worst_cluster_cohesion": min_coh,   # higher better
                "avg_kw_entropy": avg_ent,           # lower better
                "max_kw_entropy": max_ent,           # lower better
                "idx": len(rows),
                "embed_X": len(Xs)-1,
            })
            labels_list.append(labels)


    df = pd.DataFrame(rows)

    # A simple ranking heuristic:
    # - maximize silhouette and cohesion
    # - minimize davies_bouldin and entropy
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


def run_pipeline():
    if os.path.exists(cluster_report_dir):
        os.remove(cluster_report_dir)
    # fetch
    with open(papers_report_file, "r") as f:
        data = json.load(f)
    papers_data = data['papers']

    # get clusterings per month
    for month, papers_list in papers_data.items():
        print("!!!!!!!!!!!!!!! LOOKING AT ", month)
        # Create Paper objects from JSON data
        papers = serialize_to_paper_objects(papers_list)
        # embedding + clusterings
        df_best, best_labels, best_embed = _get_best_clusterings(
            papers=papers,
            embed_model_name=embed_model_name,
            ks=ks,
            top_n_keywords=top_n_keywords,
            seed=seed
        )
        # generate best clustering report
        print(f"\n{month} Top choice:", df_best)
        print(f"Appending {month} best clustering report to {cluster_report_dir}")
        _write_best_clustering_report(
            papers=papers,
            labels=best_labels,
            embed=best_embed,
            header=f"# {month} BEST CLUSTERING (mode={df_best['mode']}, k={df_best['k']})",
            max_summary_chars=350,
            report_dir=cluster_report_dir,
        )

if __name__ == "__main__":
    run_pipeline()