#!/usr/bin/env python3
"""
Lazy calibration for topic merge/create threshold T_high.

Idea:
- Treat each monthly cluster centroid as a "pseudo-topic point"
- For each cluster c in time, compute max cosine similarity to any cluster from PRIOR months
- Collect these maxima -> similarity distribution
- Choose conservative T_high near "obvious matches" tail (e.g., p95 / p97 / p99)

Input:
  JSON file with shape:
  {
    "embed_config_id": "...",   # optional (can also pass via --embed-config-id)
    "dim": 384,                # optional but recommended
    "clusters": [
      {
        "period_start": "YYYY-MM-DD",
        "period_end": "YYYY-MM-DD",
        "cluster_key": "...",
        "centroid": [float, ...],
        "size": int,            # optional
        ...
      },
      ...
    ]
  }

Usage:
  python lazy_calibrate_threshold.py --input clusters_12m.json --embed-config-id YOUR_ID
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ClusterPoint:
    period_start: datetime
    period_end: datetime
    cluster_key: str
    vec: np.ndarray
    size: int


def parse_dt(s: str) -> datetime:
    # Accept YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d")


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero / near-zero vector encountered; cannot normalize.")
    return v / n


def load_clusters(path: str, embed_config_id: Optional[str]) -> Tuple[str, List[ClusterPoint]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_embed = data.get("embed_config_id")
    if embed_config_id is None:
        embed_config_id = file_embed

    if embed_config_id is None:
        raise ValueError("embed_config_id not provided (neither in file nor via --embed-config-id).")

    dim = data.get("dim")  # optional
    clusters_raw = data.get("clusters", [])
    if not isinstance(clusters_raw, list) or not clusters_raw:
        raise ValueError("Input JSON must contain a non-empty 'clusters' list.")

    clusters: List[ClusterPoint] = []
    for c in clusters_raw:
        if not isinstance(c, dict):
            continue
        key = str(c.get("cluster_key") or "")
        if not key:
            raise ValueError("Each cluster must have a unique 'cluster_key'.")

        ps = parse_dt(c["period_start"])
        pe = parse_dt(c["period_end"])
        size = int(c.get("size", 1))

        vec_list = c.get("centroid")
        if vec_list is None:
            raise ValueError(f"Cluster {key} missing 'centroid' list.")
        vec = np.array(vec_list, dtype=np.float32)

        if dim is not None and len(vec) != int(dim):
            raise ValueError(f"Cluster {key} centroid dim {len(vec)} != declared dim {dim}")

        vec = l2_normalize(vec)
        clusters.append(ClusterPoint(ps, pe, key, vec, size))

    # Sort by time (period_start, then cluster_key for stability)
    clusters.sort(key=lambda x: (x.period_start, x.cluster_key))
    return embed_config_id, clusters


def compute_prior_max_sims(
    clusters: List[ClusterPoint],
    lookback_months: Optional[int],
) -> List[Dict[str, Any]]:
    """
    For each cluster i, compare to clusters from earlier periods.
    Optionally restrict comparisons to last `lookback_months` months.
    Returns list of per-cluster records with max_sim and argmax.
    """
    results: List[Dict[str, Any]] = []

    # Pre-stack vectors for efficiency
    vecs = np.stack([c.vec for c in clusters], axis=0)  # (N, D)

    for i, c in enumerate(clusters):
        # prior indices are [0, i)
        if i == 0:
            results.append(
                {
                    "cluster_key": c.cluster_key,
                    "period_start": c.period_start.strftime("%Y-%m-%d"),
                    "max_sim_to_prior": None,
                    "nearest_prior_cluster_key": None,
                }
            )
            continue

        prior_idxs = list(range(0, i))

        # Optional time window
        if lookback_months is not None:
            # Keep only priors whose period_start >= (c.period_start - lookback_months)
            # Approximate month as 30 days for MVP simplicity
            cutoff = c.period_start.timestamp() - float(lookback_months) * 30.0 * 24.0 * 3600.0
            prior_idxs = [j for j in prior_idxs if clusters[j].period_start.timestamp() >= cutoff]

        if not prior_idxs:
            results.append(
                {
                    "cluster_key": c.cluster_key,
                    "period_start": c.period_start.strftime("%Y-%m-%d"),
                    "max_sim_to_prior": None,
                    "nearest_prior_cluster_key": None,
                }
            )
            continue

        # Cosine sim because vectors are normalized: sim = dot
        sims = vecs[prior_idxs] @ c.vec  # (len(prior),)
        j_local = int(np.argmax(sims))
        max_sim = float(sims[j_local])
        j = prior_idxs[j_local]

        results.append(
            {
                "cluster_key": c.cluster_key,
                "period_start": c.period_start.strftime("%Y-%m-%d"),
                "max_sim_to_prior": max_sim,
                "nearest_prior_cluster_key": clusters[j].cluster_key,
            }
        )

    return results


def percentile(xs: np.ndarray, p: float) -> float:
    return float(np.percentile(xs, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to clusters JSON")
    ap.add_argument("--embed-config-id", default=None, help="Filter/verify embed_config_id")
    ap.add_argument(
        "--lookback-months",
        type=int,
        default=None,
        help="Optional lookback window (months) for prior comparison; default=None means all prior months.",
    )
    ap.add_argument(
        "--suggest",
        default="95",
        help="Comma-separated percentiles to print as candidate thresholds (e.g., '90,95,97,99')",
    )
    ap.add_argument(
        "--dump-results",
        default=None,
        help="Optional path to write per-cluster max_sim_to_prior records as JSON",
    )
    args = ap.parse_args()

    embed_id, clusters = load_clusters(args.input, args.embed_config_id)
    per_cluster = compute_prior_max_sims(clusters, args.lookback_months)

    sims = np.array([r["max_sim_to_prior"] for r in per_cluster if r["max_sim_to_prior"] is not None], dtype=np.float32)
    if sims.size == 0:
        raise RuntimeError("No prior similarities computed (need >=2 clusters total, with some priors).")

    sims_sorted = np.sort(sims)
    print(f"embed_config_id: {embed_id}")
    print(f"clusters: {len(clusters)}")
    print(f"comparisons (max-to-prior): {len(sims)}")
    print(f"min/median/mean/max: {float(sims_sorted[0]):.4f} / {percentile(sims, 50):.4f} / {float(np.mean(sims)):.4f} / {float(sims_sorted[-1]):.4f}")

    ps = [float(x.strip()) for x in args.suggest.split(",") if x.strip()]
    print("\nCandidate thresholds (percentiles of max-to-prior similarity):")
    for p in ps:
        print(f"  p{int(p):02d}: {percentile(sims, p):.4f}")

    # A conservative default suggestion:
    # - start at p95 or p97 (create-friendly) and tune online
    p95 = percentile(sims, 95)
    p97 = percentile(sims, 97)
    print("\nSuggested starting points (create-friendly):")
    print(f"  T_high ~ p95 = {p95:.4f}")
    print(f"  (more conservative) T_high ~ p97 = {p97:.4f}")

    if args.dump_results:
        out = {
            "embed_config_id": embed_id,
            "lookback_months": args.lookback_months,
            "records": per_cluster,
        }
        with open(args.dump_results, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nWrote per-cluster records to: {args.dump_results}")


if __name__ == "__main__":
    main()
