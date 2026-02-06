#!/usr/bin/env python3
"""
run after embed_cluster_12m.py and test_t_threshold.py (to get the T_attach threshold).

Close-to-real calibration: simulate merge/create as online clustering over cluster centroids.

MVP mode: compare against ALL topics ever (persistent topic set).
Future mode: compare against topics active within last N months.

Input JSON:
{
  "embed_config_id": "...",
  "dim": 384,
  "dtype": "f32_le",            # optional, used if centroid_b64 present
  "clusters": [
    {
      "period_start": "YYYY-MM-DD",
      "period_end": "YYYY-MM-DD",
      "cluster_key": "...",
      "size": 42,               # optional (default 1)
      "centroid_b64": "...",    # OR "centroid": [floats...]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Zero / near-zero vector encountered; cannot normalize.")
    return v / n


def decode_centroid_b64(b64: str, dim: int, dtype: str = "f32_le") -> np.ndarray:
    raw = base64.b64decode(b64)
    if dtype != "f32_le":
        raise ValueError(f"Unsupported dtype '{dtype}'. Use 'f32_le'.")
    vec = np.frombuffer(raw, dtype="<f4")  # little-endian float32
    if vec.size != dim:
        raise ValueError(f"Decoded centroid dim {vec.size} != expected dim {dim}")
    return vec.astype(np.float32, copy=False)


@dataclass
class ClusterPoint:
    period_start: datetime
    period_end: datetime
    key: str
    vec: np.ndarray   # normalized
    w: float          # weight (size by default)


@dataclass
class TopicState:
    topic_id: int
    mu: np.ndarray    # normalized centroid
    W: float          # total weight accumulated
    last_seen: datetime


def load_points(path: str, embed_config_id: Optional[str]) -> Tuple[str, int, str, List[ClusterPoint]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_embed = data.get("embed_config_id")
    if embed_config_id is None:
        embed_config_id = file_embed
    if embed_config_id is None:
        raise ValueError("embed_config_id not provided (neither in file nor via --embed-config-id).")

    dim = int(data.get("dim") or 0)
    if dim <= 0:
        raise ValueError("Input JSON must include a positive 'dim'.")

    dtype = data.get("dtype", "f32_le")

    pts: List[ClusterPoint] = []
    for c in data.get("clusters", []):
        key = str(c.get("cluster_key") or "")
        if not key:
            raise ValueError("Each cluster must have 'cluster_key'.")

        ps = parse_dt(c["period_start"])
        pe = parse_dt(c["period_end"])
        w = float(c.get("size", 1))

        if "centroid_b64" in c and c["centroid_b64"]:
            vec = decode_centroid_b64(c["centroid_b64"], dim=dim, dtype=dtype)
        elif "centroid" in c and c["centroid"] is not None:
            vec = np.array(c["centroid"], dtype=np.float32)
            if vec.size != dim:
                raise ValueError(f"{key}: centroid dim {vec.size} != expected {dim}")
        else:
            raise ValueError(f"{key}: must provide either centroid_b64 or centroid list")

        vec = l2_normalize(vec)
        pts.append(ClusterPoint(ps, pe, key, vec, w))

    pts.sort(key=lambda p: (p.period_start, p.key))
    return embed_config_id, dim, dtype, pts


def months_ago_cutoff(dt: datetime, months: int) -> float:
    # MVP approximation: 30 days per month is OK for gating
    return dt.timestamp() - months * 30.0 * 24.0 * 3600.0


def run_simulation(
    points: List[ClusterPoint],
    T_attach: float,
    active_months: Optional[int],
) -> Dict[str, Any]:
    topics: List[TopicState] = []
    best_sims: List[float] = []
    decisions: List[Dict[str, Any]] = []

    for p in points:
        # pick candidate topics (MVP: all; Future: only recent active)
        candidates: List[TopicState]
        if active_months is None:
            candidates = topics
        else:
            cutoff = months_ago_cutoff(p.period_start, active_months)
            candidates = [t for t in topics if t.last_seen.timestamp() >= cutoff]

        if not candidates:
            # create first topic
            topic_id = len(topics)
            topics.append(TopicState(topic_id, p.vec.copy(), p.w, p.period_start))
            best_sims.append(float("nan"))
            decisions.append({
                "cluster_key": p.key,
                "period_start": p.period_start.strftime("%Y-%m-%d"),
                "best_sim": None,
                "decision": "create_new",
                "topic_id": topic_id
            })
            continue

        # compute cosine sims: mu are normalized so dot = cosine
        mus = np.stack([t.mu for t in candidates], axis=0)           # (M, D)
        sims = mus @ p.vec                                           # (M,)
        idx = int(np.argmax(sims))
        s_best = float(sims[idx])
        t_best = candidates[idx]

        if s_best >= T_attach:
            # attach: weighted mean then renormalize
            mu_raw = (t_best.mu * t_best.W + p.vec * p.w) / (t_best.W + p.w)
            t_best.mu = l2_normalize(mu_raw)
            t_best.W += p.w
            t_best.last_seen = p.period_start
            decision = "attach"
        else:
            # create new topic
            topic_id = len(topics)
            topics.append(TopicState(topic_id, p.vec.copy(), p.w, p.period_start))
            decision = "create_new"
            t_best = topics[-1]  # for logging
        best_sims.append(s_best)
        decisions.append({
            "cluster_key": p.key,
            "period_start": p.period_start.strftime("%Y-%m-%d"),
            "best_sim": s_best,
            "decision": decision,
            "topic_id": t_best.topic_id
        })

    sims_arr = np.array([x for x in best_sims if not np.isnan(x)], dtype=np.float32)
    return {
        "n_points": len(points),
        "n_topics_final": len(topics),
        "best_sims": sims_arr,
        "decisions": decisions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--embed-config-id", default=None)
    ap.add_argument("--active-months", type=int, default=None,
                    help="Future mode: only match topics active in last N months. MVP: omit for all topics ever.")
    ap.add_argument("--T-attach", type=float, required=True,
                    help="Attach threshold used in simulation (start with your candidate T_high).") # get from running test_t_threshold.py
    ap.add_argument("--suggest", default="90,95,97,99",
                    help="Percentiles to print as candidate thresholds from best-to-topic sims.")
    ap.add_argument("--dump-decisions", default=None,
                    help="Optional JSON output with per-cluster decisions and best_sim.")
    args = ap.parse_args()

    embed_id, dim, dtype, points = load_points(args.input, args.embed_config_id)
    out = run_simulation(points, T_attach=args.T_attach, active_months=args.active_months)

    sims = out["best_sims"]
    if sims.size == 0:
        raise RuntimeError("No similarities computed (need at least 2 points).")

    print(f"embed_config_id: {embed_id}")
    print(f"points (clusters): {out['n_points']}")
    print(f"final pseudo-topics: {out['n_topics_final']}")
    print(f"active_months: {args.active_months}")
    print(f"T_attach used: {args.T_attach:.4f}")
    print(f"min/median/mean/max: {float(np.min(sims)):.4f} / {float(np.median(sims)):.4f} / {float(np.mean(sims)):.4f} / {float(np.max(sims)):.4f}")

    ps = [float(x.strip()) for x in args.suggest.split(",") if x.strip()]
    print("\nCandidate thresholds from best-to-topic similarity distribution:")
    for p in ps:
        print(f"  p{int(p):02d}: {float(np.percentile(sims, p)):.4f}")

    if args.dump_decisions:
        payload = {
            "embed_config_id": embed_id,
            "dim": dim,
            "dtype": dtype,
            "active_months": args.active_months,
            "T_attach": args.T_attach,
            "n_points": out["n_points"],
            "n_topics_final": out["n_topics_final"],
            "decisions": out["decisions"],
        }
        with open(args.dump_decisions, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote decisions to: {args.dump_decisions}")


if __name__ == "__main__":
    main()
