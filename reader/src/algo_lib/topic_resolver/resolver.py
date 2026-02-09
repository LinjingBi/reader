"""
Topic resolver implementation.

Compares cluster centroids to topic centroids using cosine similarity,
selects the best matching topic, and decides whether to merge or create
a new topic based on a similarity threshold.
"""

from __future__ import annotations

import base64

import numpy as np

from algo_lib.topic_resolver.errors import TopicResolverError
from algo_lib.topic_resolver.models import (
    ClusterInput,
    TopicInput,
    TopicResolveAction,
    TopicResolveOutput,
)


def decode_centroid_b64(b64: str) -> np.ndarray:
    """
    Decode base64-encoded centroid to numpy array.
    
    Only supports f32_le (little-endian float32) format.
    
    Args:
        b64: Base64-encoded string
    
    Returns:
        Decoded numpy array (float32)
    
    Raises:
        TopicResolverError: If base64 decode fails or unsupported dtype
    """
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise TopicResolverError(f"Failed to decode base64: {e}") from e
    
    try:
        vec = np.frombuffer(raw, dtype="<f4")  # little-endian float32
    except Exception as e:
        raise TopicResolverError(
            f"Unsupported dtype. Only 'f32_le' (little-endian float32) is supported. Error: {e}"
        ) from e
    
    return vec.astype(np.float32, copy=False)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2 normalize a vector.
    
    Args:
        v: Input vector
        eps: Minimum norm threshold (raises error if norm < eps)
    
    Returns:
        Normalized vector
    
    Raises:
        TopicResolverError: If vector norm is too small
    """
    n = float(np.linalg.norm(v))
    if n < eps:
        raise TopicResolverError(
            "Zero / near-zero vector encountered; cannot normalize."
        )
    return v / n


def encode_centroid_b64(vec: np.ndarray) -> str:
    """
    Encode numpy array to base64 string.
    
    Args:
        vec: Numpy array to encode (will be converted to float32 little-endian)
    
    Returns:
        Base64-encoded string
    """
    # Convert to float32 little-endian bytes
    centroid_bytes = vec.astype("<f4").tobytes()
    # Encode to base64
    return base64.b64encode(centroid_bytes).decode("utf-8")


def resolve_topic(
    topics: list[TopicInput],
    cluster: ClusterInput,
    resolve_threshold: float,
) -> TopicResolveOutput:
    """
    Resolve a cluster to either merge into an existing topic or create a new topic.
    
    Compares the cluster's centroid against all topic centroids using cosine similarity,
    finds the best matching topic, and decides based on the similarity threshold whether
    to merge or create a new topic.
    
    Note: Only supports f32_le (little-endian float32) format for centroid encoding.
    
    Args:
        topics: List of existing topics to compare against
        cluster: Cluster to resolve
        resolve_threshold: Similarity threshold (0-1). If best similarity >= threshold, merge; otherwise create
    
    Returns:
        TopicResolveOutput with action, merge_to_topic (if merge), new centroid, and weight
    
    Raises:
        TopicResolverError: For various error conditions (invalid base64, unsupported dtype, normalization failure, etc.)
    """
    # Validate threshold
    if not (0.0 <= resolve_threshold <= 1.0):
        raise TopicResolverError(
            f"resolve_threshold must be in [0, 1], got {resolve_threshold}"
        )
    
    # Decode and normalize cluster centroid
    try:
        cluster_vec = decode_centroid_b64(cluster.centroid_b64)
        cluster_vec = l2_normalize(cluster_vec)
    except TopicResolverError:
        raise
    except Exception as e:
        raise TopicResolverError(f"Failed to decode/normalize cluster centroid: {e}") from e
    
    # Handle empty topics list - automatically create new topic
    if not topics:
        new_centroid_b64 = encode_centroid_b64(cluster_vec)
        return TopicResolveOutput(
            action=TopicResolveAction.CREATE,
            merge_to_topic=None,
            new_topic_centroid_b64=new_centroid_b64,
            new_topic_weight=cluster.centroid_weight,
            score=1.0,
        )
    
    # Decode and normalize all topic centroids
    topic_vecs = []
    topic_weights = []
    topic_ids = []
    
    for topic in topics:
        try:
            topic_vec = decode_centroid_b64(topic.centroid_b64)
            topic_vec = l2_normalize(topic_vec)
            topic_vecs.append(topic_vec)
            topic_weights.append(topic.centroid_weight)
            topic_ids.append(topic.id)
        except TopicResolverError:
            raise
        except Exception as e:
            raise TopicResolverError(
                f"Failed to decode/normalize topic {topic.id} centroid: {e}"
            ) from e
    
    # Compute cosine similarities: stack topic centroids, compute dot product
    # Since vectors are normalized, dot product = cosine similarity
    mus = np.stack(topic_vecs, axis=0)  # (M, D)
    sims = mus @ cluster_vec  # (M,)
    
    # Find top 1 topic
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])
    best_topic_id = topic_ids[idx]
    best_topic_vec = topic_vecs[idx]
    best_topic_weight = topic_weights[idx]
    
    # Threshold decision
    if best_sim >= resolve_threshold:
        # Merge: compute weighted mean then renormalize
        mu_raw = (
            best_topic_vec * best_topic_weight + cluster_vec * cluster.centroid_weight
        ) / (best_topic_weight + cluster.centroid_weight)
        merged_vec = l2_normalize(mu_raw)
        merged_weight = best_topic_weight + cluster.centroid_weight
        new_centroid_b64 = encode_centroid_b64(merged_vec)
        
        return TopicResolveOutput(
            action=TopicResolveAction.MERGE,
            merge_to_topic=best_topic_id,
            new_topic_centroid_b64=new_centroid_b64,
            new_topic_weight=merged_weight,
            score=best_sim,
        )
    else:
        # Create new topic: use cluster's normalized centroid
        new_centroid_b64 = encode_centroid_b64(cluster_vec)
        return TopicResolveOutput(
            action=TopicResolveAction.CREATE,
            merge_to_topic=None,
            new_topic_centroid_b64=new_centroid_b64,
            new_topic_weight=cluster.centroid_weight,
            score=1.0,
        )

