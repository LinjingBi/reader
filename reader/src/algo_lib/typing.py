"""
Protocol-based typing for algo_lib.

Defines PaperLike Protocol to decouple algorithms from fetch.Paper.
"""

from __future__ import annotations
from typing import Protocol, Dict, List
from dataclasses import dataclass

class PaperLike(Protocol):
    """
    Protocol defining the interface for paper-like objects.
    
    algo_lib code should type against Sequence[PaperLike],
    never importing fetch.Paper directly.
    """
    pid: str
    title: str
    summary: str
    keywords: list[str]
    url: str
    published_at: str


@dataclass
class BestClusteringResult:
    """
    Result of best clustering selection.
    
    Used as return type for clustering selection functions.
    """
    mode: str
    k: int
    cluster_members_ordered: Dict[int, List[int]]  # cluster_label -> list of paper indices
    cluster_members_similarities: Dict[int, Dict[int, float]]  # cluster_label -> paper_index -> similarity to centroid
    cluster_cohesion: Dict[int, float]  # cluster_label -> cohesion
