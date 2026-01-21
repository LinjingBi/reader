"""
Protocol-based typing for algo_lib.

Defines PaperLike Protocol to decouple algorithms from fetch.Paper.
"""

from __future__ import annotations
from typing import Protocol


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

