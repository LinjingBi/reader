"""Domain models for reader package"""

from dataclasses import dataclass
from typing import List


@dataclass
class Paper:
    """Paper model representing a research paper"""
    pid: str
    title: str
    summary: str
    keywords: List[str]
    url: str = ""
    published_at: str = ""
