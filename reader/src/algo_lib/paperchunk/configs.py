from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Tuple

from .types import SelectorId

@dataclass(frozen=True)
class EngineConfig:
    concurrency: int = 16
    timeout_s: float = 30.0

    prefer: Literal["html", "pdf", "auto"] = "auto"

    required_selectors: Tuple[SelectorId, ...] = ("summary", "introduction", "method", "conclusion")

@dataclass(frozen=True)
class ScoreConfig:
    normalize_within_selector: bool = True

    # Applied only to mapped blocks.
    clean_text_fn: Callable[[str], str] = lambda s: s

@dataclass(frozen=True)
class TrainConfig:
    enable_unmapped_candidates: bool = True
    enable_proposals: bool = True
