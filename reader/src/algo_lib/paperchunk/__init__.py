"""paperchunk: async paper fetch + heading-block parsing + selector matching + scoring/training outputs.

This is a schema-first scaffold (MVP) based on the agreed dataclasses and interfaces.
"""

from .types import (
    PaperId, Url, SelectorId, TextId, SourceType,
    FetchResult, HeadingBlock, ParseResult,
    AliasHit, HeadingMatch, DebugHeadingEvent,
    ScoreRow, ScoreSummary, ScoreOutput,
    TrainSummary, TrainOutput,
    TextTable,
)
from .configs import EngineConfig, ScoreConfig, TrainConfig
from .scoring import run_scoring
from .training import run_training
from .clean import clean_v1

__version__ = "0.1.0"

__all__ = [
    # ids / enums
    "PaperId", "Url", "SelectorId", "TextId", "SourceType",
    # core types
    "FetchResult", "HeadingBlock", "ParseResult",
    "AliasHit", "HeadingMatch", "DebugHeadingEvent",
    # outputs
    "ScoreRow", "ScoreSummary", "ScoreOutput",
    "TrainSummary", "TrainOutput",
    "TextTable",
    # configs / entrypoints
    "EngineConfig", "ScoreConfig", "TrainConfig",
    "run_scoring", "run_training",
    "clean_v1",
]
