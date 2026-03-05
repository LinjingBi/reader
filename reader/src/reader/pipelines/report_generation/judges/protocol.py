"""Judge protocol and related types for report generation."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JudgeLoopExitCondition(str, Enum):
    """Reason the judge retry loop concluded."""

    JUDGE_ACCEPTED = "judge_accepted"
    RETRIES_EXHAUSTED = "retries_exhausted"
    LLM_ERROR = "llm_error"
    ERROR = "error"


class JudgeResultProtocol(Protocol):
    """Minimal interface for judge output - only needs overall score."""

    overall: float


class JudgeProtocol(Protocol[T]):
    """Protocol for judge used by call_structured_with_judge_retry."""

    name: str

    def judge(self, output: T) -> JudgeResultProtocol: ...

    def inject_warnings_into_prompt(
        self, prompt_base: str, judge_output: JudgeResultProtocol
    ) -> tuple[str, int]: ...

    def count_warnings(self, judge_output: JudgeResultProtocol) -> int: ...

    def log_to_jsonl(
        self,
        log_path: str,
        item_pk: str,
        output: T,
        judge_output: JudgeResultProtocol,
    ) -> None: ...
