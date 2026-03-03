"""Report generation pipeline judges."""

from reader.pipelines.report_generation.judges.judge import JudgeOutput, LLMJudge
from reader.pipelines.report_generation.judges.metrics.planner import (
    hard_validate_planner_output,
    soft_validate_planner_output,
)

planner_judge = LLMJudge(
    name="planner",
    hard_validate=hard_validate_planner_output,
    soft_validate=soft_validate_planner_output,
)

__all__ = [
    "JudgeOutput",
    "LLMJudge",
    "planner_judge",
]
