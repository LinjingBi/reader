"""Report generation pipeline judges."""

from reader.pipelines.report_generation.judges.judge import JudgeOutput, LLMJudge
from reader.pipelines.report_generation.judges.metrics.planner import (
    hard_validate_planner_output,
    soft_validate_planner_output,
)
from reader.pipelines.report_generation.judges.metrics.writer_summary import (
    hard_validate_writer_summary,
    soft_validate_writer_summary,
)
from reader.pipelines.report_generation.judges.writer_wrappers import (
    WriterSupplyJudgeWrapper,
    WriterWritingJudgeWrapper,
)

planner_judge = LLMJudge(
    name="planner",
    hard_validate=hard_validate_planner_output,
    soft_validate=soft_validate_planner_output,
)

front_matter_judge = LLMJudge(
    name="front_matter",
    hard_validate=hard_validate_writer_summary,
    soft_validate=soft_validate_writer_summary,
)

__all__ = [
    "JudgeOutput",
    "LLMJudge",
    "planner_judge",
    "front_matter_judge",
    "WriterSupplyJudgeWrapper",
    "WriterWritingJudgeWrapper",
]
