"""Context-aware judge wrappers for report writer steps.

call_structured_with_judge_retry passes raw LLM output (ReportWriterSupplementOutput /
ReportWriterSectionOutput), but writer_supply and writer_writing validators expect
WriterSupplyJudgeInput / WriterWritingJudgeInput with context. These wrappers build
the full judge input from raw output + constructor-injected context and delegate.
"""

from __future__ import annotations

from typing import List

from reader.pipelines.report_generation.judges.judge import JudgeOutput, LLMJudge
from reader.pipelines.report_generation.judges.metrics.writer_supply import (
    WriterSupplyJudgeInput,
    hard_validate_writer_supply,
    soft_validate_writer_supply,
)
from reader.pipelines.report_generation.judges.metrics.writer_writing import (
    WriterWritingJudgeInput,
    hard_validate_writer_writing,
    soft_validate_writer_writing,
)
from reader.pipelines.report_generation.report import (
    ReportWriterSectionOutput,
    ReportWriterSupplementOutput,
)


# Inner judges that expect WriterSupplyJudgeInput / WriterWritingJudgeInput
_inner_supply_judge = LLMJudge[WriterSupplyJudgeInput](
    name="writer_supply",
    hard_validate=hard_validate_writer_supply,
    soft_validate=soft_validate_writer_supply,
)

_inner_writing_judge = LLMJudge[WriterWritingJudgeInput](
    name="writer_writing",
    hard_validate=hard_validate_writer_writing,
    soft_validate=soft_validate_writer_writing,
)


class WriterSupplyJudgeWrapper:
    """
    JudgeProtocol[ReportWriterSupplementOutput] that injects available_paper_ids
    and available_history_report_ids from constructor context.
    """

    def __init__(
        self,
        available_paper_ids: List[str],
        available_history_report_ids: List[str],
    ) -> None:
        self.name = "writer_supply"
        self._available_paper_ids = available_paper_ids
        self._available_history_report_ids = available_history_report_ids

    def judge(self, output: ReportWriterSupplementOutput) -> JudgeOutput:
        judge_input = WriterSupplyJudgeInput(
            output=output,
            available_paper_ids=self._available_paper_ids,
            available_history_report_ids=self._available_history_report_ids,
        )
        return _inner_supply_judge.judge(judge_input)

    def inject_warnings_into_prompt(
        self, prompt_base: str, judge_output: JudgeOutput
    ) -> tuple[str, int]:
        return _inner_supply_judge.inject_warnings_into_prompt(
            prompt_base, judge_output
        )

    def count_warnings(self, judge_output: JudgeOutput) -> int:
        return _inner_supply_judge.count_warnings(judge_output)

    def log_to_jsonl(
        self,
        log_path: str,
        item_pk: str,
        output: ReportWriterSupplementOutput,
        judge_output: JudgeOutput,
    ) -> None:
        _inner_supply_judge.log_to_jsonl(log_path, item_pk, output, judge_output)


class WriterWritingJudgeWrapper:
    """
    JudgeProtocol[ReportWriterSectionOutput] that injects outline_item and
    allowed_citations from constructor context.
    """

    def __init__(
        self,
        outline_item: str,
        allowed_citations: List[str],
    ) -> None:
        self.name = "writer_writing"
        self._outline_item = outline_item
        self._allowed_citations = allowed_citations

    def judge(self, output: ReportWriterSectionOutput) -> JudgeOutput:
        judge_input = WriterWritingJudgeInput(
            output=output,
            outline_item=self._outline_item,
            allowed_citations=self._allowed_citations,
        )
        return _inner_writing_judge.judge(judge_input)

    def inject_warnings_into_prompt(
        self, prompt_base: str, judge_output: JudgeOutput
    ) -> tuple[str, int]:
        return _inner_writing_judge.inject_warnings_into_prompt(
            prompt_base, judge_output
        )

    def count_warnings(self, judge_output: JudgeOutput) -> int:
        return _inner_writing_judge.count_warnings(judge_output)

    def log_to_jsonl(
        self,
        log_path: str,
        item_pk: str,
        output: ReportWriterSectionOutput,
        judge_output: JudgeOutput,
    ) -> None:
        _inner_writing_judge.log_to_jsonl(log_path, item_pk, output, judge_output)
