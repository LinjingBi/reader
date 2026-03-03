"""LLMJudge class - generic judge for LLM outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Generic, List, Tuple, TypeVar

from pydantic import BaseModel

from reader.logging.logging_setup import get_logger

from reader.pipelines.report_generation.judges.metrics.planner import ValidationReport

T = TypeVar("T", bound=BaseModel)


@dataclass
class JudgeOutput:
    """Output from LLMJudge.judge() matching HeuristicResult format."""
    sub_scores: Dict[str, float]  # All rule scores (0.0 or 1.0 for bools)
    overall: float  # 0.0 if any must-pass fails, else 1.0 + soft_schema_valid.score
    reasons: Dict[str, List[Tuple[str, str]]]  # (rule_declaration, received_fact) per rule group


class LLMJudge(Generic[T]):
    """
    Generic judge for LLM outputs. Configurable via constructor-injected
    hard and soft validate functions. Works with any output class from report.py
    (e.g. LLMReportPlannerOutput, ReportWriterSupplementOutput, etc.).
    """

    def __init__(
        self,
        name: str,
        hard_validate: Callable[[T], ValidationReport],
        soft_validate: Callable[[T], ValidationReport],
    ) -> None:
        self.name = name
        self._hard_validate = hard_validate
        self._soft_validate = soft_validate

    def judge(self, judge_input: T) -> JudgeOutput:
        """
        Judge the given input using hard and soft validation. Pure logic only, no logging.
        """
        if judge_input is None:  # pyright: ignore[reportUnreachable]
            raise ValueError("judge_input must not be None")
        sub_scores: Dict[str, float] = {}
        reasons: Dict[str, List[Tuple[str, str]]] = {}

        hard_result = self._hard_validate(judge_input)
        sub_scores["hard_schema_valid"] = hard_result.score
        if hard_result.reasons:
            reasons["hard_schema_valid"] = hard_result.reasons

        soft_result = self._soft_validate(judge_input)
        sub_scores["soft_schema_valid"] = soft_result.score
        if soft_result.reasons:
            reasons["soft_schema_valid"] = soft_result.reasons

        must_pass_rules = ["hard_schema_valid"]
        must_pass_failed = any(sub_scores.get(rule, 0.0) == 0.0 for rule in must_pass_rules)

        if must_pass_failed:
            overall = 0.0
        else:
            overall = 1.0 + sub_scores["soft_schema_valid"]

        return JudgeOutput(
            sub_scores=sub_scores,
            overall=overall,
            reasons=reasons,
        )

    def log_to_jsonl(
        self,
        log_path: str,
        item_pk: str,
        judge_input: T,
        judge_output: JudgeOutput,
    ) -> None:
        """Append (judge_input, judge_output) to JSONL file. No-op if log_path is None or empty."""
        if not log_path:
            return
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "cluster_pk_hash": item_pk,
            "date": datetime.now(timezone.utc).isoformat(),
            "judge_input": judge_input.model_dump() if judge_input is not None else None,
            "judge_output": asdict(judge_output),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger = get_logger()
        logger.info(f"Successfully appended judge input for cluster {item_pk} to {log_path}")

    def count_warnings(self, judge_output: JudgeOutput) -> int:
        """Count unique rule declarations from judge_output.reasons."""
        if not judge_output.reasons:
            return 0
        seen: set[str] = set()
        count = 0
        for rule_list in judge_output.reasons.values():
            for rule_declaration, _ in rule_list:
                if rule_declaration not in seen:
                    seen.add(rule_declaration)
                    count += 1
        return count

    def inject_warnings_into_prompt(self, prompt_base: str, judge_output: JudgeOutput) -> tuple[str, int]:
        """
        Append a WARNING section with failed rule declarations to the base prompt.
        Returns (prompt_base + warning_block, num_warnings).
        """
        if not judge_output.reasons:
            return (prompt_base, 0)
        seen: set[str] = set()
        rule_declarations: list[str] = []
        for rule_list in judge_output.reasons.values():
            for rule_declaration, _ in rule_list:
                if rule_declaration not in seen:
                    seen.add(rule_declaration)
                    rule_declarations.append(rule_declaration)

        if not rule_declarations:
            return (prompt_base, 0)

        lines = ["\n\nWARNING:"]
        for rule in rule_declarations:
            lines.append(f"- {rule}")
        warning_block = "\n".join(lines)
        return (prompt_base + warning_block, len(rule_declarations))
