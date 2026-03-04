"""Writer prompt builders for report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from reader.pipelines.report_generation.report import (
    ReportWriterSectionInput,
    ReportWriterSectionOutput,
    ReportWriterSupplementInput,
)


def _load_template(template_name: str) -> str:
    """Load template from prompts/writer directory."""
    template_dir = Path(__file__).parent
    path = template_dir / template_name
    return path.read_text(encoding="utf-8")


def build_evidence_requests_prompt(
    supplement_input: ReportWriterSupplementInput,
    template_name: str,
) -> str:
    """Build prompt for supply step (decide what supplements to request)."""
    template = _load_template(template_name)
    input_dict = supplement_input.model_dump()
    input_json = json.dumps(input_dict, indent=2, ensure_ascii=False)
    return template.replace("<INPUT_JSON>", input_json)


def build_section_writing_prompt(
    section_input: ReportWriterSectionInput,
    template_name: str,
) -> str:
    """Build prompt for write step (write one section with supplements)."""
    template = _load_template(template_name)
    input_dict = section_input.model_dump()
    input_json = json.dumps(input_dict, indent=2, ensure_ascii=False)
    return template.replace("<INPUT_JSON>", input_json)


def build_summary_writing_prompt(
    sections: List[ReportWriterSectionOutput],
    template_name: str,
) -> str:
    """Build prompt for front matter (summary) step from report body sections."""
    report_body_parts = [
        f"## {s.section_name}\n\n{s.section_text}" for s in sections
    ]
    report_body_text = "\n\n".join(report_body_parts)
    template = _load_template(template_name)
    return template.replace("<REPORT_BODY_TEXT>", report_body_text)
