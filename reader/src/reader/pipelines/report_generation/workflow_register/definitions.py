"""Workflow node definitions for report generation pipelines."""

from reader.pipelines.report_generation.workflow_register.models import WorkflowNodeDef

KICK_OFF_REPORT_JOB_NODES = [
    WorkflowNodeDef(
        id="resolve_report_topic",
        kind="step",
        display_name="Resolve report topic",
    ),
    WorkflowNodeDef(
        id="fetch_report_generation_metadata",
        kind="step",
        display_name="Fetch report generation metadata",
    ),
    WorkflowNodeDef(
        id="initialize_report_generation_llm_client",
        kind="step",
        display_name="Initialize LLM client",
    ),
    WorkflowNodeDef(
        id="generate_report_plan",
        kind="step",
        display_name="Generate report plan",
        contains=["run_evidence_completion_loop"],
    ),
    WorkflowNodeDef(
        id="run_evidence_completion_loop",
        kind="loop",
        display_name="Evidence completion loop",
    ),
    WorkflowNodeDef(
        id="planner_judge_loop",
        kind="loop",
        display_name="Planner judge retry loop",
    ),
    WorkflowNodeDef(
        id="generate_report_body",
        kind="step",
        display_name="Generate report body",
        contains=["run_writing_loop"],
    ),
    WorkflowNodeDef(
        id="run_writing_loop",
        kind="loop",
        display_name="Writing loop",
    ),
    WorkflowNodeDef(
        id="writer_supply_judge_loop",
        kind="loop",
        display_name="Writer supply judge retry loop",
    ),
    WorkflowNodeDef(
        id="writer_writing_judge_loop",
        kind="loop",
        display_name="Writer writing judge retry loop",
    ),
    WorkflowNodeDef(
        id="generate_report_front_matter",
        kind="step",
        display_name="Generate report front matter",
    ),
    WorkflowNodeDef(
        id="front_matter_judge_loop",
        kind="loop",
        display_name="Front matter judge retry loop",
    ),
    WorkflowNodeDef(
        id="save_report_to_fs",
        kind="step",
        display_name="Save report to local FS",
    ),
    WorkflowNodeDef(
        id="save_report_to_db",
        kind="step",
        display_name="Save report to database",
    ),
]
