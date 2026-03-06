"""Workflow register for report generation pipeline trace recording."""

from reader.pipelines.report_generation.workflow_register.models import (
    LoopNodeRecord,
    LoopRunStatus,
    StepNodeRecord,
    StepRunStatus,
    WorkflowNodeDef,
    WorkflowTraceReport,
)
from reader.pipelines.report_generation.workflow_register.register import WorkflowRegister
from reader.pipelines.report_generation.workflow_register.decorators import (
    record_loop,
    record_step,
    with_workflow_register,
)
__all__ = [
    "LoopNodeRecord",
    "LoopRunStatus",
    "StepNodeRecord",
    "StepRunStatus",
    "WorkflowNodeDef",
    "WorkflowTraceReport",
    "WorkflowRegister",
    "record_loop",
    "record_step",
    "with_workflow_register",
]
