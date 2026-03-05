"""WorkflowRegister - records step/loop status and outputs, generates trace report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from reader.pipelines.report_generation.config.config import ReportGenerationConfig
from reader.pipelines.report_generation.workflow_register.models import (
    LoopNodeRecord,
    LoopRunStatus,
    StepNodeRecord,
    StepRunStatus,
    WorkflowNodeDef,
    WorkflowTraceNode,
    WorkflowTraceReport,
)


def _serialize_output(output: Any) -> dict | None:
    """
    Serialize output for storage in trace.
    - Pydantic models: model_dump(mode="json")
    - LLMClient: non-serializable, store placeholder
    - None: return None
    - List of models: serialize each element
    - dict: returned as-is
    Raises TypeError for unsupported types.
    """
    if output is None:
        return None

    # LLMClient is non-serializable
    if type(output).__name__ == "LLMClient":
        return {"_type": "LLMClient", "initialized": True}

    # Pydantic models
    if hasattr(output, "model_dump"):
        return output.model_dump(mode="json")

    # List of Pydantic models
    if isinstance(output, list):
        return [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in output
        ]

    # Dict is already serializable
    if isinstance(output, dict):
        return output

    raise TypeError(
        f"Unsupported output type for trace serialization: {type(output).__name__}. "
        "Supported: None, Pydantic models, list of Pydantic models, dict, LLMClient."
    )


class WorkflowRegister:
    """Records step/loop status and outputs; generates trace report for rerun."""

    def __init__(
        self,
        workflow_id: str,
        node_defs: list[WorkflowNodeDef],
        cache_path: Path,
        cluster_pk_hash: str,
    ):
        self.workflow_id = workflow_id
        self.cluster_pk_hash = cluster_pk_hash
        self.node_defs = {n.id: n for n in node_defs}
        self._node_defs_ordered = list(node_defs)
        self.cache_path = Path(cache_path)
        self.records: dict[str, Union[StepNodeRecord, LoopNodeRecord]] = {}
        for n in node_defs:
            if n.kind == "step":
                self.records[n.id] = StepNodeRecord(node_id=n.id)
            else:
                self.records[n.id] = LoopNodeRecord(node_id=n.id)

    def record_step(self, node_id: str, status: Any, output: Any) -> None:
        """Record step completion. status: StepTerminationStatus (done/error)."""
        if node_id not in self.records:
            return
        rec = self.records[node_id]
        if not isinstance(rec, StepNodeRecord):
            return
        rec.status = StepRunStatus(status.value) if hasattr(status, "value") else StepRunStatus(status)
        rec.output = _serialize_output(output) if output is not None else None

    def record_loop(self, node_id: str, status: Any, output: Any) -> None:
        """Record loop completion. status: LoopRunStatus (complete/partial/error)."""
        if node_id not in self.records:
            return
        rec = self.records[node_id]
        if not isinstance(rec, LoopNodeRecord):
            return
        rec.status = LoopRunStatus(status.value) if hasattr(status, "value") else LoopRunStatus(status)
        rec.output = _serialize_output(output) if output is not None else None

    def get_trace_report(self, config: ReportGenerationConfig) -> WorkflowTraceReport:
        """Build full trace report including config."""
        nodes: list[WorkflowTraceNode] = []
        for node_def in self._node_defs_ordered:
            rec = self.records.get(node_def.id)
            if rec is None:
                continue
            nodes.append(
                WorkflowTraceNode(
                    id=node_def.id,
                    kind=node_def.kind,
                    display_name=node_def.display_name,
                    contains=node_def.contains,
                    status=rec.status.value,
                    output=rec.output,
                )
            )
        return WorkflowTraceReport(
            workflow_id=self.workflow_id,
            cluster_pk_hash=self.cluster_pk_hash,
            timestamp=_local_iso_now(),
            config=config.model_dump(mode="json"),
            nodes=nodes,
        )

    def write_trace_to_cache(self, config: ReportGenerationConfig) -> Path:
        """Write trace to cache path. Creates parent dirs. Returns path written."""
        report = self.get_trace_report(config)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        return self.cache_path


def _local_iso_now() -> str:
    """Local time as ISO 8601 with timezone offset. Parse with datetime.fromisoformat() and convert to UTC via .astimezone(timezone.utc)."""
    return datetime.now().astimezone().isoformat()
