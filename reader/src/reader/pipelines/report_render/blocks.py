"""Core logic for report render: load, validate, signature check, TUI display."""

import hashlib
import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from reader.adapters import memo
from reader.adapters.memo import CheckReportSignatureRequest
from reader.pipelines.report_generation.report import ObservationReport
from reader.pipelines.report_render.config.config import RenderReportConfig
from reader.pipelines.report_render.model import RenderReportOutput
from reader.tui.report_viewer import display_report
from reader.logging.logging_setup import get_logger

logger = get_logger()


async def render_report(
    cluster_pk_hash: str,
    intent_mode: Optional[str],
    cfg: RenderReportConfig,
) -> RenderReportOutput:
    """
    Fetch report from memo, validate, optionally check signature, display in TUI.

    Returns:
        RenderReportOutput(status="done", message="report render finished successfully")
        on success; RenderReportOutput(status="error", message=...) on any error.
    """
    log_prefix = f"[render report] - [cluster_pk_hash={cluster_pk_hash}, intent_mode={intent_mode}]"
    logger.info(f"{log_prefix} start")
    try:
        # 2. Call memo get-report
        resp = await memo.get_report(cluster_pk_hash, cfg.memo)

        # 3. Validate memo return
        if resp.status != "ok":
            msg = resp.message or ""
            return RenderReportOutput(
                status="error",
                message=f"fetch from memo db error, memo status: {resp.status}, memo message: {msg}",
            )

        meta = resp.meta
        if meta is None:
            return RenderReportOutput(
                status="error",
                message=f"fetch from memo db error, memo status: ok but meta is None",
            )

        if intent_mode is not None:
            if meta.intent_mode != intent_mode:
                return RenderReportOutput(
                    status="error",
                    message=f"found report for cluster {cluster_pk_hash}, but intent does not match, expect: {intent_mode}, got: {meta.intent_mode}",
                )

        # 4. Check file exists
        report_path = Path(meta.report_url)
        if not report_path.exists():
            return RenderReportOutput(
                status="error",
                message=f"report file {report_path} does not exist",
            )

        # 5. Load and compute signature
        json_bytes = report_path.read_bytes()
        try:
            obs = ObservationReport.model_validate_json(json_bytes.decode("utf-8"))
        except ValidationError as e:
            logger.error(f"{log_prefix} report model validation failed: {e}", exc_info=True)
            return RenderReportOutput(
                status="error",
                message=f"report model validation failed.",
            )

        # Recompute signature same as _save_report_to_fs
        canonical_json = json.dumps(
            obs.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        signature = hashlib.sha256(canonical_json).hexdigest()

        # 6. Check signature
        sig_resp = await memo.check_report_signature(
            CheckReportSignatureRequest(cluster_pk_hash=cluster_pk_hash, signature=signature),
            cfg.memo,
        )

        # 7. Signature result – log warning only, do not fail
        if sig_resp.status in ("error", "not_match"):
            logger.warning(
                f"{log_prefix} Report signature check failed: status={sig_resp.status}, message={sig_resp.message}"
            )

        # 8. Display TUI
        await display_report(obs)

        logger.info(f"{log_prefix} finished")

        # 9. Success
        return RenderReportOutput(
            status="done",
            message="report render finished successfully",
        )

    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error: {e}", exc_info=True)
        return RenderReportOutput(
            status="error",
            message=f"unexpected error, check log {cfg.cache.render_report_log_path} for debugging",
        )
