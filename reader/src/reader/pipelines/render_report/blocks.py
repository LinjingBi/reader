"""Core logic for render report: load from file, validate, display in TUI."""

from pathlib import Path

from pydantic import ValidationError

from reader.pipelines.report_generation.report import ObservationReport
from reader.pipelines.render_report.config.config import RenderReportConfig
from reader.pipelines.render_report.model import RenderReportOutput
from reader.tui.report_viewer import display_report
from reader.logging.logging_setup import get_logger

logger = get_logger()


async def render_report(
    cfg: RenderReportConfig,
    report_file: str,
) -> RenderReportOutput:
    """
    Load report from file, validate, display in TUI.

    Returns:
        RenderReportOutput(status="done", message="...") on success;
        RenderReportOutput(status="error", message=...) on any error.
    """
    path = Path(report_file)
    log_prefix = f"[render report] - [report={path}]"
    logger.info(f"{log_prefix} start")
    try:
        # 1. Check file exists
        if not path.exists():
            return RenderReportOutput(
                status="error",
                message=f"report file {path} does not exist",
            )

        # 2. Load and validate
        json_bytes = path.read_bytes()
        try:
            obs = ObservationReport.model_validate_json(json_bytes.decode("utf-8"))
        except ValidationError as e:
            logger.error(f"{log_prefix} report file validation failed: {e}", exc_info=True)
            return RenderReportOutput(
                status="error",
                message="report file validation failed. check log {cfg.cache.render_report_log_path} for details.",
            )

        # 3. Display TUI
        await display_report(obs)

        logger.info(f"{log_prefix} finished")

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
