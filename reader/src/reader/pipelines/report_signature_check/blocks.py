"""Core logic for report signature check: load, validate, compute signature, verify via memo."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from reader.adapters import memo
from reader.adapters.memo import CheckReportSignatureRequest
from reader.pipelines.report_generation.report import ObservationReport
from reader.pipelines.report_signature_check.config.config import ReportSignatureConfig
from reader.pipelines.report_signature_check.model import CheckReportSignatureOutput
from reader.logging.logging_setup import get_logger

logger = get_logger()


async def check_report_signature(
    cfg: ReportSignatureConfig,
    report_file: str,
) -> CheckReportSignatureOutput:
    """
    Load report from file, validate, compute signature, verify via memo.

    Returns:
        CheckReportSignatureOutput with status (match/not_match/error) and message.
    """
    path = Path(report_file)
    log_prefix = f"[check report signature] - [report_file={path}]"
    logger.info(f"{log_prefix} start")
    try:
        # 1. Check file exists
        if not path.exists():
            return CheckReportSignatureOutput(
                status="error",
                message=f"report file {path} does not exist",
            )

        # 2. Load and validate
        json_bytes = path.read_bytes()
        try:
            obs = ObservationReport.model_validate_json(json_bytes.decode("utf-8"))
        except ValidationError as e:
            logger.error(f"{log_prefix} report model validation failed: {e}", exc_info=True)
            return CheckReportSignatureOutput(
                status="error",
                message="report file validation failed. check log {cfg.cache.report_signature_check_log_path} for details.",
            )

        # 3. Extract cluster_pk_hash and compute signature
        cluster_pk_hash = obs.cluster_pk_hash
        canonical_json = json.dumps(
            obs.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        signature = hashlib.sha256(canonical_json).hexdigest()

        # 4. Call memo check_report_signature
        sig_resp = await memo.check_report_signature(
            CheckReportSignatureRequest(cluster_pk_hash=cluster_pk_hash, signature=signature),
            cfg.memo,
        )

        logger.info(f"{log_prefix} finished: status={sig_resp.status}")

        return CheckReportSignatureOutput(
            status=sig_resp.status,
            message=sig_resp.message,
        )

    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error: {e}", exc_info=True)
        return CheckReportSignatureOutput(
            status="error",
            message=f"unexpected error, check log {cfg.cache.report_signature_check_log_path} for debugging",
        )
