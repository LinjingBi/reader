"""Report generation pipeline orchestration"""

from typing import Optional

from reader.pipelines.report_generation.config.config import ReportGenerationConfig
from reader.adapters import memo
from reader.pipelines.report_generation.blocks import start_generation, select_cluster_and_intent, ReportGenerationRuntimeError
from reader.pipelines.report_generation.db.store import init_report_job, InitReportJobResponseNextStatus
from reader.pipelines.report_generation.report import ReportJobAction, ReportJobOutput, ReportJobOutputMeta
from reader.pipelines.report_generation.prompts.planner.build import UserIntent
from reader.logging.logging_setup import get_logger

logger = get_logger()

# contain potential mcp servers/tool calls/skills(?)
async def generate_report(cfg: ReportGenerationConfig) -> ReportJobOutput:
    """
    Run the report generation pipeline for the configured period.
    Async and non-blocking for report generation LLM calls.

    Args:
        cfg: ReportGenerationConfig instance

    Returns:
        ReportJobOutput with action, message, and meta (only when action is FETCH_REPORT).
        On error, returns ReportJobOutput with action DEBUG_INTERNAL_ERROR or DEBUG_FOR_RERUN.
    """
    log_prefix = f"[generate report] - "
    logger.info(f"{log_prefix}start")
    
    try:
        # Get period dates and source from config
        source = cfg.run.source
        period_start = cfg.run.period_start
        period_end = cfg.run.period_end

        # fetch cluster observations
        logger.info(f"{log_prefix}Memo get-clusters-observation started: snapshot_id={source}|{period_start}|{period_end}")
        clusters_observation = await memo.get_clusters_observation(source, period_start, period_end, cfg.memo)
        logger.info(f"{log_prefix}Memo get-clusters-observation successful: snapshot_id={source}|{period_start}|{period_end}")
        # display cluster observations for user to decide which cluster and intent to generate report for in tui
        selected_pk_hash, selected_intent_enum = await select_cluster_and_intent(clusters_observation)
        logger.info(f"{log_prefix}Selected pk_hash: {selected_pk_hash}, Selected intent: {selected_intent_enum}")
    except Exception as e:
        logger.error(f"{log_prefix}Internal error. {e}", exc_info=True)
        return ReportJobOutput(action=ReportJobAction.DEBUG_INTERNAL_ERROR, message=f"unexpected error. check log under {cfg.cache.report_generation_log_path} for debugging.", meta=None)
    try:
        # start report generation
        resp = await _trigger_report_job(selected_pk_hash, selected_intent_enum, cfg)
        if resp is None:
            return ReportJobOutput(action=ReportJobAction.DEBUG_INTERNAL_ERROR, message=f"unexpected empty response. check log under {cfg.cache.report_generation_log_path} for debugging.", meta=None)
        else:
            logger.info(f"{log_prefix}finished, message: {resp.message}")
            print(f"resp: {resp}")  # debug only
            return resp
    except ReportGenerationRuntimeError as e:
        logger.error(f"{log_prefix} report generation runtime error: {e}", exc_info=True)
        return ReportJobOutput(
            action=ReportJobAction.DEBUG_FOR_RERUN,
            message=f"error happened during new report generation: {e}\n"
            f"check log under {cfg.cache.report_generation_log_path} and cache under {cfg.cache.report_generation_cache} for minimal-effort rerun.\n"
            f"search filter: cluster_pk_hash={selected_pk_hash}, user_intent={selected_intent_enum.value}",
            meta=None,
        )
    except Exception as e:
        logger.error(f"{log_prefix}Internal error: {e}", exc_info=True)
        return ReportJobOutput(action=ReportJobAction.DEBUG_INTERNAL_ERROR, message=f"unexpected error. check log under {cfg.cache.report_generation_log_path} for debugging.", meta=None)


async def _trigger_report_job(cluster_pk_hash: str, user_intent: UserIntent, cfg: ReportGenerationConfig) -> Optional[ReportJobOutput]:
    """
    trigger a report generation job based on its current status.
    Non-blocking: uses async LLM calls when executor is configured.

    Calls init_report_job (database) and handles/logs the response. Maps next_status:
    - running: kick off new job, returns ReportJobOutput(FETCH_REPORT, message, meta)
    - resuming: job ready to resume (error expired), returns ReportJobOutput(WAIT_FOR_JOB_TO_RESUME, message, None)
    - waiting: returns ReportJobOutput(WAIT_FOR_JOB_TO_FINISH, meta.message, None)
    - done: returns ReportJobOutput(FETCH_REPORT, message, meta)

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        user_intent: User intent enum
        cfg: ReportGenerationConfig instance

    Returns:
        ReportJobOutput with action, message, and meta (only when action is FETCH_REPORT).
    """
    resp = init_report_job(cluster_pk_hash, cfg)
    meta = resp.meta

    if resp.next_status == InitReportJobResponseNextStatus.RUNNING:
        await start_generation(cluster_pk_hash, user_intent, cfg)
        next_status = ReportJobAction.FETCH_REPORT
        msg = f"A new report is just generated."
        return ReportJobOutput(
            action=next_status,
            message=msg,
            meta=ReportJobOutputMeta(cluster_pk_hash=cluster_pk_hash, intent_mode=user_intent.name.lower()),
        )
    elif resp.next_status == InitReportJobResponseNextStatus.RESUMING:
        next_status = ReportJobAction.WAIT_FOR_JOB_TO_RESUME
        return ReportJobOutput(action=next_status, message=meta.message, meta=None)
    elif resp.next_status == InitReportJobResponseNextStatus.WAITING:
        next_status = ReportJobAction.WAIT_FOR_JOB_TO_FINISH
        return ReportJobOutput(action=next_status, message=meta.message, meta=None)
    elif resp.next_status == InitReportJobResponseNextStatus.DONE:
        next_status = ReportJobAction.FETCH_REPORT
        msg = f"{meta.message}"
        return ReportJobOutput(
            action=next_status,
            message=msg,
            meta=ReportJobOutputMeta(cluster_pk_hash=cluster_pk_hash, intent_mode=user_intent.name.lower()),
        )
    else:
        msg = f"Unexpected next_status={resp.next_status} from init_report_job, message={meta.message}"
        raise ValueError(f"{msg}")
