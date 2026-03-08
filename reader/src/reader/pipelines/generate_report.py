"""Report generation pipeline orchestration"""

from typing import Tuple

from reader.pipelines.report_generation.config.config import ReportGenerationConfig
from reader.adapters import memo
from reader.pipelines.report_generation.blocks import start_generation, select_cluster_and_intent
from reader.pipelines.report_generation.db.store import init_report_job
from reader.pipelines.report_generation.report import ReportJobAction
from reader.pipelines.report_generation.prompts.planner.build import UserIntent
from reader.logging.logging_setup import get_logger

logger = get_logger()

# contain potential mcp servers/tool calls/skills(?)
async def generate_report(cfg: ReportGenerationConfig) -> Tuple[ReportJobAction, str]:
    """
    Run the report generation pipeline for the configured period.
    Async and non-blocking for report generation LLM calls.

    Args:
        cfg: ReportGenerationConfig instance

    Returns:
        Tuple of (ReportJobAction, descriptive_message). Always returns a tuple;
        on error, returns (ReportJobAction.DEBUG_INTERNAL_ERROR, error_message).
    """
    log_prefix = f"[generate report] - "
    # Get period dates and source from config
    try:
        source = cfg.run.source
        period_start = cfg.run.period_start
        period_end = cfg.run.period_end

        # fetch cluster observations
        logger.info(f"{log_prefix}Memo get-clusters-observation started: snapshot_id={source}|{period_start}|{period_end}")
        clusters_observation = await memo.get_clusters_observation(source, period_start, period_end, cfg.memo)
        logger.info(f"{log_prefix}Memo get-clusters-observation successful: snapshot_id={source}|{period_start}|{period_end}")
        # display cluster observations for user to select in tui
        selected_pk_hash, selected_intent_enum = await select_cluster_and_intent(clusters_observation)
        logger.info(f"{log_prefix}Selected pk_hash: {selected_pk_hash}, Selected intent: {selected_intent_enum}")

        # start report generation
        return await trigger_report_job(selected_pk_hash, selected_intent_enum, cfg)
    except Exception as e:
        logger.error(f"{log_prefix}Internal error. {e}", exc_info=True)
        return (ReportJobAction.DEBUG_INTERNAL_ERROR, f"unexpected error. check log under {cfg.cache.report_generation_log_path} for debugging.")


async def trigger_report_job(cluster_pk_hash: str, user_intent: UserIntent, cfg: ReportGenerationConfig) -> Tuple[ReportJobAction, str]:
    """
    trigger a report generation job based on its current status.
    Non-blocking: uses async LLM calls when executor is configured.

    Calls init_report_job (database) and handles/logs the response. Maps next_status:
    - running: kick off new job, returns ('kick_off_report_job', message)
    - resuming: job ready to resume (error expired), returns ('report_job_is_ready_to_resume', message)
    - waiting: returns ('wait_for_report_job_to_finish', meta.message)
    - done: returns ('fetch_report_and_print_report_url', message)

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        user_intent: User intent enum
        cfg: ReportGenerationConfig instance

    Returns:
        Tuple of (ReportJobAction, descriptive_message). Always returns a tuple;
        on error, returns (ReportJobAction.DEBUG_INTERNAL_ERROR, error_message).
    """
    log_prefix = f"[report job trigger] - [{cluster_pk_hash}] - "
    try:
        resp = init_report_job(cluster_pk_hash, cfg)
        meta = resp.meta

        if resp.next_status == 'running':
            await start_generation(cluster_pk_hash, user_intent, cfg)
            next_status = ReportJobAction.FETCH_REPORT
            history_dir = cfg.cache.history_reports
            msg = f"A new report is just generated, search for it under {history_dir} using cluster pk hash {cluster_pk_hash}"
            logger.info(f"{log_prefix}next status {next_status}. {msg}")
            return (next_status, msg)
        elif resp.next_status == 'resuming':
            next_status = ReportJobAction.RESUME_JOB
            cache_dir = cfg.cache.report_generation_cache
            msg = f"{meta.message} Search under {cache_dir} using cluster_pk_hash: {cluster_pk_hash} for possible cache for minimal-effort rerun."
            logger.info(f"{log_prefix}next status {next_status}. {msg}")
            return (next_status, msg)
        elif resp.next_status == 'waiting':
            next_status = ReportJobAction.WAIT_FOR_JOB_TO_FINISH
            logger.info(f"{log_prefix}next status {next_status}. {meta.message}")
            return (next_status, meta.message)
        elif resp.next_status == 'done':
            next_status = ReportJobAction.FETCH_REPORT
            history_dir = cfg.cache.history_reports
            msg = f"A report for this cluster has been generated before, search for it under {history_dir} using cluster pk hash {cluster_pk_hash}"
            logger.info(f"{log_prefix}next status {next_status}. {msg}")
            return (next_status, msg)
        else:
            next_status = ReportJobAction.DEBUG_INTERNAL_ERROR
            msg = f"Unexpected next_status={resp.next_status} from init_report_job, message={meta.message}"
            logger.error(f"{log_prefix}{msg}")
            return (next_status, msg)
    except Exception as e:
        log_file_path = cfg.cache.report_generation_log_path
        logger.error(f"{log_prefix}Internal error. {e}", exc_info=True)
        error_msg = f"unexpected error. check log under {log_file_path} using cluster pk hash {cluster_pk_hash} for debugging."
        return (ReportJobAction.DEBUG_INTERNAL_ERROR, error_msg)
