"""Report generation pipeline orchestration"""

from reader.pipelines.report_generation.config.config import ReportGenerationConfig
from reader.adapters import memo
from reader.pipelines.report_generation.blocks import create_report_job
from reader.logging.logging_setup import get_logger
from reader.tui.clusters_observation import display_clusters_observation
from reader.pipelines.report_generation.prompts.planner.build import UserIntent
logger = get_logger()

# contain potential mcp servers/tool calls/skills(?)
async def generate_report(cfg: ReportGenerationConfig) -> None:
    """
    Run the report generation pipeline for the configured period.
    Async and non-blocking for report generation LLM calls.

    Args:
        cfg: ReportGenerationConfig instance
    """
    # Get period dates and source from config
    source = cfg.run.source
    period_start = cfg.run.period_start
    period_end = cfg.run.period_end

    # report generation
    logger.info(f"Memo get-clusters-observation started: snapshot_id={source}|{period_start}|{period_end}")
    clusters_observation = await memo.get_clusters_observation(source, period_start, period_end, cfg.memo)
    logger.info(f"Memo get-clusters-observation successful: snapshot_id={source}|{period_start}|{period_end}")

    # Get user intent options from enum
    user_intent_options = UserIntent.get_all_display_strings()

    selected_pk_hash, selected_intent = await display_clusters_observation(clusters_observation, user_intent_options)
    logger.info(f"Selected pk_hash: {selected_pk_hash}, Selected intent: {selected_intent}")

    # Create report generation job (first step for any report generation request)
    await create_report_job(selected_pk_hash, selected_intent, cfg)
