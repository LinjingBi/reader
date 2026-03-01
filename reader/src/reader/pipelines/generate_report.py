"""Report generation pipeline orchestration"""

from reader.pipelines.report_generation.config.config import ReportGenerationConfig
from reader.adapters import memo
from reader.pipelines.report_generation.blocks import create_report_job, select_cluster_and_intent
from reader.logging.logging_setup import get_logger
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

    # fetch cluster observations
    logger.info(f"Memo get-clusters-observation started: snapshot_id={source}|{period_start}|{period_end}")
    clusters_observation = await memo.get_clusters_observation(source, period_start, period_end, cfg.memo)
    logger.info(f"Memo get-clusters-observation successful: snapshot_id={source}|{period_start}|{period_end}")
    # display cluster observations for user to select in tui
    selected_pk_hash, selected_intent_enum = await select_cluster_and_intent(clusters_observation)
    logger.info(f"Selected pk_hash: {selected_pk_hash}, Selected intent: {selected_intent_enum}")

    # start report generation
    await create_report_job(selected_pk_hash, selected_intent_enum, cfg)
