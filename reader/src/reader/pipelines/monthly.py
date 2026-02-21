"""Monthly pipeline orchestration"""

from reader.config import ReaderConfig
from reader.adapters import memo
from reader.pipelines.blocks import create_report_job
from reader.logging.logging_setup import get_logger
from reader.tui.clusters_observation import display_clusters_observation
from reader.prompts.report_planner.build import UserIntent
logger = get_logger()

# contain potential mcp servers/tool calls/skills(?)
def run_monthly(cfg: ReaderConfig) -> None:
    """
    Run the monthly pipeline for the configured month.
    
    Args:
        cfg: ReaderConfig instance
    """
    # Get period dates from config
    period_start = cfg.run.period_start
    period_end = cfg.run.period_end
    
    # report generation
    if cfg.report_generation and cfg.report_generation.enable:
        logger.info(f"Memo get-clusters-observation started: snapshot_id=hf_monthly|{period_start}|{period_end}")
        clusters_observation = memo.get_clusters_observation('hf_monthly', period_start, period_end, cfg.memo)
        logger.info(f"Memo get-clusters-observation successful: snapshot_id=hf_monthly|{period_start}|{period_end}")
        
        # Get user intent options from enum
        user_intent_options = UserIntent.get_all_display_strings()
        
        selected_pk_hash, selected_intent = display_clusters_observation(clusters_observation, user_intent_options)
        logger.info(f"Selected pk_hash: {selected_pk_hash}, Selected intent: {selected_intent}")

        # Create report generation job (first step for any report generation request)
        create_report_job(selected_pk_hash, selected_intent, cfg)

        """
        1. compare new selected cluster observation with existing topics in memo db(cosine similarity)
            - if similarity is greater than threshold, update the topic
            - if similarity is less than threshold, create a new topic
            - delete the selected cluster observation from memo db
        2. generate report md for #1 topic
            - derive "depth"(the depth requirements in the prompt to the report llm)
            - get the report from llm(apply basic heuristic rules)
        3. write the report to local device
        async
        4. display the report in tui
        5. store the report url and update related tables in memo db
        """
        # return selected_pk_hash

    