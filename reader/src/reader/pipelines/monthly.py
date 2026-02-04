"""Monthly pipeline orchestration"""

from reader.config import ReaderConfig
from reader.adapters import memo
from reader.pipelines.blocks import get_hf_paper_metadata, generate_clustering_reports, summarize_clusters_parallel, serialize_cluster_reports, convert_cluster_reports_to_memo_payload
from reader.logging.logging_setup import get_logger
from reader.pipelines.metrics import JudgeOutput, ClusterReport
from typing import Dict, Tuple, Optional
from reader.tui.clusters_observation import display_clusters_observation
logger = get_logger()

# contain potential mcp servers
def run_monthly(cfg: ReaderConfig) -> None:
    """
    Run the monthly pipeline for the configured month.
    
    Args:
        cfg: ReaderConfig instance
    """
    # Get paper metadata from HF API or cached file
    papers, period_start, period_end = get_hf_paper_metadata(cfg)
    
    # # dump hf metadata(paper, cluster, embed config) to memo db
    # if cfg.memo.enabled:
    #     # Generate monthly clustering reports and payload
    #     fresh_paper_payload = generate_clustering_reports(cfg, papers, period_start, period_end)
    #     logger.info(f"Memo ingest started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    #     memo.fresh_paper(fresh_paper_payload, cfg)

    #     logger.info(f"Memo ingest successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    # # enrich monthly clusters with llm and dump to memo db
    # if cfg.cluster_summarization.enable and cfg.memo.enabled:
    #     logger.info("Memo get-best-run started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    #     best_cluster_run = memo.get_best_clustering(fresh_paper_payload.source, period_start, period_end, cfg)

    #     logger.info(f"Memo get-best-run successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    #     cluster_reports: Dict[str, Tuple[Optional[ClusterReport], JudgeOutput]] = summarize_clusters_parallel(cfg, best_cluster_run)
        
    #     inject_clusters_observation_payload = convert_cluster_reports_to_memo_payload(cluster_reports, cfg)
    #     logger.info(f"Memo inject-clusters-observation started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    #     memo.inject_clusters_observation(inject_clusters_observation_payload, cfg)
    #     logger.info(f"Memo inject-clusters-observation successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    
    # read 
    if cfg.memo.enabled:
        logger.info(f"Memo get-clusters-observation started: snapshot_id=hf_monthly|{period_start}|{period_end}")
        clusters_observation = memo.get_clusters_observation('hf_monthly', period_start, period_end, cfg)
        logger.info(f"Memo get-clusters-observation successful: snapshot_id=hf_monthly|{period_start}|{period_end}")
        selected_pk_hash = display_clusters_observation(clusters_observation)
        logger.info(f"Selected pk_hash: {selected_pk_hash}")

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
    