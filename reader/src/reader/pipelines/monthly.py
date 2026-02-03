"""Monthly pipeline orchestration"""

from reader.config import ReaderConfig
from reader.adapters import memo
from reader.pipelines.blocks import get_hf_paper_metadata, generate_clustering_reports, summarize_clusters_parallel, serialize_cluster_reports, convert_cluster_reports_to_memo_payload
from reader.logging.logging_setup import get_logger
from reader.pipelines.metrics import JudgeOutput, ClusterReport
from typing import Dict, Tuple, Optional

logger = get_logger()


def run_monthly(cfg: ReaderConfig) -> None:
    """
    Run the monthly pipeline for the configured month.
    
    Args:
        cfg: ReaderConfig instance
    """
    # Get paper metadata from HF API or cached file
    papers, period_start, period_end = get_hf_paper_metadata(cfg)
    
    # Generate monthly clustering reports and payload
    fresh_paper_payload = generate_clustering_reports(cfg, papers, period_start, period_end)

    # Optionally call memo adapter if enabled
    if cfg.memo.enabled:
        logger.info(f"Memo ingest started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
        memo.fresh_paper(fresh_paper_payload, cfg)

        logger.info(f"Memo ingest successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    
    if cfg.cluster_summarization.enable and cfg.memo.enabled:
        logger.info("Memo get-best-run started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
        best_cluster_run = memo.get_best_clustering(fresh_paper_payload.source, period_start, period_end, cfg)

        logger.info(f"Memo get-best-run successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
        cluster_reports: Dict[str, Tuple[Optional[ClusterReport], JudgeOutput]] = summarize_clusters_parallel(cfg, best_cluster_run)
        
        inject_clusters_observation_payload = convert_cluster_reports_to_memo_payload(cluster_reports, cfg)
        logger.info(f"Memo inject-clusters-observation started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
        memo.inject_clusters_observation(inject_clusters_observation_payload, cfg)
        logger.info(f"Memo inject-clusters-observation successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
