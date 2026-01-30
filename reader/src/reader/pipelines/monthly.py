"""Monthly pipeline orchestration"""

from reader.config import ReaderConfig
from reader.adapters import memo
from reader.pipelines.blocks import get_hf_paper_metadata, generate_clustering_reports, summarize_clusters_parallel
from reader.logging.logging_setup import get_logger

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
        memo_result = memo.fresh_paper(fresh_paper_payload, cfg)
        if memo_result:
            logger.info(f"Memo ingest successful: snapshot_id={memo_result.snapshot_id}, cluster_run_id={memo_result.cluster_run_id}")
    
    if cfg.cluster_summarization.enable and cfg.memo.enabled:
        best_cluster_run = memo.get_best_clustering(fresh_paper_payload.source, period_start, period_end, cfg)
        if best_cluster_run:
            logger.info(f"Memo get-best-run successful: snapshot_id={best_cluster_run.snapshot_id}, cluster_run_id={best_cluster_run.cluster_run_id}")
            cluster_reports = summarize_clusters_parallel(cfg, best_cluster_run)
            for cluster_report, judge_output in cluster_reports:
                if cluster_report:
                    logger.info(f"Cluster report: {cluster_report}")
                if judge_output:
                    logger.info(f"Judge output: {judge_output}")
