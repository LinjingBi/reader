"""HF data pipeline orchestration"""

from reader.pipelines.hf_data.config.config import HFDataPipeConfig, load_config
from reader.adapters import memo
from reader.pipelines.hf_data.blocks import get_hf_paper_metadata, generate_clustering_reports
from reader.logging.logging_setup import setup_logging, get_logger

logger = get_logger()
def run_hf_data(cfg: HFDataPipeConfig) -> None:
    """
    Run the HF data pipeline for the configured month.
    
    Args:
        cfg: HFDataPipeConfig instance
    """
    # Get paper metadata from HF API or cached file
    papers, period_start, period_end = get_hf_paper_metadata(cfg)
    
    # dump hf metadata(paper, cluster, embed config) to memo db
    if cfg.memo.enabled:
        # Generate monthly clustering reports and payload
        fresh_paper_payload = generate_clustering_reports(cfg, papers, period_start, period_end)
        
        logger.info(f"Memo ingest started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
        memo.fresh_paper(fresh_paper_payload, cfg)

        logger.info(f"Memo ingest successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")

