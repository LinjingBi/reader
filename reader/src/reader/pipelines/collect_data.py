"""HF data pipeline orchestration"""

import asyncio

from reader.pipelines.hf_data.config.config import HFDataPipeConfig, load_config
from reader.adapters import memo
from reader.pipelines.hf_data.blocks import (
    get_hf_paper_metadata,
    generate_clustering_reports,
    process_paper_chunks,
    summarize_clusters_parallel,
    convert_cluster_reports_to_memo_payload,
)
from reader.logging.logging_setup import get_logger

logger = get_logger()


async def _ingest_fresh_papers(cfg: HFDataPipeConfig):
    """
    Get paper metadata from HF API, embedding and clustering, write to memo db.
    
    Args:
        cfg: HFDataPipeConfig instance
        
    Returns:
        Tuple of (fresh_paper_response, fresh_paper_payload, period_start, period_end)
    """
    papers, period_start, period_end = get_hf_paper_metadata(cfg)
    fresh_paper_payload = generate_clustering_reports(cfg, papers, period_start, period_end)
    logger.info(f"Memo fresh-paper ingest started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    fresh_paper_response = await memo.fresh_paper(fresh_paper_payload, cfg.memo)
    logger.info(f"Memo fresh-paper ingest successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    return fresh_paper_response, fresh_paper_payload, period_start, period_end


async def _enrich_clusters(cfg: HFDataPipeConfig, fresh_paper_payload, period_start, period_end):
    """
    Enrich monthly clusters with LLM and dump to memo db.
    
    Args:
        cfg: HFDataPipeConfig instance
        fresh_paper_payload: Fresh paper payload from previous step
        period_start: Period start timestamp
        period_end: Period end timestamp
    """
    logger.info(f"Memo get-best-run started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    best_cluster_run = await memo.get_best_clustering(
        fresh_paper_payload.source,
        period_start,
        period_end,
        cfg.memo
    )
    logger.info(f"Memo get-best-run successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    logger.info(f"Summarizing clusters started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    cluster_reports = await summarize_clusters_parallel(cfg, best_cluster_run)
    logger.info(f"Summarizing clusters successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    inject_clusters_observation_payload = convert_cluster_reports_to_memo_payload(cluster_reports, cfg)
    logger.info(f"Memo inject-clusters-observation started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    await memo.inject_clusters_observation(inject_clusters_observation_payload, cfg.memo)
    logger.info(f"Memo inject-clusters-observation successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")


async def _process_paper_chunks(cfg: HFDataPipeConfig, fresh_paper_response):
    """
    Process paper chunks and write to memo db.
    
    Args:
        cfg: HFDataPipeConfig instance
        fresh_paper_response: Fresh paper response from previous step
    """
    inject_payload = await process_paper_chunks(cfg, fresh_paper_response)
    logger.info(f"Memo inject-papers-chunk ingest started")
    inject_response = await memo.inject_papers_chunk(inject_payload, cfg.memo)
    logger.info(f"Memo inject-papers-chunk completed: total_papers={inject_response.meta.total_papers_count}, total_chunks={inject_response.meta.total_chunks_count}")


async def run_hf_data(cfg: HFDataPipeConfig) -> None:
    """
    Run the HF data pipeline for the configured month.
    
    Args:
        cfg: HFDataPipeConfig instance
    """
    fresh_paper_response, fresh_paper_payload, period_start, period_end = await _ingest_fresh_papers(cfg)
    await _enrich_clusters(cfg, fresh_paper_payload, period_start, period_end)
    await _process_paper_chunks(cfg, fresh_paper_response)

