"""HF data pipeline orchestration"""

import asyncio
from reader.pipelines.hf_data.config.config import HFDataPipeConfig, load_config
from reader.adapters import memo
from reader.pipelines.hf_data.blocks import get_hf_paper_metadata, generate_clustering_reports, convert_papers_for_chunking
from algo_lib.paperchunk.scoring import run_scoring
from reader.logging.logging_setup import get_logger

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
    # Generate monthly clustering reports and payload
    fresh_paper_payload = generate_clustering_reports(cfg, papers, period_start, period_end)
    
    logger.info(f"Memo ingest started: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    fresh_paper_response = memo.fresh_paper(fresh_paper_payload, cfg.memo)
    logger.info(f"Memo ingest successful: snapshot_id={fresh_paper_payload.source}|{fresh_paper_payload.period_start}|{fresh_paper_payload.period_end}")
    
    # Convert papers for chunking and run scoring
    papers_dict = convert_papers_for_chunking(fresh_paper_response)

    # papers_dict = {
    #     # "2501.01234": "https://arxiv.org/abs/2501.01234",
    #     # "2501.01235": "https://arxiv.org/abs/2501.01235",

    #     "2501.01236": "https://arxiv.org/abs/2501.01236",
    # }
    score_output = asyncio.run(run_scoring(papers_dict, cfg.paper_chunk.rules_path))
    # print(score_output.score_table)
    logger.info(f"Paper chunk scoring completed: total_papers={score_output.summary.total_papers}, scored_ok={score_output.summary.scored_ok}")
    




