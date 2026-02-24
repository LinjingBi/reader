"""HF data pipeline orchestration"""

import asyncio
import calendar
import json
from dataclasses import asdict
from datetime import datetime, timezone

from reader.pipelines.hf_data.config.config import HFDataPipeConfig
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

# use a constant for the data source identifier
DATA_SOURCE = "hf_monthly"

def _get_run_metadata(cfg: HFDataPipeConfig) -> tuple[str, str, str]:
    """
    Extract run metadata from config.

    Returns:
        Tuple of (source, period_start, period_end)
    """
    source = DATA_SOURCE
    month_key = cfg.run.month_key
    parts = month_key.split("=")
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Invalid month key format: {month_key}")
    year_month = parts[1]
    year, month = map(int, year_month.split("-"))
    period_start = f"{year:04d}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    period_end = f"{year:04d}-{month:02d}-{last_day:02d}"
    return source, period_start, period_end


async def _ingest_fresh_papers(
    source: str, period_start: str, period_end: str, cfg: HFDataPipeConfig
):
    """
    Get paper metadata from HF API, embedding and clustering, write to memo db.

    Args:
        source: Data source identifier
        period_start: Period start date (YYYY-MM-DD)
        period_end: Period end date (YYYY-MM-DD)
        cfg: HFDataPipeConfig instance

    Returns:
        None (paper info for downstream tasks comes from get_best_clustering)
    """
    papers = get_hf_paper_metadata(cfg, source, period_start, period_end)
    fresh_paper_payload = generate_clustering_reports(
        cfg, papers, source, period_start, period_end
    )
    logger.info(
        f"Memo fresh-paper ingest started: snapshot_id={source}|{period_start}|{period_end}"
    )
    await memo.fresh_paper(fresh_paper_payload, cfg.memo)
    logger.info(
        f"Memo fresh-paper ingest successful: snapshot_id={source}|{period_start}|{period_end}"
    )


async def _enrich_clusters(cfg: HFDataPipeConfig, best_cluster_run):
    """
    Enrich monthly clusters with LLM and dump to memo db.

    Args:
        cfg: HFDataPipeConfig instance
        best_cluster_run: GetBestRunResponse from memo get-best-clustering

    Returns:
        Dict with "empty" and "score_zero" lists of error items, or None if task disabled
    """
    logger.info(
        f"Summarizing clusters started: snapshot_id={best_cluster_run.source}|{best_cluster_run.period_start}|{best_cluster_run.period_end}"
    )
    cluster_reports = await summarize_clusters_parallel(cfg, best_cluster_run)
    logger.info(
        f"Summarizing clusters successful: snapshot_id={best_cluster_run.source}|{best_cluster_run.period_start}|{best_cluster_run.period_end}"
    )

    # Extract error items for rerun helper (minimal change to existing flow)
    enrich_errors = {"empty": [], "score_zero": []}
    for pk_hash, (cluster_report, judge_result) in cluster_reports.items():
        judge_report_dict = asdict(judge_result)
        item = {"pk_hash": pk_hash, "judge_report": judge_report_dict}
        if cluster_report is None:
            enrich_errors["empty"].append(item)
        elif judge_result.overall == 0.0:
            item["cluster_report"] = cluster_report.model_dump()
            enrich_errors["score_zero"].append(item)

    inject_clusters_observation_payload = convert_cluster_reports_to_memo_payload(
        cluster_reports, cfg
    )
    logger.info(
        f"Memo inject-clusters-observation started: snapshot_id={best_cluster_run.source}|{best_cluster_run.period_start}|{best_cluster_run.period_end}"
    )
    await memo.inject_clusters_observation(inject_clusters_observation_payload, cfg.memo)
    logger.info(
        f"Memo inject-clusters-observation successful: snapshot_id={best_cluster_run.source}|{best_cluster_run.period_start}|{best_cluster_run.period_end}"
    )
    return enrich_errors


def _build_paper_id_to_url(best_cluster_run) -> dict:
    """Build paper_id -> url map from best_cluster_run."""
    paper_id_to_url = {}
    for cluster in best_cluster_run.clusters:
        for paper in cluster.papers:
            paper_id_to_url[paper.paper_id] = paper.url or ""
    return paper_id_to_url


async def _process_paper_chunks(cfg: HFDataPipeConfig, best_cluster_run):
    """
    Process paper chunks and write to memo db.

    Args:
        cfg: HFDataPipeConfig instance
        best_cluster_run: GetBestRunResponse from memo get-best-clustering

    Returns:
        List of error items for papers with status != "ok", or None if task disabled
    """
    inject_payload = await process_paper_chunks(cfg, best_cluster_run)
    logger.info(f"Memo inject-papers-chunk ingest started")
    inject_response = await memo.inject_papers_chunk(inject_payload, cfg.memo)
    logger.info(f"Memo inject-papers-chunk completed: total_papers={inject_response.meta.total_papers_count}, total_chunks={inject_response.meta.total_chunks_count}")

    # Extract error info for rerun helper (non-ok papers)
    paper_id_to_url = _build_paper_id_to_url(best_cluster_run)
    lib_config = inject_payload.lib_config
    paper_chunk_errors = []
    for paper in inject_payload.papers:
        if paper.status != "ok":
            paper_chunk_errors.append({
                "paper_id": paper.paper_id,
                "url": paper_id_to_url.get(paper.paper_id, ""),
                "status": paper.status,
                "lib_config_id": lib_config.lib_config_id,
                "rules_path": cfg.algos.paperchunk.rules_path,
                "rules_version": lib_config.json_payload.version,
                "compiled_regex_version": lib_config.json_payload.compiled_regex_version,
            })
    return paper_chunk_errors


def _log_rerun_helper(
    cfg: HFDataPipeConfig,
    enrich_errors: dict | None,
    paper_chunk_errors: list | None,
    task_failures: list[tuple[str, str]] | None = None,
) -> None:
    """
    Log a parseable rerun helper section for tasks with errors.

    When enrich_errors or paper_chunk_errors is None (task disabled), skip that task's section.
    task_failures: list of (task_name, error_message) for tasks that raised exceptions.
    """
    has_enrich = enrich_errors and (enrich_errors.get("empty") or enrich_errors.get("score_zero"))
    has_paper_chunk = paper_chunk_errors and len(paper_chunk_errors) > 0
    has_task_failures = task_failures and len(task_failures) > 0
    if not has_enrich and not has_paper_chunk and not has_task_failures:
        return

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "",
        "=== RERUN_HELPER_START ===",
        f"timestamp_utc: {timestamp_utc}",
        f"run: month={cfg.run.month_key}",
    ]

    if has_task_failures:
        for task_name, error_msg in task_failures:
            lines.append(f"--- task: {task_name} ---")
            lines.append("[task_failed]")
            lines.append(f"error: {error_msg}")
            lines.append("---")

    if has_enrich:
        lines.append("--- task: _enrich_clusters ---")
        if enrich_errors.get("empty"):
            lines.append("[empty_cluster_report]")
            for item in enrich_errors["empty"]:
                lines.append(f"pk_hash: {item['pk_hash']}")
                lines.append(f"judge_report: {json.dumps(item['judge_report'], ensure_ascii=False)}")
                lines.append("---")
        if enrich_errors.get("score_zero"):
            lines.append("[overall_score_zero]")
            for item in enrich_errors["score_zero"]:
                lines.append(f"pk_hash: {item['pk_hash']}")
                lines.append(f"judge_report: {json.dumps(item['judge_report'], ensure_ascii=False)}")
                lines.append(f"cluster_report: {json.dumps(item['cluster_report'], ensure_ascii=False)}")
                lines.append("---")

    if has_paper_chunk:
        lines.append("--- task: paper_chunk ---")
        for item in paper_chunk_errors:
            lines.append(f"paper_id: {item['paper_id']}")
            lines.append(f"url: {item['url']}")
            lines.append(f"status: {item['status']}")
            lines.append(f"lib_config_id: {item['lib_config_id']}")
            lines.append(f"rules_path: {item['rules_path']}")
            lines.append(f"rules_version: {item['rules_version']}")
            lines.append(f"compiled_regex_version: {item['compiled_regex_version']}")
            lines.append("---")

    lines.append("=== RERUN_HELPER_END ===")
    logger.info("\n".join(lines))


async def run_hf_data(cfg: HFDataPipeConfig) -> None:
    """
    Run the HF data pipeline for the configured month.

    Args:
        cfg: HFDataPipeConfig instance
    """
    logger.info(
        "Tasks enabled: fetch_hf_data=%s, cluster_summarization=%s, paper_chunk=%s",
        cfg.task.fetch_hf_data,
        cfg.task.cluster_summarization,
        cfg.task.paper_chunk,
    )

    source, period_start, period_end = _get_run_metadata(cfg)

    # Phase 1: fetch_hf_data must run first and complete before any other task.
    if cfg.task.fetch_hf_data:
        await _ingest_fresh_papers(source, period_start, period_end, cfg)

    # Pre-step: get best clustering when any phase 2 task is enabled
    best_cluster_run = None
    if cfg.task.cluster_summarization or cfg.task.paper_chunk:
        logger.info(
            f"Memo get-best-run started: snapshot_id={source}|{period_start}|{period_end}"
        )
        best_cluster_run = await memo.get_best_clustering(
            source,
            period_start,
            period_end,
            cfg.memo,
        )
        logger.info(
            f"Memo get-best-run successful: snapshot_id={source}|{period_start}|{period_end}"
        )

    # Phase 2: Tasks that run in parallel after pre-step (when enabled)
    tasks = []
    task_types = []
    if cfg.task.cluster_summarization:
        tasks.append(_enrich_clusters(cfg, best_cluster_run))
        task_types.append("enrich")
    if cfg.task.paper_chunk:
        tasks.append(_process_paper_chunks(cfg, best_cluster_run))
        task_types.append("paper_chunk")

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        enrich_errors = None
        paper_chunk_errors = None
        task_failures = []
        for task_type, result in zip(task_types, results):
            if isinstance(result, Exception):
                task_name = "_enrich_clusters" if task_type == "enrich" else "paper_chunk"
                task_failures.append((task_name, str(result)))
                logger.error(f"Task {task_type} failed: {result}", exc_info=True)
                continue
            if task_type == "enrich":
                enrich_errors = result
            else:
                paper_chunk_errors = result

        _log_rerun_helper(cfg, enrich_errors, paper_chunk_errors, task_failures or None)

