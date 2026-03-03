"""HF data pipeline building blocks"""

import os
import base64
import json
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any, Tuple
import numpy as np
from pydantic import ValidationError

from algo_lib.clustering import get_best_clustering
from algo_lib.typing import PaperLike
from algo_lib.paperchunk.types import PaperId, Url, PaperStatus
from algo_lib.paperchunk.scoring import run_scoring

from reader.pipelines.hf_data.config.config import (
    HFDataPipeConfig,
    LLMGeminiConfig,
    render_best_cluster_text_report_path,
    render_best_cluster_report_path,
    render_papers_scoring_summary_report_path,
    render_paper_scoring_debug_heading_events_path,
    render_cluster_summarization_events_path,
)
from reader.pipelines.hf_data.report import (
    FreshPaperPayload,
    PaperInput,
    ClusterMemberInput,
    ClusterInput,
    EmbedConfig,
    EmbedConfigPayload,
    ClusterConfig,
    ClusterConfigPayload,
    InjectPapersChunkPayload,
    PaperChunkLibConfig,
    PaperChunkLibConfigPayload,
    ChunkEntry,
    PaperChunkData,
)
from reader.adapters.hf import get_monthly_report, parse_papers, save_papers_to_file
from reader.adapters.memo import GetBestRunResponse, ClusterCard
from reader.adapters.llm import LLMClient, TokenBucket, LLMGenerationError
from reader.pipelines.hf_data.report import (
    ClusterReport,
    ClusterObservation,
    InjectClustersObservationInput,
    LLMConfigInput,
    ClusterObservationRow,
)
from reader.pipelines.hf_data.metrics import judge_output, JudgeOutput
from reader.logging.logging_setup import get_logger

logger = get_logger()


def get_hf_paper_metadata(
    cfg: HFDataPipeConfig,
    source: str,
    period_start: str,
    period_end: str,
) -> list:
    """
    Get paper metadata from HF API or cached file.

    Args:
        cfg: HFDataPipeConfig instance
        source: Data source identifier
        period_start: Period start date (YYYY-MM-DD)
        period_end: Period end date (YYYY-MM-DD)

    Returns:
        List of Paper objects
    """
    import asyncio

    month_key = cfg.run.month_key
    papers_report_file = cfg.sources.hf.output_json

    # Check if papers_report.json exists, generate if missing
    if not Path(papers_report_file).exists():
        logger.info(f"{papers_report_file} not found, generating from HF API...")
        results = asyncio.run(get_monthly_report(cfg))
        save_papers_to_file(results, cfg)
        logger.info(f"Generated {papers_report_file}")

    # Load papers_report_file
    with open(papers_report_file, "r") as f:
        data = json.load(f)
    papers_data = data["papers"]

    # Process single month
    if month_key not in papers_data:
        raise ValueError(
            f"Month {month_key} not found in papers_data. Available months: {list(papers_data.keys())}"
        )

    papers_list = papers_data[month_key]
    logger.info(f"Processing {month_key}")
    logger.info(f"Period: {period_start} to {period_end}")

    # Create Paper objects from JSON data
    papers = parse_papers(papers_list, cfg)

    return papers


def write_best_clustering_text_report(
    papers: Sequence[PaperLike],
    cluster_members_ordered: Dict[int, List[int]],
    header: str = "",
    max_summary_chars: int = 350,
    report_dir: str = 'best_clustering_reports.md',
) -> None:
    """
    Write a human-readable text report for a chosen clustering:
    - cluster sizes 
    - optional TF-IDF keyword hints
    - each paper: title + (truncated) summary + url
    
    Args:
        papers: Sequence of paper-like objects
        cluster_members_ordered: Dictionary mapping cluster_id -> list of paper indices sorted by similarity
        header: Optional header string to write at the top
        max_summary_chars: Maximum characters for summary truncation
        report_dir: Path to the report file
    """
    clusters = cluster_members_ordered
    
    with open(report_dir, 'a+') as f:
        if header:
            f.write("\n" + "=" * 90 + '\n')
            f.write(header + '\n')

        # sort by cluster size desc
        cluster_order = sorted(clusters.keys(), key=lambda c: len(clusters[c]), reverse=True)

        for c in cluster_order:
            idxs = clusters[c]
            f.write("\n" + "-" * 90 + "\n")
            f.write(f"Cluster {c} | size={len(idxs)}\n")

            for i in idxs:
                p = papers[i]
                summ = p.summary.strip() if p.summary else ""
                if max_summary_chars and len(summ) > max_summary_chars:
                    summ = summ[:max_summary_chars] + "…"
                f.write(f"\n[{p.pid}] {p.title}\n")
                if p.url:
                    f.write(f"URL: {p.url}\n")
                if summ:
                    f.write(f"Summary: {summ}\n")


def generate_fresh_paper_payload(
    papers: Sequence[PaperLike],
    member_similarities: Dict[int, Dict[int, float]],
    cluster_cohesion_dict: Dict[int, float],
    cluster_centroids: Dict[int, np.ndarray],
    source: str,
    period_start: str,
    period_end: str,
    embed_model_name: str,
    best_mode: str,
    best_k: int,
    top_n_keywords: int,
    seed: int,
    config: HFDataPipeConfig,
    raw_json: str = "",
    output_path: Optional[str] = None,
) -> FreshPaperPayload:
    """
    Generate fresh_paper_payload.json format report.
    
    Args:
        papers: Sequence of paper-like objects (must have published_at field)
        member_similarities: Dict[cluster_id] -> Dict[paper_idx] -> similarity to centroid
        cluster_cohesion_dict: Dict[cluster_id] -> average cohesion
        cluster_centroids: Dict[cluster_id] -> centroid vector (normalized numpy array)
        source: Data source identifier
        period_start: Start date in YYYY-MM-DD format
        period_end: End date in YYYY-MM-DD format
        embed_model_name: Embedding model name
        best_mode: Best embedding mode selected
        best_k: Best k value selected
        top_n_keywords: Number of top keywords used
        seed: Random seed used
        config: HFDataPipeConfig instance
        raw_json: Optional raw JSON string
        output_path: Path for JSON output file. If None, no file will be written.
    
    Returns:
        FreshPaperPayload instance matching fresh_paper_payload.json format
    """
        
    # Build papers array
    papers_list = []
    for p in papers:
        papers_list.append(PaperInput(
            raw_paper_id=p.pid,
            title=p.title,
            summary=p.summary,
            keywords=p.keywords,
            url=p.url,
            published_at=p.published_at,
        ))
    
    # Build clusters array
    clusters_list = []
    
    for c, papers_sim in member_similarities.items():
        idxs = sorted(member_similarities[c], key=member_similarities[c].get, reverse=True)
        members_list = []
        
        for rank, paper_idx in enumerate(idxs):
            sim = papers_sim[paper_idx]
            # Get formatted paper_id from PaperInput instance
            paper_input = papers_list[paper_idx]
            members_list.append(ClusterMemberInput(
                paper_id=paper_input.paper_id,
                rank_in_cluster=rank,
                sim_to_centroid=sim,
            ))
        
        # Convert centroid to base64-encoded float32 bytes
        if c not in cluster_centroids:
            raise ValueError(f"Cluster {c} missing centroid in cluster_centroids")
        centroid = cluster_centroids[c]
        # Convert numpy array to float32 bytes (little-endian)
        # Use '<f4' to explicitly ensure little-endian byte order
        centroid_bytes = centroid.astype('<f4').tobytes()
        # Encode to base64
        centroid_b64 = base64.b64encode(centroid_bytes).decode('utf-8')
        
        clusters_list.append(ClusterInput(
            cluster_index=c,
            size=len(idxs),
            cohesion=cluster_cohesion_dict[c],
            centroid_b64=centroid_b64,
            members=members_list,
        ))
    
    # Build embed_config
    embed_config = EmbedConfig(
        json_payload=EmbedConfigPayload(
            model_name=embed_model_name,
            mode=best_mode,
            top_n_keywords=top_n_keywords,
        )
    )
    
    # Build cluster_config
    cluster_config = ClusterConfig(
        json_payload=ClusterConfigPayload(
            k=best_k,
            seed=seed,
            algorithm="kmeans",
        )
    )
    
    # Build complete payload
    payload = FreshPaperPayload(
        source=source,
        period_start=period_start,
        period_end=period_end,
        raw_json=raw_json,
        embed_config=embed_config,
        cluster_config=cluster_config,
        papers=papers_list,
        clusters=clusters_list,
    )
    
    # Write JSON file only if output_path is provided
    if output_path is not None:
        logger.info(f"Generating {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(payload.model_dump_json(indent=2, exclude_none=False))
    
    return payload


def generate_clustering_reports(
    cfg: HFDataPipeConfig,
    papers: list,
    source: str,
    period_start: str,
    period_end: str,
) -> FreshPaperPayload:
    """
    Generate clustering reports and payload.
    
    Args:
        cfg: HFDataPipeConfig instance
        papers: List of Paper objects
        source: Data source identifier
        period_start: Start date in YYYY-MM-DD format
        period_end: End date in YYYY-MM-DD format

    Returns:
        FreshPaperPayload instance
    """
    month_key = cfg.run.month_key
    
    # Get best clustering with enhanced metadata
    result = get_best_clustering(
        papers=papers,
        embed_model_name=cfg.algos.embedding.model,
        modes=cfg.algos.embedding.modes,
        k_candidates=cfg.algos.clustering.k_candidates,
        top_n_keywords=cfg.algos.embedding.top_n_keywords,
        seed=cfg.algos.clustering.random_seed
    )
    logger.info(f"{month_key} Top choice: mode: {result.mode} k: {result.k} embed_model: {cfg.algos.embedding.model}")

    # Generate JSON payload
    cluster_json_report_path = render_best_cluster_report_path(cfg, month_key)
    fresh_paper_payload = generate_fresh_paper_payload(
        papers=papers,
        member_similarities=result.cluster_members_similarities,
        cluster_cohesion_dict=result.cluster_cohesion,
        cluster_centroids=result.cluster_centroids,
        source=source,
        period_start=period_start,
        period_end=period_end,
        embed_model_name=cfg.algos.embedding.model,
        best_mode=result.mode,
        best_k=result.k,
        top_n_keywords=cfg.algos.embedding.top_n_keywords,
        seed=cfg.algos.clustering.random_seed,
        config=cfg,
        raw_json="",  # Optional: can be set to actual raw JSON if available
        output_path=cluster_json_report_path,  # Will be None if not configured, which triggers default behavior
    )
    # Write text report if configured
    cluster_text_report_path = render_best_cluster_text_report_path(cfg, month_key)
    if cluster_text_report_path:
        logger.info(f"Appending {month_key} best clustering text report to {cluster_text_report_path}")
        # Remove existing report if it exists
        if os.path.exists(cluster_text_report_path):
            os.remove(cluster_text_report_path)
        
        write_best_clustering_text_report(
            papers=papers,
            cluster_members_ordered=result.cluster_members_ordered,
            header=f"# {month_key} BEST CLUSTERING (mode={result.mode}, k={result.k})",
            max_summary_chars=350,
            report_dir=cluster_text_report_path,
        )
    
    return fresh_paper_payload


def _convert_papers_for_chunking(best_run_response: GetBestRunResponse) -> Dict[PaperId, Url]:
    """
    Convert papers from get-best-run response to the format expected by run_scoring.

    Args:
        best_run_response: GetBestRunResponse instance containing cluster papers

    Returns:
        Dictionary mapping PaperId to Url
    """
    papers_dict: Dict[PaperId, Url] = {}
    for cluster in best_run_response.clusters:
        for paper in cluster.papers:
            papers_dict[paper.paper_id] = paper.url
    return papers_dict


async def process_paper_chunks(
    cfg: HFDataPipeConfig,
    best_run_response: GetBestRunResponse,
) -> InjectPapersChunkPayload:
    """
    Process paper chunks: convert papers, run scoring, and convert ScoreOutput to InjectPapersChunkPayload.

    Args:
        cfg: HFDataPipeConfig instance
        best_run_response: GetBestRunResponse instance containing cluster papers

    Returns:
        InjectPapersChunkPayload instance ready for memo injection
    """
    # Convert papers for chunking and run scoring
    papers_dict = _convert_papers_for_chunking(best_run_response)
    executor = cfg.algos.paperchunk.paper_parser_executor
    score_output = await run_scoring(papers_dict, cfg.algos.paperchunk.rules_path, executor=executor)
    logger.info(f"Paper chunk scoring completed: total_papers={score_output.summary.total_papers}, scored_ok={score_output.summary.scored_ok}")
    
    # Render output file paths
    month_key = cfg.run.month_key
    summary_report_path = render_papers_scoring_summary_report_path(cfg, month_key)
    debug_events_path = render_paper_scoring_debug_heading_events_path(cfg, month_key)
    
    # Write paper scoring summary report if path is provided
    if summary_report_path is not None:
        summary_dict = asdict(score_output.summary)
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Paper scoring summary report saved to {summary_report_path}")
    
    # Write paper scoring debug heading events if path is provided
    if debug_events_path is not None:
        with open(debug_events_path, 'a', encoding='utf-8') as f:
            for event in score_output.debug_heading_events:
                event_dict = asdict(event)
                f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
        logger.info(f"Paper scoring debug heading events saved to {debug_events_path}")
    
    # Extract version and compiled_regex_version from score_output.rules_meta
    lib_config_payload = PaperChunkLibConfigPayload(
        version=score_output.rules_meta.version,
        compiled_regex_version=score_output.rules_meta.compiled_regex_version,
    )
    
    # Convert ScoreOutput to request format
    # Group chunks by paper_id
    paper_chunks: Dict[str, List[ChunkEntry]] = {}
    for score_row in score_output.sel2texts_score_table:
        paper_id = score_row.paper_id
        if paper_id not in paper_chunks:
            paper_chunks[paper_id] = []
        
        # Get text from text_table
        text = score_output.text_table.get(score_row.text_id, "")
        
        paper_chunks[paper_id].append(ChunkEntry(
            selector_id=score_row.selector_id,
            text_id=score_row.text_id,
            text=text,
            score=score_row.score,
        ))
    
    # Build papers list
    papers = []
    for paper_id, status in score_output.papers_status.items():
        chunks = paper_chunks.get(paper_id, [])
        papers.append(PaperChunkData(
            paper_id=paper_id,
            status=status.value if isinstance(status, PaperStatus) else status,
            chunks=chunks,
        ))
    
    # Build complete payload
    payload = InjectPapersChunkPayload(
        lib_config=PaperChunkLibConfig(json_payload=lib_config_payload),
        papers=papers,
    )
    
    return payload


# ============================================================================
# LLM clusters summarization/enrichment blocks
# ============================================================================

def load_template(template_path: Path) -> str:
    """
    Load prompt template from file.
    
    Args:
        template_path: Path to template file
    
    Returns:
        Template content as string
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def render_template_per_cluster(template_content: str, cluster_data: dict) -> str:
    """
    Render template for a single cluster by replacing {{CLUSTER_JSON}} placeholder.
    
    Args:
        template_content: Template string with {{CLUSTER_JSON}} placeholder
        cluster_data: Single cluster dict with papers array
    
    Returns:
        Rendered prompt string
    """
    rendered = template_content.replace(
        "{{CLUSTER_JSON}}",
        json.dumps(cluster_data, indent=2, ensure_ascii=False)
    )
    
    return rendered


def _count_judge_warnings(judge_output: JudgeOutput) -> int:
    """
    Count unique rule clarifications from judge_output.reasons.

    Args:
        judge_output: JudgeOutput from judge_output()

    Returns:
        Number of unique rule declarations (deduplicated while preserving order)
    """
    seen: set[str] = set()
    count = 0
    for rule_list in judge_output.reasons.values():
        for rule_clarification, _ in rule_list:
            if rule_clarification not in seen:
                seen.add(rule_clarification)
                count += 1
    return count


def inject_judge_warnings_into_prompt(prompt_base: str, judge_output: JudgeOutput) -> tuple[str, int]:
    """
    Append a WARNING section with failed rule declarations to the base prompt.

    The warning list is regenerated from judge_output.reasons each call.
    Use prompt_base (warning-free) so each retry attaches a fresh warning list.

    Args:
        prompt_base: Warning-free prompt from first render
        judge_output: JudgeOutput containing reasons from failed validation

    Returns:
        Tuple of (prompt_base + warning_block, num_warnings)
    """
    seen: set[str] = set()
    rule_declarations: list[str] = []
    for rule_list in judge_output.reasons.values():
        for rule_clarification, _ in rule_list:
            if rule_clarification not in seen:
                seen.add(rule_clarification)
                rule_declarations.append(rule_clarification)

    if not rule_declarations:
        return (prompt_base, 0)

    lines = ["\n\nWARNING:"]
    for rule in rule_declarations:
        lines.append(f"- {rule}")
    warning_block = "\n".join(lines)
    return (prompt_base + warning_block, len(rule_declarations))


def _initialize_llm_client(llm_gemini_config: LLMGeminiConfig) -> LLMClient:
    """
    Initialize LLM client with rate limiting buckets.
    
    Args:
        llm_gemini_config: LLMGeminiConfig instance
    
    Returns:
        Initialized LLMClient instance
    
    Raises:
        ValueError: If API key is not found in environment variable
    """
    # Get API key from environment variable
    api_key = os.getenv(llm_gemini_config.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {llm_gemini_config.api_key_env}")
    
    # Initialize TokenBucket instances for rate limiting
    rpm_bucket = TokenBucket(
        capacity=llm_gemini_config.gemini_rpm_limit,
        refill_rate=llm_gemini_config.gemini_rpm_limit,
        name="gemini_rpm"
    )
    
    tpm_bucket = TokenBucket(
        capacity=llm_gemini_config.gemini_tpm_limit,
        refill_rate=llm_gemini_config.gemini_tpm_limit,
        name="gemini_tpm"
    )
    
    # Get executor from config
    executor = llm_gemini_config.gemini_call_executor
    
    # Create LLMClient instance
    llm_client = LLMClient(
        model=llm_gemini_config.model,
        api_key=api_key,
        rpm_bucket=rpm_bucket,
        tpm_bucket=tpm_bucket,
        executor=executor
    )
    
    logger.info(f"Initialized LLM client with model: {llm_gemini_config.model}")
    return llm_client


def _convert_cluster_card_to_dict(cluster_card: ClusterCard, top_n: int | None = None) -> Dict[str, Any]:
    """
    Convert ClusterCard to dict format expected by template.
    
    Args:
        cluster_card: ClusterCard instance with PaperCard papers
        top_n: Max papers to include. If None, include all papers.
    
    Returns:
        Dictionary with papers array in format expected by template
    """
    papers_iter = cluster_card.papers if top_n is None else cluster_card.papers[:top_n]
    papers_list = []
    for paper_card in papers_iter:
        papers_list.append({
            "paper_id": paper_card.paper_id,
            "title": paper_card.title,
            "summary": paper_card.summary,
            "keywords": paper_card.keywords,
            "url": paper_card.url,
            "rank_in_group": paper_card.rank_in_cluster,
            "sim_to_centroid": paper_card.sim_to_centroid,
        })
    
    result = {
        "papers": papers_list,
        "size": cluster_card.size,
    }
    if cluster_card.cohesion is not None:
        result["cohesion"] = cluster_card.cohesion
    
    return result


def _save_cluster_summarization_events(
    cfg: HFDataPipeConfig,
    results_dict: Dict[str, Tuple[Optional[ClusterReport], JudgeOutput]]
) -> None:
    """
    Save cluster summarization events to JSONL file.
    
    Args:
        cfg: HFDataPipeConfig instance
        results_dict: Dictionary mapping pk_hash to tuple of (Optional[ClusterReport], JudgeOutput)
    """
    month_key = cfg.run.month_key
    events_path = render_cluster_summarization_events_path(cfg, month_key)
    if events_path is not None:
        with open(events_path, 'a', encoding='utf-8') as f:
            for pk_hash, (cluster_report, judge_result) in results_dict.items():
                # Convert JudgeOutput dataclass to dict for serialization
                judge_result_dict = asdict(judge_result)
                # Create ClusterObservationRow
                row = ClusterObservationRow(
                    cluster_pk_hash=pk_hash,
                    cluster_report=cluster_report,
                    judge_result=judge_result_dict
                )
                # Write as JSON line
                row_dict = row.model_dump(mode='json')
                f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
        logger.info(f"Cluster summarization events saved to {events_path}")


async def summarize_clusters_parallel(
    cfg: HFDataPipeConfig,
    best_run_response: GetBestRunResponse
) -> Dict[str, Tuple[Optional[ClusterReport], JudgeOutput]]:
    """
    Process all clusters in parallel and generate ClusterReport summaries.
    
    Args:
        cfg: HFDataPipeConfig instance
        best_run_response: GetBestRunResponse from memo adapter
    
    Returns:
        Dictionary mapping pk_hash to tuple of (Optional[ClusterReport], JudgeOutput):
        - Each tuple contains (cluster_report_or_none, judge_output_or_failed_judge)
    """
    # Load prompt template using resolved path from config
    template_path = cfg.cluster_summarization.prompt_template_path
    template_content = load_template(template_path)
    logger.info(f"Loaded prompt template from {template_path}")
    
    # Initialize LLM client
    llm_client = _initialize_llm_client(cfg.cluster_summarization.llm_gemini)
    
    # Process clusters in parallel

    async def process_single_cluster(cluster_card: ClusterCard, top_n_paper: int | None = None) -> Tuple[Optional[ClusterReport], JudgeOutput]:
        """
        Process a single cluster and return ClusterReport and JudgeOutput.
        Retries up to 3 times if overall score is 0.0.
        
        Args:
            cluster_card: ClusterCard instance
            top_n_paper: Max papers to include in prompt. If None, include all papers.
        
        Returns:
            Tuple of (Optional[ClusterReport], JudgeOutput):
            - ClusterReport: Parsed cluster report if successful, None otherwise
            - JudgeOutput: Contains validation scores and reasons (or failed_judge on error)
        """
        pk_hash = cluster_card.pk_hash
        
        # Convert ClusterCard to dict format
        cluster_dict = _convert_cluster_card_to_dict(cluster_card, top_n=top_n_paper)
        
        # Render template with cluster data (keep warning-free base for retry injection)
        prompt_base = render_template_per_cluster(template_content, cluster_dict)
        prompt = prompt_base

        # Retry logic: up to 3 retries if score is 0.0
        max_retries = 3
        retry_threshold = 1.87
        best_cluster_report = None
        best_judge_result = None
        best_score = float('-inf')
        
        for attempt in range(max_retries + 1):  # Initial attempt + 3 retries = 4 total
            try:
                # Call LLM with structured output (returns parsed ClusterReport) using async method
                cluster_report = await llm_client.call_structured_async(
                    prompt=prompt,
                    response_model=ClusterReport,
                    temperature=cfg.cluster_summarization.llm_gemini.temperature,
                    max_tokens=cfg.cluster_summarization.llm_gemini.max_tokens
                )
                # Use judge_output to validate response
                # cluster_dict is used for citation validation
                judge_result = judge_output(cluster_report, cluster_dict)

                # Track best score across all attempts
                if judge_result.overall > best_score:
                    best_score = judge_result.overall
                    best_cluster_report = cluster_report
                    best_judge_result = judge_result

                # If score is positive, success - stop retrying
                if judge_result.overall > retry_threshold:
                    logger.info(
                        f"cluster {pk_hash} attempt {attempt+1}/{max_retries+1}: cluster_report overall score {best_score:.2f} > {retry_threshold}, accepted"
                    )
                    return best_cluster_report, best_judge_result

                # Score is at or below threshold - judge retry (inject warnings, re-call LLM)
                if attempt < max_retries:
                    prompt, num_warnings = inject_judge_warnings_into_prompt(prompt_base, judge_result)
                    logger.warning(
                        f"cluster {pk_hash} attempt {attempt+1}/{max_retries+1}: cluster_report overall score {judge_result.overall:.2f} <= {retry_threshold}, "
                        f"injecting {num_warnings} warning(s), retrying"
                    )
                else:
                    num_warnings = _count_judge_warnings(judge_result)
                    logger.warning(
                        f"cluster {pk_hash} attempt {attempt+1}/{max_retries+1}: cluster_report overall score {judge_result.overall:.2f} <= {retry_threshold}, "
                        f"{num_warnings} warning(s) (last judge retry, returning best)"
                    )

            except (LLMGenerationError, ValidationError) as e:
                logger.warning(f"cluster {pk_hash} attempt {attempt+1}/{max_retries+1}: llm call failed: {e}")
                if not best_cluster_report:
                    if isinstance(e, LLMGenerationError):
                        reasons = {"llm_generation_error": [("llm api call must succeed to return cluster report", str(e))]}
                    else:
                        reasons = {"validation_error": [("llm structured output must be valid json matching cluster report schema", str(e))]}
                    failed_judge = JudgeOutput(
                        sub_scores=None,
                        overall=0.0,
                        reasons=reasons,
                    )
                    best_score = float('-inf')  # "-inf" means a failed llm call is even worser than below the retry threshold.
                    best_judge_result = failed_judge
                continue

        # All judge retries exhausted - return best result from retry history
        logger.warning(
            f"cluster {pk_hash} attempt {max_retries+1}/{max_retries+1}: judge retries exhausted, returning best cluster_report (overall score: {best_score:.2f})"
        )
        return best_cluster_report, best_judge_result
            
    
    # Process clusters in parallel using asyncio.gather
    clusters = best_run_response.clusters
    passed_clusters = 0
    logger.info(f"Processing {len(clusters)} clusters in parallel")
    
    # Create tasks for all clusters
    top_n_paper = cfg.cluster_summarization.top_n
    tasks = []
    pk_hash_list = []
    for cluster_card in clusters:
        pk_hash = cluster_card.pk_hash
        tasks.append(process_single_cluster(cluster_card, top_n_paper=top_n_paper))
        pk_hash_list.append(pk_hash)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and build results_dict
    results_dict = {}
    for pk_hash, result in zip(pk_hash_list, results):
        if isinstance(result, Exception):
            # Handle unexpected exceptions (LLM/Validation errors are caught inside retry loop)
            logger.error(f"Unexpected error collecting cluster with pk_hash {pk_hash}: {result}", exc_info=True)
            failed_judge = JudgeOutput(
                sub_scores=None,
                overall=0.0,
                reasons={"internal_error": [("single cluster processing must not raise unexpected exceptions", f"Exception: {str(result)}")]},
            )
            results_dict[pk_hash] = (None, failed_judge)
        else:
            cluster_report, judge_result = result
            results_dict[pk_hash] = (cluster_report, judge_result)
            if cluster_report is None:
                logger.warning(f"Cluster with pk_hash {pk_hash} returned None (processing failed, overall score: {judge_result.overall})")
            else:
                passed_clusters += 1
    
    logger.info(f"cluster summarization is finished, got {passed_clusters}/{len(clusters)} non empty cluster report(s)")
    
    # Write cluster summarization events if path is provided
    _save_cluster_summarization_events(cfg, results_dict)
    
    return results_dict


def convert_cluster_reports_to_memo_payload(
    cluster_reports: Dict[str, Tuple[Optional[ClusterReport], JudgeOutput]],
    cfg: HFDataPipeConfig
) -> InjectClustersObservationInput:
    """
    Convert cluster reports from summarize_clusters_parallel to memo inject_clusters_observation payload format.
    
    Args:
        cluster_reports: Dictionary mapping pk_hash to tuple of (Optional[ClusterReport], JudgeOutput)
        cfg: HFDataPipeConfig instance
    
    Returns:
        InjectClustersObservationInput dict mapping pk_hash to ClusterObservation
    """
    # Load template content to compute prompt_hash
    template_path = cfg.cluster_summarization.prompt_template_path
    template_content = load_template(template_path)
    
    # Compute prompt_hash: SHA256 hash of template_content
    prompt_hash_hex = hashlib.sha256(template_content.encode('utf-8')).hexdigest()
    prompt_hash = f"sha256:{prompt_hash_hex}"
    
    # Build LLM config
    llm_config_id = f"{cfg.cluster_summarization.llm_gemini.model}|{prompt_hash}"
    llm_config_json_payload = {
        "provider": "google",
        "model": cfg.cluster_summarization.llm_gemini.model,
        "temperature": cfg.cluster_summarization.llm_gemini.temperature,
        "max_tokens": cfg.cluster_summarization.llm_gemini.max_tokens,
    }
    llm_config = LLMConfigInput(
        llm_config_id=llm_config_id,
        json_payload=llm_config_json_payload
    )
    
    # Build payload
    payload: InjectClustersObservationInput = {}
    
    for pk_hash, (cluster_report, judge_result) in cluster_reports.items():
        # Skip if cluster_report is None (failed clusters)
        # Note: Empty cluster reports from LLM (where cluster_report is None) are NOT injected
        # into memo DB. Only clusters with valid ClusterReport objects are included in the payload.
        if cluster_report is None:
            logger.warning(f"Skipping failed cluster with pk_hash {pk_hash}")
            continue
        
        # Use ClusterReport.model_dump() as payload_json
        payload_json = cluster_report.model_dump()
        
        # Extract fields from ClusterReport
        title = cluster_report.title
        summary = cluster_report.what_this_topic_is_about + "\n" + cluster_report.why_it_matters
        keywords_json = cluster_report.keyword_list
        
        # Extract score from judge_result
        score = judge_result.overall
        
        # Create ClusterObservation
        observation = ClusterObservation(
            llm_config=llm_config,
            payload_json=payload_json,
            summary=summary,
            title=title,
            keywords_json=keywords_json,
            score=score
        )
        
        payload[pk_hash] = observation
    
    logger.info(f"Converted {len(payload)} cluster reports to memo payload format")
    return payload

