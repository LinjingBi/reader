"""Main evaluation runner"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
from pydantic import BaseModel

# Add eval directory to path
sys.path.insert(0, str(Path(__file__).parent))

from packers.p0 import build_input, get_packer_stats
from judges.heuristics import judge_output
from utils.llm_client import LLMClient
from utils.logger import EvalLogger
from utils.template_loader import load_template, compute_template_hash, render_template, render_template_per_cluster
from schemas.cluster_response import ClusterReport
from schemas.output_models import OutputSchema, ClusterCard, RepresentativePaperOutput, ReadingOrderItemOutput


# Pydantic models for config validation
class PromptVariant(BaseModel):
    id: str
    path: str


class PromptsConfig(BaseModel):
    variants: List[PromptVariant]


class DatasetConfig(BaseModel):
    months: List[str]
    input_dir: str


class PackersConfig(BaseModel):
    variants: List[str]


class LLMConfig(BaseModel):
    provider: str
    models: List[str]
    temperature: float
    max_tokens: int
    api_key_env: str


class JudgeConfig(BaseModel):
    schema_path: str
    use_llm_judge: bool


class LoggingConfig(BaseModel):
    level: str
    console: bool
    file: bool


class EvalConfig(BaseModel):
    dataset: DatasetConfig
    packers: PackersConfig
    prompts: PromptsConfig
    llm: LLMConfig
    judge: JudgeConfig
    logging: LoggingConfig

def _load_config(config_path: Path, logger: EvalLogger) -> EvalConfig:
    """Load and validate config file with Pydantic models"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse into Pydantic models
    config = EvalConfig(**config_dict)
    
    # Filter credentials for logging
    credential_keys = {'api_key', 'api_key_env', 'password', 'secret', 'token'}
    
    def filter_dict(d, parent_key=''):
        result = {}
        for k, v in d.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            if any(cred_key in k.lower() for cred_key in credential_keys):
                continue  # Skip credential fields
            if isinstance(v, dict):
                result[k] = filter_dict(v, full_key)
            else:
                result[k] = v
        return result
    
    filtered_config = filter_dict(config_dict)
    config_yaml = yaml.dump(filtered_config, sort_keys=True, default_flow_style=False)
    logger._log("config.loaded", f"\n{config_yaml}\n", level=logging.INFO)
    
    return config


def generate_run_id(month: str, packing_variant: str, prompt_variant: str, model: str, temperature: float, prompt_hash: str) -> str:
    """Generate stable run ID"""
    # Format: {month}__{packing}__{prompt}__{model}__t{temp}__{hash}
    temp_str = f"t{temperature}".replace(".", "_")
    return f"{month}__{packing_variant}__{prompt_variant}__{model}__{temp_str}__{prompt_hash[:8]}"


def generate_snapshot_id(source: str, period_start: str, period_end: str) -> str:
    """Generate snapshot ID from input data"""
    return f"{source}|{period_start}|{period_end}"


def _pack_input_data(monthly_data: dict, logger: EvalLogger) -> Tuple[dict, dict]:
    """Pack input data using P0 packer"""
    packed_data, warnings = build_input(monthly_data)
    stats = get_packer_stats(packed_data)
    
    # Log packer stats at INFO level
    warnings_str = ",".join(warnings) if warnings else "[]"
    logger._log("packer.end", f"clusters={stats['clusters']}, papers={stats['papers']}, warnings=[{warnings_str}]", level=logging.INFO)
    
    # Log packed data content at DEBUG level only
    logger._log("packer.data", json.dumps(packed_data, indent=2, ensure_ascii=False), level=logging.DEBUG)
    
    return packed_data, stats


def _generate_prompt(template_path: Path, output_schema_path: Path, packed_data: dict, prompt_file_path: Path, logger: EvalLogger) -> Tuple[str, str]:
    """Generate complete prompt using template"""
    # Load template
    template_content = load_template(template_path)
    
    # Compute template hash
    template_hash = compute_template_hash(template_content)
    
    # Render template
    rendered_prompt = render_template(
        template_content=template_content,
        output_schema_path=output_schema_path,
        input_data=packed_data
    )
    
    # Save rendered prompt to file
    prompt_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        f.write(rendered_prompt)
    
    # Log template hash at INFO level
    logger._log("prompt.hash", f"template_hash={template_hash}", level=logging.INFO)
    
    # Log prompt content at DEBUG level only
    logger._log("prompt.content", rendered_prompt, level=logging.DEBUG)
    
    return rendered_prompt, template_hash


def _call_llm(llm_config: LLMConfig, model: str, prompt: str, logger: EvalLogger) -> str:
    """Call LLM and return raw text response"""
    # Initialize LLM client
    api_key = os.getenv(llm_config.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {llm_config.api_key_env}")
    
    llm_client = LLMClient(
        provider=llm_config.provider,
        model=model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        api_key=api_key
    )
    
    # Log call metadata at INFO level
    logger._log("llm.call", f"model={model}, api_uri={llm_client.api_uri}, args={llm_client.get_api_args()}", level=logging.INFO)
    
    # Call LLM (with retry logic built in)
    raw_response = llm_client.call(prompt)
    
    # Log response content at DEBUG level only
    logger._log("llm.response", raw_response, level=logging.DEBUG)
    
    return raw_response


def _call_llm_per_cluster(
    llm_config: LLMConfig,
    model: str,
    cluster_data: dict,
    template_content: str,
    logger: EvalLogger,
    run_results_dir: Optional[Path] = None,
    cluster_index: Optional[int] = None
) -> ClusterReport:
    """
    Call LLM for a single cluster using structured output.
    
    Args:
        llm_config: LLM configuration
        model: Model name
        cluster_data: Single cluster dict with papers array
        template_content: Prompt template content
        logger: Logger instance
        run_results_dir: Optional directory to save raw responses
        cluster_index: Optional cluster index for saving raw responses
    
    Returns:
        Parsed ClusterReport Pydantic model
    """
    # Initialize LLM client
    api_key = os.getenv(llm_config.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {llm_config.api_key_env}")
    
    llm_client = LLMClient(
        provider=llm_config.provider,
        model=model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        api_key=api_key
    )
    
    # Render template for this cluster
    prompt = render_template_per_cluster(template_content, cluster_data)
    
    # Log call metadata at INFO level
    logger._log("llm.call", f"model={model}, cluster_size={len(cluster_data.get('papers', []))}, args={llm_client.get_api_args()}", level=logging.INFO)
    
    # Get raw response first (so we can save it even if validation fails)
    raw_response = llm_client.call_structured_raw(prompt, ClusterReport)
    
    # Save raw response if we have a directory (before validation)
    if run_results_dir and cluster_index is not None:
        try:
            sanitized_model = model.replace("/", "_")
            raw_response_path = run_results_dir / f"raw_response_{sanitized_model}_cluster_{cluster_index}.json"
            with open(raw_response_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": model,
                    "cluster_index": cluster_index,
                    "raw_response": raw_response
                }, f, indent=2, ensure_ascii=False)
            logger._log("llm.raw_saved", f"Raw response saved to {raw_response_path}", level=logging.DEBUG)
        except Exception as save_error:
            logger._log("llm.raw_save_failed", f"Failed to save raw response: {str(save_error)}", level=logging.WARNING)
    
    # Validate raw response
    try:
        cluster_report = ClusterReport.model_validate(raw_response)
        
        # Log response content at DEBUG level only
        logger._log("llm.response", cluster_report.model_dump_json(indent=2, ensure_ascii=False), level=logging.DEBUG)
        
        return cluster_report
    except Exception as e:
        # Update saved raw response with validation error if we saved it
        if run_results_dir and cluster_index is not None:
            try:
                sanitized_model = model.replace("/", "_")
                raw_response_path = run_results_dir / f"raw_response_{sanitized_model}_cluster_{cluster_index}.json"
                with open(raw_response_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model": model,
                        "cluster_index": cluster_index,
                        "raw_response": raw_response,
                        "validation_error": str(e)
                    }, f, indent=2, ensure_ascii=False)
                logger._log("llm.raw_updated", f"Raw response updated with validation error at {raw_response_path}", level=logging.INFO)
            except Exception:
                pass  # Ignore errors updating the file
        raise


def _normalize_confidence(confidence: str) -> str:
    """Normalize confidence from HIGH/MEDIUM/LOW to high/medium/low"""
    confidence_map = {
        "HIGH": "high",
        "MEDIUM": "medium",
        "LOW": "low"
    }
    return confidence_map.get(confidence.upper(), confidence.lower())


def _format_confidence_rationale(rationale_list: List[str]) -> str:
    """Format confidence_rationale from list to single string"""
    return "\n".join(f"- {item}" for item in rationale_list)


def _enrich_reading_order(
    reading_order_items: List[Any],
    papers_lookup: Dict[str, dict]
) -> List[Dict[str, str]]:
    """
    Enrich reading order items with URLs from paper data.
    
    Args:
        reading_order_items: List of ReadingOrderItem from ClusterReport
        papers_lookup: Dict mapping paper_id to paper data with url
    
    Returns:
        List of dicts with paper_id, url, why_read_next
    """
    enriched = []
    for item in reading_order_items:
        paper_id = item.paper_id if hasattr(item, 'paper_id') else item.get('paper_id')
        why_read_now = item.why_read_now if hasattr(item, 'why_read_now') else item.get('why_read_now', '')
        
        paper = papers_lookup.get(paper_id, {})
        url = paper.get('url', '')
        
        enriched.append({
            "paper_id": paper_id,
            "url": url,
            "why_read_next": why_read_now
        })
    return enriched


def _map_cluster_report_to_card(
    cluster_report: ClusterReport,
    cluster_index: int,
    papers_lookup: Dict[str, dict]
) -> Dict[str, Any]:
    """
    Map ClusterReport to cluster_card dict matching output schema.
    
    Args:
        cluster_report: ClusterReport from LLM
        cluster_index: Cluster index for cluster_key
        papers_lookup: Dict mapping paper_id to paper data
    
    Returns:
        Dict matching ClusterCard structure
    """
    section_a = cluster_report.section_a
    section_b = cluster_report.section_b
    
    # Map representative papers
    representative_papers = [
        {
            "paper_id": paper.paper_id,
            "reason_representative": paper.title  # Using title as reason for now
        }
        for paper in section_a.representative_papers
    ]
    
    # Enrich reading order with URLs
    reading_order = _enrich_reading_order(section_a.reading_order, papers_lookup)
    
    # Format notes from list to string
    notes_str = "\n".join(f"- {note}" for note in section_a.notes) if section_a.notes else ""
    
    # Use keywords from section_b (5-12 items), take up to 7 to match output schema constraint (3-7)
    tags = section_b.keyword_list[:7] if len(section_b.keyword_list) > 7 else section_b.keyword_list
    
    return {
        "cluster_key": f"cluster_index:{cluster_index}",
        "topic_name": section_a.title,
        "one_liner": section_a.one_liner,
        "tags": tags,
        "what_this_cluster_is_about": section_a.what_this_cluster_is_about,
        "why_it_matters": section_a.why_it_matters,
        "confidence": _normalize_confidence(section_a.confidence),
        "confidence_rationale": _format_confidence_rationale(section_a.confidence_rationale),
        "representative_papers": representative_papers,
        "reading_order": reading_order,
        "search_query_seed": section_a.search_query_seed,
        "notes": notes_str
    }


def _synthesize_output(
    cluster_cards: List[Dict[str, Any]],
    snapshot_id: str,
    period_start: str,
    period_end: str
) -> Dict[str, Any]:
    """
    Synthesize cluster cards into final output format.
    
    Args:
        cluster_cards: List of cluster card dicts
        snapshot_id: Snapshot ID
        period_start: Period start date
        period_end: Period end date
    
    Returns:
        Dict matching OutputSchema structure
    """
    return {
        "snapshot_id": snapshot_id,
        "cluster_run_id": "",  # Will be set by caller if needed
        "period_start": period_start,
        "period_end": period_end,
        "cluster_cards": cluster_cards
    }


def _process_clusters_parallel(
    llm_config: LLMConfig,
    model: str,
    packed_data: dict,
    template_content: str,
    monthly_data: dict,
    snapshot_id: str,
    period_start: str,
    period_end: str,
    logger: EvalLogger,
    run_results_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Process all clusters in parallel and synthesize output.
    
    Args:
        llm_config: LLM configuration
        model: Model name
        packed_data: Packed data with clusters
        template_content: Prompt template content
        monthly_data: Original monthly data for paper lookup
        snapshot_id: Snapshot ID
        period_start: Period start date
        period_end: Period end date
        logger: Logger instance
    
    Returns:
        Synthesized output dict matching OutputSchema
    """
    clusters = packed_data.get("clusters", [])
    
    # Build papers lookup for enriching reading order
    papers_lookup = {paper["paper_id"]: paper for paper in monthly_data.get("papers", [])}
    
    # Process clusters in parallel
    cluster_cards = []
    cluster_reports = []
    
    def process_single_cluster(cluster_data: dict, cluster_idx: int) -> Tuple[int, Optional[ClusterReport]]:
        """Process a single cluster and return (index, report)"""
        try:
            # Extract just the papers for the prompt
            cluster_for_prompt = {"papers": cluster_data.get("papers", [])}
            report = _call_llm_per_cluster(
                llm_config=llm_config,
                model=model,
                cluster_data=cluster_for_prompt,
                template_content=template_content,
                logger=logger,
                run_results_dir=run_results_dir,
                cluster_index=cluster_idx
            )
            return cluster_idx, report
        except Exception as e:
            logger._log("cluster.error", f"Cluster {cluster_idx} failed: {str(e)}", level=logging.ERROR)
            logger._log("cluster.error", f"Cluster {cluster_idx} failed: {str(e)}\n{traceback.format_exc()}", level=logging.DEBUG)
            return cluster_idx, None
    
    # Process clusters in parallel
    with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
        futures = []
        for i, cluster in enumerate(clusters):
            cluster_index = cluster.get("cluster_index", i)
            future = executor.submit(process_single_cluster, cluster, cluster_index)
            futures.append((future, cluster_index))
        
        # Collect results
        for future, cluster_index in futures:
            try:
                idx, report = future.result()
                if report:
                    cluster_reports.append((cluster_index, report))
            except Exception as e:
                logger._log("cluster.collect_error", f"Error collecting cluster result: {str(e)}", level=logging.ERROR)
    
    # Map reports to cluster cards
    for cluster_index, report in cluster_reports:
        try:
            cluster_card = _map_cluster_report_to_card(report, cluster_index, papers_lookup)
            cluster_cards.append(cluster_card)
        except Exception as e:
            logger._log("cluster.map_error", f"Error mapping cluster {cluster_index}: {str(e)}", level=logging.ERROR)
    
    # Synthesize final output
    output = _synthesize_output(cluster_cards, snapshot_id, period_start, period_end)
    
    return output


def _judge_output(output_json: dict, monthly_data: dict, schema_path: Path, response_file_path: Path, logger: EvalLogger) -> Tuple[Optional[dict], Optional[dict]]:
    """Judge output: validate using Pydantic and heuristics"""
    # Save output JSON to file
    response_file_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = response_file_path.stem.replace("model_output_", "")
    
    json_path = response_file_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"model": model_name, **output_json}, f, indent=2, ensure_ascii=False)
    
    try:
        # Call judge (now handles Pydantic validation internally)
        judge_result = judge_output(
            output_json=output_json,
            input_data=monthly_data,
            schema_path=schema_path
        )
        
        # Save judge result
        judge_result_path = response_file_path.parent / f"judge_result_{model_name}.json"
        with open(judge_result_path, 'w', encoding='utf-8') as f:
            json.dump({"model": model_name, **judge_result}, f, indent=2, ensure_ascii=False)
        
        # Log judge results at INFO level
        logger._log("judge.results", 
                   f"schema_valid={'pass' if judge_result['schema_valid']['passed'] else 'fail'}, "
                   f"citations_ok={'pass' if judge_result['citations_ok']['passed'] else 'fail'}, "
                   f"length_ok={'pass' if judge_result['length_ok']['passed'] else 'fail'}, "
                   f"name_generic_penalty={judge_result['scores']['name_generic_penalty']}",
                   level=logging.INFO)
        
        return judge_result, output_json
        
    except Exception as e:
        # Log error at ERROR level
        logger._log("judge.error", f"error={str(e)}", level=logging.ERROR)
        # Log traceback at DEBUG level
        logger._log("judge.error", f"error={str(e)}\n{traceback.format_exc()}", level=logging.DEBUG)
        return None, None


def _generate_summary(run_id: str, snapshot_id: str, month: str, packing_variant: str, prompt_variant: str, template_hash: str, models: List[str], model_results: List[dict], response_file_paths: Dict[str, str], run_results_dir: Path, logger: EvalLogger) -> dict:
    """Generate summary of evaluation results"""
    # Aggregate scores across all models
    if model_results:
        aggregate_scores = {
            "schema_valid": sum(r["scores"]["schema_valid"] for r in model_results) / len(model_results),
            "citations_ok": sum(r["scores"]["citations_ok"] for r in model_results) / len(model_results),
            "length_ok": sum(r["scores"]["length_ok"] for r in model_results) / len(model_results),
        }
    else:
        aggregate_scores = {
            "schema_valid": 0.0,
            "citations_ok": 0.0,
            "length_ok": 0.0,
        }
    
    # Add response file paths to model results
    for result in model_results:
        model = result["model"]
        if model in response_file_paths:
            result["response_file"] = response_file_paths[model]
    
    # Create summary dict
    summary = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "month": month,
        "packing_variant": packing_variant,
        "prompt_variant": prompt_variant,
        "prompt_hash": template_hash,
        "models_evaluated": models,
        "model_results": model_results,
        "aggregate_scores": aggregate_scores,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save summary
    summary_path = run_results_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Log summary at INFO level
    logger._log("summary.generated", f"Summary saved to {summary_path}", level=logging.INFO)
    
    return summary


def load_monthly_data(data_dir: str, month: str) -> dict:
    """Load monthly clustering JSON"""
    month_file = os.path.join(data_dir, f"{month}.json")
    if not os.path.exists(month_file):
        raise FileNotFoundError(f"Monthly data file not found: {month_file}")
    
    with open(month_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_json(text: str) -> dict:
    # 1. Try fenced code block
    match = re.search(
        r"```(?:json)?\s*\n([\s\S]*?)\n```",
        text,
        re.IGNORECASE
    )
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try raw JSON object
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        print(f"Found JSON object: {match.group(1)}")
        try:
            return json.loads(match.group(1))
        except Exception:
            traceback.print_exc()
            raise

    raise ValueError("No JSON found")

def _process_single_model(
    model: str,
    llm_config: LLMConfig,
    packed_data: dict,
    template_content: str,
    monthly_data: dict,
    snapshot_id: str,
    period_start: str,
    period_end: str,
    schema_path: Path,
    response_file_path: Path,
    run_results_dir: Path,
    logger: EvalLogger
) -> Tuple[Dict[str, Any], str]:
    """
    Process a single model evaluation using per-cluster parallel processing.
    
    Returns:
        Tuple of (result_dict, response_file_path_str)
    """
    model_start_time = time.time()
    
    try:
        # Step 4: Process clusters in parallel
        output_json = _process_clusters_parallel(
            llm_config=llm_config,
            model=model,
            packed_data=packed_data,
            template_content=template_content,
            monthly_data=monthly_data,
            snapshot_id=snapshot_id,
            period_start=period_start,
            period_end=period_end,
            logger=logger,
            run_results_dir=run_results_dir
        )
        
        # Step 5: Judge output (validates using Pydantic and heuristics)
        judge_result, parsed_output = _judge_output(
            output_json=output_json,
            monthly_data=monthly_data,
            schema_path=schema_path,
            response_file_path=response_file_path,
            logger=logger
        )
        
        # Compute overall scores
        if judge_result:
            overall_scores = {
                "schema_valid": 1.0 if judge_result["schema_valid"]["passed"] else 0.0,
                "citations_ok": 1.0 if judge_result["citations_ok"]["passed"] else 0.0,
                "length_ok": 1.0 if judge_result["length_ok"]["passed"] else 0.0,
            }
        else:
            overall_scores = {
                "schema_valid": 0.0,
                "citations_ok": 0.0,
                "length_ok": 0.0,
            }
        
        duration = time.time() - model_start_time
        logger._log("run.model_end", f"[MODEL: {model}] - duration={duration:.3f}s, overall_scores={overall_scores}", level=logging.INFO)
        
        # Return relative path for summary
        response_file_rel = str(response_file_path.relative_to(run_results_dir))
        
        return {
            "model": model,
            "scores": overall_scores,
            "duration": duration,
            "parsed_output": parsed_output is not None
        }, response_file_rel
        
    except Exception as e:
        duration = time.time() - model_start_time
        logger._log("run.model_error", f"[MODEL: {model}] - Failed with error: {str(e)}", level=logging.ERROR)
        logger._log("run.model_error", f"[MODEL: {model}] - Failed with error: {str(e)}\n{traceback.format_exc()}", level=logging.DEBUG)
        
        # Return relative path for summary even on error
        response_file_rel = str(response_file_path.relative_to(run_results_dir))
        
        return {
            "model": model,
            "scores": {"schema_valid": 0.0, "citations_ok": 0.0, "length_ok": 0.0},
            "duration": duration,
            "parsed_output": False,
            "error": str(e)
        }, response_file_rel


def run_evaluation(config_path: Path, month: str, dry_run: bool) -> None:
    """
    Run evaluation for a single month.
    
    Args:
        config_path: Path to eval.yaml config file
        month: Month key (e.g., "2025-01")
        dry_run: If True, skip LLM calls
    """
    start_time = time.time()
    
    # Setup paths
    eval_dir = Path(__file__).parent
    results_dir = eval_dir / "results"
    
    # Step 1: Load config
    # Create a temporary logger for config loading (before we know run_id)
    temp_logger = EvalLogger("temp", None)
    config = _load_config(config_path, temp_logger)
    
    # Setup data directory
    data_dir = eval_dir / config.dataset.input_dir
    
    # Load monthly data
    monthly_data = load_monthly_data(str(data_dir), month)
    
    # Extract metadata
    source = monthly_data.get("source", "hf_monthly")
    period_start = monthly_data.get("period_start", "")
    period_end = monthly_data.get("period_end", "")
    snapshot_id = generate_snapshot_id(source, period_start, period_end)
    
    # Get prompt variant (first one for MVP)
    prompt_variant_config = config.prompts.variants[0]
    prompt_path = eval_dir / prompt_variant_config.path
    packing_variant = config.packers.variants[0]  # "P0"
    
    # Get models list
    models = config.llm.models
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("'models' must be a non-empty list")
    
    # Compute template hash before generating run_id (hash is based on template content, not rendered prompt)
    template_content = load_template(prompt_path)
    template_hash = compute_template_hash(template_content)
    
    # Generate base run ID
    base_run_id = generate_run_id(
        month=month,
        packing_variant=packing_variant,
        prompt_variant=prompt_variant_config.id,
        model=models[0].replace("/", "_"),  # Use first model for base ID
        temperature=config.llm.temperature,
        prompt_hash=template_hash
    )
    
    # Setup results directory
    run_results_dir = results_dir / base_run_id
    # Delete existing directory if it exists
    if run_results_dir.exists():
        shutil.rmtree(run_results_dir)
    run_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging (shared logger for all models)
    logger = EvalLogger(base_run_id, run_results_dir)
    
    # Log run start
    logger.run_start(snapshot_id)
    logger._log("run.models", f"Evaluating {len(models)} models: {', '.join(models)}", level=logging.INFO)
    
    # Step 2: Pack input data
    packed_data, stats = _pack_input_data(monthly_data, logger)
    
    # Step 3: Load template (per-cluster template, no rendering needed yet)
    output_schema_path = eval_dir / config.judge.schema_path
    
    # Log template hash at INFO level
    logger._log("prompt.hash", f"template_hash={template_hash}", level=logging.INFO)
    # Copy prompt template file to results directory for record-keeping
    prompt_template_path = run_results_dir / prompt_variant_config.path.name
    shutil.copy(prompt_path, prompt_template_path)
    logger._log("prompt.template", f"Prompt template saved to {prompt_template_path}", level=logging.INFO)
    
    if dry_run:
        logger._log("run.dry_run", "Dry run mode - skipping LLM calls", level=logging.INFO)
        return
    
    # Process all models in parallel
    logger._log("run.parallel_start", f"Starting parallel evaluation for {len(models)} models", level=logging.INFO)
    
    model_results = []
    response_file_paths = {}
    
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # Submit all model evaluation tasks
        future_to_model = {}
        for model in models:
            # Sanitize model name for file paths
            sanitized_model = model.replace("/", "_")
            response_file_path = run_results_dir / f"model_output_{sanitized_model}.json"
            
            future = executor.submit(
                _process_single_model,
                model=model,
                llm_config=config.llm,
                packed_data=packed_data,
                template_content=template_content,
                monthly_data=monthly_data,
                snapshot_id=snapshot_id,
                period_start=period_start,
                period_end=period_end,
                schema_path=output_schema_path,
                response_file_path=response_file_path,
                run_results_dir=run_results_dir,
                logger=logger
            )
            future_to_model[future] = model
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result, response_file_rel = future.result()
                model_results.append(result)
                response_file_paths[model] = response_file_rel
                logger._log("run.model_complete", f"[MODEL: {model}] - Completed successfully", level=logging.INFO)
            except Exception as e:
                logger._log("run.model_error", f"[MODEL: {model}] - Failed with error: {str(e)}", level=logging.ERROR)
                logger._log("run.model_error", f"[MODEL: {model}] - Failed with error: {str(e)}\n{traceback.format_exc()}", level=logging.DEBUG)
                # Get response file path even on error
                sanitized_model = model.replace("/", "_")
                response_file_rel = f"model_output_{sanitized_model}.json"
                response_file_paths[model] = response_file_rel
                model_results.append({
                    "model": model,
                    "scores": {"schema_valid": 0.0, "citations_ok": 0.0, "length_ok": 0.0},
                    "duration": 0.0,
                    "parsed_output": False,
                    "error": str(e)
                })
    
    # Aggregate results across all models
    logger._log("run.parallel_complete", f"All {len(models)} models completed", level=logging.INFO)
    
    # Step 6: Generate summary
    summary = _generate_summary(
        run_id=base_run_id,
        snapshot_id=snapshot_id,
        month=month,
        packing_variant=packing_variant,
        prompt_variant=prompt_variant_config.id,
        template_hash=template_hash,
        models=models,
        model_results=model_results,
        response_file_paths=response_file_paths,
        run_results_dir=run_results_dir,
        logger=logger
    )
    
    # Log run end with aggregate scores
    duration = time.time() - start_time
    logger.run_end(duration, summary["aggregate_scores"])
    
    # Log individual model scores
    for result in model_results:
        scores_str = ",".join(f"{k}={v}" for k, v in result["scores"].items())
        logger._log("run.model_summary", 
                   f"[MODEL: {result['model']}] - duration={result['duration']:.3f}s, scores={{{scores_str}}}", 
                   level=logging.INFO)
    
    print(f"\nEvaluation complete. Results saved to: {run_results_dir}")
    print(f"Evaluated {len(models)} models: {', '.join(models)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to eval config file"
    )
    parser.add_argument(
        "--month",
        type=str,
        required=True,
        help="Month key (e.g., '2025-01')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the evaluation pipeline"
    )
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    run_evaluation(Path(config_path), args.month, args.dry_run)


if __name__ == "__main__":
    main()

