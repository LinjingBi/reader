"""
Main evaluation runner.

Workflow:
1. Load configuration from YAML file
2. Load monthly clustering data (JSON)
3. Pack input data using P0 packer (transforms raw data into prompt-ready format)
4. Load prompt template and compute hash for run ID generation
5. For each model (in parallel):
   a. Process all clusters in parallel:
      - Call LLM with per-cluster prompt template
      - Validate response using judge (JSON parsing, schema validation, citations)
      - Save raw responses per cluster
   b. Synthesize cluster reports into final output JSON
   c. Save model output and compute scores
6. Generate summary with aggregate scores across all models
7. Save summary.json with run metadata and results
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
from pydantic import BaseModel
from dataclasses import asdict

# Add eval directory to path
sys.path.insert(0, str(Path(__file__).parent))

from packers.p0 import build_input, get_packer_stats
from judges.heuristics import judge_output, JudgeOutput
from utils.llm_client import LLMClient, TokenBucket
from utils.logger import EvalLogger
from utils.template_loader import load_template, compute_template_hash, render_template_per_cluster
from schemas.cluster_response import ClusterReport


# Pydantic models for config validation
class PromptVariant(BaseModel):
    id: str
    path: str


class PromptsConfig(BaseModel):
    variants: List[PromptVariant]


class DatasetConfig(BaseModel):
    input_dir: str


class PackersConfig(BaseModel):
    variants: List[str]


class LLMConfig(BaseModel):
    provider: str
    models: List[str]
    temperature: float
    max_tokens: int
    api_key_env: str
    gemini_rpm_limit: Optional[int] = 15
    gemini_tpm_limit: Optional[int] = 250000
    openai_rpm_limit: Optional[int] = 15
    openai_tpm_limit: Optional[int] = 250000


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
    """
    Load and validate config file with Pydantic models.
    
    Args:
        config_path: Path to YAML config file
        logger: Logger instance for logging config loading
        
    Returns:
        Validated EvalConfig object
    """
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
    """
    Generate stable run ID from run parameters.
    
    Args:
        month: Month key (e.g., "2025-01")
        packing_variant: Packing variant name (e.g., "P0")
        prompt_variant: Prompt variant ID
        model: Model name
        temperature: Temperature setting
        prompt_hash: Template hash (first 8 chars)
        
    Returns:
        Run ID string in format: {month}__{packing}__{prompt}__{model}__t{temp}__{hash}
    """
    # Format: {month}__{packing}__{prompt}__{model}__t{temp}__{hash}
    temp_str = f"t{temperature}".replace(".", "_")
    return f"{month}__{packing_variant}__{prompt_variant}__{model}__{temp_str}__{prompt_hash[:8]}"


def generate_snapshot_id(source: str, period_start: str, period_end: str) -> str:
    """
    Generate snapshot ID from input data metadata.
    
    Args:
        source: Data source identifier
        period_start: Period start date string
        period_end: Period end date string
        
    Returns:
        Snapshot ID string in format: {source}|{period_start}|{period_end}
    """
    return f"{source}|{period_start}|{period_end}"


def _pack_input_data(monthly_data: Dict[str, Any], logger: EvalLogger) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Pack input data using P0 packer.
    
    Args:
        monthly_data: Raw monthly clustering data dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (packed_data, stats):
        - packed_data: Packed data dictionary ready for prompts
        - stats: Statistics dictionary with cluster and paper counts
    """
    packed_data, warnings = build_input(monthly_data)
    stats = get_packer_stats(packed_data)
    
    # Log packer stats at INFO level
    warnings_str = ",".join(warnings) if warnings else "[]"
    logger._log("packer.end", f"clusters={stats['clusters']}, papers={stats['papers']}, warnings=[{warnings_str}]", level=logging.INFO)
    
    # Log packed data content at DEBUG level only
    logger._log("packer.data", json.dumps(packed_data, indent=2, ensure_ascii=False), level=logging.DEBUG)
    
    return packed_data, stats


def _call_llm_per_cluster(
    llm_client: LLMClient,
    model: str,
    cluster_data: Dict[str, Any],
    template_content: str,
    logger: EvalLogger,
    run_results_dir: Optional[Path] = None,
    cluster_index: Optional[int] = None
) -> Tuple[JudgeOutput, Optional[ClusterReport]]:
    """
    Call LLM for a single cluster using structured output.
    
    Args:
        llm_client: LLM client instance (with token buckets configured)
        model: Model name
        cluster_data: Single cluster dict with papers array
        template_content: Prompt template content
        logger: Logger instance
        run_results_dir: Optional directory to save raw responses
        cluster_index: Optional cluster index for saving raw responses
    
    Returns:
        Tuple of (JudgeOutput, Optional[ClusterReport]):
        - JudgeOutput: Contains sub_scores, overall, and reasons
        - ClusterReport: Validated Pydantic object if validation passes, None otherwise
    """
    # Render template for this cluster
    prompt = render_template_per_cluster(template_content, cluster_data)
    
    # Log call metadata at INFO level
    logger._log("llm.call", f"model={model}, cluster_size={len(cluster_data.get('papers', []))}, args={llm_client.get_api_args()}", level=logging.INFO)
    
    # Get raw response text (so we can save it even if validation fails)
    raw_response_text = llm_client.call_structured_raw(prompt, ClusterReport)

    sanitized_model = model.replace("/", "_")
    raw_response_path = run_results_dir / f"raw_response_{sanitized_model}_cluster_{cluster_index}.json"
    with open(raw_response_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": model,
            "cluster_index": cluster_index,
            "raw_response_text": raw_response_text,
        }, f, indent=2, ensure_ascii=False)
    logger._log("llm.raw_saved", f"Raw response saved to {raw_response_path}", level=logging.INFO)
    
    return judge_output(raw_response_text, cluster_data)


def _process_clusters_parallel(
    llm_config: LLMConfig,
    model: str,
    packed_data: Dict[str, Any],
    template_content: str,
    monthly_data: Dict[str, Any],
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
        run_results_dir: Optional directory to save raw responses
    
    Returns:
        Dictionary with synthesized output containing:
        - snapshot_id: Snapshot identifier
        - cluster_run_id: Cluster run identifier (empty string)
        - period_start: Period start date
        - period_end: Period end date
        - cluster_reports: List of cluster report dictionaries
        - cluster_judges: List of judge output dictionaries
    """
    clusters = packed_data.get("clusters", [])   
    
    # Get API key from environment variable
    api_key = os.getenv(llm_config.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {llm_config.api_key_env}")
    
    # Determine provider-specific rate limits
    if llm_config.provider == "gemini":
        rpm_limit = llm_config.gemini_rpm_limit
        tpm_limit = llm_config.gemini_tpm_limit
    elif llm_config.provider == "openai":
        rpm_limit = llm_config.openai_rpm_limit
        tpm_limit = llm_config.openai_tpm_limit
    else:
        raise ValueError(f"Unsupported provider: {llm_config.provider}")
    
    # Initialize token buckets per thread pool
    req_bucket = TokenBucket(
        capacity=rpm_limit,
        refill_rate=rpm_limit / 60.0,
        name=f"{llm_config.provider}_rpm"
    )
    
    tpm_bucket = TokenBucket(
        capacity=tpm_limit,
        refill_rate=tpm_limit / 60.0,
        name=f"{llm_config.provider}_tpm"
    )
    
    # Create LLM client once per thread pool
    llm_client = LLMClient(
        provider=llm_config.provider,
        model=model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        api_key=api_key,
        rpm_bucket=req_bucket,
        tpm_bucket=tpm_bucket
    )
    
    # Process clusters in parallel
    cluster_reports = []
    cluster_judges = []
    
    def process_single_cluster(cluster_data: Dict[str, Any], cluster_idx: int) -> Tuple[Optional[JudgeOutput], Optional[ClusterReport]]:
        """
        Process a single cluster and return judge output and report.
        
        Args:
            cluster_data: Cluster data dictionary
            cluster_idx: Cluster index
            
        Returns:
            Tuple of (Optional[JudgeOutput], Optional[ClusterReport]):
            - JudgeOutput: Judge results if successful, None on error
            - ClusterReport: Parsed cluster report if successful, None on error
        """
        try:
            # Extract just the papers for the prompt
            cluster_for_prompt = {"papers": cluster_data.get("papers", [])}
            judge_output, report = _call_llm_per_cluster(
                llm_client=llm_client,
                model=model,
                cluster_data=cluster_for_prompt,
                template_content=template_content,
                logger=logger,
                run_results_dir=run_results_dir,
                cluster_index=cluster_idx
            )
            return judge_output, report
        except Exception as e:
            logger._log("cluster.error", f"Cluster {cluster_idx} failed: {str(e)}", level=logging.ERROR)
            logger._log("cluster.error", f"Cluster {cluster_idx} failed: {str(e)}\n{traceback.format_exc()}", level=logging.DEBUG)
            return None, None
    
    # Process clusters in parallel
    with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
        futures = []
        for i, cluster in enumerate(clusters):
            cluster_index = cluster.get("cluster_index", i)
            future = executor.submit(process_single_cluster, cluster, cluster_index)
            futures.append((future, cluster_index))
        
        # Collect results
        missing_report = []
        missing_judge = []
        for future, cluster_index in futures:
            try:
                judge_output, report = future.result()
                if judge_output and report:
                    cluster_reports.append(report.model_dump())
                    cluster_judges.append(asdict(judge_output))
                else:
                    if not report:
                        missing_report.append(cluster_index)
                    if not judge_output:
                        missing_judge.append(cluster_index)
            except Exception as e:
                logger._log("cluster.collect_error", f"Error collecting cluster {cluster_index} result: {str(e)}", level=logging.ERROR)
                missing_report.append(cluster_index)
                missing_judge.append(cluster_index)

    
    return {
        "snapshot_id": snapshot_id,
        "cluster_run_id": "",  # Will be set by caller if needed
        "period_start": period_start,
        "period_end": period_end,
        "cluster_reports": cluster_reports,
        "cluster_judges": cluster_judges,
        "missing_report": missing_report,
        "missing_judge": missing_judge,
    }


def _generate_summary(run_id: str, snapshot_id: str, month: str, packing_variant: str, prompt_variant: str, template_hash: str, models: List[str], model_results: List[Dict[str, Any]], response_file_paths: Dict[str, str], run_results_dir: Path, logger: EvalLogger) -> Dict[str, Any]:
    """
    Generate summary of evaluation results.
    
    Args:
        run_id: Run identifier
        snapshot_id: Snapshot identifier
        month: Month key (e.g., "2025-01")
        packing_variant: Packing variant name
        prompt_variant: Prompt variant ID
        template_hash: Template hash
        models: List of model names evaluated
        model_results: List of model result dictionaries
        response_file_paths: Dictionary mapping model names to response file paths
        run_results_dir: Directory to save summary
        logger: Logger instance
        
    Returns:
        Summary dictionary with run metadata
    """
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
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save summary
    summary_path = run_results_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Log summary at INFO level
    logger._log("summary.generated", f"Summary saved to {summary_path}", level=logging.INFO)
    
    return summary


def load_monthly_data(data_dir: str, month: str) -> Dict[str, Any]:
    """
    Load monthly clustering JSON file.
    
    Args:
        data_dir: Directory containing monthly data files
        month: Month key (e.g., "2025-01")
        
    Returns:
        Monthly data dictionary
        
    Raises:
        FileNotFoundError: If monthly data file doesn't exist
    """
    month_file = os.path.join(data_dir, f"{month}.json")
    if not os.path.exists(month_file):
        raise FileNotFoundError(f"Monthly data file not found: {month_file}")
    
    with open(month_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def _process_single_model(
    model: str,
    llm_config: LLMConfig,
    packed_data: Dict[str, Any],
    template_content: str,
    monthly_data: Dict[str, Any],
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
    
    Args:
        model: Model name
        llm_config: LLM configuration
        packed_data: Packed data dictionary
        template_content: Prompt template content
        monthly_data: Original monthly data dictionary
        snapshot_id: Snapshot identifier
        period_start: Period start date
        period_end: Period end date
        schema_path: Path to output schema (unused, kept for compatibility)
        response_file_path: Path to save model output JSON
        run_results_dir: Results directory
        logger: Logger instance
        
    Returns:
        Tuple of (result_dict, response_file_path_str):
        - result_dict: Dictionary with model, scores, duration, parsed_output, and optional error
        - response_file_path_str: Relative path to response file
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
        
        # Save output JSON to file
        response_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(response_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        
        # Compute overall scores from cluster judges
        if output_json['cluster_judges']:
            overall_scores = {
                "overall": sum(judge['overall'] for judge in output_json['cluster_judges']) / len(output_json['cluster_judges'])
            }
        else:
            overall_scores = {"overall": 0.0}
        
        duration = time.time() - model_start_time
        logger._log("run.model_end", f"[MODEL: {model}] - duration={duration:.3f}s, overall_scores={overall_scores}", level=logging.INFO)
        
        # Return relative path for summary
        response_file_rel = str(response_file_path.relative_to(run_results_dir))
        
        return {
            "model": model,
            "scores": overall_scores,
            "duration": duration,
            "parsed_output": len(output_json.get('cluster_reports', [])) > 0,
            "missing_report": output_json.get('missing_report', []),
            "missing_judge": output_json.get('missing_judge', [])
        }, response_file_rel
        
    except Exception as e:
        duration = time.time() - model_start_time
        logger._log("run.model_error", f"[MODEL: {model}] - Failed with error: {str(e)}", level=logging.ERROR)
        logger._log("run.model_error", f"[MODEL: {model}] - Failed with error: {str(e)}\n{traceback.format_exc()}", level=logging.DEBUG)
        
        # Return relative path for summary even on error
        response_file_rel = str(response_file_path.relative_to(run_results_dir))
        
        return {
            "model": model,
            "scores": {"overall": 0.0},
            "duration": duration,
            "parsed_output": False,
            "error": str(e)
        }, response_file_rel


def run_evaluation(config_path: Path, month: str, dry_run: bool) -> None:
    """
    Run evaluation for a single month.
    
    Main entry point that orchestrates the entire evaluation pipeline:
    1. Loads configuration and monthly data
    2. Packs input data
    3. Processes all models in parallel (each processing clusters in parallel)
    4. Generates summary with aggregate scores
    
    Args:
        config_path: Path to eval.yaml config file
        month: Month key (e.g., "2025-01")
        dry_run: If True, skip LLM calls and exit early
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
    prompt_template_path = run_results_dir
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
                logger._log("run.model_complete", f"[MODEL: {model}] - Completed", level=logging.INFO)
            except Exception as e:
                logger._log("run.model_error", f"[MODEL: {model}] - Failed with error: {str(e)}", level=logging.ERROR)
                logger._log("run.model_error", f"[MODEL: {model}] - Failed with error: {str(e)}\n{traceback.format_exc()}", level=logging.DEBUG)
                # Get response file path even on error
                sanitized_model = model.replace("/", "_")
                response_file_rel = f"model_output_{sanitized_model}.json"
                response_file_paths[model] = response_file_rel
                model_results.append({
                    "model": model,
                    "scores": {"overall": 0.0},
                    "duration": 0.0,
                    "parsed_output": False,
                    "error": str(e)
                })
    
    # Aggregate results across all models
    logger._log("run.parallel_complete", f"All {len(models)} models completed", level=logging.INFO)
    
    # Step 6: Generate summary
    _generate_summary(
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
    
    # Log run end
    duration = time.time() - start_time
    logger.run_end(duration)
    
    # Log individual model scores
    for result in model_results:
        scores_str = ",".join(f"{k}={v}" for k, v in result["scores"].items())
        logger._log("run.model_summary", 
                   f"[MODEL: {result['model']}] - duration={result['duration']:.3f}s, scores={{{scores_str}}}, missing cluster report: {len(result.get("missing_report", []))}, missing cluster judge: {len(result.get("missing_judge", []))}", 
                   level=logging.INFO)
    
    logger._log("run.summary", f"\nEvaluation complete. Results saved to: {run_results_dir}")
    logger._log("run.summary", f"Evaluated {len(models)} models: {', '.join(models)}")


def main() -> None:
    """
    Main entry point for command-line interface.
    
    Parses command-line arguments and runs evaluation.
    """
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

