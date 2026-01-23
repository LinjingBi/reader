"""Main evaluation runner"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import yaml

# Add eval directory to path
sys.path.insert(0, str(Path(__file__).parent))

from packers.p0 import build_input, get_packer_stats
from judges.heuristics import judge_output
from utils.llm_client import LLMClient
from utils.logger import EvalLogger
from utils.template_loader import load_template, compute_template_hash, render_template, get_preview


def load_config(config_path: Path) -> dict:
    """Load and validate config file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_run_id(month: str, packing_variant: str, prompt_variant: str, model: str, temperature: float, prompt_hash: str) -> str:
    """Generate stable run ID"""
    # Format: {month}__{packing}__{prompt}__{model}__t{temp}__{hash}
    temp_str = f"t{temperature}".replace(".", "_")
    return f"{month}__{packing_variant}__{prompt_variant}__{model}__{temp_str}__{prompt_hash[:8]}"


def generate_snapshot_id(source: str, period_start: str, period_end: str) -> str:
    """Generate snapshot ID from input data"""
    return f"{source}|{period_start}|{period_end}"


def load_monthly_data(data_dir: str, month: str) -> dict:
    """Load monthly clustering JSON"""
    month_file = os.path.join(data_dir, f"{month}.json")
    if not os.path.exists(month_file):
        raise FileNotFoundError(f"Monthly data file not found: {month_file}")
    
    with open(month_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_evaluation(config_path: Path, month: str) -> None:
    """
    Run evaluation for a single month.
    
    Args:
        config_path: Path to eval.yaml config file
        month: Month key (e.g., "2025-01")
    """
    # Load config
    config = load_config(config_path)
    
    # Setup paths
    eval_dir = os.path.dirname(__file__)
    data_dir = os.path.join(eval_dir, config["dataset"]["input_dir"])
    prompts_dir = os.path.join(eval_dir, "prompts")
    schemas_dir = os.path.join(eval_dir, "schemas")
    results_dir = os.path.join(eval_dir, "results")
    
    # Load monthly data
    monthly_data = load_monthly_data(data_dir, month)
    
    # Extract metadata
    source = monthly_data.get("source", "hf_monthly")
    period_start = monthly_data.get("period_start", "")
    period_end = monthly_data.get("period_end", "")
    snapshot_id = generate_snapshot_id(source, period_start, period_end)
    
    # Get prompt variant (first one for MVP)
    prompt_variant = config["prompts"]["variants"][0]
    prompt_path = os.path.join(eval_dir, prompt_variant["path"])
    packing_variant = config["packers"]["variants"][0]  # "P0"
    
    # Load prompt template
    template_content = load_template(Path(prompt_path))
    template_hash = compute_template_hash(template_content)
    
    # Generate run ID
    llm_config = config["llm"]
    run_id = generate_run_id(
        month=month,
        packing_variant=packing_variant,
        prompt_variant=prompt_variant["id"],
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        prompt_hash=template_hash
    )
    
    # Setup logging
    run_results_dir = os.path.join(results_dir, run_id)
    logger = EvalLogger(run_id, Path(run_results_dir))
    
    start_time = time.time()
    
    # Log run start
    logger.run_start(snapshot_id)
    
    # Log config artifacts
    logger.config_artifact(config)
    
    # P0 Packer
    packed_data, warnings = build_input(monthly_data)
    stats = get_packer_stats(packed_data)
    logger.packer_end(stats["clusters"], stats["papers"], warnings)
    
    # Render prompt template
    output_schema_path = os.path.join(eval_dir, config["judge"]["schema_path"])
    rendered_prompt = render_template(
        template_content=template_content,
        output_schema_path=output_schema_path,
        input_data=packed_data
    )
    preview = get_preview(rendered_prompt)
    logger.enrich_prompt(prompt_variant["id"], template_hash, preview)
    
    # Initialize LLM client
    api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {api_key_env}")
    
    llm_client = LLMClient(
        provider=llm_config["provider"],
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
        api_key=api_key
    )
    
    # Log LLM call details
    logger.enrich_llm(
        model=llm_config["model"],
        api_uri=llm_client.api_uri,
        args=llm_client.get_api_args()
    )
    
    # Ensure results directory exists
    os.makedirs(run_results_dir, exist_ok=True)
    model_output_path = os.path.join(run_results_dir, "model_output.json")
    
    # Call LLM
    try:
        raw_response = llm_client.call(rendered_prompt)
        logger.enrich_response_raw(len(raw_response), saved_to=str(model_output_path))
        
        # Save raw response first
        with open(model_output_path, 'w', encoding='utf-8') as f:
            json.dump({"raw_response": raw_response}, f, indent=2, ensure_ascii=False)
        
        # Parse response
        try:
            parsed_output = json.loads(raw_response)
            cluster_cards_count = len(parsed_output.get("cluster_cards", []))
            logger.enrich_response_parsed(cluster_cards_count, valid=True)
            
            # Save parsed output (overwrite raw_response dict)
            with open(model_output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_output, f, indent=2, ensure_ascii=False)
        
        except json.JSONDecodeError as e:
            logger.enrich_parse_error(str(e))
            parsed_output = None
            # Keep raw_response saved in file
    
    except Exception as e:
        logger.enrich_parse_error(f"LLM call failed: {str(e)}")
        parsed_output = None
        raw_response = ""
        # Save error to file
        with open(model_output_path, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e), "raw_response": ""}, f, indent=2, ensure_ascii=False)
    
    # Heuristic judge
    if parsed_output:
        judge_result = judge_output(
            output_json=parsed_output,
            input_data=monthly_data,
            schema_path=Path(output_schema_path)
        )
        
        # Log heuristic metrics
        logger.heuristic_metrics(
            schema_valid=judge_result["schema_valid"]["passed"],
            citations_ok=judge_result["citations_ok"]["passed"],
            length_ok=judge_result["length_ok"]["passed"],
            name_generic_penalty=judge_result["scores"]["name_generic_penalty"]
        )
        
        # Save judge result
        judge_result_path = os.path.join(run_results_dir, "judge_result.json")
        with open(judge_result_path, 'w', encoding='utf-8') as f:
            json.dump(judge_result, f, indent=2, ensure_ascii=False)
        
        # Compute overall scores
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
    
    # Save run metadata
    run_metadata = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "month": month,
        "packing_variant": packing_variant,
        "prompt_variant": prompt_variant["id"],
        "prompt_hash": template_hash,
        "llm_config": {
            "provider": llm_config["provider"],
            "model": llm_config["model"],
            "temperature": llm_config["temperature"],
            "max_tokens": llm_config["max_tokens"],
        },
        "timestamp": datetime.now().isoformat(),
        "heuristic_scores": overall_scores,
    }
    metadata_path = os.path.join(run_results_dir, "run_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)
    
    # Log run end
    duration = time.time() - start_time
    logger.run_end(duration, overall_scores)
    
    print(f"\nEvaluation complete. Results saved to: {run_results_dir}")


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
    
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    run_evaluation(Path(config_path), args.month)


if __name__ == "__main__":
    main()

