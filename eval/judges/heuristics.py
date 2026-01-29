"""Heuristic judge for validating LLM outputs"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import output models for Pydantic validation
# Use absolute import from eval directory
import sys
eval_dir = Path(__file__).parent.parent
if str(eval_dir) not in sys.path:
    sys.path.insert(0, str(eval_dir))
from schemas.cluster_response import ClusterReport
from judges.hard_rules import hard_validate_cluster_report, try_parse_cluster_report
from judges.soft_rules import soft_validate_cluster_report


@dataclass
class JudgeOutput:
    """Output from judge_output function matching HeuristicResult format"""
    sub_scores: Dict[str, float]  # All rule scores (0.0 or 1.0 for bools)
    overall: float  # 0.0 if any must-pass fails, else 1.0 + soft_schema_valid.score
    reasons: Dict[str, str]  # Human-readable reasons for each rule

def judge_output(raw_text: str, input_data: dict) -> Tuple[JudgeOutput, Optional[ClusterReport]]:
    """
    Judge LLM output using JSON parsing, Pydantic validation, and heuristic checks.
    
    Args:
        raw_text: Raw text response from LLM (may contain JSON in markdown or raw format)
        input_data: Original input data (for citation validation)
    
    Returns:
        Tuple of (JudgeOutput, Optional[ClusterReport])
        - JudgeOutput: Contains sub_scores, overall, and reasons
        - ClusterReport: Validated Pydantic object if validation passes, None otherwise
    """
    sub_scores: Dict[str, float] = {}
    reasons: Dict[str, str] = {}
    
    # 1. Parse JSON to get ClusterReport (needed for return value and soft validation)
    cluster_report, parse_error = try_parse_cluster_report(raw_text)
    if cluster_report is not None:
        sub_scores["json_valid"] = 1.0
        reasons["json_valid"] = "OK"
    else:
        sub_scores["json_valid"] = 0.0
        error_msg = str(parse_error) if parse_error else "Unknown cluster report parse error"
        reasons["json_valid"] = f"JSON parse error: {error_msg}"
    
    # 2. Hard validation (includes parsing, field validation, and citation checks)
    hard_result = hard_validate_cluster_report(cluster_report, input_data)
    sub_scores["hard_schema_valid"] = hard_result.score
    # Only include reasons if there are failure messages (not empty string)
    if hard_result.reasons:
        reasons["hard_schema_valid"] = hard_result.reasons
    
    # 3. Soft validation (includes name_generic check)

    soft_result = soft_validate_cluster_report(cluster_report)
    sub_scores["soft_schema_valid"] = soft_result.score
    # Only include reasons if there are failure messages (not empty string)
    if soft_result.reasons:
        reasons["soft_schema_valid"] = soft_result.reasons

    
    # Compute overall score
    # Must-pass rules: json_valid, hard_schema_valid
    must_pass_rules = ["json_valid", "hard_schema_valid"]
    must_pass_failed = any(sub_scores.get(rule, 0.0) == 0.0 for rule in must_pass_rules)
    
    if must_pass_failed:
        overall = 0.0
    else:
        # If all must-pass rules pass, overall is 1 + soft_schema_valid.score
        overall = 1.0 + sub_scores.get("soft_schema_valid", 0.0)
    
    return JudgeOutput(
        sub_scores=sub_scores,
        overall=overall,
        reasons=reasons
    ), cluster_report
