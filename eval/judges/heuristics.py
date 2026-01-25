"""Heuristic judge for validating LLM outputs"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple

from jsonschema import validate, ValidationError
from pydantic import ValidationError as PydanticValidationError

# Import output models for Pydantic validation
# Use absolute import from eval directory
import sys
from pathlib import Path
eval_dir = Path(__file__).parent.parent
if str(eval_dir) not in sys.path:
    sys.path.insert(0, str(eval_dir))
from schemas.output_models import OutputSchema


def _validate_schema_pydantic(output_json: dict) -> tuple[bool, list[str]]:
    """
    Validate output using Pydantic models (primary validation method).
    
    Returns:
        Tuple of (passed: bool, errors: List[str])
    """
    try:
        OutputSchema.model_validate(output_json)
        return True, []
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"{field_path}: {error['msg']}")
        return False, errors
    except Exception as e:
        return False, [f"Pydantic validation error: {str(e)}"]


def _validate_schema_jsonschema(output_json: dict, schema_path: Path) -> tuple[bool, list[str]]:
    """
    Validate output using JSON schema (fallback validation method).
    
    Returns:
        Tuple of (passed: bool, errors: List[str])
    """
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        validate(instance=output_json, schema=schema)
        return True, []
    except ValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"JSON schema validation error: {str(e)}"]


def _validate_citations(output_json: dict, input_data: dict) -> tuple[bool, list[str]]:
    """
    Validate that all cited paper_ids exist in input data.
    
    Returns:
        Tuple of (passed: bool, invalid_citations: List[str])
    """
    input_paper_ids = _extract_input_paper_ids(input_data)
    output_paper_ids = _extract_output_paper_ids(output_json)
    
    invalid_citations = []
    for paper_id in output_paper_ids:
        if paper_id not in input_paper_ids:
            invalid_citations.append(paper_id)
    
    return len(invalid_citations) == 0, invalid_citations


def _validate_lengths(output_json: dict) -> tuple[bool, list[str]]:
    """
    Validate length constraints (kept for compatibility, but Pydantic should cover most).
    
    Returns:
        Tuple of (passed: bool, violations: List[str])
    """
    violations = []
    cluster_cards = output_json.get("cluster_cards", [])
    
    for i, card in enumerate(cluster_cards):
        # Most length checks are now handled by Pydantic, but keep some for edge cases
        # Only check fields that might not be fully covered
        notes = card.get("notes", "")
        if len(notes) > 600:
            violations.append(f"cluster_cards[{i}].notes: {len(notes)} chars (max 600)")
    
    return len(violations) == 0, violations


def judge_output(output_json: dict, input_data: dict, schema_path: Path) -> dict:
    """
    Judge LLM output using Pydantic validation (primary) and heuristic checks.
    
    Args:
        output_json: Parsed LLM output JSON
        input_data: Original input data (for citation validation)
        schema_path: Path to JSON schema file (fallback validation)
    
    Returns:
        Dict with validation results:
        {
            "schema_valid": {"passed": bool, "errors": []},
            "citations_ok": {"passed": bool, "invalid_citations": []},
            "length_ok": {"passed": bool, "violations": []},
            "scores": {"name_generic_penalty": float}
        }
    """
    results = {
        "schema_valid": {"passed": False, "errors": []},
        "citations_ok": {"passed": False, "invalid_citations": []},
        "length_ok": {"passed": False, "violations": []},
        "scores": {"name_generic_penalty": 0.0}
    }
    
    # 1. Schema validation - try Pydantic first (takes precedence)
    pydantic_passed, pydantic_errors = _validate_schema_pydantic(output_json)
    
    if pydantic_passed:
        results["schema_valid"]["passed"] = True
    else:
        # Fall back to JSON schema validation
        jsonschema_passed, jsonschema_errors = _validate_schema_jsonschema(output_json, schema_path)
        if jsonschema_passed:
            results["schema_valid"]["passed"] = True
            results["schema_valid"]["errors"] = pydantic_errors  # Still report Pydantic errors
        else:
            results["schema_valid"]["errors"] = pydantic_errors + jsonschema_errors
    
    # 2. Citation validation (only if schema is valid)
    if not results["schema_valid"]["passed"]:
        return results
    
    citations_passed, invalid_citations = _validate_citations(output_json, input_data)
    results["citations_ok"]["passed"] = citations_passed
    results["citations_ok"]["invalid_citations"] = invalid_citations
    
    # 3. Length validation (mostly covered by Pydantic, but check edge cases)
    length_passed, violations = _validate_lengths(output_json)
    results["length_ok"]["passed"] = length_passed
    results["length_ok"]["violations"] = violations
    
    # 4. Score features (heuristic scoring)
    cluster_cards = output_json.get("cluster_cards", [])
    results["scores"]["name_generic_penalty"] = _compute_name_generic_penalty(cluster_cards)
    
    return results


def _extract_input_paper_ids(input_data: dict) -> Set[str]:
    """Extract all paper_ids from input papers"""
    paper_ids = set()
    papers = input_data.get("papers", [])
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def _extract_output_paper_ids(output_json: dict) -> Set[str]:
    """Extract all paper_ids from output (representative_papers and reading_order)"""
    paper_ids = set()
    cluster_cards = output_json.get("cluster_cards", [])
    
    for card in cluster_cards:
        # From representative_papers
        rep_papers = card.get("representative_papers", [])
        for paper in rep_papers:
            paper_id = paper.get("paper_id", "")
            if paper_id:
                paper_ids.add(paper_id)
        
        # From reading_order
        reading_order = card.get("reading_order", [])
        for item in reading_order:
            paper_id = item.get("paper_id", "")
            if paper_id:
                paper_ids.add(paper_id)
    
    return paper_ids


def _compute_name_generic_penalty(cluster_cards: List[dict]) -> float:
    """
    Compute penalty for generic topic names.
    
    Generic names: "AI", "LLM", "Vision", "Machine Learning", etc.
    Returns penalty score (0.0 = no penalty, higher = more generic)
    """
    generic_terms = {
        "ai", "llm", "vision", "machine learning", "deep learning",
        "neural network", "nlp", "computer vision", "ml", "dl"
    }
    
    penalty = 0.0
    for card in cluster_cards:
        topic_name = card.get("topic_name", "").lower()
        # Check if topic name is exactly a generic term or starts with it
        if topic_name in generic_terms:
            penalty += 1.0
        elif any(topic_name.startswith(term + " ") for term in generic_terms):
            penalty += 0.5
    
    # Normalize by number of clusters
    if len(cluster_cards) > 0:
        penalty = penalty / len(cluster_cards)
    
    return penalty

