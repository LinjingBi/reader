"""Heuristic judge for validating LLM outputs"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set

from jsonschema import validate, ValidationError


def judge_output(output_json: dict, input_data: dict, schema_path: Path) -> dict:
    """
    Judge LLM output using heuristic validation gates.
    
    Args:
        output_json: Parsed LLM output JSON
        input_data: Original input data (for citation validation)
        schema_path: Path to JSON schema file
    
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
    
    # 1. Schema validation
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        validate(instance=output_json, schema=schema)
        results["schema_valid"]["passed"] = True
    except ValidationError as e:
        results["schema_valid"]["errors"] = [str(e)]
    except Exception as e:
        results["schema_valid"]["errors"] = [f"Schema validation error: {str(e)}"]
    
    # 2. Citation validation
    if not results["schema_valid"]["passed"]:
        # Skip citation validation if schema is invalid
        return results
    
    input_paper_ids = _extract_input_paper_ids(input_data)
    output_paper_ids = _extract_output_paper_ids(output_json)
    
    invalid_citations = []
    for paper_id in output_paper_ids:
        if paper_id not in input_paper_ids:
            invalid_citations.append(paper_id)
    
    results["citations_ok"]["passed"] = len(invalid_citations) == 0
    results["citations_ok"]["invalid_citations"] = invalid_citations
    
    # 3. Length validation
    violations = []
    
    cluster_cards = output_json.get("cluster_cards", [])
    for i, card in enumerate(cluster_cards):
        # topic_name: <= 100 chars
        topic_name = card.get("topic_name", "")
        if len(topic_name) > 100:
            violations.append(f"cluster_cards[{i}].topic_name: {len(topic_name)} chars (max 100)")
        
        # one_liner: <= 220 chars
        one_liner = card.get("one_liner", "")
        if len(one_liner) > 220:
            violations.append(f"cluster_cards[{i}].one_liner: {len(one_liner)} chars (max 220)")
        
        # tags: 3-7 items, each <= 40 chars
        tags = card.get("tags", [])
        if len(tags) < 3:
            violations.append(f"cluster_cards[{i}].tags: {len(tags)} items (min 3)")
        if len(tags) > 7:
            violations.append(f"cluster_cards[{i}].tags: {len(tags)} items (max 7)")
        for j, tag in enumerate(tags):
            if len(tag) > 40:
                violations.append(f"cluster_cards[{i}].tags[{j}]: {len(tag)} chars (max 40)")
        
        # what_this_cluster_is_about: <= 1200 chars
        what_about = card.get("what_this_cluster_is_about", "")
        if len(what_about) > 1200:
            violations.append(f"cluster_cards[{i}].what_this_cluster_is_about: {len(what_about)} chars (max 1200)")
        
        # why_it_matters: <= 1200 chars
        why_matters = card.get("why_it_matters", "")
        if len(why_matters) > 1200:
            violations.append(f"cluster_cards[{i}].why_it_matters: {len(why_matters)} chars (max 1200)")
        
        # confidence_rationale: <= 300 chars
        confidence_rationale = card.get("confidence_rationale", "")
        if len(confidence_rationale) > 300:
            violations.append(f"cluster_cards[{i}].confidence_rationale: {len(confidence_rationale)} chars (max 300)")
        
        # representative_papers: check reason_representative <= 300 chars
        rep_papers = card.get("representative_papers", [])
        for j, paper in enumerate(rep_papers):
            reason = paper.get("reason_representative", "")
            if len(reason) > 300:
                violations.append(f"cluster_cards[{i}].representative_papers[{j}].reason_representative: {len(reason)} chars (max 300)")
        
        # reading_order: check why_read_next <= 220 chars
        reading_order = card.get("reading_order", [])
        for j, item in enumerate(reading_order):
            why_read = item.get("why_read_next", "")
            if len(why_read) > 220:
                violations.append(f"cluster_cards[{i}].reading_order[{j}].why_read_next: {len(why_read)} chars (max 220)")
        
        # search_query_seed: <= 200 chars
        search_query = card.get("search_query_seed", "")
        if len(search_query) > 200:
            violations.append(f"cluster_cards[{i}].search_query_seed: {len(search_query)} chars (max 200)")
        
        # notes: <= 600 chars
        notes = card.get("notes", "")
        if len(notes) > 600:
            violations.append(f"cluster_cards[{i}].notes: {len(notes)} chars (max 600)")
    
    results["length_ok"]["passed"] = len(violations) == 0
    results["length_ok"]["violations"] = violations
    
    # 4. Score features
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

