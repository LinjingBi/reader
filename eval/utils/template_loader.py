"""Prompt template loader and renderer"""

import hashlib
import json
from pathlib import Path
from typing import Dict


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


def compute_template_hash(template_content: str) -> str:
    """
    Compute SHA256 hash of template content for reproducibility.
    
    Args:
        template_content: Template content string
    
    Returns:
        Hex digest of SHA256 hash (first 16 chars)
    """
    return hashlib.sha256(template_content.encode('utf-8')).hexdigest()[:16]


def render_template(
    template_content: str,
    output_schema_path: Path,
    input_data: dict
) -> str:
    """
    Render template by replacing placeholders.
    
    Args:
        template_content: Template string with {{OUTPUT_JSON}} and {{INPUT_JSON}} placeholders
        output_schema_path: Path to output JSON schema file
        input_data: Packed input data (dict) to insert
    
    Returns:
        Rendered prompt string
    """
    # Load output schema
    with open(output_schema_path, 'r', encoding='utf-8') as f:
        output_schema = json.load(f)
    
    # Replace placeholders
    rendered = template_content.replace(
        "{{OUTPUT_JSON}}",
        json.dumps(output_schema, indent=2)
    )
    
    rendered = rendered.replace(
        "{{INPUT_JSON}}",
        json.dumps(input_data, indent=2, ensure_ascii=False)
    )
    
    return rendered


def render_template_per_cluster(
    template_content: str,
    cluster_data: dict
) -> str:
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

