from __future__ import annotations

import json
import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from google import genai


# ----------------------------
# Pydantic models (A + B)
# ----------------------------

class RepresentativePaper(BaseModel):
    paper_id: str = Field(..., description="paper_id, referenced like [paper_id] in the report")
    title: str = Field(..., description="paper title")


class ReadingOrderItem(BaseModel):
    paper_id: str = Field(..., description="paper_id")
    why_read_now: str = Field(..., description='Short reason, <= 12 words (best-effort)')


class ClusterCardSectionA(BaseModel):
    # SECTION A fields
    title: str
    one_liner: str
    keyword_list: List[str] = Field(..., min_length=3, max_length=8)
    what_this_cluster_is_about: str
    why_it_matters: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    confidence_rationale: List[str] = Field(..., min_length=2, max_length=4)
    representative_papers: List[RepresentativePaper] = Field(..., min_length=2, max_length=5)
    reading_order: List[ReadingOrderItem] = Field(..., min_length=3, max_length=7)
    search_query_seed: str
    notes: List[str] = Field(..., max_length=5)


# SECTION B model from your previous step (paste yours here if you want stricter validators)
class ClusterIndexSectionB(BaseModel):
    title: str
    summary: str
    keyword_list: List[str] = Field(..., min_length=5, max_length=12)


class ClusterReport(BaseModel):
    """
    One response object that contains:
      - section_a: structured fields (you can render these into Markdown cards yourself)
      - section_b: JSON index for DB storage
    """
    section_a: ClusterCardSectionA
    section_b: ClusterIndexSectionB


# ----------------------------
# Gemini call + parsing
# ----------------------------

PROMPT_TEMPLATE = """\
You are a technical research analyst. You will be given ONE cluster of ML papers as JSON.
Each paper has: paper_id, title, summary, keywords, url, sim_to_centroid, rank_in_cluster.

Hard rules:
- Use ONLY the provided titles/summaries/keywords. Do NOT invent methods, results, datasets, numbers, or claims not present.
- If the cluster is thematically mixed, say so explicitly in notes and lower confidence.
- When referencing a paper, cite it using its paper_id (e.g., [paper_id]).
- Keep the output compact and decision-oriented.

Task:
Return a SINGLE JSON object that matches the provided JSON Schema.
- section_a must include the report fields.
- section_b is a compact DB index derived from section_a.
- section_b.title MUST equal section_a.title exactly.
- section_b.keyword_list must be consistent with section_a.keyword_list and paper keywords.

INPUT (JSON):
{cluster_json}
"""


def generate_cluster_report(
    cluster_json: dict,
    model: str = "gemini-3-flash-preview",
) -> ClusterReport:
    """
    Calls Gemini with structured outputs (JSON Schema) and parses the result into Pydantic models.
    Requires: pip install google-genai pydantic
    Env: GEMINI_API_KEY must be set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")

    client = genai.Client(api_key=api_key)

    prompt = PROMPT_TEMPLATE.format(cluster_json=json.dumps(cluster_json, ensure_ascii=False))

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": ClusterReport.model_json_schema(),
        },
    )

    # response.text is a JSON string
    data = json.loads(response.text)

    # Parse & validate with Pydantic
    report = ClusterReport.model_validate(data)

    # Enforce cross-field equality (schema can’t do this reliably)
    if report.section_b.title != report.section_a.title:
        raise ValueError(
            f"title mismatch: section_a.title={report.section_a.title!r} "
            f"section_b.title={report.section_b.title!r}"
        )

    return report


if_name__ == "__main__":
    # Example input payload (shape matches what you described)
    cluster = {
        "papers": [
            {
                "paper_id": "p1",
                "title": "Example Paper 1",
                "summary": "Short provided summary...",
                "keywords": ["example", "topic"],
                "url": "https://example.com/p1",
                "sim_to_centroid": 0.77,
                "rank_in_cluster": 0,
            },
            {
                "paper_id": "p2",
                "title": "Example Paper 2",
                "summary": "Another provided summary...",
                "keywords": ["example", "method"],
                "url": "https://example.com/p2",
                "sim_to_centroid": 0.71,
                "rank_in_cluster": 1,
            },
            {
                "paper_id": "p3",
                "title": "Example Paper 3",
                "summary": "Another provided summary...",
                "keywords": ["related", "topic"],
                "url": "https://example.com/p3",
                "sim_to_centroid": 0.69,
                "rank_in_cluster": 2,
            },
        ]
    }

    report = generate_cluster_report(cluster)
    print("Parsed OK ✅")
    print(report.model_dump_json(indent=2, ensure_ascii=False))

