#!/usr/bin/env python3
"""
dspy_topic_report_demo.py

Input JSON shape:
  [
    {"cluster_id": "...", "papers": [ {paper_id,title,summary,keywords,rank_in_cluster}, ... ]},
    ...
  ]

This script:
  1) Baseline DSPy program: one cluster_json -> one output_json (ClusterOutput)
  2) Heuristic-only evaluation (no LLM judge)
  3) DSPy optimization (compile) using BootstrapFewShot
  4) Save optimized artifact + production load demo
  5) Print ALL heuristic averages baseline vs optimized

Install:
  pip install dspy-ai pydantic

Run:
  export OPENAI_API_KEY=... or GEMINI_API_KEY
  export DSPY_MODEL=openai/gpt-4.1-mini   # optional
  python dspy_topic_report_demo.py /mnt/data/2025-01.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

import dspy


# =========================
# Config
# =========================

OUT_DIR = "./dspy_out"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_CLUSTERS_FOR_DEV = 8  # local quick test

# Minimal, stable-ish decoding (no multi-run stability checks requested)
LM = dspy.LM(model='gemini/gemini-2.5-flash-lite', temperature=0.2)
dspy.configure(lm=LM)

# Weighted scoring (must sum to 1.0)
W_VALID_JSON = 0.35
W_TITLE_MATCH = 0.35
W_CROSS_LINKS = 0.20
W_REST = 0.10  # schema_valid + citations_ok + forbidden_words_ok (+ optional ordering softness)


# =========================
# Helpers
# =========================

def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

def is_title_case_no_colon(title: str) -> bool:
    t = (title or "").strip()
    if ":" in t:
        return False
    return t == t.title()

def contains_forbidden_cluster_words(s: str) -> bool:
    return bool(re.search(r"\b(cluster|clustering)\b", s or "", flags=re.IGNORECASE))

def extract_bracket_citations(text: str) -> List[str]:
    """Extract [paper_id] citations."""
    return re.findall(r"\[([A-Za-z0-9._:-]+)\]", text or "")

def parse_bracket_item(s: str) -> Tuple[str, str]:
    """Parse '[paper_id] rest...' -> (paper_id, rest)."""
    m = re.match(r"^\[([A-Za-z0-9._:-]+)\]\s+(.+)$", (s or "").strip())
    if not m:
        return ("", "")
    return (m.group(1), m.group(2).strip())

def split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]

def get_cluster_id_set(cluster: dict) -> set:
    return {p["paper_id"] for p in cluster.get("papers", []) if "paper_id" in p}

def get_rank_map(cluster: dict) -> Dict[str, int]:
    m = {}
    for p in cluster.get("papers", []):
        pid = p.get("paper_id")
        r = p.get("rank_in_cluster")
        if pid is not None and isinstance(r, int):
            m[pid] = r
    return m

def load_clusters(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("clusters"), list):
        return data["clusters"]
    raise ValueError("Expected a list of clusters or {'clusters': [...]}.")


# =========================
# Pydantic Schemas (field rules live here)
# =========================

class WhyItMatters(BaseModel):
    practical_significance: str
    research_significance: str

    @field_validator("practical_significance", "research_significance")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("why_it_matters parts must be non-empty")
        return v


class ClusterCardSectionA(BaseModel):
    """SECTION A — Human-readable report fields"""
    title: str
    one_liner: str
    what_this_topic_is_about: str
    why_it_matters: WhyItMatters
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    confidence_rationale: List[str] = Field(..., min_length=2, max_length=4)
    representative_papers: List[str] = Field(..., min_length=2, max_length=5)  # "[paper_id] title"
    reading_order: List[str] = Field(..., min_length=3, max_length=7)          # "[paper_id] reason"
    search_query_seed: str
    notes: List[str] = Field(..., max_length=5)

    # ---- title ----
    @field_validator("title")
    @classmethod
    def _title_rules(cls, v: str) -> str:
        v = (v or "").strip()
        if count_words(v) > 12:
            raise ValueError("title: max 12 words")
        if not is_title_case_no_colon(v):
            raise ValueError("title: must be Title Case and contain no colon")
        return v

    # ---- one_liner ----
    @field_validator("one_liner")
    @classmethod
    def _one_liner_rules(cls, v: str) -> str:
        v = (v or "").strip()
        if count_words(v) > 25:
            raise ValueError("one_liner: max 25 words")
        return v

    # ---- what_this_topic_is_about ----
    @field_validator("what_this_topic_is_about")
    @classmethod
    def _about_rules(cls, v: str) -> str:
        v = (v or "").strip()
        wc = count_words(v)
        if not (80 <= wc <= 140):
            raise ValueError("what_this_topic_is_about: 80–140 words")
        return v

    # ---- why_it_matters (combined length 60–120) ----
    @model_validator(mode="after")
    def _why_total_len(self) -> "ClusterCardSectionA":
        total = count_words(self.why_it_matters.practical_significance) + count_words(self.why_it_matters.research_significance)
        if not (60 <= total <= 120):
            raise ValueError("why_it_matters: combined practical+research must be 60–120 words")
        return self

    # ---- confidence_rationale ----
    @field_validator("confidence_rationale")
    @classmethod
    def _confidence_rationale_rules(cls, v: List[str]) -> List[str]:
        if not (2 <= len(v) <= 4):
            raise ValueError("confidence_rationale: 2–4 bullet points")
        for bullet in v:
            if count_words(bullet) > 18:
                raise ValueError("confidence_rationale: each bullet <= 18 words")
        return v

    # ---- representative_papers ----
    @field_validator("representative_papers")
    @classmethod
    def _representative_papers_rules(cls, v: List[str]) -> List[str]:
        for item in v:
            pid, rest = parse_bracket_item(item)
            if not pid or not rest:
                raise ValueError("representative_papers: format must be '[paper_id] title'")
        return v

    # ---- reading_order ----
    @field_validator("reading_order")
    @classmethod
    def _reading_order_rules(cls, v: List[str]) -> List[str]:
        for item in v:
            pid, rest = parse_bracket_item(item)
            if not pid or not rest:
                raise ValueError("reading_order: format must be '[paper_id] reason'")
            if count_words(rest) > 12:
                raise ValueError("reading_order: reason <= 12 words")
        return v

    # ---- search_query_seed ----
    @field_validator("search_query_seed")
    @classmethod
    def _search_query_seed_rules(cls, v: str) -> str:
        v = (v or "").strip()
        if "\n" in v:
            raise ValueError("search_query_seed: one line only")
        terms = [t for t in re.split(r"\s+", v) if t]
        if not (2 <= len(terms) <= 5):
            raise ValueError("search_query_seed: 2–5 key terms")
        return v

    # ---- notes ----
    @field_validator("notes")
    @classmethod
    def _notes_rules(cls, v: List[str]) -> List[str]:
        if len(v) > 5:
            raise ValueError("notes: up to 5 bullet points")
        for bullet in v:
            if count_words(bullet) > 20:
                raise ValueError("notes: each bullet <= 20 words")
        return v


class ClusterIndexSectionB(BaseModel):
    """SECTION B — JSON index for database storage"""
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    keyword_list: List[str] = Field(..., min_length=5, max_length=12)

    @field_validator("summary")
    @classmethod
    def _summary_len(cls, v: str) -> str:
        v = (v or "").strip()
        wc = count_words(v)
        if not (60 <= wc <= 110):
            raise ValueError("section_b.summary: 60–110 words")
        return v

    @field_validator("keyword_list")
    @classmethod
    def _keyword_list_rules(cls, v: List[str]) -> List[str]:
        out = []
        seen = set()
        for k in v:
            k2 = (k or "").strip()
            if not k2:
                continue
            if not (1 <= len(k2.split()) <= 3):
                raise ValueError("keyword_list: each keyword must be 1–3 words")
            if k2 in seen:
                raise ValueError("keyword_list: must be deduped")
            if "#" in k2:
                raise ValueError("keyword_list: no hashtags")
            seen.add(k2)
            out.append(k2)
        if not (5 <= len(out) <= 12):
            raise ValueError("keyword_list: 5–12 items after normalization")
        return out


class ClusterOutput(BaseModel):
    section_a: ClusterCardSectionA
    section_b: ClusterIndexSectionB


# =========================
# DSPy Signature (minimal + optimization-friendly)
# =========================

class ClusterReportSig(dspy.Signature):
    """
    Synthesize ONE topic from a cluster_json (papers with paper_id/title/summary/keywords/ranks).

    Output MUST match ClusterOutput schema exactly.

    Grounding rules:
      - Use only provided titles/summaries/keywords (no invented claims).
      - Reference papers via inline [paper_id] only (no URLs).
      - Use the word "topic" (avoid "cluster/clustering") in narrative.

    Content rules:
      - what_this_topic_is_about: explain the shared theme and connect multiple papers.
      - why_it_matters: fill practical_significance + research_significance (no hype/speculation).
      - section_b.summary: derived from section_a (theme + significance).

    Structural rules are enforced by schema validators; do not add extra fields.
    """
    cluster_json: str = dspy.InputField(desc="JSON for one topic: {cluster_id, papers:[...]} ")
    output: ClusterOutput = dspy.OutputField(desc="ClusterOutput matching the schema.")


class ClusterReporter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(ClusterReportSig)

    def forward(self, cluster_json: str) -> ClusterOutput:
        return self.gen(cluster_json=cluster_json).output


# =========================
# Heuristics (weighted)
# =========================

@dataclass
class HeuristicResult:
    sub_scores: Dict[str, float]
    overall: float
    reasons: Dict[str, str]


def cross_paper_links_score(about_text: str) -> Tuple[float, str]:
    """
    Require 2–4 sentences in what_this_topic_is_about that include >=2 distinct [paper_id] citations.
    """
    sentences = split_sentences(about_text)
    link_count = 0
    for s in sentences:
        cited = extract_bracket_citations(s)
        if len(set(cited)) >= 2:
            link_count += 1

    if 2 <= link_count <= 4:
        return (1.0, f"OK (link_sentences={link_count})")
    if link_count in (1, 5):
        return (0.5, f"Borderline (link_sentences={link_count}, want 2–4)")
    return (0.0, f"Fail (link_sentences={link_count}, want 2–4)")


def heuristic_eval(cluster_json: str, output: ClusterOutput | str) -> HeuristicResult:
    """
    Evaluate output against heuristics.
    
    Args:
        cluster_json: JSON string of the input cluster
        output: Either ClusterOutput model or JSON string (for backward compatibility)
    """
    sub: Dict[str, float] = {}
    reasons: Dict[str, str] = {}

    # Parse cluster_json to get allowed paper_ids and ranks
    try:
        cluster = json.loads(cluster_json)
        cluster_ids = get_cluster_id_set(cluster)
        rank_map = get_rank_map(cluster)
    except Exception as e:
        return HeuristicResult(
            sub_scores={"cluster_json_valid": 0.0},
            overall=0.0,
            reasons={"cluster_json_valid": f"cluster_json parse error: {e}"}
        )

    # Handle both ClusterOutput model and JSON string (for backward compatibility)
    if isinstance(output, str):
        # (1) valid_json_format
        try:
            data = json.loads(output)
            sub["valid_json_format"] = 1.0
            reasons["valid_json_format"] = "OK"
        except Exception as e:
            sub["valid_json_format"] = 0.0
            reasons["valid_json_format"] = f"Output JSON parse error: {e}"
            return HeuristicResult(sub_scores=sub, overall=0.0, reasons=reasons)

        # (2) schema_valid
        try:
            parsed = ClusterOutput.model_validate(data)
            sub["schema_valid"] = 1.0
            reasons["schema_valid"] = "OK"
        except ValidationError as e:
            sub["schema_valid"] = 0.0
            reasons["schema_valid"] = f"Pydantic validation error: {str(e)[:450]}"
            overall = W_VALID_JSON * sub["valid_json_format"]
            return HeuristicResult(sub_scores=sub, overall=overall, reasons=reasons)
    else:
        # Already a ClusterOutput model
        parsed = output
        sub["valid_json_format"] = 1.0
        reasons["valid_json_format"] = "OK"
        sub["schema_valid"] = 1.0
        reasons["schema_valid"] = "OK"

    a = parsed.section_a
    b = parsed.section_b

    # (3) sectionB_title_matches_ok (high weight)
    tmatch_ok = (b.title.strip() == a.title.strip())
    sub["sectionB_title_matches_ok"] = 1.0 if tmatch_ok else 0.0
    reasons["sectionB_title_matches_ok"] = "OK" if tmatch_ok else "SectionB.title must equal SectionA.title exactly"

    # (4) citations_ok — require inline [paper_id] in:
    #   A.what_this_topic_is_about
    #   A.why_it_matters.practical_significance
    #   A.why_it_matters.research_significance
    #   B.summary
    fields_for_citations = {
        "A.what_this_topic_is_about": a.what_this_topic_is_about,
        "A.why.practical_significance": a.why_it_matters.practical_significance,
        "A.why.research_significance": a.why_it_matters.research_significance,
        "B.summary": b.summary,
    }

    cit_ok = True
    cit_detail = []
    for fname, txt in fields_for_citations.items():
        cited = extract_bracket_citations(txt)
        if len(cited) < 1:
            cit_ok = False
            cit_detail.append(f"{fname}: no [paper_id] citations")
            continue
        if not set(cited).issubset(cluster_ids):
            cit_ok = False
            bad = list(set(cited) - cluster_ids)[:5]
            cit_detail.append(f"{fname}: unknown paper_ids cited {bad}")

    sub["citations_ok"] = 1.0 if cit_ok else 0.0
    reasons["citations_ok"] = "OK" if cit_ok else "; ".join(cit_detail)

    # # (5) forbidden_words_ok — no "cluster/clustering" in narrative fields
    # fields_for_forbidden = {
    #     "A.what_this_topic_is_about": a.what_this_topic_is_about,
    #     "A.why.practical_significance": a.why_it_matters.practical_significance,
    #     "A.why.research_significance": a.why_it_matters.research_significance,
    #     "B.summary": b.summary,
    # }

    # forbid_ok = True
    # forbid_detail = []
    # for fname, txt in fields_for_forbidden.items():
    #     if contains_forbidden_cluster_words(txt):
    #         forbid_ok = False
    #         forbid_detail.append(f"{fname}: contains 'cluster/clustering'")
    # sub["forbidden_words_ok"] = 1.0 if forbid_ok else 0.0
    # reasons["forbidden_words_ok"] = "OK" if forbid_ok else "; ".join(forbid_detail)

    # (6) cross_paper_links_ok (second weight)
    cps, cpr = cross_paper_links_score(a.what_this_topic_is_about)
    sub["cross_paper_links_ok"] = cps
    reasons["cross_paper_links_ok"] = cpr

    # Optional soft checks (folded into "rest"): centrality ordering preference
    def ranks_non_decreasing(ids: List[str]) -> bool:
        ranks = [rank_map.get(pid, 10**9) for pid in ids]
        return all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))

    rep_ids = [parse_bracket_item(x)[0] for x in a.representative_papers]
    ro_ids = [parse_bracket_item(x)[0] for x in a.reading_order]

    sub["rep_central_first_ok"] = 1.0 if ranks_non_decreasing(rep_ids) else 0.7
    reasons["rep_central_first_ok"] = "OK" if sub["rep_central_first_ok"] == 1.0 else "Not strictly central-first by rank_in_cluster"
    sub["reading_order_central_to_detailed_ok"] = 1.0 if ranks_non_decreasing(ro_ids) else 0.7
    reasons["reading_order_central_to_detailed_ok"] = "OK" if sub["reading_order_central_to_detailed_ok"] == 1.0 else "Not strictly central-to-detailed by rank_in_cluster"

    # Weighted overall
    rest_keys = ["schema_valid", "citations_ok", "rep_central_first_ok", "reading_order_central_to_detailed_ok"]
    rest_avg = sum(sub.get(k, 0.0) for k in rest_keys) / len(rest_keys)

    overall = (
        W_VALID_JSON * sub["valid_json_format"]
        + W_TITLE_MATCH * sub["sectionB_title_matches_ok"]
        + W_CROSS_LINKS * sub["cross_paper_links_ok"]
        + W_REST * rest_avg
    )

    return HeuristicResult(sub_scores=sub, overall=overall, reasons=reasons)


def metric_for_optimizer(example: dspy.Example, pred_output: ClusterOutput | str) -> float:
    """Metric function for optimizer - accepts ClusterOutput or JSON string."""
    return heuristic_eval(example.cluster_json, pred_output).overall


def build_devset(clusters: List[dict], max_items: int) -> List[dspy.Example]:
    dev = []
    for c in clusters[:max_items]:
        dev.append(dspy.Example(cluster_json=json.dumps(c, ensure_ascii=False)).with_inputs("cluster_json"))
    return dev


def evaluate_program(program: ClusterReporter, devset: List[dspy.Example]) -> Dict[str, float]:
    per_rule_sum: Dict[str, float] = {}
    overall_sum = 0.0
    n = 0

    for ex in devset:
        n += 1
        try:
            output = program(cluster_json=ex.cluster_json)
            res = heuristic_eval(ex.cluster_json, output)
        except Exception:
            res = HeuristicResult(
                sub_scores={"valid_json_format": 0.0},
                overall=0.0,
                reasons={"exception": "program forward() exception"}
            )

        overall_sum += res.overall
        for k, v in res.sub_scores.items():
            per_rule_sum[k] = per_rule_sum.get(k, 0.0) + v

    metrics = {"avg_overall": overall_sum / max(1, n)}
    for k, s in sorted(per_rule_sum.items()):
        metrics[f"avg_{k}"] = s / max(1, n)
    return metrics


# =========================
# Main
# =========================

def main():
    if len(sys.argv) < 2:
        print("Usage: python dspy_topic_report_demo.py /path/to/clusters.json")
        sys.exit(1)

    clusters = load_clusters(sys.argv[1])
    devset = build_devset(clusters, max_items=min(MAX_CLUSTERS_FOR_DEV, len(clusters)))
    sample_ex = devset[0]

    # ---- Baseline ----
    baseline = ClusterReporter()
    print("\n=== Baseline evaluation (weighted heuristics) ===")
    baseline_metrics = evaluate_program(baseline, devset)
    print(json.dumps(baseline_metrics, indent=2))

    baseline_out = baseline(cluster_json=sample_ex.cluster_json)
    with open(os.path.join(OUT_DIR, "baseline_sample_output.json"), "w", encoding="utf-8") as f:
        f.write(baseline_out.model_dump_json(indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_DIR}/baseline_sample_output.json")

    # ---- Optimization ----
    # Alternatives:
    #   dspy.BootstrapFewShotWithRandomSearch(...)
    #   dspy.BootstrapRS(...)
    #   dspy.MIPROv2(...)   # often stronger, potentially slower/more expensive
    optimizer = dspy.BootstrapFewShot(
        metric=lambda ex, pred, trace: metric_for_optimizer(ex, pred.output if hasattr(pred, 'output') else pred),
        max_bootstrapped_demos=6
    )

    print("\n=== Optimization (compile) ===")
    optimized = optimizer.compile(baseline, trainset=devset)

    print("\n=== Optimized evaluation (weighted heuristics) ===")
    optimized_metrics = evaluate_program(optimized, devset)
    print(json.dumps(optimized_metrics, indent=2))

    artifact_path = os.path.join(OUT_DIR, "cluster_reporter_optimized.json")
    optimized.save(artifact_path)
    print(f"Saved optimized artifact: {artifact_path}")

    opt_out = optimized(cluster_json=sample_ex.cluster_json)
    with open(os.path.join(OUT_DIR, "optimized_sample_output.json"), "w", encoding="utf-8") as f:
        f.write(opt_out.model_dump_json(indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_DIR}/optimized_sample_output.json")

    # ---- Production demo ----
    print("\n=== Production demo (load artifact, run one cluster) ===")
    prod = ClusterReporter()
    prod.load(artifact_path)

    prod_ex = devset[min(1, len(devset) - 1)]
    prod_out = prod(cluster_json=prod_ex.cluster_json)

    with open(os.path.join(OUT_DIR, "prod_output.json"), "w", encoding="utf-8") as f:
        f.write(prod_out.model_dump_json(indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_DIR}/prod_output.json")

    # Print a tiny parsed preview
    try:
        print("[PROD] Title:", prod_out.section_a.title)
        print("[PROD] Practical:", prod_out.section_a.why_it_matters.practical_significance[:120] + ("..." if len(prod_out.section_a.why_it_matters.practical_significance) > 120 else ""))
        print("[PROD] Research:", prod_out.section_a.why_it_matters.research_significance[:120] + ("..." if len(prod_out.section_a.why_it_matters.research_significance) > 120 else ""))
        print("[PROD] Keywords:", prod_out.section_b.keyword_list)
    except Exception:
        print("[PROD] Output not parseable by schema (unexpected).")

    # ---- Show all per-rule averages baseline vs optimized ----
    print("\n=== Baseline vs Optimized (per-rule averages) ===")
    all_keys = sorted({k.replace("avg_", "") for k in baseline_metrics if k.startswith("avg_")}
                      | {k.replace("avg_", "") for k in optimized_metrics if k.startswith("avg_")})

    for key in all_keys:
        b = baseline_metrics.get(f"avg_{key}", None)
        o = optimized_metrics.get(f"avg_{key}", None)
        if b is None or o is None:
            continue
        print(f"{key:36s} baseline={b:.3f}  optimized={o:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
