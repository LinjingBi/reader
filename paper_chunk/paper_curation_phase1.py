#!/usr/bin/env python3
"""
paper_curation_phase1.py

Standalone Phase 1 paper curation runner:
- HTML first (arXiv /html/<id>), PDF fallback only if HTML unavailable
- Extract headings + section text into canonical selectors using rules.yaml aliases
- Emit JSONL logs + aggregated report.json + proposals.yaml

Run:
  python paper_curation_phase1.py run --papers papers.json --rules rules.yaml --out runs
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import yaml
from selectolax.parser import HTMLParser

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


SELECTORS = [
    "summary",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "results",
    "discussion",
    "limitations",
    "conclusion",
]

# Required selectors that must be present for a paper to be marked as "ok" in HTML extraction.
REQUIRED_SELECTORS_FOR_OK = ("summary", "introduction", "method", "conclusion")

CAPTION_PAT = re.compile(r"^\s*(figure|fig\.|table)\s*\d+[:\.\s]", re.I)


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def utc_run_id() -> str:
    # Friendly folder name
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "-")


def arxiv_urls(paper_id: str) -> Dict[str, str]:
    pid = paper_id.replace("arxiv:", "").strip()
    return {
        "abs_url": f"https://arxiv.org/abs/{pid}",
        "html_url": f"https://arxiv.org/html/{pid}",
        "pdf_url": f"https://arxiv.org/pdf/{pid}.pdf",
    }


def normalize_heading_key(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # Strip leading numbering like "1.", "2.3", "III.", "A.1" (best-effort)
    s = re.sub(r"^\s*([IVXLC]+|\d+)([\.\)]\s*|\s+)", "", s, flags=re.I)
    s = re.sub(r"^\s*\d+(\.\d+)*\s*[\.\)]?\s*", "", s)
    s = s.strip().lower()
    # remove wrapping punctuation
    s = s.strip(":-–—. ")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# def compile_alias_regex(alias: str) -> re.Pattern:
#     """
#     Compile a safe regex for an alias:
#     - case-insensitive
#     - anchors full line
#     - whitespace flexible
#     - escapes special chars
#     """
#     alias = alias.strip().lower()
#     # split on whitespace to allow flexible spaces
#     parts = [re.escape(p) for p in re.split(r"\s+", alias) if p]
#     if not parts:
#         parts = [re.escape(alias)]
#     pat = r"^\s*" + r"\s+".join(parts) + r"\s*$"
#     return re.compile(pat, re.I)
def compile_alias_regex(alias: str) -> re.Pattern:
    """
    Compile alias regex as *whole-phrase containment* within a heading_key.

    This enables:
      - "experiments and results" to match "experiments" and "results"
      - "summary and discussion" to match "summary" and "discussion"
      - numbered headings are already stripped by normalize_heading_key()

    Also supports a small whitelist of safe pluralization on the last token.
    """
    alias = alias.strip().lower()
    tokens = [t for t in re.split(r"\s+", alias) if t]
    if not tokens:
        tokens = [alias]

    plural_whitelist = {
        "work": r"work(?:s)?",
        "experiment": r"experiment(?:s)?",
        "conclusion": r"conclusion(?:s)?",
        "method": r"method(?:s)?",
        "preliminary": r"preliminar(?:y|ies)",
        "result": r"result(?:s)?",
        "limitation": r"limitation(?:s)?",
        "discussion": r"discussion(?:s)?",
        "setting": r"setting(?:s)?",
        "dataset": r"dataset(?:s)?",
        "evaluation": r"evaluation(?:s)?",
    }

    parts: List[str] = []
    for i, tok in enumerate(tokens):
        tok_l = tok.lower()
        if i == len(tokens) - 1 and tok_l in plural_whitelist:
            parts.append(plural_whitelist[tok_l])
        else:
            parts.append(re.escape(tok_l))

    # phrase with flexible whitespace
    phrase = r"\s+".join(parts)

    # "word-ish" boundaries:
    # - avoid matching inside longer tokens
    # - but allow punctuation/whitespace around
    pat = rf"(?:^|[^\w]){phrase}(?:$|[^\w])"
    return re.compile(pat, re.I)



@dataclass
class Rules:
    version: int
    compiled_regex_version: int
    selectors: Dict[str, Dict[str, Any]]
    combined_join_tokens: List[str]
    stop_headings: List[str]
    # compiled
    alias_regex: Dict[str, List[Tuple[str, re.Pattern]]]  # selector -> [(alias, regex), ...]

    @staticmethod
    def load(path: Path) -> "Rules":
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
        version = int(obj.get("version", 1))
        crv = int(obj.get("compiled_regex_version", 1))
        selectors = obj.get("selectors", {})
        join_tokens = obj.get("combined_heading_policy", {}).get("join_tokens", ["and", "&", "/"])
        stop_headings = obj.get("ignore_policy", {}).get("stop_headings", ["references", "bibliography"])

        alias_regex: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        for sel, meta in selectors.items():
            aliases = meta.get("aliases", []) or []
            alias_regex[sel] = [(a, compile_alias_regex(a)) for a in aliases]

        return Rules(
            version=version,
            compiled_regex_version=crv,
            selectors=selectors,
            combined_join_tokens=list(join_tokens),
            stop_headings=[s.lower() for s in stop_headings],
            alias_regex=alias_regex,
        )

    def stop_heading(self, heading_key: str) -> bool:
        return heading_key in self.stop_headings

    def match_selectors(self, heading_raw: str) -> Tuple[List[str], List[str]]:
        """
        Return (selectors_matched, aliases_matched) for heading_raw using alias regex.
        """
        hk = normalize_heading_key(heading_raw)
        matched_sels: List[str] = []
        matched_aliases: List[str] = []
        for sel, pairs in self.alias_regex.items():
            for alias, rgx in pairs:
                if rgx.search(hk):
                    matched_sels.append(sel)
                    matched_aliases.append(alias)
                    break
        return matched_sels, matched_aliases

    def is_combined_heading(self, heading_raw: str) -> bool:
        hk = normalize_heading_key(heading_raw)
        for tok in self.combined_join_tokens:
            if f" {tok} " in f" {hk} ":
                return True
        return False


def fetch(url: str, timeout_s: float = 25.0) -> Tuple[int, bytes, Dict[str, str]]:
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout_s,
        headers={"User-Agent": "paper-curation-phase1/0.1"},
    ) as client:
        r = client.get(url)
        return r.status_code, r.content, dict(r.headers)


async def async_fetch(url: str, timeout_s: float = 25.0) -> Tuple[int, bytes, Dict[str, str]]:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout_s,
        headers={"User-Agent": "paper-curation-phase1/0.1"},
    ) as client:
        r = await client.get(url)
        return r.status_code, r.content, dict(r.headers)


def strip_references_block(text: str, rules: Rules) -> str:
    if not text:
        return text
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if normalize_heading_key(ln) in rules.stop_headings:
            return normalize_whitespace("\n".join(lines[:i]))
    return text


# ----------------------------
# HTML extraction
# ----------------------------
def html_extract_heading_blocks(html_bytes: bytes, rules: Rules) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns:
      blocks: list of {raw_heading, heading_key, index, level, text}
      warnings: list
    """
    doc = HTMLParser(html_bytes)

    # Remove figures/tables and their content.
    for node in doc.css("figure, table"):
        node.decompose()

    warnings: List[str] = []
    blocks: List[Dict[str, Any]] = []

    # Abstract as a pseudo-block (summary)
    abstract_text = ""
    abs_node = doc.css_first(".ltx_abstract")
    if abs_node:
        abstract_text = abs_node.text(separator="\n").strip()
        abstract_text = normalize_whitespace(abstract_text)
    if abstract_text:
        blocks.append({
            "raw_heading": "Abstract",
            "heading_key": normalize_heading_key("Abstract"),
            "index": 0,
            "level": 1,
            "text": abstract_text,
            "is_pseudo": True,
        })

    # Headings in arXiv HTML are usually h1-h4, often with class ltx_title
    heading_nodes = doc.css("h1, h2, h3, h4")
    # Collect until references/bibliography
    real_headings: List[Tuple[Any, str, int]] = []
    for idx, hn in enumerate(heading_nodes, start=1):
        htxt = (hn.text() or "").strip()
        if not htxt:
            continue
        hk = normalize_heading_key(htxt)
        if rules.stop_heading(hk):
            real_headings.append((hn, htxt, idx))
            break
        real_headings.append((hn, htxt, idx))

    if not real_headings and not abstract_text:
        warnings.append("No headings/abstract detected in HTML.")
        return blocks, warnings

    # For each heading, collect sibling text until next heading
    for i, (hn, htxt, idx) in enumerate(real_headings):
        hk = normalize_heading_key(htxt)
        if rules.stop_heading(hk):
            break

        next_node = real_headings[i + 1][0] if i + 1 < len(real_headings) else None
        buf: List[str] = []
        cur = hn.next
        while cur is not None and cur is not next_node:
            try:
                t = cur.text(separator="\n").strip()
            except Exception:
                t = ""
            if t:
                buf.append(t)
            cur = cur.next

        text = normalize_whitespace("\n".join(buf))
        if text:
            blocks.append({
                "raw_heading": htxt,
                "heading_key": hk,
                "index": idx,
                "level": int(hn.tag[1]) if hn.tag and hn.tag.startswith("h") and hn.tag[1:].isdigit() else None,
                "text": strip_references_block(text, rules),
                "is_pseudo": False,
            })

    return blocks, warnings


# ----------------------------
# PDF extraction (fallback)
# ----------------------------
def pdf_extract_heading_blocks(pdf_bytes: bytes, rules: Rules) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if fitz is None:
        return [], ["PyMuPDF not installed; cannot fallback to PDF."]
    t0 = time.time()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[str] = []
    for p in doc:
        pages.append(p.get_text("text"))
    raw = normalize_whitespace("\n".join(pages))
    raw = strip_references_block(raw, rules)

    # filter caption lines
    lines = [ln for ln in raw.splitlines() if not CAPTION_PAT.match(ln)]
    lines = [ln for ln in lines if ln.strip()]
    # find heading candidates
    hits: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        # keep short-ish lines as potential headings
        if len(ln) > 80:
            continue
        hk = normalize_heading_key(ln)
        if not hk:
            continue
        if rules.stop_heading(hk):
            hits.append((i, ln))
            break
        # heuristic: line looks like a heading if it matches ANY alias OR is title-cased-ish
        matched_sels, _ = rules.match_selectors(ln)
        if matched_sels:
            hits.append((i, ln))
        else:
            # allow common headings even if unmapped (to generate events)
            if re.match(r"^\s*\d+(\.\d+)*\s+([A-Z][A-Za-z0-9\- ]{2,})\s*$", ln):
                hits.append((i, ln))

    blocks: List[Dict[str, Any]] = []
    if not hits:
        warnings.append("No heading candidates detected in PDF text (heuristic).")
        return blocks, warnings

    for j, (start_i, raw_head) in enumerate(hits):
        hk = normalize_heading_key(raw_head)
        if rules.stop_heading(hk):
            break
        end_i = hits[j + 1][0] if j + 1 < len(hits) else len(lines)
        chunk = normalize_whitespace("\n".join(lines[start_i + 1:end_i]))
        if not chunk:
            continue
        blocks.append({
            "raw_heading": raw_head.strip(),
            "heading_key": hk,
            "index": j + 1,
            "level": None,
            "text": chunk,
            "is_pseudo": False,
        })

    # Attempt abstract
    if not any(b["heading_key"] in ("abstract", "summary") for b in blocks):
        for i, ln in enumerate(lines[:200]):
            if normalize_heading_key(ln) == "abstract":
                # up to next hit or 80 lines
                next_hit = None
                for hi, _ in hits:
                    if hi > i:
                        next_hit = hi
                        break
                end = next_hit if next_hit is not None else min(i + 80, len(lines))
                abs_txt = normalize_whitespace("\n".join(lines[i + 1:end]))
                if abs_txt:
                    blocks.insert(0, {
                        "raw_heading": "Abstract",
                        "heading_key": "abstract",
                        "index": 0,
                        "level": None,
                        "text": abs_txt,
                        "is_pseudo": True,
                    })
                break

    warnings.append(f"PDF parse heuristic (ms={int((time.time()-t0)*1000)}) may be imperfect.")
    return blocks, warnings


# ----------------------------
# Heuristic auto-suggestions (Phase 1)
# ----------------------------
KW_RULES = {
    "experiment": ["experiment", "evaluation", "benchmark", "dataset", "datasets", "metric", "metrics", "implementation", "ablation"],
    "method": ["method", "approach", "model", "architecture", "framework", "algorithm"],
    "introduction": ["introduction", "motivation", "overview", "contribution"],
    "related_work": ["related", "background", "prelim", "prior"],
    "results": ["results", "performance", "accuracy", "comparison"],
    "discussion": ["discussion", "analysis", "insight", "interpretation"],
    "limitations": ["limitation", "caveat", "failure"],
    "conclusion": ["conclusion", "future"],
    "summary": ["abstract", "summary"],
}

def suggest_candidates(heading_key: str, snippet: str, heading_index: int) -> List[Dict[str, Any]]:
    """
    Produce heuristic candidates with a simple scoring:
    - heading keywords
    - snippet keywords
    - position prior (early intro/related; late conclusion/limitations)
    """
    hk = heading_key.lower()
    sn = (snippet or "").lower()

    scores = {sel: 0.0 for sel in SELECTORS}

    def add_kw_score(text: str, weight: float):
        for sel, kws in KW_RULES.items():
            for kw in kws:
                if kw in text:
                    scores[sel] += weight

    add_kw_score(hk, 2.0)
    add_kw_score(sn, 1.0)

    # position priors
    # index 1-3: likely intro/related/method
    if heading_index <= 2:
        scores["introduction"] += 0.6
        scores["related_work"] += 0.4
        scores["summary"] += 0.2
    elif heading_index <= 5:
        scores["method"] += 0.4
        scores["experiment"] += 0.2
    else:
        scores["results"] += 0.2
        scores["discussion"] += 0.2
        scores["conclusion"] += 0.4
        scores["limitations"] += 0.2

    # normalize to confidences with winner-focused amplification
    items = [(sel, sc) for sel, sc in scores.items() if sc > 0.0]
    if not items:
        return []
    items.sort(key=lambda x: x[1], reverse=True)
    
    # Winner detection: if top score is 2x the second score, return only top candidate
    if len(items) >= 2:
        top_score = items[0][1]
        second_score = items[1][1]
        if second_score > 0 and top_score >= 2.0 * second_score:
            # Clear winner - return only top candidate with high confidence
            sel = items[0][0]
            reasons = []
            if any(kw in hk for kw in KW_RULES.get(sel, [])):
                reasons.append("heading keyword")
            if any(kw in sn for kw in KW_RULES.get(sel, [])):
                reasons.append("snippet keyword")
            reasons.append("position prior")
            return [{"selector": sel, "confidence": 0.95, "reasons": reasons}]
    
    # Amplification: square scores before normalizing to amplify winners
    top = items[:3]
    squared_scores = [(sel, sc * sc) for sel, sc in top]
    total = sum(sc for _, sc in squared_scores) or 1.0
    candidates = []
    for sel, sc_squared in squared_scores:
        conf = sc_squared / total
        reasons = []
        if any(kw in hk for kw in KW_RULES.get(sel, [])):
            reasons.append("heading keyword")
        if any(kw in sn for kw in KW_RULES.get(sel, [])):
            reasons.append("snippet keyword")
        reasons.append("position prior")
        candidates.append({"selector": sel, "confidence": round(conf, 3), "reasons": reasons})
    return candidates


# ----------------------------
# Core processing
# ----------------------------
def blocks_to_selector_text(blocks: List[Dict[str, Any]], rules: Rules) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Map heading blocks to canonical selectors using Phase 1 rules.
    Returns:
      selector_text: selector -> concatenated text
      heading_events: list of heading event dicts (without run_id/paper fields filled)
    """
    selector_text = {s: "" for s in SELECTORS}
    heading_events: List[Dict[str, Any]] = []

    for b in blocks:
        raw = b["raw_heading"]
        hk = b["heading_key"]
        idx = int(b["index"])
        txt = b.get("text", "") or ""
        snippet = txt[:300].strip().replace("\n", " ")
        matched_sels, matched_aliases = rules.match_selectors(raw)

        # Combined heading multi-map proposal: only when heading contains join token AND matches >=2 selectors
        is_combined = False
        multi_map: List[str] = []
        if rules.is_combined_heading(raw) and len(set(matched_sels)) >= 2:
            is_combined = True
            multi_map = sorted(list(set(matched_sels)))

        phase1_rule_matched = len(matched_sels) > 0

        # Apply mapping to selector_text
        if is_combined:
            for sel in multi_map:
                if txt:
                    selector_text[sel] = normalize_whitespace((selector_text[sel] + "\n\n" + txt).strip())
        elif phase1_rule_matched:
            # take first matched selector (rules are designed to be unambiguous; if not, that's a design smell)
            sel = matched_sels[0]
            if txt:
                selector_text[sel] = normalize_whitespace((selector_text[sel] + "\n\n" + txt).strip())

        # suggestions only if unmapped (and not combined)
        candidates = []
        if (not phase1_rule_matched) and (not is_combined):
            candidates = suggest_candidates(hk, snippet, idx)

        heading_events.append({
            "source_used": None,  # fill later
            "heading": {
                "raw": raw,
                "key": hk,
                "index": idx,
                "level": b.get("level", None),
            },
            "match": {
                "phase1_rule_matched": phase1_rule_matched,
                "matched_selectors": multi_map if is_combined else matched_sels,
                "matched_aliases": matched_aliases if not is_combined else [],
                "is_combined_heading": is_combined,
            },
            "auto_suggest": {"candidates": candidates},
            "content_preview": {
                "snippet": snippet,
                "char_count": len(txt),
            },
        })

    # Stop headings: if references appear as a heading block, it should have been truncated upstream.
    return selector_text, heading_events


async def process_one_paper(paper_id: str, url_hint: str, rules: Rules, executor: Optional[ThreadPoolExecutor] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns:
      paper_event dict
      heading_event list (paper-level enriched)
    """
    urls = arxiv_urls(paper_id)
    # prefer given url for abs_url if present
    abs_url = url_hint or urls["abs_url"]

    fetch_ms = 0
    parse_ms = 0
    warnings: List[str] = []
    source_used = "none"
    status = "error"

    selector_text: Dict[str, str] = {s: "" for s in SELECTORS}
    heading_events: List[Dict[str, Any]] = []
    
    # Get event loop once for executor calls
    loop = asyncio.get_event_loop()

    # HTML first
    t0 = time.time()
    try:
        code, content, _hdr = await async_fetch(urls["html_url"])
        fetch_ms = int((time.time() - t0) * 1000)
        if code == 200 and content and (b"<html" in content[:2000].lower() or b"<!doctype" in content[:2000].lower()):
            source_used = "html"
            t1 = time.time()
            # Run CPU-bound parsing in thread pool
            blocks, w = await loop.run_in_executor(executor, html_extract_heading_blocks, content, rules)
            parse_ms = int((time.time() - t1) * 1000)
            warnings.extend(w)
            selector_text, heading_events = await loop.run_in_executor(executor, blocks_to_selector_text, blocks, rules)
            status = "ok" if all(selector_text[s].strip() for s in REQUIRED_SELECTORS_FOR_OK) else "partial"
        else:
            warnings.append(f"HTML unavailable (status={code}) -> PDF fallback.")
    except Exception as e:
        warnings.append(f"HTML fetch/parse failed -> PDF fallback: {e}")

    html_available = source_used == "html"
    pdf_available = True

    # PDF fallback only if HTML unavailable
    if source_used != "html":
        t0 = time.time()
        try:
            code, content, _hdr = await async_fetch(urls["pdf_url"])
            fetch_ms = int((time.time() - t0) * 1000)
            if code == 200 and content[:4] == b"%PDF":
                source_used = "pdf"
                t1 = time.time()
                # Run CPU-bound parsing in thread pool
                blocks, w = await loop.run_in_executor(executor, pdf_extract_heading_blocks, content, rules)
                parse_ms = int((time.time() - t1) * 1000)
                warnings.extend(w)
                selector_text, heading_events = await loop.run_in_executor(executor, blocks_to_selector_text, blocks, rules)
                # PDF extraction: "ok" if any selector has text, otherwise "partial"
                status = "ok" if any(selector_text.values()) else "partial"
            else:
                pdf_available = False
                status = "error"
                warnings.append(f"PDF unavailable (status={code}).")
        except Exception as e:
            pdf_available = False
            status = "error"
            warnings.append(f"PDF fetch/parse failed: {e}")

    # Fill per-selector coverage
    selector_coverage: Dict[str, Dict[str, Any]] = {}
    total_chars = 0
    for sel in SELECTORS:
        txt = selector_text.get(sel, "") or ""
        cc = len(txt)
        total_chars += cc
        selector_coverage[sel] = {"found": bool(txt.strip()), "char_count": cc}

    # Enrich heading events with paper + source
    for ev in heading_events:
        ev["source_used"] = source_used

    paper_event = {
        "paper": {
            "paper_id": paper_id,
            "abs_url": abs_url,
            "html_url": urls["html_url"],
            "pdf_url": urls["pdf_url"],
            "html_available": html_available,
            "pdf_available": pdf_available,
        },
        "source_used": source_used,
        "status": status,
        "selector_coverage": selector_coverage,
        "heading_stats": {
            "total_headings": len(heading_events),
            "mapped_headings": sum(1 for e in heading_events if e["match"]["phase1_rule_matched"] or e["match"]["is_combined_heading"]),
            "unmapped_headings": sum(1 for e in heading_events if (not e["match"]["phase1_rule_matched"]) and (not e["match"]["is_combined_heading"])),
        },
        "metrics": {"fetch_ms": fetch_ms, "parse_ms": parse_ms, "total_chars_extracted": total_chars},
        "warnings": warnings,
    }
    return paper_event, heading_events


# ----------------------------
# Aggregation -> report + proposals
# ----------------------------
def percentile(xs: List[int], p: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    k = int(math.ceil((p / 100.0) * len(xs))) - 1
    k = max(0, min(k, len(xs) - 1))
    return xs[k]


def aggregate_report(run_id: str, rules: Rules, paper_events: List[Dict[str, Any]], heading_events: List[Dict[str, Any]], top_k: int = 50) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # summary counts
    ok_p = sum(1 for p in paper_events if p["status"] == "ok")
    partial_p = sum(1 for p in paper_events if p["status"] == "partial")
    err_p = sum(1 for p in paper_events if p["status"] == "error")
    html_used = sum(1 for p in paper_events if p["source_used"] == "html")
    pdf_used = sum(1 for p in paper_events if p["source_used"] == "pdf")

    # selector coverage stats
    selector_chars: Dict[str, List[int]] = {s: [] for s in SELECTORS}
    selector_found: Dict[str, int] = {s: 0 for s in SELECTORS}
    total_chars_list: List[int] = []

    for p in paper_events:
        total_chars_list.append(int(p["metrics"].get("total_chars_extracted", 0)))
        cov = p["selector_coverage"]
        for sel in SELECTORS:
            cc = int(cov[sel]["char_count"])
            selector_chars[sel].append(cc)
            if cov[sel]["found"]:
                selector_found[sel] += 1

    n = max(1, len(paper_events))
    selector_coverage = {}
    for sel in SELECTORS:
        xs = selector_chars[sel]
        selector_coverage[sel] = {
            "found_rate": round(selector_found[sel] / n, 4),
            "mean_chars": int(sum(xs) / len(xs)) if xs else 0,
            "p50_chars": percentile(xs, 50),
            "p95_chars": percentile(xs, 95),
        }

    # unmapped headings aggregation
    # key -> stats
    unmapped = {}
    ambiguous = {}
    combined = {}

    def add_example(bucket: Dict[str, Any], ex: Dict[str, Any], limit: int = 5):
        bucket.setdefault("examples", [])
        if len(bucket["examples"]) < limit:
            bucket["examples"].append(ex)

    for ev in heading_events:
        hk = ev["heading"]["key"]
        raw = ev["heading"]["raw"]
        src = ev["source_used"]
        idx = ev["heading"]["index"]
        snippet = ev["content_preview"]["snippet"]
        paper_id = ev.get("paper", {}).get("paper_id")  # may be absent in raw list
        abs_url = ev.get("paper", {}).get("abs_url")

        is_comb = ev["match"]["is_combined_heading"]
        phase1_matched = ev["match"]["phase1_rule_matched"]
        candidates = ev.get("auto_suggest", {}).get("candidates", [])

        ex = {
            "paper_id": paper_id,
            "abs_url": abs_url,
            "raw_heading": raw,
            "heading_index": idx,
            "source_used": src,
            "snippet": snippet,
        }

        if is_comb:
            # combined heading candidates
            b = combined.setdefault(hk, {"heading_key": hk, "count": 0, "proposed_multi_map": ev["match"]["matched_selectors"], "representative_raw": {}, "examples": []})
            b["count"] += 1
            b["representative_raw"][raw] = b["representative_raw"].get(raw, 0) + 1
            add_example(b, ex)
            continue

        if not phase1_matched:
            b = unmapped.setdefault(hk, {"heading_key": hk, "unmapped_count": 0, "unique_papers": set(), "representative_raw": {}, "suggest_votes": {}, "examples": []})
            b["unmapped_count"] += 1
            if paper_id:
                b["unique_papers"].add(paper_id)
            b["representative_raw"][raw] = b["representative_raw"].get(raw, 0) + 1
            add_example(b, ex)

            # collect suggestion distribution (weighted by confidence scores)
            for c in candidates:
                sel = c["selector"]
                conf = c.get("confidence", 0.0)
                b["suggest_votes"][sel] = b["suggest_votes"].get(sel, 0.0) + conf

    # build top_unmapped list
    top_unmapped = []
    for hk, b in unmapped.items():
        unique_papers = len(b["unique_papers"])
        score = b["unmapped_count"] * math.log(1 + unique_papers)
        reps = sorted(b["representative_raw"].items(), key=lambda x: x[1], reverse=True)[:3]
        # suggestion distribution
        votes = b["suggest_votes"]
        total_votes = sum(votes.values()) or 0
        dist = {}
        if total_votes:
            dist = {k: votes[k] / total_votes for k in sorted(votes, key=votes.get, reverse=True)}
        # top suggestion
        top_sel = None
        top_share = 0.0
        if dist:
            top_sel = next(iter(dist.keys()))
            top_share = dist[top_sel]
        top_unmapped.append({
            "heading_key": hk,
            "unmapped_count": b["unmapped_count"],
            "unique_papers": unique_papers,
            "score": round(score, 3),
            "representative_raw": [x for x, _ in reps],
            "auto_suggest_distribution": {k: round(v, 3) for k, v in dist.items()},
            "top_suggestion": {"selector": top_sel, "share": round(top_share, 3)} if top_sel else None,
            "examples": b["examples"],
        })

    top_unmapped.sort(key=lambda x: x["score"], reverse=True)
    top_unmapped = top_unmapped[:top_k]

    # ambiguous headings from unmapped suggestion distributions
    ambiguous_headings = []
    for item in top_unmapped:
        dist = item.get("auto_suggest_distribution", {})
        if not dist:
            continue
        # ambiguous if top share < 0.75 or top2 close
        sels = list(dist.keys())
        top_share = dist[sels[0]]
        second_share = dist[sels[1]] if len(sels) > 1 else 0.0
        if top_share < 0.75 or (top_share - second_share) < 0.15:
            ambiguous_headings.append({
                "heading_key": item["heading_key"],
                "count": item["unmapped_count"],
                "suggestion_distribution": dist,
                "examples": item["examples"],
            })

    # combined candidates list
    combined_list = []
    for hk, b in combined.items():
        reps = sorted(b["representative_raw"].items(), key=lambda x: x[1], reverse=True)[:3]
        combined_list.append({
            "heading_key": hk,
            "count": b["count"],
            "proposed_multi_map": b["proposed_multi_map"],
            "representative_raw": [x for x, _ in reps],
            "examples": b["examples"],
        })
    combined_list.sort(key=lambda x: x["count"], reverse=True)

    # proposals: high-confidence alias additions (from top_unmapped)
    proposals = {
        "version": 1,
        "run_id": run_id,
        "rules_base_version": rules.version,
        "proposals": []
    }

    MIN_COUNT = 5
    MIN_SHARE = 0.65  # Adjusted for weighted voting + winner-focused normalization

    ambiguous_set = set(a["heading_key"] for a in ambiguous_headings)

    pid = 1
    for item in top_unmapped:
        hk = item["heading_key"]
        if hk in ambiguous_set:
            continue
        top = item.get("top_suggestion") or {}
        sel = top.get("selector")
        share = float(top.get("share") or 0.0)
        if not sel:
            continue
        if item["unmapped_count"] < MIN_COUNT or share < MIN_SHARE:
            continue
        proposals["proposals"].append({
            "proposal_id": f"p{pid:03d}",
            "kind": "add_alias",
            "selector": sel,
            "add_alias": hk,
            "confidence": round(share, 3),
            "support": {
                "unmapped_count": item["unmapped_count"],
                "unique_papers": item["unique_papers"],
                "representative_raw": item["representative_raw"],
                "examples": item["examples"],
            },
            "notes": ["Auto-suggest is heuristic; review examples before accepting."],
        })
        pid += 1

    # proposals for combined headings
    for c in combined_list:
        if c["count"] < MIN_COUNT:
            continue
        proposals["proposals"].append({
            "proposal_id": f"p{pid:03d}",
            "kind": "add_combined_heading",
            "heading_key": c["heading_key"],
            "multi_map": c["proposed_multi_map"],
            "confidence": 0.88,
            "support": {
                "count": c["count"],
                "representative_raw": c["representative_raw"],
                "examples": c["examples"],
            },
            "notes": ["Explicit combined heading; duplication across selectors is acceptable for evidence selectors."],
        })
        pid += 1

    report = {
        "run_id": run_id,
        "rules_snapshot": {"version": rules.version, "compiled_regex_version": rules.compiled_regex_version},
        "input": {"papers_count": len(paper_events)},
        "summary": {
            "ok_papers": ok_p,
            "partial_papers": partial_p,
            "error_papers": err_p,
            "html_used": html_used,
            "pdf_used": pdf_used,
            "mean_total_chars_extracted": int(sum(total_chars_list) / len(total_chars_list)) if total_chars_list else 0,
        },
        "selector_coverage": selector_coverage,
        "top_unmapped_headings": top_unmapped,
        "ambiguous_headings": ambiguous_headings,
        "combined_heading_candidates": combined_list[:top_k],
        "proposals_emitted": {"count": len(proposals["proposals"]), "path": "proposals.yaml"},
    }
    return report, proposals


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def run(papers_path: Path, rules_path: Path, out_dir: Path, top_k: int = 50, max_papers: Optional[int] = None) -> Path:
    run_id = utc_run_id()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rules = Rules.load(rules_path)
    # snapshot the rules into the run dir
    (run_dir / "rules.yaml").write_text(rules_path.read_text(encoding="utf-8"), encoding="utf-8")

    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    
    # Apply max_papers limit if specified
    if max_papers is not None:
        papers = papers[:max_papers]
    
    paper_events: List[Dict[str, Any]] = []
    heading_events_all: List[Dict[str, Any]] = []

    # Auto-detect concurrency based on CPU cores
    concurrency = min(os.cpu_count() or 4, len(papers))
    
    # Create thread pool executor for CPU-bound work
    executor = ThreadPoolExecutor(max_workers=concurrency)
    
    # Process papers concurrently
    async def process_with_enrichment(i: int, p: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        pid = p.get("id") or p.get("paper_id")
        url = p.get("url", "")
        if not pid:
            eprint(f"[skip] missing id in papers.json entry: {p}")
            return None, None
        
        eprint(f"[{i}/{len(papers)}] {pid}")
        pe, hes = await process_one_paper(str(pid), str(url), rules, executor)
        
        # enrich heading events with paper object
        for ev in hes:
            ev["run_id"] = run_id
            ev["rules_version"] = rules.version
            ev["paper"] = {
                "paper_id": pe["paper"]["paper_id"],
                "abs_url": pe["paper"]["abs_url"],
                "html_url": pe["paper"]["html_url"],
                "pdf_url": pe["paper"]["pdf_url"],
            }
        pe["run_id"] = run_id
        pe["rules_version"] = rules.version
        
        return pe, hes
    
    # Process papers in batches to respect concurrency limit
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_with_semaphore(i: int, p: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        async with semaphore:
            return await process_with_enrichment(i, p)
    
    # Gather all results
    tasks = [process_with_semaphore(i, p) for i, p in enumerate(papers, start=1)]
    results = await asyncio.gather(*tasks)
    
    # Collect results
    for pe, hes in results:
        if pe is not None and hes is not None:
            paper_events.append(pe)
            heading_events_all.extend(hes)
    
    executor.shutdown(wait=True)

    # write logs
    write_jsonl(run_dir / "paper_events.jsonl", paper_events)
    write_jsonl(run_dir / "heading_events.jsonl", heading_events_all)

    # aggregate
    report, proposals = aggregate_report(run_id, rules, paper_events, heading_events_all, top_k=top_k)
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "proposals.yaml").write_text(yaml.safe_dump(proposals, sort_keys=False, allow_unicode=True), encoding="utf-8")

    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run", help="Run Phase 1 extraction + reporting")
    ap_run.add_argument("--papers", type=str, required=True, help="Path to 'papers.json'")
    ap_run.add_argument("--rules", type=str, required=True, help="Path to rules.yaml")
    ap_run.add_argument("--out", type=str, default="runs", help="Output directory")
    ap_run.add_argument("--top-k", type=int, default=50, help="Top K unmapped headings to report")
    ap_run.add_argument("--max", type=int, default=None, help="Maximum number of papers to process (for testing/debugging)")

    args = ap.parse_args()

    if args.cmd == "run":
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = asyncio.run(run(Path(args.papers), Path(args.rules), out_dir, top_k=int(args.top_k), max_papers=args.max))
        print("Check: " + str(run_dir))


if __name__ == "__main__":
    main()
