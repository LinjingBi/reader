from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from .types import PaperId, Url, DebugHeadingEvent, TrainOutput, TrainSummary
from .configs import EngineConfig, TrainConfig
from .rules import Rules
from .core import fetch_papers_async, parse_paper, arxiv_urls
from .clean import clean_v1

def selectors_from_rules(rules: Rules) -> List[str]:
    return list(rules.selectors.keys())

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

SELECTORS = list(KW_RULES.keys())

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

def percentile(xs: List[int], p: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    k = int(math.ceil((p / 100.0) * len(xs))) - 1
    k = max(0, min(k, len(xs) - 1))
    return xs[k]

def aggregate_report(run_id: str, rules: Rules, paper_events: List[Dict[str, Any]], heading_events: List[Dict[str, Any]], top_k: int = 50):
    selectors = selectors_from_rules(rules)

    ok_p = sum(1 for p in paper_events if p["status"] == "ok")
    partial_p = sum(1 for p in paper_events if p["status"] == "partial")
    err_p = sum(1 for p in paper_events if p["status"] == "error")
    html_used = sum(1 for p in paper_events if p["source_used"] == "html")
    pdf_used = sum(1 for p in paper_events if p["source_used"] == "pdf")

    selector_chars: Dict[str, List[int]] = {s: [] for s in selectors}
    selector_found: Dict[str, int] = {s: 0 for s in selectors}
    total_chars_list: List[int] = []

    for p in paper_events:
        total_chars_list.append(int(p["metrics"].get("total_chars_extracted", 0)))
        cov = p["selector_coverage"]
        for sel in selectors:
            cc = int(cov.get(sel, {}).get("char_count", 0))
            selector_chars[sel].append(cc)
            if cov.get(sel, {}).get("found", False):
                selector_found[sel] += 1

    n = max(1, len(paper_events))
    selector_coverage = {}
    for sel in selectors:
        xs = selector_chars[sel]
        selector_coverage[sel] = {
            "found_rate": round(selector_found[sel] / n, 4),
            "mean_chars": int(sum(xs) / len(xs)) if xs else 0,
            "p50_chars": percentile(xs, 50),
            "p95_chars": percentile(xs, 95),
        }

    unmapped: Dict[str, Any] = {}
    combined: Dict[str, Any] = {}

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
        paper_id = ev.get("paper", {}).get("paper_id")
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
            for c in candidates:
                sel = c["selector"]
                conf = c.get("confidence", 0.0)
                b["suggest_votes"][sel] = b["suggest_votes"].get(sel, 0.0) + conf

    top_unmapped = []
    for hk, b in unmapped.items():
        unique_papers = len(b["unique_papers"])
        score = b["unmapped_count"] * math.log(1 + unique_papers)
        reps = sorted(b["representative_raw"].items(), key=lambda x: x[1], reverse=True)[:3]
        votes = b["suggest_votes"]
        total_votes = sum(votes.values()) or 0
        dist = {}
        if total_votes:
            dist = {k: votes[k] / total_votes for k in sorted(votes, key=votes.get, reverse=True)}
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

    proposals = {"version": 1, "run_id": run_id, "rules_base_version": rules.version, "proposals": []}
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
            "add_alias": item["heading_key"],
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

    for c in combined_list:
        if c["count"] < MIN_COUNT:
            continue
        proposals["proposals"].append({
            "proposal_id": f"p{pid:03d}",
            "kind": "add_combined_heading",
            "heading_key": c["heading_key"],
            "multi_map": c["proposed_multi_map"],
            "confidence": 0.88,
            "support": {"count": c["count"], "representative_raw": c["representative_raw"], "examples": c["examples"]},
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
    unmapped_candidates = {"top_unmapped_headings": top_unmapped, "combined_heading_candidates": combined_list[:top_k]}
    return report, proposals, unmapped_candidates

async def run_training(
    papers: Dict[PaperId, Url],
    rules_path: str,
    engine: EngineConfig = EngineConfig(),
    train: TrainConfig = TrainConfig(),
    *,
    top_k: int = 50,
    run_id: str = "train",
) -> TrainOutput:
    rules = Rules.load(rules_path)
    selectors = selectors_from_rules(rules)

    fetch_map = await fetch_papers_async(
        papers,
        concurrency=engine.concurrency,
        timeout_s=engine.timeout_s,
        mode=engine.mode,
    )

    paper_events: List[Dict[str, Any]] = []
    heading_events_all: List[Dict[str, Any]] = []
    debug_events: List[DebugHeadingEvent] = []

    fetched_ok = parsed_ok = used_html = used_pdf = 0

    for pid, fr in fetch_map.items():
        urls = arxiv_urls(pid)

        if not fr.ok:
            paper_events.append({
                "paper": {"paper_id": pid, "abs_url": fr.url or urls["abs_url"], "html_url": urls["html_url"], "pdf_url": urls["pdf_url"]},
                "status": "error",
                "source_used": "none",
                "warnings": [fr.error or "fetch failed"],
                "metrics": {"total_chars_extracted": 0},
                "selector_coverage": {s: {"found": False, "char_count": 0} for s in selectors},
            })
            debug_events.append(DebugHeadingEvent(
                paper_id=pid, url=fr.url or urls["abs_url"], source_used=None, block_index=None,
                heading_raw=None, heading_key=None, status="fetch_fail", note=fr.error
            ))
            continue

        fetched_ok += 1
        pr = parse_paper(fr, rules, mode=engine.mode)
        if not pr.ok:
            paper_events.append({
                "paper": {"paper_id": pid, "abs_url": fr.url or urls["abs_url"], "html_url": urls["html_url"], "pdf_url": urls["pdf_url"]},
                "status": "error",
                "source_used": pr.source_used or "none",
                "warnings": [pr.error or "parse failed"],
                "metrics": {"total_chars_extracted": 0},
                "selector_coverage": {s: {"found": False, "char_count": 0} for s in selectors},
            })
            debug_events.append(DebugHeadingEvent(
                paper_id=pid, url=fr.url or urls["abs_url"], source_used=pr.source_used, block_index=None,
                heading_raw=None, heading_key=None, status="parse_fail", note=pr.error
            ))
            continue

        parsed_ok += 1
        used_html += 1 if pr.used_html else 0
        used_pdf += 1 if pr.used_pdf else 0

        selector_text: Dict[str, str] = {s: "" for s in selectors}
        total_chars = 0
        source_used = pr.source_used or "none"
        warnings: List[str] = [pr.error] if pr.error else []

        for b in pr.blocks or []:
            matched_sels, matched_aliases = rules.match_selectors(b.heading_raw)
            is_comb = rules.is_combined_heading(b.heading_raw)

            snippet = (b.text_raw or "")[:240].replace("\n", " ").strip()
            hev = {
                "run_id": run_id,
                "rules_version": rules.version,
                "paper": {"paper_id": pid, "abs_url": fr.url or urls["abs_url"], "html_url": urls["html_url"], "pdf_url": urls["pdf_url"]},
                "source_used": source_used,
                "heading": {"raw": b.heading_raw, "key": b.heading_key, "index": b.block_index},
                "match": {
                    "phase1_rule_matched": bool(matched_sels),
                    "matched_selectors": matched_sels,
                    "matched_aliases": matched_aliases,
                    "is_combined_heading": bool(is_comb),
                },
                "content_preview": {"snippet": snippet},
            }

            if not matched_sels:
                if train.enable_unmapped_candidates:
                    hev["auto_suggest"] = {"candidates": suggest_candidates(b.heading_key, snippet, b.block_index)}
                heading_events_all.append(hev)
                debug_events.append(DebugHeadingEvent(
                    paper_id=pid, url=fr.url or urls["abs_url"], source_used=pr.source_used, block_index=b.block_index,
                    heading_raw=b.heading_raw, heading_key=b.heading_key, status="unmapped_heading"
                ))
                continue

            heading_events_all.append(hev)
            debug_events.append(DebugHeadingEvent(
                paper_id=pid, url=fr.url or urls["abs_url"], source_used=pr.source_used, block_index=b.block_index,
                heading_raw=b.heading_raw, heading_key=b.heading_key,
                matched_selectors=tuple(matched_sels),
                matched_aliases=tuple(matched_aliases),
                status="mapped_combined_heading" if is_comb else "mapped_heading"
            ))

            cleaned = clean_v1(b.text_raw)
            for sel in matched_sels:
                selector_text[sel] += ("\n\n" if selector_text[sel] else "") + cleaned
                total_chars += len(cleaned)

        coverage = {}
        for sel in selectors:
            cc = len(selector_text[sel])
            coverage[sel] = {"found": cc > 0, "char_count": cc}

        status = "ok" if all(selector_text[s].strip() for s in engine.required_selectors) else "partial"

        paper_events.append({
            "paper": {"paper_id": pid, "abs_url": fr.url or urls["abs_url"], "html_url": urls["html_url"], "pdf_url": urls["pdf_url"]},
            "status": status,
            "source_used": source_used,
            "warnings": warnings,
            "metrics": {"total_chars_extracted": total_chars},
            "selector_coverage": coverage,
        })

    report, proposals, unmapped_candidates = aggregate_report(run_id, rules, paper_events, heading_events_all, top_k=top_k)

    summary = TrainSummary(
        total_papers=len(papers),
        fetched_ok=fetched_ok,
        parsed_ok=parsed_ok,
        used_html=used_html,
        used_pdf=used_pdf,
        required_selectors=engine.required_selectors,
        papers_with_required_all=report["summary"]["ok_papers"],
        papers_with_required_partial=report["summary"]["partial_papers"],
        papers_failed=report["summary"]["error_papers"],
    )

    return TrainOutput(
        summary=summary,
        debug_heading_events=debug_events,
        unmapped_candidates=unmapped_candidates if train.enable_unmapped_candidates else {},
        proposals=proposals if train.enable_proposals else {},
    )
