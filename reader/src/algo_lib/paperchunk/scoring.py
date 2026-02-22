from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .types import (
    PaperId, Url, SelectorId, TextId,
    DebugHeadingEvent, ScoreRow, TextTable, ScoreSummary, ScoreOutput,
    PaperStatus, PapersStatus, RulesMeta,
)
from .configs import EngineConfig, ScoreConfig
from .rules import Rules
from .core import fetch_papers_async, parse_paper, FetchResult
from .clean import clean_v1

def make_text_id(paper_id: PaperId, source: str, block_index: int) -> TextId:
    return f"{paper_id}:{source}:{block_index}"

def base_mass(matched_selectors: Tuple[SelectorId, ...]) -> Dict[SelectorId, float]:
    """
    Calculate base mass (weight) distribution for matched selectors.
    
    Distributes a total weight of 1.0 equally among all unique matched selectors.
    This ensures that when a heading matches multiple selectors, each selector
    receives an equal share of the weight. assume the mapped selectors are all equal confidence score
    
    Examples:
        - Single match: ["method"] -> {"method": 1.0}
        - Two matches: ["method", "experiment"] -> {"method": 0.5, "experiment": 0.5}
        - Three matches: ["method", "experiment", "results"] -> 
          {"method": 0.333..., "experiment": 0.333..., "results": 0.333...}
        - Duplicate matches: ["method", "method"] -> {"method": 1.0} (deduplicated)
    
    Args:
        matched_selectors: Tuple of selector IDs that matched a heading
        
    Returns:
        Dictionary mapping each unique selector ID to its weight (sums to 1.0)
    """
    if not matched_selectors:
        return {}
    k = len(set(matched_selectors))
    w = 1.0 / k
    return {s: w for s in set(matched_selectors)}

def process_one_paper_scoring(
    pid: PaperId,
    fr: FetchResult,
    rules: Rules,
    engine: EngineConfig,
    score: ScoreConfig,
) -> Tuple[
    int,  # fetched_ok
    int,  # parsed_ok
    int,  # used_html
    int,  # used_pdf
    int,  # scored_ok
    int,  # ok_papers
    int,  # partial_papers
    int,  # fail_papers
    List[DebugHeadingEvent],
    TextTable,
    Dict[SelectorId, List[Tuple[PaperId, TextId, float]]],
    PapersStatus,
]:
    """Process a single paper synchronously. Intended to be called from within a thread pool executor."""
    debug_events: List[DebugHeadingEvent] = []
    text_table: TextTable = {}
    selector_contrib: Dict[SelectorId, List[Tuple[PaperId, TextId, float]]] = defaultdict(list)
    papers_status: PapersStatus = {}
    
    fetched_ok = parsed_ok = used_html = used_pdf = scored_ok = 0
    ok_papers = partial_papers = fail_papers = 0

    if not fr.ok:
        fail_papers += 1
        papers_status[pid] = PaperStatus.error
        debug_events.append(DebugHeadingEvent(
            paper_id=pid, url=fr.url, source_used=None, block_index=None,
            heading_raw=None, heading_key=None, status="fetch_fail", note=fr.error
        ))
        return (fetched_ok, parsed_ok, used_html, used_pdf, scored_ok, ok_papers, partial_papers, fail_papers,
                debug_events, text_table, selector_contrib, papers_status)

    fetched_ok += 1
    pr = parse_paper(fr, rules, mode=engine.mode)
    if not pr.ok:
        fail_papers += 1
        papers_status[pid] = PaperStatus.error
        debug_events.append(DebugHeadingEvent(
            paper_id=pid, url=fr.url, source_used=pr.source_used, block_index=None,
            heading_raw=None, heading_key=None, status="parse_fail", note=pr.error
        ))
        return (fetched_ok, parsed_ok, used_html, used_pdf, scored_ok, ok_papers, partial_papers, fail_papers,
                debug_events, text_table, selector_contrib, papers_status)

    parsed_ok += 1
    used_html += 1 if pr.used_html else 0
    used_pdf += 1 if pr.used_pdf else 0

    blocks = pr.blocks or []
    if not blocks:
        fail_papers += 1
        papers_status[pid] = PaperStatus.error
        debug_events.append(DebugHeadingEvent(
            paper_id=pid, url=fr.url, source_used=pr.source_used, block_index=None,
            heading_raw=None, heading_key=None, status="no_blocks",
            note="parsed ok but produced no heading blocks"
        ))
        return (fetched_ok, parsed_ok, used_html, used_pdf, scored_ok, ok_papers, partial_papers, fail_papers,
                debug_events, text_table, selector_contrib, papers_status)

    found_required = set()
    mapped_any = False

    for b in blocks:
        is_comb = rules.is_combined_heading(b.heading_raw)
        
        if is_comb:
            # For combined headings, match the whole heading first, then match each part
            matched_sels, matched_aliases = rules.match_selectors(b.heading_raw)
            part_sels, part_aliases = rules.match_combined_parts(b.heading_raw)
            # Combine matches from whole heading and parts
            matched_sels = list(set(matched_sels + part_sels))
            matched_aliases = list(set(matched_aliases + part_aliases))
        else:
            matched_sels, matched_aliases = rules.match_selectors(b.heading_raw)

        if not matched_sels:
            debug_events.append(DebugHeadingEvent(
                paper_id=pid, url=fr.url, source_used=pr.source_used, block_index=b.block_index,
                heading_raw=b.heading_raw, heading_key=b.heading_key, status="unmapped_heading"
            ))
            continue

        mapped_any = True
        debug_events.append(DebugHeadingEvent(
            paper_id=pid, url=fr.url, source_used=pr.source_used, block_index=b.block_index,
            heading_raw=b.heading_raw, heading_key=b.heading_key,
            matched_selectors=tuple(matched_sels),
            matched_aliases=tuple(matched_aliases),
            status="mapped_combined_heading" if is_comb else "mapped_heading"
        ))

        tid = make_text_id(pid, b.source, b.block_index)
        cleaned = score.clean_text_fn(b.text_raw)
        text_table[tid] = cleaned

        # Score calculation: distribute weight equally among matched selectors
        # For a heading matching N selectors, each selector gets weight 1/N
        # Example: "experiments and results" matching both "experiment" and "results"
        #          -> each gets weight 0.5
        bm = base_mass(tuple(matched_sels))
        for sel, w in bm.items():
            # Accumulate contributions: each (paper_id, text_id, weight) tuple
            # represents a portion of text content assigned to this selector
            selector_contrib[sel].append((pid, tid, w))
            if sel in engine.required_selectors:
                found_required.add(sel)

    if not mapped_any:
        fail_papers += 1
        papers_status[pid] = PaperStatus.error
        return (fetched_ok, parsed_ok, used_html, used_pdf, scored_ok, ok_papers, partial_papers, fail_papers,
                debug_events, text_table, selector_contrib, papers_status)

    scored_ok += 1
    if engine.required_selectors and set(engine.required_selectors).issubset(found_required):
        ok_papers += 1
        papers_status[pid] = PaperStatus.ok
    else:
        partial_papers += 1
        papers_status[pid] = PaperStatus.partial

    return (fetched_ok, parsed_ok, used_html, used_pdf, scored_ok, ok_papers, partial_papers, fail_papers,
            debug_events, text_table, selector_contrib, papers_status)


async def run_scoring(
    papers: Dict[PaperId, Url],
    rules_path: str,
    engine: EngineConfig = EngineConfig(),
    score: ScoreConfig = ScoreConfig(clean_text_fn=clean_v1),
    executor: Optional[ThreadPoolExecutor] = None,
) -> ScoreOutput:
    rules = Rules.load(rules_path)

    fetch_map = await fetch_papers_async(
        papers,
        concurrency=engine.concurrency,
        timeout_s=engine.timeout_s,
        mode=engine.mode,
    )

    debug_events: List[DebugHeadingEvent] = []
    text_table: TextTable = {}
    selector_contrib: Dict[SelectorId, List[Tuple[PaperId, TextId, float]]] = defaultdict(list)
    papers_status: PapersStatus = {}

    fetched_ok = parsed_ok = used_html = used_pdf = scored_ok = 0
    ok_papers = partial_papers = fail_papers = 0

    # Use provided executor or create new one for CPU-bound parsing
    loop = asyncio.get_event_loop()
    use_provided_executor = executor is not None
    
    if use_provided_executor:
        # Use provided executor
        futures = [
            loop.run_in_executor(executor, process_one_paper_scoring, pid, fr, rules, engine, score)
            for pid, fr in fetch_map.items()
        ]
        results = await asyncio.gather(*futures)
    else:
        # Create new executor with CPU count as default
        max_workers = os.cpu_count() or 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process all papers concurrently using thread pool
            futures = [
                loop.run_in_executor(executor, process_one_paper_scoring, pid, fr, rules, engine, score)
                for pid, fr in fetch_map.items()
            ]
            results = await asyncio.gather(*futures)

    # Aggregate results from all papers
    for (f_ok, p_ok, u_html, u_pdf, s_ok, o_p, part_p, fail_p,
         events, t_table, s_contrib, p_status) in results:
        fetched_ok += f_ok
        parsed_ok += p_ok
        used_html += u_html
        used_pdf += u_pdf
        scored_ok += s_ok
        ok_papers += o_p
        partial_papers += part_p
        fail_papers += fail_p
        debug_events.extend(events)
        text_table.update(t_table)
        for sel, contribs in s_contrib.items():
            selector_contrib[sel].extend(contribs)
        papers_status.update(p_status)

    # Final selector -> texts score normalization: optionally normalize weights within each selector
    # If normalize_within_selector=True: weights are normalized so they sum to 1.0 per selector
    # If normalize_within_selector=False: raw weights are used (may sum to >1.0 if multiple matches)
    score_rows: List[ScoreRow] = []
    for sel, contribs in selector_contrib.items():
        # Calculate denominator: sum of all weights for this selector (if normalizing)
        denom = sum(w for _, _, w in contribs) if score.normalize_within_selector else 1.0
        if denom <= 0:
            continue
        for pid, tid, w in contribs:
            # Normalize weight: divide by denominator if normalization enabled, else use raw weight
            s = (w / denom) if score.normalize_within_selector else w
            score_rows.append(ScoreRow(paper_id=pid, selector_id=sel, text_id=tid, score=float(s)))

    summary = ScoreSummary(
        total_papers=len(papers),
        fetched_ok=fetched_ok,
        parsed_ok=parsed_ok,
        scored_ok=scored_ok,
        used_html=used_html,
        used_pdf=used_pdf,
        ok_papers=ok_papers,
        partial_papers=partial_papers,
        fail_papers=fail_papers,
        required_selectors=engine.required_selectors,
    )
    rules_meta = RulesMeta(
        version=rules.version,
        compiled_regex_version=rules.compiled_regex_version,
    )
    return ScoreOutput(
        summary=summary,
        debug_heading_events=debug_events,
        text_table=text_table,
        sel2texts_score_table=score_rows,
        papers_status=papers_status,
        rules_meta=rules_meta,
    )
