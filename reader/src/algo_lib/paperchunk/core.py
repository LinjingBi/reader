from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import httpx
from selectolax.parser import HTMLParser

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from .types import PaperId, Url, FetchResult, ParseResult, HeadingBlock, SourceType
from .rules import Rules, normalize_heading_key

CAPTION_PAT = re.compile(r"^\s*(figure|fig\.|table)\s*\d+[:\.\s]", re.I)

def arxiv_urls(paper_id: str) -> Dict[str, str]:
    pid = paper_id.replace("arxiv:", "").strip()
    return {
        "abs_url": f"https://arxiv.org/abs/{pid}",
        "html_url": f"https://arxiv.org/html/{pid}",
        "pdf_url": f"https://arxiv.org/pdf/{pid}.pdf",
    }

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def strip_references_block(text: str, rules: Rules) -> str:
    if not text:
        return text
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if normalize_heading_key(ln) in rules.stop_headings:
            return normalize_whitespace("\n".join(lines[:i]))
    return text

def html_extract_heading_blocks(html_bytes: bytes, rules: Rules) -> Tuple[List[Dict[str, Any]], List[str]]:
    doc = HTMLParser(html_bytes)

    # Remove figures/tables
    for node in doc.css("figure, table"):
        node.decompose()

    warnings: List[str] = []
    blocks: List[Dict[str, Any]] = []

    # Abstract pseudo-block
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

    heading_nodes = doc.css("h1, h2, h3, h4")
    real_headings: List[Tuple[Any, str, int]] = []
    for idx, hn in enumerate(heading_nodes, start=1):
        htxt = (hn.text() or "").strip()
        if not htxt:
            continue
        hk = normalize_heading_key(htxt)
        real_headings.append((hn, htxt, idx))
        if rules.stop_heading(hk):
            break

    if not real_headings and not abstract_text:
        warnings.append("No headings/abstract detected in HTML.")
        return blocks, warnings

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

    lines = [ln for ln in raw.splitlines() if not CAPTION_PAT.match(ln)]
    lines = [ln for ln in lines if ln.strip()]

    hits: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        if len(ln) > 80:
            continue
        hk = normalize_heading_key(ln)
        if not hk:
            continue
        if rules.stop_heading(hk):
            hits.append((i, ln))
            break

        matched_sels, _ = rules.match_selectors(ln)
        if matched_sels:
            hits.append((i, ln))
        else:
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

    if not any(b["heading_key"] in ("abstract", "summary") for b in blocks):
        for i, ln in enumerate(lines[:200]):
            if normalize_heading_key(ln) == "abstract":
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

async def fetch_papers_async(
    papers: Dict[PaperId, Url],
    *,
    concurrency: int = 16,
    timeout_s: float = 30.0,
    prefer: str = "auto",
) -> Dict[PaperId, FetchResult]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: Dict[PaperId, FetchResult] = {}
    headers = {"User-Agent": "paperchunk/0.0.2"}

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
        async def fetch_one(pid: PaperId, url_hint: Url) -> None:
            async with sem:
                urls = arxiv_urls(pid)
                html = None
                pdfb = None
                fetched_html = False
                fetched_pdf = False
                status_code = None
                err = None

                try:
                    if prefer in ("auto", "html"):
                        r = await client.get(urls["html_url"])
                        status_code = r.status_code
                        if r.status_code == 200 and r.content and (b"<html" in r.content[:2000].lower() or b"<!doctype" in r.content[:2000].lower()):
                            html = r.content.decode("utf-8", errors="ignore")
                            fetched_html = True
                        elif prefer == "html":
                            err = f"HTML unavailable (status={r.status_code})"
                    if (not fetched_html) and prefer in ("auto", "pdf"):
                        r = await client.get(urls["pdf_url"])
                        status_code = r.status_code
                        if r.status_code == 200 and r.content[:4] == b"%PDF":
                            pdfb = bytes(r.content)
                            fetched_pdf = True
                        else:
                            err = err or f"PDF unavailable (status={r.status_code})"
                except Exception as e:
                    err = str(e)

                ok = fetched_html or fetched_pdf
                results[pid] = FetchResult(
                    paper_id=pid,
                    url=url_hint or urls["abs_url"],
                    ok=ok,
                    status_code=status_code,
                    error=None if ok else err,
                    html=html,
                    pdf_bytes=pdfb,
                    fetched_html=fetched_html,
                    fetched_pdf=fetched_pdf,
                )

        await asyncio.gather(*(fetch_one(pid, url) for pid, url in papers.items()))
    return results

def parse_paper(
    fetch: FetchResult,
    rules: Rules,
    *,
    prefer: str = "auto",
    executor: Optional[ThreadPoolExecutor] = None,
) -> ParseResult:
    if not fetch.ok:
        return ParseResult(paper_id=fetch.paper_id, url=fetch.url, ok=False, error=fetch.error, blocks=[])

    source_used: Optional[SourceType] = None
    warnings: List[str] = []
    blocks_dict: List[Dict[str, Any]] = []

    if prefer == "pdf" and fetch.pdf_bytes:
        source_used = "pdf"
        blocks_dict, warnings = pdf_extract_heading_blocks(fetch.pdf_bytes, rules)
    elif prefer == "html" and fetch.html:
        source_used = "html"
        blocks_dict, warnings = html_extract_heading_blocks(fetch.html.encode("utf-8", errors="ignore"), rules)
    else:
        if fetch.html:
            source_used = "html"
            blocks_dict, warnings = html_extract_heading_blocks(fetch.html.encode("utf-8", errors="ignore"), rules)
        elif fetch.pdf_bytes:
            source_used = "pdf"
            blocks_dict, warnings = pdf_extract_heading_blocks(fetch.pdf_bytes, rules)
        else:
            return ParseResult(paper_id=fetch.paper_id, url=fetch.url, ok=False, error="no payload", blocks=[])

    blocks: List[HeadingBlock] = []
    for b in blocks_dict:
        blocks.append(
            HeadingBlock(
                paper_id=fetch.paper_id,
                source=source_used,
                block_index=int(b.get("index", 0)),
                heading_raw=str(b.get("raw_heading", "")),
                heading_key=str(b.get("heading_key", "")),
                text_raw=str(b.get("text", "")),
            )
        )

    return ParseResult(
        paper_id=fetch.paper_id,
        url=fetch.url,
        ok=True,
        source_used=source_used,
        blocks=blocks,
        used_html=(source_used == "html"),
        used_pdf=(source_used == "pdf"),
        error="; ".join(warnings) if warnings else None,
    )
