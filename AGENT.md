# LLM-powered SFT Data Factory — High-level design + extractor dev task (for Cursor)

## Goal

Build a **data pipeline powered by LLMs** that constructs **two SFT datasets** to support:

1. **Daily AI Briefing Assistant** (curates/ranks candidates)
2. **Research Buddy** (deep-reads selected items and produces detailed writeups)

The pipeline initially uses **closed-source LLMs as “teacher models”** for both curation and deep summarization, then optionally distills into **open-source student models** later for cheaper inference.

## Sources (for demo)

We will ignore X/Twitter for now. Use:

1. **Hacker News** (interesting tech, not necessarily AI): [https://news.ycombinator.com/](https://news.ycombinator.com/)
2. **Anthropic Engineering** (agents updates): [https://www.anthropic.com/engineering](https://www.anthropic.com/engineering)
3. **Hugging Face Daily Papers + paper detail pages (and possibly comments later)**
   Example list page: [https://huggingface.co/papers/date/2026-01-06](https://huggingface.co/papers/date/2026-01-06)
   Example paper page: [https://huggingface.co/papers/2512.20578](https://huggingface.co/papers/2512.20578)

## Key insight: two page types

### A) List/feed pages (HN front page, HF Daily Papers date page, Anthropic engineering index)

* These pages are **lists**, not content.
* We **do NOT** run “main-text extraction” for ranking on them.
* We parse them deterministically into **candidate objects** (DOM parsing).

### B) Detail pages (single blog post / single paper page)

* These contain real content.
* We extract main text using FineWeb-style approach:
  **raw HTML → trafilatura → cleaned text**
* Full deep reading is only done for **top N** (e.g., top 1–3) candidates after ranking to control cost.

---

# Outputs: two SFT datasets

## Dataset A — Curator/Selector SFT (Daily Briefing Assistant)

Purpose: teach a model/policy to **rank + select + tag** items for a daily briefing.

* Input: a **batch** of candidates (comparison-based ranking is better than single-item classification)
* Candidate fields include:

  * `title`, `url`, `source`, `date`
  * lightweight preview snippet (abstract/intro or first ~1–2 paragraphs)
  * cheap signals (votes/comments, headings count, code links count, math density, length)
* Output: JSON decision:

  * `top_k` with rank, score, reason, tags, tasks, followup_depth (`deep_read` vs `shallow`)
  * discarded items + reason
  * briefing theme

This dataset is useful now even if we don’t train immediately (prompt iteration + auditability).
Selection is harder to distill to small models due to broad knowledge, so we’ll rely on teacher LLM + cheap structural signals.

## Dataset B — Deep Read / Research Buddy SFT

Purpose: teach a model to **read a full paper/blog and produce detailed, evidence-grounded outputs**.

* Input: full extracted content from detail page, **chunked** with stable chunk IDs (e.g., `CHUNK_01...`)
* Output: structured summary with **evidence anchors** (claims cite chunk IDs)

  * TL;DR
  * Contributions / method details / results (papers)
  * Design choices / tradeoffs / checklists (blogs)
  * Limitations
  * Follow-up questions / experiments
    This dataset is more distillable into an open student model.

---

# Pipeline (teacher LLMs first, student later)

1. **Parse list pages** → produce candidate list (no LLM, no trafilatura)
2. **Light preview enrichment** for all candidates (still cheap):

   * fetch candidate detail page lightly and extract short snippet (abstract/intro)
   * compute cheap structural signals
3. **Teacher LLM #1 (Curator)** ranks batch → produces Dataset A + selects top N for deep read
4. For top N:

   * fetch full detail page
   * **extract main text** (trafilatura)
   * chunk text
5. **Teacher LLM #2 (Research Buddy)** produces deep structured outputs → Dataset B
6. Optional later:

   * train student models:

     * student curator on Dataset A (or policy over features)
     * student research buddy on Dataset B (summarization/extraction)

---

# CURRENT DEVELOPMENT TASK: build extractor layer

## What “extractor” means here

We need two extractor types:

1. **List parsers (DOM)**: HN front page / HF daily papers page / Anthropic engineering index → candidates
2. **Detail extractor (FineWeb-style)**: raw HTML → trafilatura → cleaned text + metadata signals

## Immediate task focus: Detail extractor (FineWeb-style)

Implement a Python module/script that:

* Input: raw HTML (string) + optional URL
* Output:

  * `text`: extracted main text (cleaned)
  * `signals`: cheap metadata used for filtering + later student selector features
  * `fetch_meta`: status_code, final_url, bytes, content_type (if fetched via requests)

### FineWeb-style extraction

Use the same core approach described in FineWeb pipeline:
**WARC HTML → trafilatura → text**
(We’re not using CommonCrawl, but the idea is identical: take raw HTML response and run trafilatura.)

### Metadata signals to compute (cheap, not perfect)

These help prefilter, debugging, and later student model selection:

* `text_chars`, `text_words`, `text_lines`
* `heading_count` + sample headings (from HTML h1/h2/h3)
* `code_link_count` + sample code links (github/gitlab/paperswithcode/colab/hf spaces/datasets)
* `math_density` heuristic (latex patterns + math symbols / text length)
* `dup_line_ratio` (rough repetition/junk measure)
* `has_paper_sections` (headings include Abstract/Method/Results/etc.)
* `link_count`, `domain`

### Behavior

* Works for detail pages (Anthropic blog post, HF paper page)
* Might produce noisy output for list pages — that’s okay; list pages will be handled by separate DOM parsers.

---

# Code sketch (starter)

Dependencies:

* `requests`, `trafilatura`, `beautifulsoup4`

Install:

```bash
pip install requests trafilatura beautifulsoup4
```

Single-script starter (can be turned into modules later):

* fetch HTML from URL OR read HTML from file/stdin
* run trafilatura extraction
* compute HTML-based signals (headings, links)
* compute text-based signals (length, duplication, symbol ratio)
* emit JSON to stdout or file

(Use the draft code below as a starting point.)

```python
import requests
from bs4 import BeautifulSoup
import trafilatura
import re, json, time
from urllib.parse import urlparse

def fetch_html(url):
    r = requests.get(url, headers={"User-Agent":"daily-ai-briefing/0.1"}, timeout=30)
    r.raise_for_status()
    return r.text, {"final_url": r.url, "status_code": r.status_code, "content_type": r.headers.get("content-type",""), "bytes": len(r.content)}

def extract_main_text(html, url=None):
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False, favor_precision=True)
    return (text or "").strip()

def compute_signals(html, text, url=None):
    soup = BeautifulSoup(html, "html.parser")
    headings = [h.get_text(" ", strip=True)[:160] for h in soup.find_all(["h1","h2","h3"]) if h.get_text(strip=True)]
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    code_links = [l for l in links if isinstance(l,str) and any(s in l.lower() for s in ["github.com","gitlab.com","paperswithcode.com","colab.research.google.com","huggingface.co/spaces","huggingface.co/datasets"])]
    latex_hits = len(re.findall(r"(\$\$.*?\$\$|\$.*?\$|\\begin\{.*?\}|\\end\{.*?\})", soup.get_text("\n", strip=True), re.DOTALL))
    math_symbol_hits = len(re.findall(r"[=+\-*/^_]", soup.get_text("\n", strip=True)))
    math_density = (latex_hits*20 + math_symbol_hits) / max(1, len(soup.get_text("\n", strip=True)))

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    dup_line_ratio = 0.0 if len(lines) < 10 else 1.0 - (len(set(lines)) / max(1, len(lines)))

    paperish = {"abstract","introduction","method","methods","experiments","results","conclusion"}
    has_paper_sections = any(h.lower() in paperish for h in headings)

    domain = urlparse(url).netloc if url else ""
    return {
        "domain": domain,
        "heading_count": len(headings),
        "headings": headings[:30],
        "link_count": len(links),
        "code_link_count": len(code_links),
        "code_links_sample": code_links[:10],
        "math_density": float(math_density),
        "text_chars": len(text),
        "text_words": len(text.split()),
        "dup_line_ratio": float(dup_line_ratio),
        "has_paper_sections": bool(has_paper_sections),
    }

def run(url):
    html, meta = fetch_html(url)
    text = extract_main_text(html, url=url)
    sig = compute_signals(html, text, url=url)
    out = {"source":"unknown","url":url,"fetched_at":int(time.time()),"fetch_meta":meta,"text":text,"signals":sig}
    print(json.dumps(out, ensure_ascii=False, indent=2))
```

---

# Next after extractor

After detail extractor is ready, implement list parsers:

* `parse_hn_frontpage()` -> candidates
* `parse_hf_daily(date_url)` -> candidates (paper_id/title/votes/comments/submitted_by/org)
* `parse_anthropic_engineering_index()` -> candidates

Then implement “light preview enrichment” (snippet extraction) and curator prompt generation (Dataset A).
