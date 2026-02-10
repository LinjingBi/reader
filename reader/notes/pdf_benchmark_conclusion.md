# PDF Benchmark Conclusion — Intro Extraction Without ML/OCR

## Goal and use case

We explored whether we can **extract structured evidence from academic papers (PDF)** *on-the-fly* (during report planning/writing) **without ML/OCR**, and use that extracted text to improve **LLM prompt generation**.

**Why this seemed plausible**
- Academic papers often follow consistent layout patterns (title/abstract/sections), so basic rule-based extraction can be fast.
- For this project, we explicitly want **extraction** (retrieving text faithfully), not summarization.
- Using ML/NLP to “interpret” PDFs (as opposed to deterministic extraction) can introduce **hallucinations** when the source text is noisy or incomplete—especially for deep technical content.

## Dataset

- **50 PDFs** randomly sampled from **Hugging Face 2025 monthly paper snapshots**  
  (conceptually: *50 per month × 12 months*; the benchmark corpus is 50 PDFs total sampled across that year).

## What we benchmarked

We benchmarked the “find Introduction + slice intro text” task using three Python libraries:

1. **PyMuPDF (`fitz`)** — plain text extraction
2. **PyMuPDF “style-based”** — PyMuPDF structured extraction that tracks line font size/boldness and stops at the next heading with matching style (keyword-free stop rule)
3. **pdfplumber** — layout-aware extraction (pdfminer-based)
4. **pymupdf4llm** — Markdown-oriented conversion (heavier)

### Evaluation metrics

- `intro_found_rate`: fraction of PDFs where an “Introduction” heading was detected
- `seconds_mean`, `seconds_p95`: runtime per PDF
- `intro_chars_mean`: size of extracted intro slice (bounded by caps in the script)
- `errors`: extraction failures

The benchmark script iterates over PDFs, extracts the first *N* pages, detects the “Introduction” heading via regex, and slices text until a stopping rule or a character cap is reached. (See the accompanying `benchmark.py` for implementation details and tunable parameters.)

## Results (50 PDFs)

| extractor | n | intro_found_rate | seconds_mean | seconds_p95 | errors | intro_chars_mean |
|---|---:|---:|---:|---:|---:|---:|
| pymupdf | 50 | 0.98 | 0.014985 | 0.034563 | 0 | 4902.44 |
| pymupdf-style | 50 | 0.98 | 0.116722 | 0.328418 | 0 | 4827.70 |
| pdfplumber | 50 | 0.84 | 0.264537 | 0.583617 | 0 | 5584.08 |
| pymupdf4llm | 50 | 0.20 | 0.723158 | 1.936383 | 0 | 1228.86 |

### Interpretation

- **PyMuPDF** was the clear winner for on-the-fly extraction: *very fast* with a *high intro hit-rate*.
- The **style-based PyMuPDF** approach maintained a similar hit-rate, but was slower due to structured parsing and additional logic.
- **pdfplumber** was slower and less reliable for intro detection in this setup (still usable, but not competitive here).
- **pymupdf4llm** underperformed for this specific “Intro slicing” task (low hit-rate and high runtime).

## Key failure mode we observed

Even when “Introduction” is found reliably, many academic PDFs contain **embedded figures/diagrams with rich vector text** (axis labels, diagram callouts, figure annotations)(f.e. 2510.16872).  
Because this figure text is still “text objects” in the PDF, rule-based extractors often include it, and it can **dominate the extracted character budget**—polluting LLM inputs.

Filtering figure text robustly (without OCR/ML) is possible in some cases (e.g., using image/drawing bounding boxes), but:
- increases complexity,
- adds new failure modes,
- and still cannot guarantee clean separation for all publishing templates.

## Decision

**We will *not* do PDF intro extraction on-the-fly inside Call 1/Call 2.**

Instead:
- Call 1 (planner) and Call 2 (writer) will rely on **clean, well-structured metadata already stored in DB/memo**:
  - title, summary, keywords, one-liner (and other stable fields)
- PDF extraction (intro/method/results, etc.) will be moved into the upstream **fetch → embed → cluster** pipeline step, where:
  - runtime constraints are looser,
  - extraction quality can be iterated and cached,
  - and the output can be normalized/validated before it ever reaches prompt construction.

This keeps the report pipeline deterministic and reduces hallucination risk from noisy extraction artifacts.

## Next step (recommended)

Add a dedicated “paper enrichment” stage in the ingestion pipeline:
- extract structured paper sections (or high-quality excerpts),
- store them in DB with a versioned extractor config,
- and only include them in prompts when present and validated.

---

**Artifacts**
- Benchmark implementation: `<project_root>/reader/scripts/pdflib_benchmark/benchmark.py`
