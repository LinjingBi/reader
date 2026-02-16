# Paper Curation (Phase 1) — Standalone Local Script

This package contains a standalone Python script for **Phase 1 rule iteration** for arXiv paper section extraction.

It:
1) reads `papers.json` (list of paper ids + urls)
2) loads `rules.yaml` (aliases per selector; compiled to regex internally)
3) fetches arXiv HTML first, and falls back to PDF only if HTML is unavailable
4) extracts headings + section text (text-only; best-effort ignores figures/tables; stops at references)
5) writes lossless run logs (`heading_events.jsonl`, `paper_events.jsonl`)
6) aggregates a `report.json` + `proposals.yaml` for human review

## Data files

### clusters_data/
The `clusters_data/` directory contains 2025 all year HF monthly paper clustering results, saved in JSON format. These files contain the clustering output from the reader system.

### extract_reader_papers.py
The `extract_reader_papers.py` script converts the reader clustering JSON output to paper id and URL format that is required by `paper_curation_phase1.py`. It processes all JSON files in `clusters_data/`, extracts papers from cluster members, removes the "hf:" prefix from paper IDs, deduplicates papers across all files, and generates the `papers.json`.

## Quick start

### 1) Install dependencies with uv
```bash
uv sync
```

Or if you prefer to use a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2) Prepare input
Edit `papers.json` (example included):
```json
[
  {"id":"2502.01637","url":"https://arxiv.org/abs/2502.01637"}
]
```

### 3) Run
```bash
python paper_curation_phase1.py run --papers papers.json --rules rules.yaml --out runs
```

This will create a folder like:
```
runs/2026-02-16T18-40-12Z/
  rules.yaml
  heading_events.jsonl
  paper_events.jsonl
  report.json
  proposals.yaml
```

## Key outputs

### report.json
Human review artifact:
- selector coverage stats
- top unmapped headings + examples + auto-suggestions
- ambiguous headings
- combined heading candidates
- proposal count

### proposals.yaml
Suggested patches to `rules.yaml`:
- `add_alias` proposals (high-confidence only)
- `add_combined_heading` proposals (explicit “A and B” headings)

> NOTE: The script does **not** modify your rules automatically. It emits proposals so you can review.

## Notes / Limitations

- HTML parsing works best on arXiv HTML pages (`/html/<id>`). Some papers won’t have HTML.
- PDF fallback is heuristic. Expect imperfect heading detection.
- “Ignore figures/tables” is best-effort:
  - HTML: removes `<figure>` and `<table>`
  - PDF: filters obvious caption lines (Figure/Table)
- References are truncated when a “References”/“Bibliography” heading is detected.

## Typical Phase 1 workflow
1) Run on 100–1000 papers
2) Inspect `report.json`:
   - top unmapped headings
   - ambiguous headings
3) Accept/merge the best proposals into `rules.yaml` (aliases)
4) Re-run; repeat until diminishing returns
5) Then consider Phase 2 (embedding-based scoring) using `heading_events.jsonl` as training/eval data

---

If you want, you can later add a small `apply_proposals.py` tool to merge accepted proposals into `rules.yaml`.
