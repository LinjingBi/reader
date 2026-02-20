# paperchunk (schema-first + working Phase-1 engine)

This package implements a **Phase-1 (lexical alias match)** paper chunker with two modes:

- **Scoring mode** (`paperchunk.run_scoring`)
  - Async fetch arXiv HTML (preferred) with PDF fallback
  - Parse into heading blocks (HTML via selectolax; PDF heuristic via PyMuPDF)
  - Match headings to selectors via `rules.yaml` aliases
  - Emit:
    - `text_table: {text_id: cleaned_text}` (cleaned mapped blocks only)
    - `sel2texts_score_table: [(paper_id, selector_id, text_id, score), ...]`
      - score is **selector→texts score** derived from **1/N base mass** and (optionally) normalized within selector
    - `debug_heading_events`
    - `summary`

- **Training mode** (`paperchunk.run_training`)
  - Same fetch/parse/match engine
  - Additionally computes:
    - unmapped heading aggregates
    - heuristic auto-suggest candidates
    - proposals dict for alias additions and combined headings

## Notes

- The library outputs do **not** include `run_id` / `source` in score rows; callers add those externally.
- Text cleaning is applied only to mapped chunks (default: `clean_v1`).

## Example

```python
import asyncio
from paperchunk import run_scoring, EngineConfig

papers = {"2501.01234": "https://arxiv.org/abs/2501.01234"}
out = asyncio.run(run_scoring(papers, "rules.yaml", EngineConfig()))
print(out.summary)
print(list(out.text_table.items())[:1])
print(out.sel2texts_score_table[:3])
```
