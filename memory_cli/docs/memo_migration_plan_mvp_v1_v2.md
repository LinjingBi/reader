# Short Migration Plan (MVP → v1 → v2) for RDB → Retrieval/RAG → memo CLI

This plan maps the **3-stage architecture** (canonical storage → retrieval/ranking → operational CLI/receipts) into concrete milestones. It also notes how each step naturally extends to **LLM tool calling** later.

---

## MVP (Now): Stable contract + minimal planning/writing

### Stage 1 — RDB contract (truth)
- [ ] Add/confirm tables for **paper**, **report metadata**, and **new_observation** (your renamed “cluster observation”).
- [ ] Decide canonical key strategy (composite vs synthetic ids) and lock it.
- [ ] Store **report metadata in DB**, store **full report markdown on filesystem**.
  - [ ] Replace `report_md` with `report_url` (or `report_path`) + optional `sha256/size_bytes`.
  - [ ] Keep a small optional `report_excerpt` if you want preview-only in DB.

### Stage 2 — Retrieval (very light)
- [ ] Don’t do semantic search yet; instead, support **simple deterministic evidence packs**:
  - quick_background: `new_observation` + optional `history_reports`
  - research_briefing: Top-K summary-level metadata (K≤5)
  - brainstorm/implementation: enabled but intentionally “insufficient” → push into `evidence_request`

### Stage 3 — memo CLI boundary
- [ ] Ensure every write command outputs a **receipt JSON** to stdout (logs to stderr):
  - report insert receipt (report_id/report_url + keys)
  - observation insert receipt
- [ ] Add “get planner metadata” (you already did) and keep it bounded.

### Tool-calling extension (not implemented)
- [ ] Start logging “intent → evidence pack → output” so you have future training/eval traces.

---

## v1 (Next): Paper chunk storage + lexical retrieval + writer-ready evidence packs

### Stage 1 — RDB contract expands (still truth)
- [ ] Add `paper_chunk` table storing **raw text chunks** (section/page offsets).
- [ ] Add SQLite **FTS5** virtual table for chunk text (or an external FTS index):
  - query = keywords, section filters, paper_id filters

### Stage 2 — Retrieval becomes real (RAG-lite)
- [ ] Add **retrieval API** that supports:
  - `paper_id + section` fetch (intro/method/conclusion)
  - FTS search over chunks scoped by paper/topic/period
- [ ] Add **Call1 → Call2 evidence preparation**:
  - Keep `evidence_request` as “what’s missing”
  - Add `call2_evidence_plan` (new): a *small list of retrieval directives* (paper_id/section + keyword queries)
  - Build Call2 pack from: (selected chunks + minimal metadata + citations to chunk ids)

### Stage 3 — CLI evolves into orchestration
- [ ] Add CLI commands:
  - `ingest-pdf-chunks` (no OCR) → paper_chunk rows
  - `query-chunks` (FTS) → chunk ids + text
  - `build-call2-evidence` (from `call2_evidence_plan`) → bounded evidence pack JSON + chunk ids

### Tool-calling extension (v1-ready conceptually)
- [ ] Introduce a “tool catalog” table/schema in DB (name, io, constraints, cost).
- [ ] Log tool selection receipts (candidates + chosen + outputs) even if selection is still rule-based.

---

## v2 (Later): Semantic ranking + learning-to-rank + history-aware planning

### Stage 1 — Add derived indexes (still not truth)
- [ ] Store chunk embeddings in `paper_chunk_embedding` (optional, derived).
- [ ] Store topic/observation embeddings and stability stats (derived).

### Stage 2 — Hybrid retrieval + reranking
- [ ] Candidate generation via SQL + FTS (deterministic).
- [ ] Semantic rerank (embeddings) for top-N candidates.
- [ ] Add “learned reranker” option (model ranks candidates; hard constraints stay outside).
- [ ] Planner becomes history-aware (uses objective recency/coverage/stability signals).

### Stage 3 — memo CLI as an “agent runtime” (local)
- [ ] Make CLI capable of:
  - running planner → retrieval → writer in a single pipeline
  - storing all receipts and provenance
  - replay/backfill with new prompts/models safely

### Tool-calling extension (v2)
- [ ] Use the same retrieval/ranking shell to select **tools**:
  - candidates (constraints) → rank (heuristics/learned) → select → receipt
- [ ] Train rerankers on logged trajectories (offline eval first).

---

## What to measure at each step (so drift stays outside core code)
- MVP: schema stability, receipt completeness, zero hallucination (via evidence_request usage)
- v1: retrieval precision (FTS hit quality), evidence pack boundedness, writer citation correctness (chunk_id coverage)
- v2: reranker uplift vs heuristics, reduced “ask user” rate without increased hallucination, stability of history-aware depth decisions
