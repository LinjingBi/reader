# Memo: 3‑Stage Design (RDB → RAG/Ranking → CLI/Receipts) and Extension to Tool Calls 

This memo summarizes a **three-stage design** for a local-first paper reader / report system, based on our discussion about:
- **RDB as the source of truth**
- **RAG-like retrieval without polluting evidence**
- **memo CLI as the operational “front door”**
…and how the same pattern naturally extends to **LLM tool calling**.

[human] this memo is summarized by ChatGPT Thinking from our conversation wrt memo future design and a discussion wrt RAG for AI.

---

## Stage 1 — RDB as the Source of Truth (Structure-Preserving Storage)

### Goal
Make your system **auditable, reproducible, and drift-resistant** by storing **raw, high-fidelity evidence** with strong provenance.

### Core idea
Treat the relational database as a **contract**, not a cache:
- store **paper chunks** (raw text segments) and **structured metadata**
- keep stable keys, invariants, and migration history
- everything downstream (summaries, reports, embeddings) is **derived**

### Why this matters
Academic PDFs are already “high quality data”. If you only store LLM summaries, you lose:
- traceability (“where did this claim come from?”)
- robustness (summaries drift/hallucinate)
- evolution safety (schema changes become chaotic)

### Practical storage model
Store chunked evidence as rows:

- **paper**: paper_id, title, url, source, period, etc.
- **paper_chunk**: paper_id, chunk_id, section, page_start/page_end, char offsets, text
- optional: **paper_chunk_fts** (SQLite FTS5) for lexical search
- optional: **paper_chunk_embedding** for semantic ranking (do not replace raw text)

> Key principle: **RDB holds the canonical text**; embeddings are an index/ranker, not the truth.

---

## Stage 2 — Retrieval & Ranking (RAG Without Polluting Evidence)

### Goal
Answer “What evidence should Call 2 (Writer) see?” with a higher-resolution, writer-ready pack.

### Why RAG shows up here
As your plans get more complex, a fixed enumeration like “Top-3 papers + intro/conclusion” becomes brittle:
- the writer may need very specific details (“datasets”, “limitations”, “failure modes”)
- the set of needed evidence varies by intent and by the plan

So you want the planner to express needs as **queries**, not rigid slots/selectors.

### Hybrid approach (recommended)
1. **Deterministic filtering via SQL**
   - constrain by paper_id, section, time period, topic, etc.
2. **Lexical retrieval via FTS**
   - “ablation”, “baseline”, “limitations”, “failure mode”
3. **Semantic reranking**
   - embed query + candidate chunks and rank by similarity

For example:
- RDB stores canonical chunks + structure + provenance

Store chunks in SQLite:

paper_chunk(paper_id, chunk_id, section, page_start, page_end, text, …)

- FTS (SQLite FTS5 / Postgres tsvector / Elastic) for lexical recall

Add FTS5 virtual table:

paper_chunk_fts(text, paper_id UNINDEXED, section UNINDEXED, chunk_id UNINDEXED)

- (optional)Vector index for semantic ranking

Embed those candidates and rank by similarity


This produces a “RAG” effect while preserving a clean truth boundary:
- **evidence remains raw chunks**
- ranking/search is a derived layer
- you can always show what chunks were used

### Why not “vector DB only”?
Vector DBs are good at fast approximate ranking, but weak as a canonical store:
- limited joins/constraints compared to SQL
- harder provenance and migration story
- less transparent debugging

The stable design is:
- **RDB = canonical evidence store**
- **FTS + embeddings = retrieval/ranking accelerators**
- **planner outputs evidence gaps and retrieval queries**

---

## Stage 3 — memo CLI as the Operational Boundary (Execution + Receipts)

### Goal
Turn “storage + retrieval” into a **reliable system**: stable commands, receipts, and audit trails.

### What the memo CLI really is
The CLI is the *front door* to the storage contract:
- it enforces invariants (keys, schema expectations, required metadata)
- it standardizes “write actions” (insert observation, link topic, save report metadata)
- it produces machine-readable receipts (JSON outputs, logs to stderr)

### Why it’s high leverage
Prompts/models/UIs change. A stable CLI + schema is what makes the project durable:
- you can swap report generation strategies
- you can re-run or backfill reports
- you can audit what happened and when
- you can add new pipelines without rewriting everything

### Receipts as first-class objects
For each action, store (or emit):
- input identifiers
- tool/model config id
- derived outputs (paths/hashes)
- time
- errors/fallbacks

This mirrors how you want to treat reports: **metadata in DB, full content on disk, provenance everywhere**.

---

## How this Extends to LLM Tool Calling (Same Pattern)

Tool calling is the same retrieval+ranking problem, but over **actions** rather than text.

### 1) Tool Catalog = “RDB contract for tools”
Treat tools like structured entities:
- name, IO schema, constraints/preconditions
- costs (latency/$), reliability stats
- permissions/safety constraints
- examples, versioning

This catalog is your “truth”.

### 2) Tool Retrieval & Ranking = “RAG for actions”
Given an intent, do:
- candidate generation (rule-based constraints: required inputs, allowed tools)
- ranking (heuristics now, learned reranker later)
- selection policy (top-1, probe-first, ask user, fallback)

### 3) Tool Receipts = “memo CLI for tool execution”
Every tool call should produce an auditable trace:
- candidates + scores
- chosen tool + why
- inputs/outputs
- errors/fallback path
- tool catalog version

This makes the agent robust and debuggable—just like your paper reader.

---

## Summary: The Three Stages in One Line Each

1. **RDB contract:** store raw chunks + structured metadata as canonical truth.
2. **RAG/ranking layer:** retrieve and rank evidence from the RDB (FTS + embeddings), without replacing raw text.
3. **memo CLI boundary:** enforce invariants and emit receipts so the system stays auditable as it evolves.

---

## Suggested “North Star” Principle

**Raw evidence is sacred.**
Everything else—summaries, reports, rankings, tool choices—is derived and must be traceable back to canonical storage.

