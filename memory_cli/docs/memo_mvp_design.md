# Memo CLI (Memory Service) — MVP Design

## Purpose
Memo CLI is a local, safety-oriented memory service for **Reader**.

Key goals:
- Provide a **narrow, audited command surface** (no arbitrary SQL).
- Ensure **atomic ingestion** for monthly paper snapshots + best clustering artifacts (Reader Step 1–2).
- Preserve **provenance**: embedding config, clustering config, and (later) LLM config.
- Be callable as a **subprocess tool** by Python pipelines and (future) coding agents.

## Non-goals (MVP)
- No long-running daemon/server.
- No evolution.
- No saved embeddings by default (recompute on Reader side).

## Concurrency model
- Multiple processes may call the CLI concurrently.
- SQLite is configured with:
  - `journal_mode=WAL`
  - `foreign_keys=ON`
  - `synchronous=NORMAL`
  - `busy_timeout=5000ms`

This supports many concurrent readers and serialized writers with reasonable throughput.

## CLI surface (MVP)

All updates occur in **one SQLite transaction**. If any statement fails, the DB remains unchanged.

## Schema
The SQLite schema is in `schemas/schema.sql` (idempotent). For MVP the CLI executes it on startup.

## Extension points
Placeholders exist for:
- `evolution pipeline`

These will be implemented as additional safe commands once Reader’s downstream pipeline stabilizes.
