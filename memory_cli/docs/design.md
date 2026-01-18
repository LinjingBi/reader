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
- No topic attach / evolution / report writing (Step 4–6 are placeholders only).
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

### 1) `memo fresh-paper --input <payload.json|->`
Atomic write of:
- `source_snapshot`
- `paper`
- `snapshot_paper`
- `cluster_run` (selected_best=1)
- `cluster`
- `cluster_member`
- `embed_config`, `cluster_config` upsert

All updates occur in **one SQLite transaction**. If any statement fails, the DB remains unchanged.

Output (JSON):
```json
{
  "snapshot_id": "hf_monthly|2025-01-01|2025-01-31",
  "cluster_run_id": "hf_monthly|2025-01-01|2025-01-31|..."
}
```

### 2) `memo get-best-run --source ... --period-start ... --period-end ... --top-n 10`
Read-only query that returns the stored best run and top N papers per cluster.
Used to build the LLM prompt for Step 3.

## ID strategy
For MVP, Memo derives a deterministic `snapshot_id` when absent:
```
snapshot_id = "{source}|{period_start}|{period_end}"
```

`cluster_run_id` is derived deterministically to make the command rerunnable:
```
cluster_run_id = "{snapshot_id}|{embed_config_id}|{cluster_config_id}|{role}"
```

Each cluster has:
```
cluster_id = "{cluster_run_id}|c{cluster_index}"
```

## Schema
The SQLite schema is in `schemas/schema.sql` (idempotent). For MVP the CLI executes it on startup.

## Extension points (Step 4–6)
Placeholders exist for:
- `topic-attach`
- `report-write`
- `topic-event-propose`

These will be implemented as additional safe commands once Reader’s downstream pipeline stabilizes.


## Notes
- Step 4–6 commands are tracked in TODO.md (no placeholder Rust implementations).
