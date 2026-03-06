# memo-cli

Local memory CLI for the **Reader** paper-to-report pipeline.

## Build
```bash
cargo build --release
```

## Bootstrap DB
Schema is applied automatically on each command (idempotent): `schemas/schema.sql`.

## Commands

### 1) Fresh monthly ingest + best clustering (Step 1–2)
```bash
./target/release/memo-cli fresh-paper --input examples/fresh_paper_payload.json --db memo.sqlite
```

### 2) Inject paper chunks (Step 2.5)
```bash
./target/release/memo-cli inject-papers-chunk --input examples/inject_papers_chunk_input.json --db memo.sqlite
cat chunks.json | ./target/release/memo-cli inject-papers-chunk --input -
```

Ingests paper chunk data from the Python scoring pipeline. The command:
- Upserts chunk library configuration
- For each paper: creates/updates paper run mapping, deletes old chunks, and inserts new chunks
- Processes all papers in a single transaction (all-or-nothing)
- Outputs metadata: `total_papers_count` and `total_chunks_count`

Input format: JSON with `lib_config` and `papers` array. Each paper has `paper_id`, `status` ("ok" | "partial" | "error"), and `chunks` array. Each chunk contains `selector_id`, `text_id`, `text`, and `score`.

See `examples/inject_papers_chunk_input.json` for input format and `examples/inject_papers_chunk_output.json` for output format.

### 3) Read best clustering for LLM prompt (Step 3)
```bash
./target/release/memo-cli get-best-run --source hf_monthly --period-start 2025-01-01 --period-end 2025-01-31 --top-n 10 --db memo.sqlite
```

### 4) Inject cluster observations (LLM enrichment results)
```bash
./target/release/memo-cli inject-clusters-observation --input observations.json --db memo.sqlite
cat observations.json | ./target/release/memo-cli inject-clusters-observation --input -
```

### 5) Get cluster observations for clusters within a period range
```bash
./target/release/memo-cli get-clusters-observation --source hf_monthly --period-start 2025-01-01 --period-end 2025-01-31 --db memo.sqlite
```

### 6) Initialize a report generation job for a cluster
```bash
./target/release/memo-cli init-report-job --cluster-pk-hash abc123def456 --db memo.sqlite
```

Returns `next_status` and `meta` (message, optional report_url/report_signature/last_update_utc). States:
- **No job exists**: Creates job, `next_status=running`
- **Job running**: `next_status=waiting`, message includes estimated time remaining
- **Job done**: `next_status=done`, meta includes report_url and report_signature
- **Job error**: 
  - Within 5 minutes: `next_status=waiting`, message includes remaining cooldown
  - Beyond 5 minutes: Updates job to running, `next_status=resuming`

See `examples/init_report_job_output_*.json` for output formats.

### 7) Get topic resolver metadata
```bash
./target/release/memo-cli --db memo.sqlite get-topic-resolver-metadata --cluster-pk-hash abc123def456
./target/release/memo-cli get-topic-resolver-metadata --cluster-pk-hash abc123def456
```

Returns a JSON object containing:
- `topics`: List of all topics with their centroid data (id, centroid_b64, centroid_weight)
- `cluster`: Cluster metadata with centroid and centroid_weight (cluster size) for the specified cluster_pk_hash

### 8) Get report generation metadata
```bash
./target/release/memo-cli get-report-generation-metadata --cluster-pk-hash abc123def456
./target/release/memo-cli get-report-generation-metadata --cluster-pk-hash abc123def456 --add-top-papers
./target/release/memo-cli get-report-generation-metadata --cluster-pk-hash abc123def456 --add-topic-reports 42 --add-top-papers
```

Returns a JSON object containing:
- `new_observation`: Cluster observation data (name, summary, keywords, key_paper_keywords)
- `new_observation_key_paper_details`: Optional Top-K papers (K≤5) with full details (paper_id, title, summary, keywords, rank)
- `history_reports`: Optional top ≤3 reports for the specified topic (report_id, title, summary, keywords_json, intent_mode, declared_level, depth_mode)

## Docs
- `docs/design.md`
- `docs/contracts.md`
