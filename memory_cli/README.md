# memo

Local memory CLI in Rust.

## Build

```bash
cargo build --release
```

## Bootstrap DB

Schema is applied automatically on each command (idempotent): `schemas/schema.sql`.

## Commands

Commands are grouped by pipeline: **Data Pipeline** (HF fetch, cluster, chunk, enrich) and **Report Generation** (topic resolution, metadata, evidence supply, persist, verify).

---

### Commands for Data Pipeline

Used by the HF data pipeline (`hf-data`): fetch, cluster, chunk, enrich.

#### fresh-paper

Phase 1: atomic monthly ingest + best clustering write.

```bash
memo fresh-paper --input examples/fresh_paper_payload.json --db memo.sqlite
cat papers.json | memo fresh-paper --input -
memo fresh-paper --input papers.json --no-details
```

- `--no-details`: Skip querying paper details (faster, smaller output).

#### get-best-run

Read the selected best clustering run for a period (for LLM enrichment prompt).

```bash
memo get-best-run --source hf_monthly --period-start 2025-01-01 --period-end 2025-01-31 --db memo.sqlite
memo get-best-run --source hf_monthly --period-start 2025-01-01 --period-end 2025-01-31 --top-n 10 --db memo.sqlite
memo get-best-run --source hf_monthly --period-start 2025-01-01 --period-end 2025-01-31 --empty-cluster-observation-only --db memo.sqlite
```

- `--top-n`: Max papers per cluster to include. If omitted, returns all papers.
- `--empty-cluster-observation-only`: Only return clusters that have no cluster_observation.

#### inject-clusters-observation

Write LLM enrichment results (cluster summarization) back into DB as cluster-attached semantic records.

```bash
memo inject-clusters-observation --input observations.json --db memo.sqlite
cat observations.json | memo inject-clusters-observation --input -
```

#### inject-papers-chunk

Ingest paper chunk data from the Python scoring pipeline.

```bash
memo inject-papers-chunk --input examples/inject_papers_chunk_input.json --db memo.sqlite
cat chunks.json | memo inject-papers-chunk --input -
```

The command:

- Upserts chunk library configuration
- For each paper: creates/updates paper run mapping, deletes old chunks, and inserts new chunks
- Processes all papers in a single transaction (all-or-nothing)
- Outputs metadata: `total_papers_count` and `total_chunks_count`

Input format: JSON with `lib_config` and `papers` array. Each paper has `paper_id`, `status` ("ok" | "partial" | "error"), and `chunks` array. Each chunk contains `selector_id`, `text_id`, `text`, and `score`.

See `examples/inject_papers_chunk_input.json` and `examples/inject_papers_chunk_output.json`.

#### get-clusters-observation

Get cluster observations for clusters within a period range.

```bash
memo get-clusters-observation --source hf_monthly --period-start 2025-01-01 --period-end 2025-01-31 --db memo.sqlite
```

---

### Commands for Report Generation

Used by report generation and signature-check pipelines.

#### get-topic-resolver-metadata

Get topic resolver metadata (topics and cluster centroid data for merge/create).

```bash
memo get-topic-resolver-metadata --cluster-pk-hash abc123def456 --db memo.sqlite
```

Returns a JSON object containing:

- `topics`: List of all topics with their centroid data (id, centroid_b64, centroid_weight)
- `cluster`: Cluster metadata with centroid and centroid_weight (cluster size) for the specified cluster_pk_hash

See `examples/get_topic_resolver_metadata_output.json`.

#### get-report-generation-metadata

Get cluster observation, optional top papers, and optional topic reports for report generation.

```bash
memo get-report-generation-metadata --cluster-pk-hash abc123def456 --db memo.sqlite
memo get-report-generation-metadata --cluster-pk-hash abc123def456 --add-top-papers --db memo.sqlite
memo get-report-generation-metadata --cluster-pk-hash abc123def456 --add-topic-reports 42 --add-top-papers --db memo.sqlite
```

- `--add-top-papers`: Include top-K papers (K≤5) with full details.
- `--add-topic-reports <topic_id>`: Include top ≤3 reports for the specified topic.

Returns:

- `new_observation`: Cluster observation data (name, summary, keywords, key_paper_keywords)
- `new_observation_key_paper_details`: Optional top papers (paper_id, title, summary, keywords, rank)
- `history_reports`: Optional top reports (report_id, title, summary, keywords_json, intent_mode, declared_level, depth_mode)

See `examples/get_report_generation_metadata_output.json`.

#### get-report-generation-supply

Fetch evidence (paper chunks and history report fields) to fill evidence gaps from planner output.

```bash
memo get-report-generation-supply --input supplement_request.json --db memo.sqlite
echo '{"paper_requests":[],"report_requests":[]}' | memo get-report-generation-supply --input -
```

Input format: JSON with `paper_requests` (paper_id, selectors) and `report_requests` (report_id, selectors).

See `examples/get_report_generation_supply_input.json` and `examples/get_report_generation_supply_output.json`.

#### new-memory

Persist report generation results (topic, report, links) to the database in a single transaction.

```bash
memo new-memory --input payload.json --db memo.sqlite
cat payload.json | memo new-memory --input -
```

Input format: JSON with `cluster_pk_hash`, `intent_mode`, `resolved_topic`, `plan`, `front_matter`, `save_output`, `topic_resolver_config`.

Returns: `report_id`.

See `examples/new_memory_input.json` and `examples/new_memory_output.json`.

#### get-report

Fetch report metadata (report_id, report_url, intent_mode) by cluster pk_hash.

```bash
memo get-report --cluster-pk-hash abc123def456 --db memo.sqlite
```

Returns status `ok` with `meta` (report_id, report_url, intent_mode), or `not_found` if no report exists.

See `examples/get_report_output.json` and `examples/get_report_output_not_found.json`.

#### check-report-signature

Verify report file signature. Use `report_id` or `cluster_pk_hash` to identify the report. If both present, `report_id` takes precedence.

```bash
memo check-report-signature --input payload.json --db memo.sqlite
echo '{"cluster_pk_hash":"abc123","signature":"sha256hex"}' | memo check-report-signature --input -
```

Input format: JSON with `signature` and either `report_id` or `cluster_pk_hash`.

Returns status: `match`, `not_match`, or `error`.

See `examples/check_report_signature_input.json`, `examples/check_report_signature_input_by_report_id.json`, and `examples/check_report_signature_output.json`.

---

## Interesting Docs

- `docs/memo_mvp_design.md`
- `docs/contracts.md`
- `docs/memo_3_stage_design.md`
- `docs/memo_mvp23stage_migration_plan.md`

