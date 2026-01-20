# JSON Contracts (Reader ↔ Memo CLI)

This document specifies the MVP data contract for **Reader Step 1–2** (persist monthly snapshot + best clustering).

## Command: `memo fresh-paper`

### Input payload: `FreshPaperRequest`

Top-level fields:
- `source`: string (e.g., `hf_monthly`)
- `period_start`: string (YYYY-MM-DD)
- `period_end`: string (YYYY-MM-DD)
- `snapshot_id`: optional string. If omitted, Memo derives `source|period_start|period_end`.
- `raw_json`: optional string. Original HF API response serialized as JSON.
- `embed_config`: object with `embed_config_id` and `json_payload`.
- `cluster_config`: object with `cluster_config_id` and `json_payload`.
- `papers`: array of paper objects
- `clusters`: array of clusters, each containing ordered members
- `role`: optional string. Defaults to `hf_batch`.

### Example payload
See: `examples/fresh_paper_payload.json`.

### Output (JSON)
```json
{
  "snapshot_id": "hf_monthly|2025-01-01|2025-01-31",
  "cluster_run_id": "hf_monthly|2025-01-01|2025-01-31|e5-base-v2|kmeans-k4|hf_batch"
}
```

## Command: `memo get-best-run`
Returns a structured payload used to build the LLM enrichment prompt (Step 3).

Example output:
- `examples/get_best_run_output.json`

## Reader-side config files
Reader should keep explicit config files for reproducibility:
- `examples/embed_config.json`
- `examples/cluster_config.json`

Memo stores these as JSON blobs in `embed_config.json_payload` and `cluster_config.json_payload`.
