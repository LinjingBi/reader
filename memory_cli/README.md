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

### 2) Read best clustering for LLM prompt (Step 3)
```bash
./target/release/memo-cli get-best-run --source hf_monthly --period-start 2025-01-01 --period-end 2025-01-31 --top-n 10 --db memo.sqlite
```

### 3) Inject cluster observations (LLM enrichment results)
```bash
./target/release/memo-cli inject-clusters-observation --input observations.json --db memo.sqlite
cat observations.json | ./target/release/memo-cli inject-clusters-observation --input -
```

### 4) Get cluster observations for clusters within a period range
```bash
./target/release/memo-cli get-clusters-observation --source hf_monthly --period-start 2025-01-01 --period-end 2025-01-31 --db memo.sqlite
```

### 5) Start a report generation job for a cluster
```bash
./target/release/memo-cli start-report-job --cluster-pk-hash abc123def456 --db memo.sqlite
```

The command checks for existing jobs and handles different states:
- **No job exists**: Creates a new job with status `running`
- **Job exists with status `running`**: Returns existing job info
- **Job exists with status `done`**: Returns the completed report_id
- **Job exists with status `error`**: 
  - If error occurred within 5 minutes: Returns remaining wait time
  - If error occurred more than 5 minutes ago: Resets job to `running` status

## Docs
- `docs/design.md`
- `docs/contracts.md`
