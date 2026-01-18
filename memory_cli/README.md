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

## Docs
- `docs/design.md`
- `docs/contracts.md`
