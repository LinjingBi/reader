use anyhow::{Context, Result};
use rusqlite::Connection;
use std::fs;

/// Apply schema SQL (idempotent). For MVP we execute the full schema on each start.
///
/// In production you would replace this with explicit migrations.
pub fn apply_schema(conn: &Connection, schema_path: &str) -> Result<()> {
    let sql = fs::read_to_string(schema_path)
        .with_context(|| format!("failed to read schema file: {schema_path}"))?;
    let tx = conn.unchecked_transaction()?;
    tx.execute_batch(&sql)?;
    tx.commit()?;
    Ok(())
}
