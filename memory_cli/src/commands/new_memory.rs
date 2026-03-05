use crate::contracts::{NewMemoryRequest, NewMemoryResponse};
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::{Context, Result};
use std::io::{self, Read, Write};

fn validate_new_memory(
    req: &NewMemoryRequest,
    db_path: &str,
    schema_path: Option<&str>,
) -> ValidationResult {
    let mut validation = ValidationResult::new();

    eprintln!("Validation starts...");

    match validation::validate_db_path(db_path) {
        Ok(()) => validation.add_pass("Checking database path"),
        Err(e) => validation.add_fail("Checking database path", e.to_string()),
    }

    if let Some(schema_path) = schema_path {
        match validation::validate_schema_path(schema_path) {
            Ok(()) => validation.add_pass("Checking schema path"),
            Err(e) => validation.add_fail("Checking schema path", e.to_string()),
        }
    }

    if req.cluster_pk_hash.is_empty() {
        validation.add_fail("cluster_pk_hash", "must be non-empty".to_string());
    }
    if req.resolved_topic.action != "create" && req.resolved_topic.action != "merge" {
        validation.add_fail(
            "resolved_topic.action",
            "must be 'create' or 'merge'".to_string(),
        );
    }
    if req.resolved_topic.action == "merge" && req.resolved_topic.merge_to_topic.is_none() {
        validation.add_fail(
            "resolved_topic.merge_to_topic",
            "required when action is 'merge'".to_string(),
        );
    }

    validation.print_summary_to_stderr();
    validation
}

pub fn handle(
    dry_run: bool,
    db_path: &str,
    schema_path: Option<&str>,
    input_path: &str,
) -> Result<()> {
    let mut input_str = String::new();
    if input_path == "-" {
        io::stdin().read_to_string(&mut input_str)?;
    } else {
        input_str = std::fs::read_to_string(input_path)
            .with_context(|| format!("failed to read input: {}", input_path))?;
    }

    let req: NewMemoryRequest = serde_json::from_str(&input_str)
        .with_context(|| "invalid JSON payload for new-memory")?;

    let validation = validate_new_memory(&req, db_path, schema_path);

    if !validation.is_all_passed() {
        return Err(validation.to_error().unwrap());
    }

    if dry_run {
        eprintln!("\nAll validations passed (dry-run mode)");
        return Ok(());
    }

    let conn = db::open(db_path)?;
    if let Some(sp) = schema_path {
        db::migrate::apply_schema(&conn, sp)?;
    }

    let store = db::store::Store::new(&conn);
    let report_id = store.new_memory(&req).context("new_memory failed")?;

    let resp = NewMemoryResponse { report_id };
    let out = serde_json::to_string(&resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;

    Ok(())
}
