use crate::commands::validation::{self, ValidationResult};
use crate::contracts::CheckReportSignatureRequest;
use crate::db;
use anyhow::{Context, Result};
use std::io::{self, Read, Write};

fn validate_check_report_signature(
    db_path: &str,
    schema_path: Option<&str>,
    req: &CheckReportSignatureRequest,
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

    let has_id = req.report_id.is_some();
    let has_hash = req.cluster_pk_hash.as_ref().map_or(false, |s| !s.is_empty());
    if has_id || has_hash {
        validation.add_pass("At least one of report_id or cluster_pk_hash provided");
    } else {
        validation.add_fail(
            "Request validation",
            "At least one of report_id or cluster_pk_hash is required".to_string(),
        );
    }

    if !req.signature.is_empty() {
        validation.add_pass("Signature provided");
    } else {
        validation.add_fail("Request validation", "signature is required".to_string());
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
        std::io::stdin().read_to_string(&mut input_str)?;
    } else {
        input_str = std::fs::read_to_string(input_path)
            .with_context(|| format!("failed to read input: {}", input_path))?;
    }

    let req: CheckReportSignatureRequest = serde_json::from_str(&input_str)
        .with_context(|| "invalid JSON payload for check-report-signature")?;

    let validation = validate_check_report_signature(db_path, schema_path, &req);

    if !validation.is_all_passed() {
        return Err(validation.to_error().unwrap());
    }

    if dry_run {
        eprintln!("\nAll validations passed (dry-run mode)");
        return Ok(());
    }

    let conn = db::open(db_path)?;
    if let Some(schema_path) = schema_path {
        db::migrate::apply_schema(&conn, schema_path)?;
    }

    let store = db::store::Store::new(&conn);
    let resp = store.check_report_signature(&req)?;

    let out = serde_json::to_string(&resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;

    Ok(())
}
