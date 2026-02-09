use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::Result;
use std::io::{self, Write};

fn validate_get_topic_resolver_metadata(cluster_pk_hash: &str, db_path: &str, schema_path: Option<&str>) -> ValidationResult {
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

    validation.print_summary_to_stderr();
    validation
}

pub fn handle(dry_run: bool, db_path: &str, schema_path: Option<&str>, cluster_pk_hash: &str) -> Result<()> {
    let validation = validate_get_topic_resolver_metadata(cluster_pk_hash, db_path, schema_path);

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
    let resp = store.get_topic_resolver_metadata(cluster_pk_hash)?;
    
    // Output JSON to stdout (compact format for machine parsing)
    let out = serde_json::to_string(&resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;
    
    Ok(())
}

