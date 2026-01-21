use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::Result;

fn validate_get_best_run(period_start: &str, period_end: &str, db_path: &str, schema_path: &str) -> ValidationResult {
    let mut validation = ValidationResult::new();

    println!("Validation starts...");
    
    let mut date_validation_passed = true;
    
    match validation::validate_date_format(period_start, "period_start") {
        Ok(()) => {}
        Err(e) => {
            validation.add_fail("Checking date format (period_start)", e.to_string());
            date_validation_passed = false;
        }
    }
    
    match validation::validate_date_format(period_end, "period_end") {
        Ok(()) => {}
        Err(e) => {
            validation.add_fail("Checking date format (period_end)", e.to_string());
            date_validation_passed = false;
        }
    }
    
    if date_validation_passed {
        validation.add_pass("Checking date formats");
    }

    match validation::validate_db_path(db_path) {
        Ok(()) => validation.add_pass("Checking database path"),
        Err(e) => validation.add_fail("Checking database path", e.to_string()),
    }

    match validation::validate_schema_path(schema_path) {
        Ok(()) => validation.add_pass("Checking schema path"),
        Err(e) => validation.add_fail("Checking schema path", e.to_string()),
    }

    validation.print_summary();
    validation
}

pub fn handle(dry_run: bool, db_path: &str, schema_path: &str, source: &str, period_start: &str, period_end: &str, top_n: usize) -> Result<()> {
    let validation = validate_get_best_run(period_start, period_end, db_path, schema_path);

    if !validation.is_all_passed() {
        return Err(validation.to_error().unwrap());
    }

    if dry_run {
        println!("\nAll validations passed (dry-run mode)");
        return Ok(());
    }

    let conn = db::open(db_path)?;
    db::migrate::apply_schema(&conn, schema_path)?;

    let store = db::store::Store::new(&conn);
    let resp = store.get_best_run(source, period_start, period_end, top_n)?;
    let out = serde_json::to_string_pretty(&resp)?;
    println!("{out}");
    Ok(())
}
