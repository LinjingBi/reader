use crate::contracts::{FreshPaperRequest, FreshPaperResponse};
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::Result;
use std::io::{self, Write};

fn validate_fresh_paper(input_path: &str, db_path: &str, schema_path: Option<&str>) -> (ValidationResult, Option<FreshPaperRequest>) {
    let mut validation = ValidationResult::new();

    eprintln!("Validation starts...");
    
    match validation::validate_input_file(input_path) {
        Ok(()) => validation.add_pass("Checking input file exists"),
        Err(e) => validation.add_fail("Checking input file exists", e.to_string()),
    }

    let req: Option<FreshPaperRequest> = match validation::validate_json_format::<FreshPaperRequest>(input_path) {
        Ok(r) => {
            validation.add_pass("Checking JSON format");
            Some(r)
        }
        Err(e) => {
            validation.add_fail("Checking JSON format", format!("{:#}", e));
            None
        }
    };

    // Validate date formats if we have a valid request
    if let Some(ref req) = req {
        let mut date_validation_passed = true;
        
        match validation::validate_date_format(&req.period_start, "period_start") {
            Ok(()) => {}
            Err(e) => {
                validation.add_fail("Checking date format (period_start)", e.to_string());
                date_validation_passed = false;
            }
        }
        
        match validation::validate_date_format(&req.period_end, "period_end") {
            Ok(()) => {}
            Err(e) => {
                validation.add_fail("Checking date format (period_end)", e.to_string());
                date_validation_passed = false;
            }
        }
        
        if date_validation_passed {
            validation.add_pass("Checking date formats");
        }
    }

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
    (validation, req)
}

pub fn handle(dry_run: bool, db_path: &str, schema_path: Option<&str>, input_path: &str) -> Result<()> {
    let (validation, req) = validate_fresh_paper(input_path, db_path, schema_path);

    if !validation.is_all_passed() {
        return Err(validation.to_error().unwrap());
    }

    if dry_run {
        eprintln!("\nAll validations passed (dry-run mode)");
        return Ok(());
    }

    let req = req.expect("request should be valid at this point");
    
    let conn = db::open(db_path)?;
    if let Some(schema_path) = schema_path {
        db::migrate::apply_schema(&conn, schema_path)?;
    }

    let store = db::store::Store::new(&conn);
    store.fresh_paper(&req)?;
    
    eprintln!("Successfully ingested papers and clusters");
    
    // Output JSON response to stdout
    let resp = FreshPaperResponse {
        success: true,
        source: req.source.clone(),
        period_start: req.period_start.clone(),
        period_end: req.period_end.clone(),
        papers_count: req.papers.len(),
        clusters_count: req.clusters.len(),
    };
    
    let out = serde_json::to_string(&resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;
    
    Ok(())
}
