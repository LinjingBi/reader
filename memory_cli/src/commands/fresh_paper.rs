use crate::contracts::FreshPaperRequest;
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::Result;
use std::fs;
use std::path::Path;

#[derive(serde::Serialize)]
struct FreshPaperResponse {
    snapshot_id: String,
    cluster_run_id: String,
}

fn validate_fresh_paper(input_path: &str, db_path: &str, schema_path: &str) -> (ValidationResult, Option<FreshPaperRequest>) {
    let mut validation = ValidationResult::new();

    println!("Validation starts...");
    
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

    match validation::validate_schema_path(schema_path) {
        Ok(()) => validation.add_pass("Checking schema path"),
        Err(e) => validation.add_fail("Checking schema path", e.to_string()),
    }

    validation.print_summary();
    (validation, req)
}

pub fn handle(dry_run: bool, db_path: &str, schema_path: &str, input_path: &str, out_dir: Option<&str>) -> Result<()> {
    let (validation, req) = validate_fresh_paper(input_path, db_path, schema_path);

    if !validation.is_all_passed() {
        return Err(validation.to_error().unwrap());
    }

    if dry_run {
        println!("\nAll validations passed (dry-run mode)");
        return Ok(());
    }

    let req = req.expect("request should be valid at this point");
    
    let conn = db::open(db_path)?;
    db::migrate::apply_schema(&conn, schema_path)?;

    let store = db::store::Store::new(&conn);
    let (snapshot_id, cluster_run_id) = store.fresh_paper(&req)?;

    let resp = FreshPaperResponse { snapshot_id: snapshot_id.clone(), cluster_run_id: cluster_run_id.clone() };
    let out = serde_json::to_string_pretty(&resp)?;
    
    if let Some(dir) = out_dir {
        // Create directory if it doesn't exist
        fs::create_dir_all(dir)?;
        
        // Create filename: fresh_paper_{snapshot_id}_{cluster_run_id}.json
        let filename = format!("fresh_paper_{}_{}.json", snapshot_id, cluster_run_id);
        let file_path = Path::new(dir).join(&filename);
        
        fs::write(&file_path, &out)?;
        println!("Result JSON written to: {}", file_path.display());
    } else {
        println!("{out}");
    }
    
    Ok(())
}
