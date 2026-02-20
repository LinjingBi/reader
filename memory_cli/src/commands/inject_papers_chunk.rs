use crate::contracts::{InjectPapersChunkRequest, InjectPapersChunkResponse, InjectPapersChunkMeta};
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::Result;
use std::io::{self, Write};

fn validate_inject_papers_chunk(input_path: &str, db_path: &str, schema_path: Option<&str>) -> (ValidationResult, Option<InjectPapersChunkRequest>) {
    let mut validation = ValidationResult::new();

    eprintln!("Validation starts...");
    
    match validation::validate_input_file(input_path) {
        Ok(()) => validation.add_pass("Checking input file exists"),
        Err(e) => validation.add_fail("Checking input file exists", e.to_string()),
    }

    let req: Option<InjectPapersChunkRequest> = match validation::validate_json_format::<InjectPapersChunkRequest>(input_path) {
        Ok(r) => {
            validation.add_pass("Checking JSON format");
            Some(r)
        }
        Err(e) => {
            validation.add_fail("Checking JSON format", format!("{:#}", e));
            None
        }
    };

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
    let (validation, req) = validate_inject_papers_chunk(input_path, db_path, schema_path);

    if !validation.is_all_passed() {
        return Err(validation.to_error().unwrap());
    }

    if dry_run {
        eprintln!("\nAll validations passed (dry-run mode)");
        return Ok(());
    }

    let req = req.expect("request should be valid at this point");
    
    // Calculate counts from input request (like fresh_paper cmd)
    let total_papers_count = req.papers.len();
    let total_chunks_count: usize = req.papers.iter()
        .map(|paper| paper.chunks.len())
        .sum();
    
    eprintln!("Progress: Starting inject transaction for {} papers...", total_papers_count);
    
    let conn = db::open(db_path)?;
    if let Some(schema_path) = schema_path {
        db::migrate::apply_schema(&conn, schema_path)?;
    }

    let store = db::store::Store::new(&conn);
    store.inject_papers_chunk(&req)?;
    
    eprintln!("Progress: Successfully injected {} chunks from {} papers", total_chunks_count, total_papers_count);
    
    // Build response
    let resp = InjectPapersChunkResponse {
        success: true,
        meta: InjectPapersChunkMeta {
            total_papers_count,
            total_chunks_count,
        },
    };
    
    let out = serde_json::to_string(&resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;
    
    Ok(())
}

