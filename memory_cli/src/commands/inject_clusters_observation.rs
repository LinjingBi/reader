use crate::contracts::{InjectClustersObservationInput, InjectClustersObservationResponse};
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::{Context, Result};
use std::io::{self, Write};

fn validate_inject_clusters_observation(input_path: &str, db_path: &str, schema_path: Option<&str>) -> (ValidationResult, Option<InjectClustersObservationInput>) {
    let mut validation = ValidationResult::new();

    eprintln!("Validation starts...");
    
    match validation::validate_input_file(input_path) {
        Ok(()) => validation.add_pass("Checking input file exists"),
        Err(e) => validation.add_fail("Checking input file exists", e.to_string()),
    }

    let req: Option<InjectClustersObservationInput> = match validation::validate_json_format::<InjectClustersObservationInput>(input_path) {
        Ok(r) => {
            validation.add_pass("Checking JSON format");
            Some(r)
        }
        Err(e) => {
            validation.add_fail("Checking JSON format", format!("{:#}", e));
            None
        }
    };

    // Validate required fields for each observation
    if let Some(ref req) = req {
        let mut all_valid = true;
        for (pk_hash, observation) in req {
            if observation.llm_config.llm_config_id.is_empty() {
                validation.add_fail(
                    &format!("Checking observation for pk_hash={}", pk_hash),
                    "llm_config.llm_config_id is required".to_string(),
                );
                all_valid = false;
            }
            // payload_json is always present if deserialization succeeded (it's serde_json::Value)
        }
        if all_valid {
            validation.add_pass("Checking required fields");
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
    let (validation, req) = validate_inject_clusters_observation(input_path, db_path, schema_path);

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
    store.upsert_cluster_observation(&req)
        .context("failed to upsert cluster observations")?;
    
    eprintln!("Successfully injected cluster observations");
    
    // Output JSON response to stdout
    let resp = InjectClustersObservationResponse {
        status: "ok".to_string(),
    };
    
    let out = serde_json::to_string(&resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;
    
    Ok(())
}

