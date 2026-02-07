use crate::contracts::{StartReportJobResponse, ReportJobStatus};
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use std::io::{self, Write};

fn validate_start_report_job(cluster_pk_hash: &str, db_path: &str, schema_path: Option<&str>) -> ValidationResult {
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

    // Validate cluster exists in database
    match db::open(db_path) {
        Ok(conn) => {
            if let Some(schema_path) = schema_path {
                if let Err(e) = db::migrate::apply_schema(&conn, schema_path) {
                    validation.add_fail("Applying schema", e.to_string());
                    validation.print_summary_to_stderr();
                    return validation;
                }
            }
            let store = db::store::Store::new(&conn);
            match store.check_cluster_exists(cluster_pk_hash) {
                Ok(true) => validation.add_pass("Checking cluster exists"),
                Ok(false) => validation.add_fail("Checking cluster exists", format!("Cluster with pk_hash '{}' does not exist", cluster_pk_hash)),
                Err(e) => validation.add_fail("Checking cluster exists", e.to_string()),
            }
        }
        Err(e) => validation.add_fail("Opening database", e.to_string()),
    }

    validation.print_summary_to_stderr();
    validation
}

fn format_remaining_time(remaining: Duration) -> String {
    let total_seconds = remaining.num_seconds();
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    
    if minutes > 0 && seconds > 0 {
        format!("{} minutes {} seconds", minutes, seconds)
    } else if minutes > 0 {
        format!("{} minutes", minutes)
    } else {
        format!("{} seconds", seconds)
    }
}

pub fn handle(dry_run: bool, db_path: &str, schema_path: Option<&str>, cluster_pk_hash: &str) -> Result<()> {
    let validation = validate_start_report_job(cluster_pk_hash, db_path, schema_path);

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

    let now = Utc::now();
    let now_str = now.to_rfc3339();

    // Check for existing job
    match store.get_report_job(cluster_pk_hash)? {
        None => {
            // No record: create new job
            store.create_report_job(cluster_pk_hash, ReportJobStatus::Running, &now_str)?;
            let resp = StartReportJobResponse {
                status: ReportJobStatus::Running.as_str().to_string(),
                new_job: true,
                report_id: None,
                message: "a new job is running.".to_string(),
            };
            let out = serde_json::to_string(&resp)?;
            io::stdout().write_all(out.as_bytes())?;
            io::stdout().flush()?;
            Ok(())
        }
        Some((status, report_id, updated_at_str)) => {
            match status {
                ReportJobStatus::Running => {
                    // Status is running: return existing job info
                    let resp = StartReportJobResponse {
                        status: ReportJobStatus::Running.as_str().to_string(),
                        new_job: false,
                        report_id: None,
                        message: format!("an existing job for cluster-id {} is already running", cluster_pk_hash),
                    };
                    let out = serde_json::to_string(&resp)?;
                    io::stdout().write_all(out.as_bytes())?;
                    io::stdout().flush()?;
                    Ok(())
                }
                ReportJobStatus::Done => {
                    // Status is done: return report_id
                    let resp = StartReportJobResponse {
                        status: ReportJobStatus::Done.as_str().to_string(),
                        new_job: false,
                        report_id: report_id.clone(),
                        message: "a report is already generated.".to_string(),
                    };
                    let out = serde_json::to_string(&resp)?;
                    io::stdout().write_all(out.as_bytes())?;
                    io::stdout().flush()?;
                    Ok(())
                }
                ReportJobStatus::Error => {
                    // Status is error: check if within 5 minutes
                    let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                        .with_context(|| format!("Failed to parse updated_at timestamp: {}", updated_at_str))?
                        .with_timezone(&Utc);
                    
                    let elapsed = now.signed_duration_since(updated_at);
                    let five_minutes = Duration::minutes(5);
                    
                    if elapsed < five_minutes {
                        // Within 5 minutes: calculate remaining time
                        let remaining = five_minutes - elapsed;
                        let remaining_str = format_remaining_time(remaining);
                        let resp = StartReportJobResponse {
                            status: ReportJobStatus::Error.as_str().to_string(),
                            new_job: false,
                            report_id: None,
                            message: format!("an error happened before, please wait for {} to trigger a new one.", remaining_str),
                        };
                        let out = serde_json::to_string(&resp)?;
                        io::stdout().write_all(out.as_bytes())?;
                        io::stdout().flush()?;
                        Ok(())
                    } else {
                        // Beyond 5 minutes: update to running
                        store.update_report_job_to_running(cluster_pk_hash, &now_str)?;
                        let resp = StartReportJobResponse {
                            status: ReportJobStatus::Running.as_str().to_string(),
                            new_job: true,
                            report_id: None,
                            message: "errored expired, job is running now.".to_string(),
                        };
                        let out = serde_json::to_string(&resp)?;
                        io::stdout().write_all(out.as_bytes())?;
                        io::stdout().flush()?;
                        Ok(())
                    }
                }
            }
        }
    }
}

