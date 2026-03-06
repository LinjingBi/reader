use crate::contracts::{InitReportJobResponse, InitReportJobResponseMeta, ReportJobStatus};
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use std::io::{self, Write};

fn validate_init_report_job(_cluster_pk_hash: &str, db_path: &str, schema_path: Option<&str>) -> ValidationResult {
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

fn write_response(resp: &InitReportJobResponse) -> Result<()> {
    let out = serde_json::to_string(resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;
    Ok(())
}

pub fn handle(dry_run: bool, db_path: &str, schema_path: Option<&str>, cluster_pk_hash: &str) -> Result<()> {
    let validation = validate_init_report_job(cluster_pk_hash, db_path, schema_path);

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

    match store.get_report_job(cluster_pk_hash)? {
        None => {
            store.create_report_job(cluster_pk_hash, ReportJobStatus::Running, &now_str)?;
            let resp = InitReportJobResponse {
                next_status: ReportJobStatus::Running.as_str().to_string(),
                meta: InitReportJobResponseMeta {
                    report_url: None,
                    report_signature: None,
                    last_update_utc: None,
                    message: "A new job is running.".to_string(),
                },
            };
            write_response(&resp)
        }
        Some((status, report_id, created_at_str, updated_at_str)) => {
            match status {
                ReportJobStatus::Running => {
                    let avg_runtime = store.get_avg_runtime_of_latest_done_jobs()?;
                    let wait_duration = if let Some(avg) = avg_runtime {
                        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                            .with_context(|| format!("Failed to parse created_at: {}", created_at_str))?
                            .with_timezone(&Utc);
                        let expected_done = created_at + avg;
                        if expected_done < now {
                            Duration::minutes(5)
                        } else {
                            expected_done.signed_duration_since(now)
                        }
                    } else {
                        Duration::minutes(5)
                    };
                    let wait_str = format_remaining_time(wait_duration);
                    let resp = InitReportJobResponse {
                        next_status: ReportJobStatus::Waiting.as_str().to_string(),
                        meta: InitReportJobResponseMeta {
                            report_url: None,
                            report_signature: None,
                            last_update_utc: Some(updated_at_str),
                            message: format!(
                                "An existing job is already running. Estimated time remaining: {}.",
                                wait_str
                            ),
                        },
                    };
                    write_response(&resp)
                }
                ReportJobStatus::Done => {
                    let (report_url, report_signature) = match report_id {
                        Some(rid) => store
                            .get_report_url_and_signature(rid)
                            .ok()
                            .flatten()
                            .map(|(url, sig)| (Some(url), Some(sig)))
                            .unwrap_or((None, None)),
                        None => (None, None),
                    };
                    let resp = InitReportJobResponse {
                        next_status: ReportJobStatus::Done.as_str().to_string(),
                        meta: InitReportJobResponseMeta {
                            report_url,
                            report_signature,
                            last_update_utc: Some(updated_at_str),
                            message: "Report already generated by a previous run; no new run created. Returning existing report details.".to_string(),
                        },
                    };
                    write_response(&resp)
                }
                ReportJobStatus::Error => {
                    let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                        .with_context(|| format!("Failed to parse updated_at: {}", updated_at_str))?
                        .with_timezone(&Utc);
                    let elapsed = now.signed_duration_since(updated_at);
                    let five_minutes = Duration::minutes(5);

                    if elapsed < five_minutes {
                        let remaining = five_minutes - elapsed;
                        let remaining_str = format_remaining_time(remaining);
                        let resp = InitReportJobResponse {
                            next_status: ReportJobStatus::Waiting.as_str().to_string(),
                            meta: InitReportJobResponseMeta {
                                report_url: None,
                                report_signature: None,
                                last_update_utc: Some(updated_at_str),
                                message: format!(
                                    "An error happened before. Please wait {} to trigger a new job.",
                                    remaining_str
                                ),
                            },
                        };
                        write_response(&resp)
                    } else {
                        store.update_report_job_to_running(cluster_pk_hash, &now_str)?;
                        let resp = InitReportJobResponse {
                            next_status: ReportJobStatus::Resuming.as_str().to_string(),
                            meta: InitReportJobResponseMeta {
                                report_url: None,
                                report_signature: None,
                                last_update_utc: None,
                                message: "Previous error expired; job is resuming now.".to_string(),
                            },
                        };
                        write_response(&resp)
                    }
                }
                _ => unreachable!("DB only stores running, done, error"),
            }
        }
    }
}
