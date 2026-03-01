use crate::contracts::{GetReportPlannerSupplementRequest, GetReportPlannerSupplementResponse};
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{self, Read, Write};

/// Map semantic input selectors -> DB filters for paper chunks (chunk_selector.name).
/// v0: 1:1 mapping.
fn paper_selector_to_db_map() -> HashMap<&'static str, &'static str> {
    [
        ("introduction", "introduction"),
        ("related_work", "related_work"),
        ("method", "method"),
        ("experiment", "experiment"),
        ("results", "results"),
        ("discussion", "discussion"),
        ("limitations", "limitations"),
        ("conclusion", "conclusion"),
    ]
    .into_iter()
    .collect()
}

/// Map semantic input selectors -> DB filters for report fields (report table columns).
/// v0: 1:1 mapping.
fn report_selector_to_db_map() -> HashMap<&'static str, &'static str> {
    [
        ("covered_bullets", "covered_bullets"),
        ("next_targets", "next_targets"),
        ("subthreads", "subthreads"),
        ("outline", "outline"),
        ("evidence_gaps", "evidence_gaps"),
        ("sufficiency", "sufficiency"),
    ]
    .into_iter()
    .collect()
}

/// Validate and convert semantic selectors to DB filters.
/// Returns error if any selector is unknown.
fn convert_selectors(
    selectors: &[String],
    map: &HashMap<&'static str, &'static str>,
    kind: &str,
) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(selectors.len());
    for s in selectors {
        let lower = s.to_lowercase();
        match map.get(lower.as_str()) {
            Some(&db_filter) => out.push(db_filter.to_string()),
            None => {
                return Err(anyhow::anyhow!("Unknown {} selector: '{}'", kind, s));
            }
        }
    }
    Ok(out)
}

fn validate_request(req: &GetReportPlannerSupplementRequest) -> Result<()> {
    if req.paper_requests.is_empty() && req.report_requests.is_empty() {
        return Err(anyhow::anyhow!(
            "Both paper_requests and report_requests are empty; at least one request required"
        ));
    }

    let paper_map = paper_selector_to_db_map();
    let report_map = report_selector_to_db_map();

    for pr in &req.paper_requests {
        convert_selectors(&pr.selectors, &paper_map, "paper")?;
    }
    for rr in &req.report_requests {
        convert_selectors(&rr.selectors, &report_map, "report")?;
    }

    Ok(())
}

fn validate_get_report_planner_supplement(
    db_path: &str,
    schema_path: Option<&str>,
    req: &GetReportPlannerSupplementRequest,
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

    match validate_request(req) {
        Ok(()) => validation.add_pass("Validating request payload"),
        Err(e) => validation.add_fail("Validating request payload", e.to_string()),
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

    let req: GetReportPlannerSupplementRequest = serde_json::from_str(&input_str)
        .with_context(|| "invalid JSON payload for get-report-planner-supplement")?;

    let validation = validate_get_report_planner_supplement(db_path, schema_path, &req);

    if !validation.is_all_passed() {
        return Err(validation.to_error().unwrap());
    }

    if dry_run {
        eprintln!("\nAll validations passed (dry-run mode)");
        return Ok(());
    }

    let paper_map = paper_selector_to_db_map();
    let report_map = report_selector_to_db_map();

    // Convert semantic selectors to DB filters
    let paper_requests: Vec<(String, Vec<String>)> = req
        .paper_requests
        .iter()
        .map(|pr| {
            let db_filters = convert_selectors(&pr.selectors, &paper_map, "paper")
                .expect("validated in validate_request");
            (pr.paper_id.clone(), db_filters)
        })
        .collect();
    let report_requests: Vec<(i64, Vec<String>)> = req
        .report_requests
        .iter()
        .map(|rr| {
            let db_filters = convert_selectors(&rr.selectors, &report_map, "report")
                .expect("validated in validate_request");
            (rr.report_id, db_filters)
        })
        .collect();

    let conn = db::open(db_path)?;
    if let Some(ref sp) = schema_path {
        db::migrate::apply_schema(&conn, sp)?;
    }
    let store = db::store::Store::new(&conn);

    let mut paper_supplements: HashMap<String, HashMap<String, String>> = HashMap::new();
    for (paper_id, db_filters) in &paper_requests {
        match store.get_paper_chunks_supplement(paper_id, db_filters) {
            Ok(ps) => {
                let selector_map: HashMap<String, String> = ps
                    .chunks
                    .iter()
                    .map(|c| (c.selector.clone(), c.text.clone()))
                    .collect();
                paper_supplements.insert(ps.paper_id.clone(), selector_map);
            }
            Err(e) => {
                eprintln!("Warning: Failed to fetch paper {}: {}", paper_id, e);
            }
        }
    }

    let mut report_supplements: HashMap<String, HashMap<String, String>> = HashMap::new();
    for (report_id, db_filters) in &report_requests {
        match store.get_report_fields_supplement(*report_id, db_filters) {
            Ok(rs) => {
                if rs.fields.is_empty() {
                    eprintln!("Progress: Report {} not found or returned no fields", report_id);
                }
                report_supplements.insert(report_id.to_string(), rs.fields);
            }
            Err(e) => {
                eprintln!("Warning: Failed to fetch report {}: {}", report_id, e);
            }
        }
    }

    eprintln!(
        "Progress: Fetched {} paper supplements, {} report supplements",
        paper_supplements.len(),
        report_supplements.len()
    );

    let resp = GetReportPlannerSupplementResponse {
        paper_supplements,
        report_supplements,
    };

    let out = serde_json::to_string(&resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;

    Ok(())
}
