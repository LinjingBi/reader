use crate::contracts::{GetReportGenerationSupplyRequest, GetReportGenerationSupplyResponse};
use crate::commands::validation::{self, ValidationResult};
use crate::db;
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
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
/// Conversion is case insensitive (input selectors are lowercased before lookup).
/// Returns error if any selector is unknown; collects all unknowns and reports them together.
fn convert_selectors(
    selectors: &[String],
    map: &HashMap<&'static str, &'static str>,
    kind: &str,
) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(selectors.len());
    let mut unknown: Vec<String> = Vec::new();
    for s in selectors {
        let lower = s.to_lowercase();
        match map.get(lower.as_str()) {
            Some(&db_filter) => out.push(db_filter.to_string()),
            None => unknown.push(s.clone()),
        }
    }
    if unknown.is_empty() {
        Ok(out)
    } else {
        Err(anyhow::anyhow!(
            "Unknown {} selector(s): {}",
            kind,
            unknown.join(", ")
        ))
    }
}

/// Merge requests by id and convert selectors to DB filters.
/// - Merges by paper_id/report_id with case-insensitive selector deduplication.
/// - Converts semantic selectors to DB filters (case-insensitive lookup).
/// - Logs before/after deduplication stats.
fn to_db_selectors(
    req: &GetReportGenerationSupplyRequest,
) -> Result<(Vec<(String, Vec<String>)>, Vec<(i64, Vec<String>)>)> {
    let paper_map = paper_selector_to_db_map();
    let report_map = report_selector_to_db_map();

    let before_paper_reqs = req.paper_requests.len();
    let before_report_reqs = req.report_requests.len();
    let before_paper_selectors: usize = req.paper_requests.iter().map(|pr| pr.selectors.len()).sum();
    let before_report_selectors: usize = req.report_requests.iter().map(|rr| rr.selectors.len()).sum();

    // Merge by id; selectors deduplicated case-insensitively (lowercase as key)
    let mut paper_merge: HashMap<String, HashSet<String>> = HashMap::new();
    for pr in &req.paper_requests {
        let entry = paper_merge.entry(pr.paper_id.clone()).or_default();
        for s in &pr.selectors {
            entry.insert(s.to_lowercase());
        }
    }
    let mut report_merge: HashMap<i64, HashSet<String>> = HashMap::new();
    for rr in &req.report_requests {
        let entry = report_merge.entry(rr.report_id).or_default();
        for s in &rr.selectors {
            entry.insert(s.to_lowercase());
        }
    }

    // Convert to DB filters
    let mut paper_requests: Vec<(String, Vec<String>)> = Vec::new();
    for (paper_id, sels) in paper_merge {
        let mut selectors: Vec<String> = sels.into_iter().collect();
        selectors.sort();
        let db_filters = convert_selectors(&selectors, &paper_map, "paper")?;
        paper_requests.push((paper_id, db_filters));
    }
    let mut report_requests: Vec<(i64, Vec<String>)> = Vec::new();
    for (report_id, sels) in report_merge {
        let mut selectors: Vec<String> = sels.into_iter().collect();
        selectors.sort();
        let db_filters = convert_selectors(&selectors, &report_map, "report")?;
        report_requests.push((report_id, db_filters));
    }

    let after_paper_reqs = paper_requests.len();
    let after_report_reqs = report_requests.len();
    let after_paper_selectors: usize = paper_requests.iter().map(|(_, f)| f.len()).sum();
    let after_report_selectors: usize = report_requests.iter().map(|(_, f)| f.len()).sum();
    eprintln!(
        "Supply deduplication: paper_requests {}->{}, report_requests {}->{}, \
         paper_selectors {}->{}, report_selectors {}->{}",
        before_paper_reqs,
        after_paper_reqs,
        before_report_reqs,
        after_report_reqs,
        before_paper_selectors,
        after_paper_selectors,
        before_report_selectors,
        after_report_selectors,
    );

    Ok((paper_requests, report_requests))
}

fn validate_request(req: &GetReportGenerationSupplyRequest) -> Result<()> {
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

fn validate_get_report_generation_supply(
    db_path: &str,
    schema_path: Option<&str>,
    req: &GetReportGenerationSupplyRequest,
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

    let req: GetReportGenerationSupplyRequest = serde_json::from_str(&input_str)
        .with_context(|| "invalid JSON payload for get-report-generation-supply")?;

    let validation = validate_get_report_generation_supply(db_path, schema_path, &req);

    if !validation.is_all_passed() {
        return Err(validation.to_error().unwrap());
    }

    if dry_run {
        eprintln!("\nAll validations passed (dry-run mode)");
        return Ok(());
    }

    let (paper_requests, report_requests) = to_db_selectors(&req)?;

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

    let resp = GetReportGenerationSupplyResponse {
        paper_supplements,
        report_supplements,
    };

    let out = serde_json::to_string(&resp)?;
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()?;

    Ok(())
}
