use anyhow::{Context, Result};
use chrono::NaiveDate;
use std::fs;
use std::io::Read;
use std::path::Path;

/// Tracks validation results for all validation steps.
pub struct ValidationResult {
    passed: Vec<String>,
    failed: Vec<(String, String)>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            passed: Vec::new(),
            failed: Vec::new(),
        }
    }

    pub fn add_pass(&mut self, step: &str) {
        self.passed.push(step.to_string());
    }

    pub fn add_fail(&mut self, step: &str, error: String) {
        self.failed.push((step.to_string(), error));
    }

    pub fn is_all_passed(&self) -> bool {
        self.failed.is_empty()
    }

    pub fn print_summary(&self) {
        println!("\nValidation Summary:");
        for step in &self.passed {
            println!("  ✓ {}... PASSED", step);
        }
        for (step, error) in &self.failed {
            println!("  ✗ {}... FAILED: {}", step, error);
        }
    }

    pub fn to_error(&self) -> Option<anyhow::Error> {
        if self.failed.is_empty() {
            None
        } else {
            let count = self.failed.len();
            let msg = format!("{} validation(s) failed", count);
            Some(anyhow::anyhow!(msg))
        }
    }
}

/// Validate date format (YYYY-MM-DD).
pub fn validate_date_format(date_str: &str, field_name: &str) -> Result<()> {
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .with_context(|| format!("Invalid date format for {}: expected YYYY-MM-DD, got {}", field_name, date_str))?;
    Ok(())
}

/// Validate DB path: file exists.
pub fn validate_db_path(db_path: &str) -> Result<()> {
    let path = Path::new(db_path);
    if !path.exists() {
        return Err(anyhow::anyhow!("Database file does not exist: {}", db_path));
    }
    Ok(())
}

/// Validate schema path: file exists.
pub fn validate_schema_path(schema_path: &str) -> Result<()> {
    let path = Path::new(schema_path);
    if !path.exists() {
        return Err(anyhow::anyhow!("Schema file does not exist: {}", schema_path));
    }
    Ok(())
}

/// Validate input file exists (or stdin is available if path is "-").
pub fn validate_input_file(input_path: &str) -> Result<()> {
    if input_path == "-" {
        // stdin case - just verify stdin is available (can't really test without consuming)
        Ok(())
    } else {
        let path = Path::new(input_path);
        if !path.exists() {
            return Err(anyhow::anyhow!("Input file does not exist: {}", input_path));
        }
        let metadata = fs::metadata(path)
            .with_context(|| format!("Cannot access input file: {}", input_path))?;
        if !metadata.is_file() {
            return Err(anyhow::anyhow!("Input path is not a file: {}", input_path));
        }
        Ok(())
    }
}

/// Validate JSON format and deserialize.
pub fn validate_json_format<T: serde::de::DeserializeOwned>(input_path: &str) -> Result<T> {
    let mut s = String::new();
    if input_path == "-" {
        std::io::stdin().read_to_string(&mut s)?;
    } else {
        s = std::fs::read_to_string(input_path)
            .with_context(|| format!("failed to read input JSON: {}", input_path))?;
    }
    serde_json::from_str(&s)
        .with_context(|| format!("invalid JSON payload in {}", input_path))
}

