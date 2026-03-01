use serde::{Deserialize, Serialize};

/// Input payload for `get-report-planner-supplement` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetReportPlannerSupplementRequest {
    /// Paper lookup requests.
    #[serde(default)]
    pub paper_requests: Vec<PaperSupplementRequest>,
    /// Report field lookup requests.
    #[serde(default)]
    pub report_requests: Vec<ReportSupplementRequest>,
}

/// Per-paper supplement request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperSupplementRequest {
    pub paper_id: String,
    pub selectors: Vec<String>,
}

/// Per-report supplement request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSupplementRequest {
    pub report_id: i64,
    pub selectors: Vec<String>,
}

/// Response for `get-report-planner-supplement` command.
/// Matches phase2_supplement structure: paper_id/report_id -> selector -> value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetReportPlannerSupplementResponse {
    #[serde(default)]
    pub paper_supplements: std::collections::HashMap<String, std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub report_supplements: std::collections::HashMap<String, std::collections::HashMap<String, String>>,
}

/// Paper supplement: chunks by selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperSupplement {
    pub paper_id: String,
    pub chunks: Vec<PaperChunk>,
}

/// Single paper chunk (selector + text).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperChunk {
    pub selector: String,
    pub text: String,
}

/// Report supplement: fields by name (JSON string values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSupplement {
    pub report_id: i64,
    pub fields: std::collections::HashMap<String, String>,
}
