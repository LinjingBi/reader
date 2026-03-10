use serde::{Deserialize, Serialize};

/// Metadata for a report when found.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetReportMeta {
    pub report_id: i64,
    pub report_url: String,
    pub intent_mode: String,
}

/// Response for `get-report` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetReportResponse {
    pub status: String, // "ok" | "not_found" | "error"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<GetReportMeta>,
}
