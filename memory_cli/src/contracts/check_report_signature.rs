use serde::{Deserialize, Serialize};

/// Request for `check-report-signature` command.
/// At least one of report_id or cluster_pk_hash required. If both present, use report_id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckReportSignatureRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_pk_hash: Option<String>,
    pub signature: String,
}

/// Response for `check-report-signature` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckReportSignatureResponse {
    pub status: String, // "match" | "not_match" | "error"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}
