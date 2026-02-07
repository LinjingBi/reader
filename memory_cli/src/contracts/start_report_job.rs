use serde::{Deserialize, Serialize};

/// Report job status enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportJobStatus {
    Running,
    Done,
    Error,
}

impl ReportJobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReportJobStatus::Running => "running",
            ReportJobStatus::Done => "done",
            ReportJobStatus::Error => "error",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "running" => Ok(ReportJobStatus::Running),
            "done" => Ok(ReportJobStatus::Done),
            "error" => Ok(ReportJobStatus::Error),
            _ => Err(format!("Unknown status: {}", s)),
        }
    }
}

/// Response for `start-report-job` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartReportJobResponse {
    pub status: String,
    pub new_job: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_id: Option<String>,
    pub message: String,
}

