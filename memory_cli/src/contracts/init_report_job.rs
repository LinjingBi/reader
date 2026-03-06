use serde::{Deserialize, Serialize};

/// Report job status enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportJobStatus {
    Running,
    Resuming,
    Waiting,
    Done,
    Error,
}

impl ReportJobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReportJobStatus::Running => "running",
            ReportJobStatus::Resuming => "resuming",
            ReportJobStatus::Waiting => "waiting",
            ReportJobStatus::Done => "done",
            ReportJobStatus::Error => "error",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "running" => Ok(ReportJobStatus::Running),
            "resuming" => Ok(ReportJobStatus::Resuming),
            "waiting" => Ok(ReportJobStatus::Waiting),
            "done" => Ok(ReportJobStatus::Done),
            "error" => Ok(ReportJobStatus::Error),
            _ => Err(format!("Unknown status: {}", s)),
        }
    }
}

/// Response for `init-report-job` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitReportJobResponse {
    pub next_status: String,
    pub meta: InitReportJobResponseMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitReportJobResponseMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_update_utc: Option<String>,
    pub message: String,
}
