use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// New observation data from cluster observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewObservation {
    /// Cluster observation title (name).
    pub name: String,
    /// Cluster observation summary.
    pub summary: String,
    /// Keywords from cluster observation.
    pub keywords: Vec<String>,
    /// Keywords from top ≤5 papers, keyed by paper_id.
    pub key_paper_keywords: HashMap<String, Vec<String>>,
}

/// Top paper data for the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopPaper {
    /// Paper ID.
    pub paper_id: String,
    /// Paper title.
    pub title: String,
    /// Paper summary.
    pub summary: String,
    /// Paper keywords.
    pub keywords: Vec<String>,
    /// Rank in cluster (0 = most representative).
    pub rank_in_cluster: i64,
    /// Similarity to cluster centroid.
    pub sim_to_centroid: Option<f64>,
}

/// History report data for a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryReport {
    /// Report ID.
    pub report_id: i64,
    /// Report title.
    pub title: String,
    /// Report summary.
    pub summary: String,
    /// Report keywords as JSON.
    pub keywords_json: Value,
    /// Report depth context as JSON.
    pub depth_context_json: Value,
}

/// Response for `get-report-planner-metadata` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetReportPlannerMetadataResponse {
    /// New observation data from cluster.
    pub new_observation: NewObservation,
    /// Optional Top-K papers (K≤5) for the cluster.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_papers_from_new_observation: Option<Vec<TopPaper>>,
    /// Optional top ≤3 reports for the specified topic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_reports: Option<Vec<HistoryReport>>,
}

