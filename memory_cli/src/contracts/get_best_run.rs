use serde::{Deserialize, Serialize};

/// Response for `get-best-run` used to build the LLM enrichment prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetBestRunResponse {
    pub snapshot_id: String,
    pub cluster_run_id: String,
    pub source: String,
    pub period_start: String,
    pub period_end: String,
    pub embed_config_id: String,
    pub cluster_config_id: String,
    pub score_json: String,
    pub clusters: Vec<ClusterCard>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCard {
    pub cluster_id: String,
    pub cluster_index: i64,
    pub size: i64,
    pub cohesion: Option<f64>,
    pub papers: Vec<PaperCard>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperCard {
    pub paper_id: String,
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub url: String,
    pub rank_in_cluster: i64,
    pub sim_to_centroid: Option<f64>,
}
