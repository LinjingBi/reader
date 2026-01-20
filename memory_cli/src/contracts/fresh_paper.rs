use serde::{Deserialize, Serialize};

/// Reader -> Memo CLI payload for Step 1–2:
/// - persist source snapshot + papers
/// - persist best clustering run + clusters + membership ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshPaperRequest {
    pub source: String,
    pub period_start: String,
    pub period_end: String,
    pub snapshot_id: Option<String>,
    pub raw_json: Option<String>,
    pub embed_config: EmbedConfig,
    pub cluster_config: ClusterConfig,
    pub papers: Vec<PaperInput>,
    pub clusters: Vec<ClusterInput>,
    pub role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedConfig {
    pub embed_config_id: String,
    pub json_payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub cluster_config_id: String,
    pub json_payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperInput {
    pub paper_id: String,
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub url: String,
    pub published_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInput {
    pub cluster_index: i64,
    pub size: i64,
    pub centroid_b64: Option<String>,
    pub cohesion: Option<f64>,
    pub members: Vec<ClusterMemberInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMemberInput {
    pub paper_id: String,
    pub rank_in_cluster: i64,
    pub sim_to_centroid: Option<f64>,
}
