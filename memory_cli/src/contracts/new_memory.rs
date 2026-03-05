use serde::{Deserialize, Serialize};

/// Request payload for `new-memory` command.
/// Persists report generation results to the database in a single transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemoryRequest {
    pub cluster_pk_hash: String,
    pub intent_mode: String,
    pub resolved_topic: ResolvedTopic,
    pub plan: serde_json::Value,
    pub front_matter: FrontMatter,
    pub save_output: SaveOutput,
    pub topic_resolver_config: TopicResolverConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTopic {
    pub action: String, // "create" | "merge"
    pub merge_to_topic: Option<String>,
    pub new_topic_centroid_b64: String,
    pub new_topic_weight: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontMatter {
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveOutput {
    pub report_path: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicResolverConfig {
    pub topic_resolver_config_id: String,
    pub json_payload: serde_json::Value,
}

/// Response for `new-memory` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemoryResponse {
    pub report_id: i64,
}
