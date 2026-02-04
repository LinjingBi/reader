use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Input payload for `inject-clusters-observation` command.
/// Map of pk_hash -> observation data.
/// The JSON input is a flat object: {"<pk_hash>": {...}, "<pk_hash>": {...}}
pub type InjectClustersObservationInput = HashMap<String, ClusterObservation>;

/// Observation data for a single cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterObservation {
    pub llm_config: LLMConfigInput,
    pub payload_json: serde_json::Value,
}

/// LLM config input matching the llm_config table structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfigInput {
    pub llm_config_id: String,
    pub json_payload: serde_json::Value,
}

/// Response for `inject-clusters-observation` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectClustersObservationResponse {
    pub status: String,
}

