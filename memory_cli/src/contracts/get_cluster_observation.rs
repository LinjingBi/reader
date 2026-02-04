use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Response for `get-clusters-observation` command.
/// Maps pk_hash to observation data.
pub type GetClusterObservationResponse = HashMap<String, ClusterObservationData>;

/// Observation data for a single cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterObservationData {
    /// Observation creation time, formatted as "%Y-%m-%d" (YYYY-MM-DD).
    pub observation_created_time: String,
    /// LLM output payload (opaque JSON).
    pub json_payload: serde_json::Value,
    /// Cluster period start date, formatted as "%Y-%m-%d" (YYYY-MM-DD).
    pub cluster_period_start: String,
    /// Cluster period end date, formatted as "%Y-%m-%d" (YYYY-MM-DD).
    pub cluster_period_end: String,
}

