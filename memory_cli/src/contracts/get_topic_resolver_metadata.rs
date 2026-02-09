use serde::{Deserialize, Serialize};

/// Topic centroid data matching the Python TopicInput model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicCentroid {
    /// Topic ID (from topic_id).
    pub id: String,
    /// Topic centroid as base64-encoded float32 bytes.
    pub centroid_b64: String,
    /// Topic centroid weight (must be positive).
    pub centroid_weight: f64,
}

/// Cluster metadata for topic resolver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMetadata {
    /// Cluster centroid as base64-encoded float32 bytes.
    pub centroid: String,
    /// Cluster centroid weight (cluster size).
    pub centroid_weight: f64,
}

/// Response for `get-topic-resolver-metadata` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTopicResolverMetadataResponse {
    /// List of topics with their centroid data.
    pub topics: Vec<TopicCentroid>,
    /// Cluster metadata.
    pub cluster: ClusterMetadata,
}

