pub mod fresh_paper;
pub mod get_best_run;
pub mod get_cluster_observation;
pub mod inject_clusters_observation;

pub use fresh_paper::{
    ClusterConfig,
    ClusterInput,
    ClusterMemberInput,
    EmbedConfig,
    FreshPaperRequest,
    FreshPaperResponse,
    PaperInput,
};
pub use get_best_run::{ClusterCard, GetBestRunResponse, PaperCard};
pub use get_cluster_observation::{ClusterObservationData, GetClusterObservationResponse};
pub use inject_clusters_observation::{
    ClusterObservation,
    InjectClustersObservationInput,
    InjectClustersObservationResponse,
    LLMConfigInput,
};
