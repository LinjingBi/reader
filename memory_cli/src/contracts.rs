pub mod fresh_paper;
pub mod get_best_run;
pub mod new_memory;
pub mod get_cluster_observation;
pub mod get_report;
pub mod get_report_generation_metadata;
pub mod get_report_generation_supply;
pub mod get_topic_resolver_metadata;
pub mod inject_clusters_observation;
pub mod inject_papers_chunk;
pub mod check_report_signature;

pub use fresh_paper::{
    ClusterConfig,
    ClusterInput,
    ClusterMemberInput,
    EmbedConfig,
    FreshPaperRequest,
    FreshPaperResponse,
    FreshPaperResponseWithDetails,
    FreshPaperMeta,
    PaperInput,
    PaperOutput,
};
pub use get_best_run::{ClusterCard, GetBestRunResponse, PaperCard};
pub use get_cluster_observation::{ClusterObservationData, GetClusterObservationResponse};
pub use inject_clusters_observation::{
    ClusterObservation,
    InjectClustersObservationInput,
    InjectClustersObservationResponse,
    LLMConfigInput,
};
pub use get_topic_resolver_metadata::{TopicCentroid, ClusterMetadata, GetTopicResolverMetadataResponse};
pub use get_report_generation_metadata::{GetReportGenerationMetadataResponse, NewObservation, TopPaper, HistoryReport};
pub use inject_papers_chunk::{
    InjectPapersChunkRequest,
    InjectPapersChunkResponse,
    InjectPapersChunkMeta,
    LibConfig,
    PaperChunkData,
    ChunkEntry,
};
pub use new_memory::{NewMemoryRequest, NewMemoryResponse};
pub use get_report_generation_supply::{
    GetReportGenerationSupplyRequest,
    GetReportGenerationSupplyResponse,
    PaperSupplementRequest,
    ReportSupplementRequest,
    PaperSupplement,
    PaperChunk,
    ReportSupplement,
};
pub use get_report::{GetReportResponse, GetReportMeta};
pub use check_report_signature::{CheckReportSignatureRequest, CheckReportSignatureResponse};
