pub mod fresh_paper;
pub mod get_best_run;

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
