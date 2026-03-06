mod fresh_paper;
mod get_best_run;
mod new_memory;
mod get_cluster_observation;
mod get_report_generation_metadata;
mod get_report_generation_supply;
mod get_topic_resolver_metadata;
mod inject_clusters_observation;
mod inject_papers_chunk;
mod validation;

use crate::cli::{Args, Command};
use anyhow::Result;

pub use validation::ValidationResult;

pub fn dispatch(args: Args) -> Result<()> {
    match args.cmd {
        Command::FreshPaper { input, no_details } => {
            fresh_paper::handle(args.dry_run, &args.db, args.schema.as_deref(), &input, no_details)
        }
        Command::GetBestRun { source, period_start, period_end, top_n, empty_cluster_observation_only } => {
            get_best_run::handle(args.dry_run, &args.db, args.schema.as_deref(), &source, &period_start, &period_end, top_n, empty_cluster_observation_only)
        }
        Command::GetClustersObservation { source, period_start, period_end } => {
            get_cluster_observation::handle(args.dry_run, &args.db, args.schema.as_deref(), &source, &period_start, &period_end)
        }
        Command::InjectClustersObservation { input } => {
            inject_clusters_observation::handle(args.dry_run, &args.db, args.schema.as_deref(), &input)
        }
        Command::GetTopicResolverMetadata { cluster_pk_hash } => {
            get_topic_resolver_metadata::handle(args.dry_run, &args.db, args.schema.as_deref(), &cluster_pk_hash)
        }
        Command::GetReportGenerationMetadata { cluster_pk_hash, add_topic_reports, add_top_papers } => {
            get_report_generation_metadata::handle(args.dry_run, &args.db, args.schema.as_deref(), &cluster_pk_hash, add_topic_reports, add_top_papers)
        }
        Command::InjectPapersChunk { input } => {
            inject_papers_chunk::handle(args.dry_run, &args.db, args.schema.as_deref(), &input)
        }
        Command::GetReportGenerationSupply { input } => {
            get_report_generation_supply::handle(args.dry_run, &args.db, args.schema.as_deref(), &input)
        }
        Command::NewMemory { input } => {
            new_memory::handle(args.dry_run, &args.db, args.schema.as_deref(), &input)
        }
    }
}
