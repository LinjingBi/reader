mod fresh_paper;
mod get_best_run;
mod get_cluster_observation;
mod get_topic_resolver_metadata;
mod inject_clusters_observation;
mod start_report_job;
mod validation;

use crate::cli::{Args, Command};
use anyhow::Result;

pub use validation::ValidationResult;

pub fn dispatch(args: Args) -> Result<()> {
    match args.cmd {
        Command::FreshPaper { input } => {
            fresh_paper::handle(args.dry_run, &args.db, args.schema.as_deref(), &input)
        }
        Command::GetBestRun { source, period_start, period_end, top_n } => {
            get_best_run::handle(args.dry_run, &args.db, args.schema.as_deref(), &source, &period_start, &period_end, top_n)
        }
        Command::GetClustersObservation { source, period_start, period_end } => {
            get_cluster_observation::handle(args.dry_run, &args.db, args.schema.as_deref(), &source, &period_start, &period_end)
        }
        Command::InjectClustersObservation { input } => {
            inject_clusters_observation::handle(args.dry_run, &args.db, args.schema.as_deref(), &input)
        }
        Command::StartReportJob { cluster_pk_hash } => {
            start_report_job::handle(args.dry_run, &args.db, args.schema.as_deref(), &cluster_pk_hash)
        }
        Command::GetTopicResolverMetadata { cluster_pk_hash } => {
            get_topic_resolver_metadata::handle(args.dry_run, &args.db, args.schema.as_deref(), &cluster_pk_hash)
        }
    }
}
