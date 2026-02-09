use clap::{Parser, Subcommand};

/// Memo CLI: safe, narrow command surface over an SQLite memory DB.
#[derive(Parser, Debug)]
#[command(
    name = "memo",
    version,
    about,
    long_about = "Memo CLI: safe, narrow command surface over an SQLite memory DB.

EXAMPLES:
  memo fresh-paper --input papers.json
  memo get-best-run --source hf_monthly --period-start 2024-01-01 --period-end 2024-01-31"
)]
pub struct Args {
    /// Path to SQLite DB file.
    #[arg(long, env = "MEMO_DB", default_value = "memo.sqlite")]
    pub db: String,

    /// Optional Path to schema SQL (used for bootstrap/migrations).
    #[arg(long, env = "MEMO_SCHEMA")]
    pub schema: Option<String>,

    /// Dry-run mode: validate inputs without performing DB operations. false by default.
    #[arg(long)]
    pub dry_run: bool,

    #[command(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Atomic monthly ingest + best clustering write.
    #[command(
        long_about = "Atomic monthly ingest + best clustering write.

EXAMPLES:
  memo fresh-paper --input papers.json
  cat papers.json | memo fresh-paper --input -"
    )]
    FreshPaper {
        /// JSON payload path. Use '-' to read from stdin.
        #[arg(long)]
        input: String,
    },

    /// Read the selected best clustering run for a period (for LLM enrichment prompt).
    #[command(
        long_about = "Read the selected best clustering run for a period (for LLM enrichment prompt).

EXAMPLES:
  memo get-best-run --source hf_monthly --period-start 2024-01-01 --period-end 2024-01-31
  memo get-best-run --source hf_monthly --period-start 2024-01-01 --period-end 2024-01-31 --top-n 5"
    )]
    GetBestRun {
        /// Snapshot source, e.g., 'hf_monthly'.
        #[arg(long)]
        source: String,
        /// Period start date (YYYY-MM-DD).
        #[arg(long)]
        period_start: String,
        /// Period end date (YYYY-MM-DD).
        #[arg(long)]
        period_end: String,
        /// Max papers per cluster to include.
        #[arg(long, default_value_t = 10)]
        top_n: usize,
    },

    /// Get cluster observations for clusters within a period range.
    #[command(
        long_about = "Get cluster observations for clusters within a period range.

EXAMPLES:
  memo get-clusters-observation --source hf_monthly --period-start 2024-01-01 --period-end 2024-01-31"
    )]
    GetClustersObservation {
        /// Snapshot source, e.g., 'hf_monthly'.
        #[arg(long)]
        source: String,
        /// Period start date (YYYY-MM-DD) - start of range filter.
        #[arg(long)]
        period_start: String,
        /// Period end date (YYYY-MM-DD) - end of range filter.
        #[arg(long)]
        period_end: String,
    },

    /// Write LLM enrichment results back into DB as cluster-attached semantic records.
    #[command(
        long_about = "Write LLM enrichment results back into DB as cluster-attached semantic records.

EXAMPLES:
  memo inject-clusters-observation --input observations.json
  cat observations.json | memo inject-clusters-observation --input -"
    )]
    InjectClustersObservation {
        /// JSON payload path. Use '-' to read from stdin.
        #[arg(long)]
        input: String,
    },

    /// Start a report generation job for a cluster.
    #[command(
        long_about = "Start a report generation job for a cluster.

EXAMPLES:
  memo start-report-job --cluster-pk-hash abc123def456"
    )]
    StartReportJob {
        /// Cluster pk_hash (primary key hash from cluster table).
        #[arg(long)]
        cluster_pk_hash: String,
    },

    /// Get topic resolver metadata (topics and cluster data).
    #[command(
        long_about = "Get topic resolver metadata including all topics with their centroid data and cluster metadata.

EXAMPLES:
  memo get-topic-resolver-metadata --cluster-pk-hash abc123def456
  memo --db memo.sqlite get-topic-resolver-metadata --cluster-pk-hash abc123def456"
    )]
    GetTopicResolverMetadata {
        /// Cluster pk_hash (primary key hash from cluster table).
        #[arg(long)]
        cluster_pk_hash: String,
    },
}

impl Args {
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }
}