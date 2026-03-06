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
        /// Skip querying paper details (faster, smaller output).
        #[arg(long)]
        no_details: bool,
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
        /// Max papers per cluster to include. If omitted, returns all papers.
        #[arg(long)]
        top_n: Option<usize>,
        /// Only return clusters that have no cluster_observation (checking via pk_hash).
        /// When false (default), returns all clusters matching the period and source.
        #[arg(long)]
        empty_cluster_observation_only: bool,
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

    /// Inlcude a cluster's observation and its top papers (≤5) for the given cluster pk_hash, and topic details for the given topic id.
    #[command(
        long_about = "Report generation metadata includes a cluster's observation and its top papers (≤5)(optional) for the given cluster pk_hash, and topic details for the given topic id.

EXAMPLES:
  memo get-report-generation-metadata --cluster-pk-hash abc123def456
  memo get-report-generation-metadata --cluster-pk-hash abc123def456 --add-top-papers
  memo get-report-generation-metadata --cluster-pk-hash abc123def456 --add-topic-reports 42 --add-top-papers"
    )]
    GetReportGenerationMetadata {
        /// Cluster pk_hash (primary key hash from cluster table).
        #[arg(long)]
        cluster_pk_hash: String,
        /// Optional: Include top ≤3 reports for a topic.
        #[arg(long)]
        add_topic_reports: Option<i64>,
        /// Include top-K papers details (≤5) for the cluster.
        #[arg(long)]
        add_top_papers: bool,
    },

    /// Inject paper chunks into the database.
    #[command(
        long_about = "Inject paper chunks into the database from Python scoring pipeline output.

EXAMPLES:
  memo inject-papers-chunk --input chunks.json
  cat chunks.json | memo inject-papers-chunk --input -"
    )]
    InjectPapersChunk {
        /// JSON payload path. Use '-' to read from stdin.
        #[arg(long)]
        input: String,
    },

    /// Persist report generation results to the database.
    #[command(
        long_about = "Persist report generation results (topic, report, links) in a single transaction.

EXAMPLES:
  memo new-memory --input payload.json
  cat payload.json | memo new-memory --input -"
    )]
    NewMemory {
        /// JSON payload path. Use '-' to read from stdin.
        #[arg(long)]
        input: String,
    },

    /// Get report generation supply (paper chunks and history report fields for evidence gaps).
    #[command(
        long_about = "Fetch evidence (paper chunks and history report fields) to fill evidence gaps from planner output.

EXAMPLES:
  memo get-report-generation-supply --input supplement_request.json
  echo '{\"paper_requests\":[],\"report_requests\":[]}' | memo get-report-generation-supply --input -"
    )]
    GetReportGenerationSupply {
        /// JSON payload path. Use '-' to read from stdin.
        #[arg(long)]
        input: String,
    },
}

impl Args {
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }
}