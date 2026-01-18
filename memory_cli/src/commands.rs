mod fresh_paper;
mod get_best_run;

use crate::cli::{Args, Command};
use anyhow::Result;

pub fn dispatch(args: Args) -> Result<()> {
    match args.cmd {
        Command::FreshPaper { input } => fresh_paper::handle(&args.db, &args.schema, &input),
        Command::GetBestRun { source, period_start, period_end, top_n } => {
            get_best_run::handle(&args.db, &args.schema, &source, &period_start, &period_end, top_n)
        }
    }
}
