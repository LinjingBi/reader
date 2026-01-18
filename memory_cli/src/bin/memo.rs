use anyhow::Result;

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e:?}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = memo_cli::cli::Args::parse();
    memo_cli::commands::dispatch(args)
}
