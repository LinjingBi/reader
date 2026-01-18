use crate::contracts::FreshPaperRequest;
use crate::db;
use anyhow::{Context, Result};
use std::io::Read;

#[derive(serde::Serialize)]
struct FreshPaperResponse {
    snapshot_id: String,
    cluster_run_id: String,
}

pub fn handle(db_path: &str, schema_path: &str, input_path: &str) -> Result<()> {
    let req: FreshPaperRequest = read_json(input_path)?;

    let conn = db::open(db_path)?;
    db::migrate::apply_schema(&conn, schema_path)?;

    let store = db::store::Store::new(&conn);
    let (snapshot_id, cluster_run_id) = store.fresh_paper(&req)?;

    let resp = FreshPaperResponse { snapshot_id, cluster_run_id };
    let out = serde_json::to_string_pretty(&resp)?;
    println!("{out}");
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    let mut s = String::new();
    if path == "-" {
        std::io::stdin().read_to_string(&mut s)?;
    } else {
        s = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read input JSON: {path}"))?;
    }
    let v: T = serde_json::from_str(&s).context("invalid JSON payload")?;
    Ok(v)
}
