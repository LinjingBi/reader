use crate::db;
use anyhow::Result;

pub fn handle(db_path: &str, schema_path: &str, source: &str, period_start: &str, period_end: &str, top_n: usize) -> Result<()> {
    let conn = db::open(db_path)?;
    db::migrate::apply_schema(&conn, schema_path)?;

    let store = db::store::Store::new(&conn);
    let resp = store.get_best_run(source, period_start, period_end, top_n)?;
    let out = serde_json::to_string_pretty(&resp)?;
    println!("{out}");
    Ok(())
}
