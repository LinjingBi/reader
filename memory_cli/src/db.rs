pub mod migrate;
pub mod store;

use anyhow::Result;
use rusqlite::Connection;

/// Open a SQLite connection and apply required PRAGMAs.
pub fn open(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    // Concurrency + integrity defaults
    conn.pragma_update(None, "foreign_keys", &"ON")?;
    conn.pragma_update(None, "journal_mode", &"WAL")?;
    conn.pragma_update(None, "synchronous", &"NORMAL")?;
    conn.pragma_update(None, "busy_timeout", &5000i64)?;
    Ok(conn)
}
