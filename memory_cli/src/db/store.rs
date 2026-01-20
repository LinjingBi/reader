use crate::contracts::{FreshPaperRequest, GetBestRunResponse, ClusterCard, PaperCard};
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection, Transaction};

/// Thin repository layer. Exposes only safe, pre-defined operations.
pub struct Store<'a> {
    conn: &'a Connection,
}

impl<'a> Store<'a> {
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    /// Atomic ingest of month snapshot + papers + best clustering (Step 1–2).
    pub fn fresh_paper(&self, req: &FreshPaperRequest) -> Result<(String, String)> {
        let tx = self.conn.unchecked_transaction()?;
        let now = Utc::now().to_rfc3339();

        // Upsert configs
        self.upsert_embed_config(&tx, &req.embed_config.embed_config_id, &req.embed_config.json_payload.to_string(), &now)?;
        self.upsert_cluster_config(&tx, &req.cluster_config.cluster_config_id, &req.cluster_config.json_payload.to_string(), &now)?;

        // Snapshot
        let snapshot_id = req.snapshot_id.clone().unwrap_or_else(|| {
            format!("{}|{}|{}", req.source, req.period_start, req.period_end)
        });
        let raw_json = req.raw_json.clone().unwrap_or_else(|| "{}".to_string());
        self.upsert_snapshot(&tx, &snapshot_id, &req.source, &req.period_start, &req.period_end, &raw_json, &now)?;

        // Papers + snapshot links
        for p in &req.papers {
            self.upsert_paper(&tx, p, &now)?;
            self.link_snapshot_paper(&tx, &snapshot_id, &p.paper_id)?;
        }

        // Cluster run (selected_best=1)
        let role = req.role.clone().unwrap_or_else(|| "hf_batch".to_string());
        let cluster_run_id = format!(
            "{}|{}|{}|{}",
            snapshot_id, req.embed_config.embed_config_id, req.cluster_config.cluster_config_id, role
        );

        // Ensure only one selected_best per snapshot+role (MVP)
        tx.execute(
            "UPDATE cluster_run SET selected_best=0 WHERE snapshot_id=?1 AND role=?2",
            params![snapshot_id, role],
        )?;

        self.upsert_cluster_run(
            &tx,
            &cluster_run_id,
            &snapshot_id,
            &req.embed_config.embed_config_id,
            &req.cluster_config.cluster_config_id,
            &role,
            &now,
        )?;

        // Delete old clusters for this run id (idempotent rerun)
        tx.execute("DELETE FROM cluster WHERE cluster_run_id=?1", params![cluster_run_id])?;
        // cluster_member is cascaded by cluster delete.

        // Insert clusters + members
        for c in &req.clusters {
            let cluster_id = format!("{}|c{}", cluster_run_id, c.cluster_index);
            self.insert_cluster(
                &tx,
                &cluster_id,
                &cluster_run_id,
                c.cluster_index,
                c.size,
                c.centroid_b64.as_deref(),
                c.cohesion,
                &now,
            )?;

            for m in &c.members {
                self.insert_cluster_member(
                    &tx,
                    &cluster_id,
                    &m.paper_id,
                    m.rank_in_cluster,
                    m.sim_to_centroid,
                )?;
            }
        }

        tx.commit()?;
        Ok((snapshot_id, cluster_run_id))
    }

    /// Read the selected best run for a snapshot period.
    pub fn get_best_run(&self, source: &str, period_start: &str, period_end: &str, top_n: usize) -> Result<GetBestRunResponse> {
        let snapshot_id = format!("{}|{}|{}", source, period_start, period_end);

        let (cluster_run_id, embed_config_id, cluster_config_id) = self.conn.query_row(
            "SELECT cluster_run_id, embed_config_id, cluster_config_id
             FROM cluster_run
             WHERE snapshot_id=?1 AND role='hf_batch' AND selected_best=1
             ORDER BY created_at DESC
             LIMIT 1",
            params![snapshot_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)),
        ).with_context(|| format!("no selected best run found for snapshot_id={snapshot_id}"))?;

        // Load clusters
        let mut stmt = self.conn.prepare(
            "SELECT cluster_id, cluster_index, size, cohesion
             FROM cluster
             WHERE cluster_run_id=?1
             ORDER BY cluster_index ASC"
        )?;

        let clusters_iter = stmt.query_map(params![cluster_run_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<f64>>(3)?,
            ))
        })?;

        let mut clusters: Vec<ClusterCard> = Vec::new();
        for r in clusters_iter {
            let (cluster_id, cluster_index, size, cohesion) = r?;
            let papers = self.get_cluster_papers(&cluster_id, top_n)?;
            clusters.push(ClusterCard { cluster_id, cluster_index, size, cohesion, papers });
        }

        Ok(GetBestRunResponse {
            snapshot_id,
            cluster_run_id,
            source: source.to_string(),
            period_start: period_start.to_string(),
            period_end: period_end.to_string(),
            embed_config_id,
            cluster_config_id,
            clusters,
        })
    }

    fn get_cluster_papers(&self, cluster_id: &str, top_n: usize) -> Result<Vec<PaperCard>> {
        let mut stmt = self.conn.prepare(
            "SELECT p.paper_id, p.title, p.summary, p.keywords_json, p.url, cm.rank_in_cluster, cm.sim_to_centroid
             FROM cluster_member cm
             JOIN paper p ON p.paper_id = cm.paper_id
             WHERE cm.cluster_id=?1
             ORDER BY cm.rank_in_cluster ASC
             LIMIT ?2"
        )?;

        let iter = stmt.query_map(params![cluster_id, top_n as i64], |row| {
            let kw_json: String = row.get(3)?;
            let keywords: Vec<String> = serde_json::from_str(&kw_json).unwrap_or_default();
            Ok(PaperCard {
                paper_id: row.get(0)?,
                title: row.get(1)?,
                summary: row.get(2)?,
                keywords,
                url: row.get(4)?,
                rank_in_cluster: row.get(5)?,
                sim_to_centroid: row.get(6)?,
            })
        })?;

        let mut out = Vec::new();
        for r in iter { out.push(r?); }
        Ok(out)
    }

    // ---------- SQL helpers ----------

    fn upsert_embed_config(&self, tx: &Transaction, id: &str, json_payload: &str, now: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO embed_config(embed_config_id, json_payload, created_at)
             VALUES(?1, ?2, ?3)
             ON CONFLICT(embed_config_id) DO UPDATE SET json_payload=excluded.json_payload",
            params![id, json_payload, now],
        )?;
        Ok(())
    }

    fn upsert_cluster_config(&self, tx: &Transaction, id: &str, json_payload: &str, now: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO cluster_config(cluster_config_id, json_payload, created_at)
             VALUES(?1, ?2, ?3)
             ON CONFLICT(cluster_config_id) DO UPDATE SET json_payload=excluded.json_payload",
            params![id, json_payload, now],
        )?;
        Ok(())
    }

    fn upsert_snapshot(&self, tx: &Transaction, snapshot_id: &str, source: &str, start: &str, end: &str, raw_json: &str, now: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO source_snapshot(snapshot_id, source, period_start, period_end, raw_json, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(snapshot_id) DO UPDATE SET raw_json=excluded.raw_json",
            params![snapshot_id, source, start, end, raw_json, now],
        )?;
        Ok(())
    }

    fn upsert_paper(&self, tx: &Transaction, p: &crate::contracts::PaperInput, now: &str) -> Result<()> {
        let kw_json = serde_json::to_string(&p.keywords)?;
        tx.execute(
            "INSERT INTO paper(paper_id, title, summary, keywords_json, url, source, published_at, ingested_at)
             VALUES(?1, ?2, ?3, ?4, ?5, 'hf', ?6, ?7)
             ON CONFLICT(paper_id) DO UPDATE SET
               title=excluded.title,
               summary=excluded.summary,
               keywords_json=excluded.keywords_json,
               url=excluded.url,
               published_at=excluded.published_at",
            params![p.paper_id, p.title, p.summary, kw_json, p.url, p.published_at, now],
        )?;
        Ok(())
    }

    fn link_snapshot_paper(&self, tx: &Transaction, snapshot_id: &str, paper_id: &str) -> Result<()> {
        tx.execute(
            "INSERT OR IGNORE INTO snapshot_paper(snapshot_id, paper_id) VALUES(?1, ?2)",
            params![snapshot_id, paper_id],
        )?;
        Ok(())
    }

    fn upsert_cluster_run(&self, tx: &Transaction, cluster_run_id: &str, snapshot_id: &str, embed_config_id: &str, cluster_config_id: &str, role: &str, now: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO cluster_run(cluster_run_id, snapshot_id, embed_config_id, cluster_config_id, role, selected_best, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5, 1, ?6)
             ON CONFLICT(cluster_run_id) DO UPDATE SET
               selected_best=1",
            params![cluster_run_id, snapshot_id, embed_config_id, cluster_config_id, role, now],
        )?;
        Ok(())
    }

    fn insert_cluster(&self, tx: &Transaction, cluster_id: &str, cluster_run_id: &str, cluster_index: i64, size: i64, centroid_b64: Option<&str>, cohesion: Option<f64>, now: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO cluster(cluster_id, cluster_run_id, cluster_index, size, centroid_b64, cohesion, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![cluster_id, cluster_run_id, cluster_index, size, centroid_b64, cohesion, now],
        )?;
        Ok(())
    }

    fn insert_cluster_member(&self, tx: &Transaction, cluster_id: &str, paper_id: &str, rank_in_cluster: i64, sim_to_centroid: Option<f64>) -> Result<()> {
        tx.execute(
            "INSERT INTO cluster_member(cluster_id, paper_id, rank_in_cluster, sim_to_centroid)
             VALUES(?1, ?2, ?3, ?4)",
            params![cluster_id, paper_id, rank_in_cluster, sim_to_centroid],
        )?;
        Ok(())
    }
}
