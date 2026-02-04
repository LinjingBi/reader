use crate::contracts::{
    FreshPaperRequest, GetBestRunResponse, ClusterCard, PaperCard,
    InjectClustersObservationInput,
    GetClusterObservationResponse, ClusterObservationData,
};
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection, Transaction};
use sha2::{Sha256, Digest};
use hex;

/// Thin repository layer. Exposes only safe, pre-defined operations.
pub struct Store<'a> {
    conn: &'a Connection,
}

impl<'a> Store<'a> {
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    /// Compute SHA256 hash of cluster primary key fields
    fn compute_cluster_pk_hash(
        source: &str,
        period_start: &str,
        period_end: &str,
        embed_config_id: &str,
        cluster_config_id: &str,
        role: &str,
        cluster_index: i64,
    ) -> String {
        let input = format!(
            "{}|{}|{}|{}|{}|{}|{}",
            source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index
        );
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Atomic ingest of month snapshot + papers + best clustering (Step 1–2).
    /// Split into two transactions: Tx A (ingest) and Tx B (clusters).
    pub fn fresh_paper(&self, req: &FreshPaperRequest) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let role = req.role.clone().unwrap_or_else(|| "hf_batch".to_string());

        // Tx A: Ingest (configs, snapshot, papers)
        {
            let tx_a = self.conn.unchecked_transaction()?;
            
            // Upsert configs
            self.upsert_embed_config(&tx_a, &req.embed_config.embed_config_id, &req.embed_config.json_payload.to_string(), &now)?;
            self.upsert_cluster_config(&tx_a, &req.cluster_config.cluster_config_id, &req.cluster_config.json_payload.to_string(), &now)?;

            // Snapshot
            let raw_json = req.raw_json.clone().unwrap_or_else(|| "{}".to_string());
            self.upsert_snapshot(&tx_a, &req.source, &req.period_start, &req.period_end, &raw_json, &now)?;

            // Papers + snapshot links
            for p in &req.papers {
                self.upsert_paper(&tx_a, p, &now)?;
                self.link_snapshot_paper(&tx_a, &req.source, &req.period_start, &req.period_end, &p.paper_id)?;
            }

            tx_a.commit()?;
        }

        // Tx B: Clusters
        {
            let tx_b = self.conn.unchecked_transaction()?;

            // Ensure only one selected_best per snapshot+role (enforced by partial unique index)
            tx_b.execute(
                "UPDATE cluster_run SET selected_best=0 WHERE source=?1 AND period_start=?2 AND period_end=?3 AND role=?4",
                params![req.source, req.period_start, req.period_end, role],
            )?;

            self.upsert_cluster_run(
                &tx_b,
                &req.source,
                &req.period_start,
                &req.period_end,
                &req.embed_config.embed_config_id,
                &req.cluster_config.cluster_config_id,
                &role,
                &now,
            )?;

            // Delete old clusters for this run (idempotent rerun)
            tx_b.execute(
                "DELETE FROM cluster WHERE source=?1 AND period_start=?2 AND period_end=?3 AND embed_config_id=?4 AND cluster_config_id=?5 AND role=?6",
                params![req.source, req.period_start, req.period_end, req.embed_config.embed_config_id, req.cluster_config.cluster_config_id, role],
            )?;
            // cluster_member is cascaded by cluster delete.

            // Update cluster_updated_at timestamp when clusters are regenerated
            let cluster_update_time = Utc::now().to_rfc3339();
            tx_b.execute(
                "UPDATE cluster_run SET updated_at=?1 WHERE source=?2 AND period_start=?3 AND period_end=?4 AND embed_config_id=?5 AND cluster_config_id=?6 AND role=?7",
                params![cluster_update_time, req.source, req.period_start, req.period_end, req.embed_config.embed_config_id, req.cluster_config.cluster_config_id, role],
            )?;

            // Insert clusters + members
            for c in &req.clusters {
                self.insert_cluster(
                    &tx_b,
                    &req.source,
                    &req.period_start,
                    &req.period_end,
                    &req.embed_config.embed_config_id,
                    &req.cluster_config.cluster_config_id,
                    &role,
                    c.cluster_index,
                    c.size,
                    c.centroid_b64.as_deref(),
                    c.cohesion,
                    &now,
                )?;

                for m in &c.members {
                    self.insert_cluster_member(
                        &tx_b,
                        &req.source,
                        &req.period_start,
                        &req.period_end,
                        &req.embed_config.embed_config_id,
                        &req.cluster_config.cluster_config_id,
                        &role,
                        c.cluster_index,
                        &m.paper_id,
                        m.rank_in_cluster,
                        m.sim_to_centroid,
                    )?;
                }
            }

            tx_b.commit()?;
        }

        Ok(())
    }

    /// Read the selected best run for a snapshot period.
    pub fn get_best_run(&self, source: &str, period_start: &str, period_end: &str, top_n: usize) -> Result<GetBestRunResponse> {
        let (embed_config_id, cluster_config_id) = self.conn.query_row(
            "SELECT embed_config_id, cluster_config_id
             FROM cluster_run
             WHERE source=?1 AND period_start=?2 AND period_end=?3 AND role='hf_batch' AND selected_best=1
             ORDER BY created_at DESC
             LIMIT 1",
            params![source, period_start, period_end],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        ).with_context(|| format!("no selected best run found for source={source}, period_start={period_start}, period_end={period_end}"))?;

        // Load clusters
        let mut stmt = self.conn.prepare(
            "SELECT cluster_index, pk_hash, size, cohesion
             FROM cluster
             WHERE source=?1 AND period_start=?2 AND period_end=?3 AND embed_config_id=?4 AND cluster_config_id=?5 AND role='hf_batch'
             ORDER BY cluster_index ASC"
        )?;

        let clusters_iter = stmt.query_map(params![source, period_start, period_end, embed_config_id, cluster_config_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<f64>>(3)?,
            ))
        })?;

        let mut clusters: Vec<ClusterCard> = Vec::new();
        for r in clusters_iter {
            let (cluster_index, pk_hash, size, cohesion) = r?;
            let papers = self.get_cluster_papers(source, period_start, period_end, &embed_config_id, &cluster_config_id, cluster_index, top_n)?;
            clusters.push(ClusterCard { cluster_index, pk_hash, size, cohesion, papers });
        }

        Ok(GetBestRunResponse {
            source: source.to_string(),
            period_start: period_start.to_string(),
            period_end: period_end.to_string(),
            embed_config_id,
            cluster_config_id,
            clusters,
        })
    }

    fn get_cluster_papers(&self, source: &str, period_start: &str, period_end: &str, embed_config_id: &str, cluster_config_id: &str, cluster_index: i64, top_n: usize) -> Result<Vec<PaperCard>> {
        let mut stmt = self.conn.prepare(
            "SELECT p.paper_id, p.title, p.summary, p.keywords_json, p.url, cm.rank_in_cluster, cm.sim_to_centroid
             FROM cluster_member cm
             JOIN paper p ON p.paper_id = cm.paper_id
             WHERE cm.source=?1 AND cm.period_start=?2 AND cm.period_end=?3 AND cm.embed_config_id=?4 AND cm.cluster_config_id=?5 AND cm.role='hf_batch' AND cm.cluster_index=?6
             ORDER BY cm.rank_in_cluster ASC
             LIMIT ?7"
        )?;

        let iter = stmt.query_map(params![source, period_start, period_end, embed_config_id, cluster_config_id, cluster_index, top_n as i64], |row| {
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

    fn upsert_snapshot(&self, tx: &Transaction, source: &str, start: &str, end: &str, raw_json: &str, now: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO source_snapshot(source, period_start, period_end, raw_json, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(source, period_start, period_end) DO UPDATE SET raw_json=excluded.raw_json",
            params![source, start, end, raw_json, now],
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

    fn link_snapshot_paper(&self, tx: &Transaction, source: &str, period_start: &str, period_end: &str, paper_id: &str) -> Result<()> {
        tx.execute(
            "INSERT OR IGNORE INTO snapshot_paper(source, period_start, period_end, paper_id) VALUES(?1, ?2, ?3, ?4)",
            params![source, period_start, period_end, paper_id],
        )?;
        Ok(())
    }

    fn upsert_cluster_run(&self, tx: &Transaction, source: &str, period_start: &str, period_end: &str, embed_config_id: &str, cluster_config_id: &str, role: &str, now: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO cluster_run(source, period_start, period_end, embed_config_id, cluster_config_id, role, selected_best, created_at, updated_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, 1, ?7, ?7)
             ON CONFLICT(source, period_start, period_end, embed_config_id, cluster_config_id, role) DO UPDATE SET
               selected_best=1",
            params![source, period_start, period_end, embed_config_id, cluster_config_id, role, now],
        )?;
        Ok(())
    }

    fn insert_cluster(&self, tx: &Transaction, source: &str, period_start: &str, period_end: &str, embed_config_id: &str, cluster_config_id: &str, role: &str, cluster_index: i64, size: i64, centroid_b64: Option<&str>, cohesion: Option<f64>, now: &str) -> Result<()> {
        let pk_hash = Self::compute_cluster_pk_hash(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index);
        tx.execute(
            "INSERT INTO cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index, pk_hash, size, centroid_b64, cohesion, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) DO UPDATE SET
               pk_hash=excluded.pk_hash,
               size=excluded.size,
               centroid_b64=excluded.centroid_b64,
               cohesion=excluded.cohesion",
            params![source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index, pk_hash, size, centroid_b64, cohesion, now],
        )?;
        Ok(())
    }

    fn insert_cluster_member(&self, tx: &Transaction, source: &str, period_start: &str, period_end: &str, embed_config_id: &str, cluster_config_id: &str, role: &str, cluster_index: i64, paper_id: &str, rank_in_cluster: i64, sim_to_centroid: Option<f64>) -> Result<()> {
        tx.execute(
            "INSERT INTO cluster_member(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index, paper_id, rank_in_cluster, sim_to_centroid)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index, paper_id, rank_in_cluster, sim_to_centroid],
        )?;
        Ok(())
    }

    /// Upsert cluster observations (LLM enrichment results).
    /// Upserts llm_config entries and cluster_observation rows in a single transaction.
    pub fn upsert_cluster_observation(&self, input: &InjectClustersObservationInput) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let tx = self.conn.unchecked_transaction()?;

        for (pk_hash, observation) in input {
            // Upsert llm_config
            let json_payload_str = serde_json::to_string(&observation.llm_config.json_payload)
                .context("failed to serialize llm_config json_payload")?;
            self.upsert_llm_config(
                &tx,
                &observation.llm_config.llm_config_id,
                &json_payload_str,
                &now,
            )?;

            // Serialize payload_json to compact JSON string
            let payload_json_str = serde_json::to_string(&observation.payload_json)
                .context("failed to serialize payload_json")?;

            // Upsert cluster_observation
            tx.execute(
                "INSERT INTO cluster_observation(pk_hash, created_at, llm_config_id, payload_json)
                 VALUES(?1, ?2, ?3, ?4)
                 ON CONFLICT(pk_hash) DO UPDATE SET
                   created_at=excluded.created_at,
                   llm_config_id=excluded.llm_config_id,
                   payload_json=excluded.payload_json",
                params![pk_hash, now, observation.llm_config.llm_config_id, payload_json_str],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    fn upsert_llm_config(&self, tx: &Transaction, id: &str, json_payload: &str, now: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO llm_config(llm_config_id, json_payload, created_at)
             VALUES(?1, ?2, ?3)
             ON CONFLICT(llm_config_id) DO UPDATE SET json_payload=excluded.json_payload",
            params![id, json_payload, now],
        )?;
        Ok(())
    }

    /// Get cluster observations for clusters within a period range.
    /// Returns observations for clusters that:
    /// - Match the source
    /// - Have period_start >= query.period_start AND period_end <= query.period_end
    /// - Belong to best runs (selected_best=1)
    pub fn get_cluster_observation(&self, source: &str, period_start: &str, period_end: &str) -> Result<GetClusterObservationResponse> {
        // Use optimized JOIN query to get cluster observations
        let mut stmt = self.conn.prepare(
            "SELECT co.pk_hash, co.created_at, co.payload_json, c.period_start, c.period_end
             FROM cluster_observation co
             JOIN cluster c ON co.pk_hash = c.pk_hash
             JOIN cluster_run cr ON (
                 c.source = cr.source AND
                 c.period_start = cr.period_start AND
                 c.period_end = cr.period_end AND
                 c.embed_config_id = cr.embed_config_id AND
                 c.cluster_config_id = cr.cluster_config_id AND
                 c.role = cr.role
             )
             WHERE c.source = ?1
               AND c.period_start >= ?2
               AND c.period_end <= ?3
               AND cr.selected_best = 1
               AND c.role = 'hf_batch'"
        )?;

        let rows = stmt.query_map(params![source, period_start, period_end], |row| {
            let pk_hash: String = row.get(0)?;
            let created_at: String = row.get(1)?;
            let payload_json_str: String = row.get(2)?;
            let cluster_period_start: String = row.get(3)?;
            let cluster_period_end: String = row.get(4)?;

            // Parse RFC3339 timestamp and format as YYYY-MM-DD
            let observation_created_time = match chrono::DateTime::parse_from_rfc3339(&created_at) {
                Ok(dt) => dt.format("%Y-%m-%d").to_string(),
                Err(_) => {
                    // Fallback: try to extract date part if parsing fails
                    created_at.split('T').next().unwrap_or(&created_at).to_string()
                }
            };

            // Parse payload_json string to serde_json::Value
            let json_payload: serde_json::Value = serde_json::from_str(&payload_json_str)
                .map_err(|_e| rusqlite::Error::InvalidColumnType(3, "payload_json".to_string(), rusqlite::types::Type::Text))?;

            Ok((pk_hash, ClusterObservationData {
                observation_created_time,
                json_payload,
                cluster_period_start,
                cluster_period_end,
            }))
        })?;

        let mut result = GetClusterObservationResponse::new();
        for row_result in rows {
            let (pk_hash, data) = row_result?;
            result.insert(pk_hash, data);
        }

        Ok(result)
    }
}
