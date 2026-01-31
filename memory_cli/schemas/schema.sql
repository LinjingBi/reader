-- reader-memory MVP schema (SQLite)
-- Notes:
-- - Canonical topics are language-first objects. Clusters are geometry-first artifacts of a run.
-- - Embeddings are OPTIONAL in MVP. This schema supports both "store vectors" and "recompute on the fly".
-- - All comparisons must be done within the same embed_config_id (recorded on runs/events).

PRAGMA foreign_keys = ON;

-- -----------------------------
-- Config provenance
-- -----------------------------

CREATE TABLE IF NOT EXISTS embed_config (
  embed_config_id TEXT PRIMARY KEY,
  json_payload    TEXT NOT NULL,  -- model name, revision, text template version, normalize, etc.
  created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_config (
  cluster_config_id TEXT PRIMARY KEY,
  json_payload      TEXT NOT NULL, -- algo name, params, scoring weights version, etc.
  created_at        TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS llm_config (
  llm_config_id TEXT PRIMARY KEY,
  provider      TEXT NOT NULL,     -- e.g., 'openai'|'anthropic'|'google'|'openrouter'
  model         TEXT NOT NULL,     -- model name as used by the API
  endpoint      TEXT,              -- base URL or routing identifier (nullable)
  params_json   TEXT NOT NULL,     -- JSON: temperature, top_p, max_tokens, etc.
  created_at    TEXT NOT NULL
);


-- -----------------------------
-- Source snapshots (HF pulls)
-- -----------------------------

CREATE TABLE IF NOT EXISTS source_snapshot (
  source        TEXT NOT NULL,     -- e.g., 'hf_monthly'
  period_start  TEXT NOT NULL,
  period_end    TEXT NOT NULL,
  raw_json      TEXT NOT NULL,     -- store HF response (optionally compressed at the app level)
  created_at    TEXT NOT NULL,
  PRIMARY KEY (source, period_start, period_end)
);

-- -----------------------------
-- Papers (canonical objects)
-- -----------------------------

CREATE TABLE IF NOT EXISTS paper (
  paper_id      TEXT PRIMARY KEY, -- arXiv id or HF id (normalize early)
  title         TEXT NOT NULL,
  summary       TEXT NOT NULL,     -- abstract (or HF summary field)
  keywords_json TEXT NOT NULL,     -- JSON list
  url           TEXT NOT NULL,
  source        TEXT NOT NULL,     -- 'hf'
  published_at  TEXT,              -- ISO date, nullable
  ingested_at   TEXT NOT NULL
);

-- Paper belongs to a snapshot period (many-to-many in case you ingest from multiple sources/periods)
CREATE TABLE IF NOT EXISTS snapshot_paper (
  source       TEXT NOT NULL,
  period_start TEXT NOT NULL,
  period_end   TEXT NOT NULL,
  paper_id     TEXT NOT NULL,
  PRIMARY KEY (source, period_start, period_end, paper_id),
  FOREIGN KEY (source, period_start, period_end) REFERENCES source_snapshot(source, period_start, period_end) ON DELETE CASCADE,
  FOREIGN KEY (paper_id) REFERENCES paper(paper_id) ON DELETE CASCADE
);

-- OPTIONAL: store embeddings (recommended once stable)
CREATE TABLE IF NOT EXISTS paper_embedding (
  paper_id        TEXT NOT NULL,
  embed_config_id TEXT NOT NULL,
  dim             INTEGER NOT NULL,
  vector_b64      TEXT NOT NULL,   -- base64-encoded float32 bytes OR JSON array; base64 is smaller
  created_at      TEXT NOT NULL,
  PRIMARY KEY (paper_id, embed_config_id),
  FOREIGN KEY (paper_id)        REFERENCES paper(paper_id)        ON DELETE CASCADE,
  FOREIGN KEY (embed_config_id) REFERENCES embed_config(embed_config_id) ON DELETE RESTRICT
);

-- -----------------------------
-- Cluster runs (geometry-first)
-- -----------------------------

CREATE TABLE IF NOT EXISTS cluster_run (
  source           TEXT NOT NULL,
  period_start     TEXT NOT NULL,
  period_end       TEXT NOT NULL,
  embed_config_id  TEXT NOT NULL,
  cluster_config_id TEXT NOT NULL,
  role             TEXT NOT NULL,   -- 'hf_batch' (MVP), future: 'reader_view'
  selected_best    INTEGER NOT NULL DEFAULT 0, -- 1 if chosen as best clustering for the snapshot
  created_at       TEXT NOT NULL,
  updated_at       TEXT,              -- timestamp when clusters were last regenerated
  PRIMARY KEY (source, period_start, period_end, embed_config_id, cluster_config_id, role),
  FOREIGN KEY (source, period_start, period_end) REFERENCES source_snapshot(source, period_start, period_end) ON DELETE CASCADE,
  FOREIGN KEY (embed_config_id)   REFERENCES embed_config(embed_config_id)      ON DELETE RESTRICT,
  FOREIGN KEY (cluster_config_id) REFERENCES cluster_config(cluster_config_id)  ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS cluster (
  source           TEXT NOT NULL,
  period_start     TEXT NOT NULL,
  period_end       TEXT NOT NULL,
  embed_config_id  TEXT NOT NULL,
  cluster_config_id TEXT NOT NULL,
  role             TEXT NOT NULL,
  cluster_index    INTEGER NOT NULL, -- 0..k-1
  size             INTEGER NOT NULL,
  -- OPTIONAL geometry artifacts for display/matching/debug
  centroid_b64     TEXT,             -- base64 float32 bytes (nullable if not storing)
  cohesion         REAL,             -- avg cosine to centroid (nullable)
  created_at       TEXT NOT NULL,
  PRIMARY KEY (source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index),
  FOREIGN KEY (source, period_start, period_end, embed_config_id, cluster_config_id, role) 
    REFERENCES cluster_run(source, period_start, period_end, embed_config_id, cluster_config_id, role) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cluster_member (
  source           TEXT NOT NULL,
  period_start     TEXT NOT NULL,
  period_end       TEXT NOT NULL,
  embed_config_id  TEXT NOT NULL,
  cluster_config_id TEXT NOT NULL,
  role             TEXT NOT NULL,
  cluster_index    INTEGER NOT NULL,
  paper_id         TEXT NOT NULL,
  rank_in_cluster  INTEGER NOT NULL, -- 0=most representative
  sim_to_centroid  REAL,             -- nullable if you want only ordering
  PRIMARY KEY (source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index, paper_id),
  FOREIGN KEY (source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) 
    REFERENCES cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) ON DELETE CASCADE,
  FOREIGN KEY (paper_id) REFERENCES paper(paper_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cluster_member_rank
  ON cluster_member(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index, rank_in_cluster);

-- -----------------------------
-- Topics (language-first canonical objects)
-- -----------------------------

CREATE TABLE IF NOT EXISTS topic (
  topic_id           TEXT PRIMARY KEY,
  canonical_name     TEXT NOT NULL,
  canonical_summary  TEXT NOT NULL,
  labels_json        TEXT NOT NULL,  -- JSON list
  status             TEXT NOT NULL,  -- 'active' | 'merged' | 'deprecated'
  created_at         TEXT NOT NULL,
  updated_at         TEXT NOT NULL
);

-- OPTIONAL: embed canonical topic card for quick matching (can also compute on the fly)
CREATE TABLE IF NOT EXISTS topic_embedding (
  topic_id        TEXT NOT NULL,
  embed_config_id TEXT NOT NULL,
  dim             INTEGER NOT NULL,
  vector_b64      TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  PRIMARY KEY (topic_id, embed_config_id),
  FOREIGN KEY (topic_id)        REFERENCES topic(topic_id)        ON DELETE CASCADE,
  FOREIGN KEY (embed_config_id) REFERENCES embed_config(embed_config_id) ON DELETE RESTRICT
);

-- Observations are per-period semantic snapshots produced by LLM during attachment
CREATE TABLE IF NOT EXISTS topic_observation (
  observation_id       TEXT PRIMARY KEY,
  topic_id             TEXT NOT NULL,
  source               TEXT NOT NULL,
  period_start         TEXT NOT NULL,
  period_end           TEXT NOT NULL,
  embed_config_id      TEXT NOT NULL,
  cluster_config_id    TEXT NOT NULL,
  role                 TEXT NOT NULL,
  cluster_index        INTEGER NOT NULL,
  proposed_name        TEXT NOT NULL,
  proposed_summary     TEXT NOT NULL,
  proposed_labels_json TEXT NOT NULL,
  produced_by          TEXT NOT NULL,  -- 'llm' | 'human',
  created_at           TEXT NOT NULL,
  llm_config_id        TEXT,           -- provenance of proposed_* fields (nullable),
  FOREIGN KEY (topic_id)   REFERENCES topic(topic_id)   ON DELETE CASCADE,
  FOREIGN KEY (source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) 
    REFERENCES cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) ON DELETE CASCADE,
  FOREIGN KEY (llm_config_id) REFERENCES llm_config(llm_config_id) ON DELETE SET NULL
);

-- Link clusters (from runs) to topics; captures add/create decision + matching provenance
CREATE TABLE IF NOT EXISTS topic_cluster_link (
  topic_id          TEXT NOT NULL,
  source            TEXT NOT NULL,
  period_start      TEXT NOT NULL,
  period_end        TEXT NOT NULL,
  embed_config_id   TEXT NOT NULL,
  cluster_config_id TEXT NOT NULL,
  role              TEXT NOT NULL,
  cluster_index     INTEGER NOT NULL,
  decision          TEXT NOT NULL,  -- 'matched_existing' | 'created_new' | 'attached'
  match_score       REAL,           -- cosine sim between topic vector and cluster/topic card (nullable)
  created_at        TEXT NOT NULL,
  PRIMARY KEY (topic_id, source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index),
  FOREIGN KEY (topic_id)   REFERENCES topic(topic_id)   ON DELETE CASCADE,
  FOREIGN KEY (source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) 
    REFERENCES cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) ON DELETE CASCADE
);

-- Topic events: canonical changes (rename/merge/split/refresh) with provenance
CREATE TABLE IF NOT EXISTS topic_event (
  event_id            TEXT PRIMARY KEY,
  event_type          TEXT NOT NULL,  -- 'rename'|'merge'|'split'|'refresh'|'deprecate',
  topic_ids_json      TEXT NOT NULL,  -- JSON list of involved topic_ids,
  produced_by         TEXT NOT NULL,  -- 'llm'|'human'|'system',
  embed_config_id     TEXT,           -- config used for any shortlisting/confirmation (optional),
  proposal_text       TEXT NOT NULL,  -- rationale / prompt output,
  geometry_check_json TEXT NOT NULL,  -- confirmation signals (can be {} in MVP),
  approved            INTEGER NOT NULL DEFAULT 0,
  created_at          TEXT NOT NULL,
  approved_at         TEXT,
  llm_config_id       TEXT,           -- provenance for proposal_text (nullable),
  FOREIGN KEY (embed_config_id) REFERENCES embed_config(embed_config_id) ON DELETE SET NULL,
  FOREIGN KEY (llm_config_id) REFERENCES llm_config(llm_config_id) ON DELETE SET NULL
);

-- Lineage edges to preserve history without rewriting old links
CREATE TABLE IF NOT EXISTS topic_lineage (
  topic_id          TEXT NOT NULL, -- current/survivor topic
  ancestor_topic_id TEXT NOT NULL, -- predecessor
  relation          TEXT NOT NULL, -- 'merged_from'|'split_from'|'renamed_from'
  effective_from    TEXT NOT NULL,
  PRIMARY KEY (topic_id, ancestor_topic_id, relation),
  FOREIGN KEY (topic_id)          REFERENCES topic(topic_id)          ON DELETE CASCADE,
  FOREIGN KEY (ancestor_topic_id) REFERENCES topic(topic_id)          ON DELETE CASCADE
);

-- -----------------------------
-- Reports and depth annotations
-- -----------------------------

CREATE TABLE IF NOT EXISTS report (
  report_id        TEXT PRIMARY KEY,
  period_start     TEXT NOT NULL,
  period_end       TEXT NOT NULL,
  source           TEXT NOT NULL,
  embed_config_id  TEXT NOT NULL,
  cluster_config_id TEXT NOT NULL,
  role             TEXT NOT NULL,
  cluster_index    INTEGER NOT NULL,
  report_md        TEXT NOT NULL,   -- full report content,
  created_at       TEXT NOT NULL,
  llm_config_id    TEXT,           -- provenance for report_md (nullable),
  FOREIGN KEY (source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) 
    REFERENCES cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index) ON DELETE CASCADE,
  FOREIGN KEY (llm_config_id) REFERENCES llm_config(llm_config_id) ON DELETE SET NULL
);

-- Reports can link to multiple topics (primary/secondary/related)
CREATE TABLE IF NOT EXISTS report_topic_link (
  report_id      TEXT NOT NULL,
  topic_id       TEXT NOT NULL,
  role           TEXT NOT NULL,   -- 'primary'|'secondary'|'related'
  match_score    REAL,            -- optional
  created_at     TEXT NOT NULL,
  PRIMARY KEY (report_id, topic_id),
  FOREIGN KEY (report_id) REFERENCES report(report_id) ON DELETE CASCADE,
  FOREIGN KEY (topic_id)  REFERENCES topic(topic_id)  ON DELETE CASCADE
);

-- Depth annotations: editorial signal from LLM, grounded by objective stats
-- Objective stats are computed from report_topic_link + report.
CREATE TABLE IF NOT EXISTS topic_depth_annotation (
  depth_id        TEXT PRIMARY KEY,
  topic_id        TEXT NOT NULL,
  as_of_date      TEXT NOT NULL,
  depth_level     TEXT NOT NULL, -- e.g., 'intro'|'intermediate'|'advanced',
  rationale       TEXT NOT NULL,
  based_on_report_ids_json TEXT NOT NULL, -- JSON list,
  produced_by     TEXT NOT NULL, -- 'llm'|'human',
  created_at      TEXT NOT NULL,
  llm_config_id   TEXT,           -- provenance for depth_level/rationale (nullable),
  FOREIGN KEY (topic_id) REFERENCES topic(topic_id) ON DELETE CASCADE,
  FOREIGN KEY (llm_config_id) REFERENCES llm_config(llm_config_id) ON DELETE SET NULL
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_report_period ON report(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_report_topic_role ON report_topic_link(topic_id, role);
CREATE INDEX IF NOT EXISTS idx_topic_status ON topic(status);

-- Performance indexes for fresh_paper operations
CREATE INDEX IF NOT EXISTS idx_cluster_run_snapshot_role ON cluster_run(source, period_start, period_end, role);
CREATE INDEX IF NOT EXISTS idx_cluster_cluster_run ON cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role);

-- Partial unique index to enforce single best run per snapshot+role
CREATE UNIQUE INDEX IF NOT EXISTS ux_cluster_run_best 
  ON cluster_run(source, period_start, period_end, role) 
  WHERE selected_best=1;
