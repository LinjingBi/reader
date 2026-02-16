-- Fresh-paper command schema (SQLite)
-- Minimal schema containing only tables and indexes required for memo fresh-paper command
-- Extracted from memory_cli/schemas/schema.sql

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
  pk_hash          TEXT NOT NULL UNIQUE, -- SHA256 hex hash of primary key fields
  size             INTEGER NOT NULL,
  -- Geometry artifacts for display/matching/debug
  centroid_b64     TEXT NOT NULL,    -- base64 float32 bytes. normalized
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

-- -----------------------------
-- Indexes for fresh_paper operations
-- -----------------------------

CREATE INDEX IF NOT EXISTS idx_cluster_member_rank
  ON cluster_member(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index, rank_in_cluster);

CREATE INDEX IF NOT EXISTS idx_cluster_run_snapshot_role ON cluster_run(source, period_start, period_end, role);

CREATE INDEX IF NOT EXISTS idx_cluster_cluster_run ON cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role);

-- Partial unique index to enforce single best run per snapshot+role
CREATE UNIQUE INDEX IF NOT EXISTS ux_cluster_run_best 
  ON cluster_run(source, period_start, period_end, role) 
  WHERE selected_best=1;

-- Note: pk_hash has UNIQUE constraint which automatically creates an index

