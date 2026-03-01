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
  json_payload  TEXT NOT NULL,  -- model name, provider, endpoint, params, etc.
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

CREATE INDEX IF NOT EXISTS idx_cluster_member_rank
  ON cluster_member(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index, rank_in_cluster);

-- -----------------------------
-- Cluster observations (LLM enrichment results)
-- -----------------------------  
CREATE TABLE IF NOT EXISTS cluster_observation (
  -- same pk_hash as in cluster table
  pk_hash TEXT PRIMARY KEY,

  -- provenance / record
  created_at        TEXT NOT NULL,
  llm_config_id     TEXT NOT NULL,

  -- LLM output (opaque JSON)
  payload_json      TEXT NOT NULL,

  -- Extracted fields from payload_json
  summary           TEXT NOT NULL,
  title             TEXT NOT NULL,
  keywords_json     TEXT NOT NULL,
  score             REAL NOT NULL,  -- Judge output overall score


  -- Consumption tracking
  consumed          INTEGER NOT NULL DEFAULT 0,  -- 0 = false, 1 = true

  FOREIGN KEY (pk_hash)
    REFERENCES cluster(pk_hash)
    ON DELETE CASCADE,

  FOREIGN KEY (llm_config_id)
    REFERENCES llm_config(llm_config_id)
);

-- Performance indexes for fresh_paper operations
CREATE INDEX IF NOT EXISTS idx_cluster_run_snapshot_role ON cluster_run(source, period_start, period_end, role);
CREATE INDEX IF NOT EXISTS idx_cluster_cluster_run ON cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role);
-- Note: pk_hash has UNIQUE constraint which automatically creates an index

-- Partial unique index to enforce single best run per snapshot+role
CREATE UNIQUE INDEX IF NOT EXISTS ux_cluster_run_best 
  ON cluster_run(source, period_start, period_end, role) 
  WHERE selected_best=1;

-- -----------------------------
-- Paper chunking (v2 schema: chunk_text + map_id-based scoring)
-- -----------------------------

-- 1) Which chunker / extractor config produced the chunks
CREATE TABLE IF NOT EXISTS chunk_lib_config (
  lib_config_id   TEXT PRIMARY KEY,
  json_payload    TEXT NOT NULL DEFAULT '{}',
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- 2) One run per paper per lib version (you can allow multiple reruns)
CREATE TABLE IF NOT EXISTS paper_run_map (
  run_id          INTEGER PRIMARY KEY,
  paper_id        TEXT NOT NULL,                -- FK to paper
  lib_config_id   TEXT NOT NULL,                -- FK to chunk_lib_config
  status          TEXT NOT NULL,   -- ok|partial|error
  is_latest       INTEGER NOT NULL DEFAULT 0,   -- 0/1 wrt the runs for the same paper_id
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE(paper_id, lib_config_id),
  FOREIGN KEY(paper_id)      REFERENCES paper(paper_id) ON DELETE CASCADE,
  FOREIGN KEY(lib_config_id) REFERENCES chunk_lib_config(lib_config_id) ON DELETE RESTRICT,
  CHECK (status IN ('ok','partial','error')),
  CHECK (is_latest IN (0,1))
);

CREATE INDEX IF NOT EXISTS idx_paper_run_map_latest
  ON paper_run_map(paper_id, lib_config_id, is_latest);

CREATE INDEX IF NOT EXISTS idx_paper_run_map_paper_created
  ON paper_run_map(paper_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_run_map_one_latest
  ON paper_run_map(paper_id, lib_config_id)
  WHERE is_latest = 1;

-- 3) Selector dimension (DB layer selectors)
CREATE TABLE IF NOT EXISTS chunk_selector (
  selector_id     INTEGER PRIMARY KEY,
  name            TEXT NOT NULL UNIQUE,         -- e.g. summary/introduction/method/...
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_chunk_selector_name
  ON chunk_selector(name);

-- 4) Chunk text store: pure text + metadata (source of truth for content)
CREATE TABLE IF NOT EXISTS chunk_text (
  run_id          INTEGER NOT NULL,
  text_id         TEXT NOT NULL,
  text            TEXT NOT NULL,
  char_count      INTEGER NOT NULL,
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  PRIMARY KEY(run_id, text_id),
  FOREIGN KEY(run_id) REFERENCES paper_run_map(run_id) ON DELETE CASCADE,
  CHECK (char_count >= 0)
);

-- 5) Mapping table: ties a run+selector to a text_id. Uses surrogate map_id for tight downstream binding.
CREATE TABLE IF NOT EXISTS paper_chunk_map (
  map_id          INTEGER PRIMARY KEY,
  run_id          INTEGER NOT NULL,
  selector_id     INTEGER NOT NULL,
  text_id         TEXT NOT NULL,
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  FOREIGN KEY(run_id)      REFERENCES paper_run_map(run_id) ON DELETE CASCADE,
  FOREIGN KEY(selector_id) REFERENCES chunk_selector(selector_id) ON DELETE RESTRICT,
  FOREIGN KEY(run_id, text_id) REFERENCES chunk_text(run_id, text_id) ON DELETE CASCADE,
  -- natural uniqueness for correctness:
  UNIQUE(run_id, selector_id, text_id)
);

CREATE INDEX IF NOT EXISTS idx_pcm_run_selector
  ON paper_chunk_map(run_id, selector_id);

CREATE INDEX IF NOT EXISTS idx_pcm_run_text
  ON paper_chunk_map(run_id, text_id);

CREATE INDEX IF NOT EXISTS idx_pcm_text
  ON paper_chunk_map(text_id);

-- 6) Selector -> texts scoring table (only scoring table in MVP)
CREATE TABLE IF NOT EXISTS selector_texts_score (
  map_id          INTEGER PRIMARY KEY,
  score           REAL NOT NULL,                -- e.g. 1.0 or 1/N, optionally normalized within selector
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  FOREIGN KEY(map_id) REFERENCES paper_chunk_map(map_id) ON DELETE CASCADE
);