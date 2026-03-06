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

CREATE TABLE IF NOT EXISTS topic_resolver_config (
  topic_resolver_config_id TEXT PRIMARY KEY,
  json_payload             TEXT NOT NULL,  -- algo name, params, threshold, etc.
  created_at               TEXT NOT NULL
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
-- -----------------------------
-- Topics (language-first canonical objects)
-- -----------------------------

CREATE TABLE IF NOT EXISTS topic (
  topic_id           INTEGER PRIMARY KEY,   -- rowid-backed, auto assigns
  canonical_name     TEXT NOT NULL,
  canonical_summary  TEXT NOT NULL,
  labels_json        TEXT NOT NULL,  -- JSON list
  status             TEXT NOT NULL,  -- 'active' | 'merged' | 'deprecated'
  created_at         TEXT NOT NULL,
  updated_at         TEXT NOT NULL,
  centroid_b64       TEXT NOT NULL,  -- base64 float32 bytes (same dim as cluster centroid). normalized
  centroid_weight    REAL NOT NULL,  -- centroid weight used in cluster centroid computation
  centroid_updated_at TEXT           -- ISO timestamp when centroid last updated
);

-- Link clusters (from runs) to topics; captures add/create decision + matching provenance
CREATE TABLE IF NOT EXISTS topic_cluster_link (
  topic_id                 INTEGER NOT NULL,
  cluster_pk_hash           TEXT NOT NULL UNIQUE,
  topic_resolver_config_id TEXT,  -- config used for resolution (nullable)
  decision                 TEXT NOT NULL,  -- 'created' | 'merged'
  match_score              REAL NOT NULL,  -- cosine sim between topic vector and cluster/topic card
  created_at               TEXT NOT NULL,
  PRIMARY KEY (topic_id, cluster_pk_hash),
  FOREIGN KEY (topic_id)                 REFERENCES topic(topic_id)                 ON DELETE CASCADE,
  FOREIGN KEY (cluster_pk_hash)         REFERENCES cluster(pk_hash)                 ON DELETE CASCADE,
  FOREIGN KEY (topic_resolver_config_id) REFERENCES topic_resolver_config(topic_resolver_config_id) ON DELETE SET NULL
);

-- TBD for evlution pipeline
-- Topic events: canonical changes (rename/merge/split) with provenance
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

-- TBD for evlution pipeline
-- Lineage edges to preserve history without rewriting old links
CREATE TABLE IF NOT EXISTS topic_lineage (
  topic_id          INTEGER NOT NULL, -- current/survivor topic
  ancestor_topic_id INTEGER NOT NULL, -- predecessor
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
  report_id        INTEGER PRIMARY KEY,   -- rowid-backed, auto assigns
  cluster_pk_hash  TEXT UNIQUE,                        -- nullable reference to cluster
  created_at       TEXT NOT NULL,
  intent_mode      TEXT NOT NULL,                    -- quick_background|research_briefing|brainstorm_directions|implementation_angle
  report_url       TEXT NOT NULL,
  -- report summary-level fields
  title            TEXT NOT NULL,
  summary          TEXT NOT NULL,                    -- 80-120 words target (enforce in app)
  keywords_json    TEXT NOT NULL DEFAULT '[]',        -- JSON list of strings
  signature        TEXT NOT NULL,
  -- Fields from report plan (not actual report content)
  declared_level   TEXT NOT NULL,                    -- intro|intermediate|deep-dive
  depth_mode       TEXT NOT NULL,                      -- Onboard|Continue|Deepen|Restructure
  covered_bullets  TEXT NOT NULL,                      -- JSON array of strings
  next_targets     TEXT NOT NULL,                      -- JSON array of strings
  subthreads       TEXT NOT NULL,                      -- JSON array of {name, paper_ids}
  outline          TEXT NOT NULL,                      -- JSON array of strings
  sufficiency      TEXT NOT NULL,                      -- sufficient|borderline|insufficient
  plan             TEXT NOT NULL,                      -- JSON object (LLMReportPlannerPlan)
  --
  FOREIGN KEY (cluster_pk_hash) REFERENCES cluster(pk_hash) ON DELETE SET NULL
);

-- Reports can link to multiple topics (primary/secondary/related)
CREATE TABLE IF NOT EXISTS report_topic_link (
  report_id      INTEGER NOT NULL,
  topic_id       INTEGER NOT NULL,
  role           TEXT NOT NULL,   -- 'primary'|'secondary'|'related'
  created_at     TEXT NOT NULL,
  PRIMARY KEY (report_id, topic_id),
  FOREIGN KEY (report_id) REFERENCES report(report_id) ON DELETE CASCADE,
  FOREIGN KEY (topic_id)  REFERENCES topic(topic_id)  ON DELETE CASCADE
);

-- TBD for evlution pipeline
-- Depth annotations: editorial signal from LLM, grounded by objective stats
-- Objective stats are computed from report_topic_link + report.
CREATE TABLE IF NOT EXISTS topic_depth_annotation (
  depth_id        TEXT PRIMARY KEY,
  topic_id        INTEGER NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_report_topic_role ON report_topic_link(topic_id, role);
CREATE INDEX IF NOT EXISTS idx_topic_status ON topic(status);

-- Performance indexes for fresh_paper operations
CREATE INDEX IF NOT EXISTS idx_cluster_run_snapshot_role ON cluster_run(source, period_start, period_end, role);
CREATE INDEX IF NOT EXISTS idx_cluster_cluster_run ON cluster(source, period_start, period_end, embed_config_id, cluster_config_id, role);
-- Note: pk_hash has UNIQUE constraint which automatically creates an index

-- Partial unique index to enforce single best run per snapshot+role
CREATE UNIQUE INDEX IF NOT EXISTS ux_cluster_run_best 
  ON cluster_run(source, period_start, period_end, role) 
  WHERE selected_best=1;

-- Faster "latest report per topic" + coverage counts
CREATE INDEX IF NOT EXISTS idx_report_created_at ON report(created_at);
CREATE INDEX IF NOT EXISTS idx_report_intent_mode ON report(intent_mode);
CREATE INDEX IF NOT EXISTS idx_report_declared_level ON report(declared_level);

-- topic_cluster_link queries (find all clusters attached to a topic / find match_score distributions)
CREATE INDEX IF NOT EXISTS idx_topic_cluster_link_topic
  ON topic_cluster_link(topic_id);

CREATE INDEX IF NOT EXISTS idx_topic_cluster_link_match_score
  ON topic_cluster_link(match_score);

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