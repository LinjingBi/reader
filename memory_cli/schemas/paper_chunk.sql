PRAGMA foreign_keys = ON;

-- 1) Which chunker / extractor config produced the chunks
CREATE TABLE IF NOT EXISTS chunk_lib_config (
  lib_config_id   TEXT PRIMARY KEY,
  json_payload    TEXT NOT NULL DEFAULT '{}',

  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))

  -- Optional if you have SQLite JSON1 enabled:
  -- , CHECK (json_valid(json_payload))
);


-- 2) One run per paper per lib version (you can allow multiple reruns)
CREATE TABLE IF NOT EXISTS paper_chunk_run (
  run_id          INTEGER PRIMARY KEY,
  paper_id        TEXT NOT NULL,                -- FK to paper
  lib_config_id   TEXT NOT NULL,                -- FK to chunk_lib_config

  UNIQUE(paper_id, lib_config_id),

  status          TEXT NOT NULL DEFAULT 'ok',   -- ok|partial|error (or your enum)
  is_latest       INTEGER NOT NULL DEFAULT 0,   -- 0/1

  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),

  FOREIGN KEY(paper_id)      REFERENCES paper(paper_id) ON DELETE CASCADE,
  FOREIGN KEY(lib_config_id) REFERENCES chunk_lib_config(lib_config_id) ON DELETE RESTRICT,

  CHECK (status IN ('ok','partial','error')),
  CHECK (is_latest IN (0,1))
);

-- Fast “latest by default” fetches:
CREATE INDEX IF NOT EXISTS idx_paper_chunk_run_latest
  ON paper_chunk_run(paper_id, lib_config_id, is_latest);

CREATE INDEX IF NOT EXISTS idx_paper_chunk_run_paper_created
  ON paper_chunk_run(paper_id, created_at DESC);

-- Enforce at most ONE latest run per (paper_id, lib_config_id):
CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_chunk_run_one_latest
  ON paper_chunk_run(paper_id, lib_config_id)
  WHERE is_latest = 1;


-- 3) Selector dimension (DB layer selectors)
CREATE TABLE IF NOT EXISTS chunk_selector (
  selector_id     INTEGER PRIMARY KEY,
  name            TEXT NOT NULL UNIQUE,         -- e.g. summary/introduction/method/...
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_chunk_selector_name
  ON chunk_selector(name);


-- 4) The actual chunk text per selector per run
CREATE TABLE IF NOT EXISTS paper_chunk (
  run_id          INTEGER NOT NULL,
  selector_id     INTEGER NOT NULL,
  text            TEXT NOT NULL,
  char_count      INTEGER NOT NULL DEFAULT 0,

  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),

  PRIMARY KEY(run_id, selector_id),

  FOREIGN KEY(run_id)      REFERENCES paper_chunk_run(run_id) ON DELETE CASCADE,
  FOREIGN KEY(selector_id) REFERENCES chunk_selector(selector_id) ON DELETE RESTRICT,

  CHECK (char_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_paper_chunk_selector
  ON paper_chunk(selector_id);

-- Optional: if you frequently fetch all selectors for a run quickly, PK already helps.
