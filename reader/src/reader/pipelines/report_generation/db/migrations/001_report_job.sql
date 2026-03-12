CREATE TABLE IF NOT EXISTS report_job (
  cluster_pk_hash TEXT PRIMARY KEY,
  status          TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);
