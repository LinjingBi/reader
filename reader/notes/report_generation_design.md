# Reader-Memory Report Generation Design Doc (MVP → History‑Aware)

**Scope:** This document captures the decisions and architecture for (1) **topic merge/create** and (2) **two-call report generation**. It includes MVP-first defaults and the future history-aware version.

**Key principles**
- **Geometry-first for matching:** merge/create decisions are driven by **cluster centroids** in a specific `embed_config_id` space.
- **Language-first for topics:** topic canonical semantics are **stable** and only updated by the **evolution pipeline**.
- **MVP bias:** prefer **creating new topics** over risky merges; refine later via evolution (merge/split/rename).
- **Client-side storage:** DB is an index + metadata store; full report content can live on local filesystem.

---

## 0) Objects and invariants

### Cluster geometry
- Each `cluster` may store a centroid vector: `cluster.centroid_b64` (base64 float32 bytes) and optional `cluster.cohesion`.
- Each cluster is tied to an embedding space via `cluster.embed_config_id`.
- Cluster membership and representativeness are captured by `cluster_member.rank_in_cluster`, `cluster_member.sim_to_centroid`.

### Topics (canonical semantics)
- `topic` holds canonical semantic fields:
  - `canonical_name`, `canonical_summary`, `labels_json`, `status`
- Canonical semantic fields **must not be mutated** during attachment/matching, except at topic creation.
- After creation, canonical semantic updates happen **only** via evolution pipeline (`topic_event`, `topic_lineage`) with approval.

### Topic observations (per-period semantic snapshots)
- `topic_observation` stores **proposed** name/summary/labels for a specific period+cluster attachment.
- These are “semantic snapshots” of what the period’s cluster *meant*, without changing canonical topic semantics.

### Reports
- `report` is produced for a specific `(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index)`.
- Reports link to topics via `report_topic_link` (primary/secondary/related).

---

## 1) Topic resolution: merge vs create (geometry-driven)

### 1.1 Inputs
Given a **user-chosen cluster** identified by:
- `(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index)`
and data:
- `cluster.centroid_b64` (required for geometry matching; if NULL, fallback is text embedding)
- `cluster.size` (for weighted centroid updates, optional but recommended)

Candidate topic set:
- Topics that share the same `embed_config_id` (once topics record their embed config).

### 1.2 Topic centroid geometry (source of truth for matching)
All geometry comparisons are in the same `embed_config_id`.

- **New topic:** topic centroid = chosen cluster centroid.
- **Existing topic:** topic centroid = **average of attached clusters’ centroids** (observations are clusters),
  ideally weighted by cluster size.

> Geometry thresholding should use `cluster` centroids.  
> For existing topics, centroid is a function of attached cluster centroids; for new topics, centroid is exactly the chosen cluster centroid.

### 1.3 Similarity metric
Primary metric:
- `sim = cosine(topic.topic_centroid_b64, cluster.centroid_b64)`

Optional later:
- secondary lexical overlap gate (labels/keywords), or an alternate embedding config.

### 1.4 Decision policy (MVP conservative)
Let `t*` be the nearest topic by cosine similarity (top-1 pick).

- If `sim(t*, cluster) >= T_high` → **matched_existing**
- Else → **created_new**

MVP philosophy:
- **Prefer create** early to avoid contaminating canonical topics with accidental merges.
- Evolution pipeline later can merge/split/rename based on richer evidence.

### 1.5 “Lazy topic embedding” threshold calibration (initial T_high)
Goal: get a reasonable starting point for `T_high` using your existing 12 months of HF data.

**Lazy calibration idea:** treat clusters as pseudo-topics and measure nearest-neighbor similarity.
1) Collect all clusters across the 12 months for a fixed `embed_config_id`.
2) For each cluster `c`, compute the max cosine similarity to clusters from prior months (or all other clusters excluding itself).
3) This yields a similarity distribution with a “high tail” of very similar pairs.
4) Choose a conservative starting threshold:
   - `T_high` near where the “obvious matches” begin (e.g., a high percentile of cross-cluster similarities).
5) Log monthly nearest-topic similarities once pipeline runs, then adjust `T_high`.

**Text-based fallback (if centroids are missing):**
- Embed `cluster_title + summary + keyword_list` using the exact same embedding model + fixed text template;
- run the same nearest-neighbor distribution analysis;
- use the resulting scale only for that `embed_config_id` and text template.

### 1.6 Merge/create write actions (canonical semantics rule)

#### Case A: `matched_existing` (merge / attach to an existing topic)
**Do:**
- Insert `topic_cluster_link`:
  - `decision='matched_existing'`
  - `match_score = sim` (cosine topic centroid vs cluster centroid)
- Insert `topic_observation`:
  - `proposed_name`, `proposed_summary`, `proposed_labels_json`
  - `produced_by='llm'` (or human), `llm_config_id` if applicable
- Optionally update topic geometry centroid (recommended):
  - update `topic.topic_centroid_b64` (incremental mean)
  - update `topic.topic_centroid_updated_at`
  - set `topic.embed_config_id` if currently NULL (otherwise must match)

**Do NOT:**
- Do **not** modify canonical topic semantics:
  - `topic.canonical_name`, `topic.canonical_summary`, `topic.labels_json`

#### Case B: `created_new`
**Do:**
- Create `topic` canonical semantics **once** at birth:
  - `canonical_name/summary/labels_json` from cluster observation or a topic-card generation step
- Initialize topic geometry centroid:
  - `topic.embed_config_id = cluster.embed_config_id`
  - `topic.topic_centroid_b64 = cluster.centroid_b64`
  - `topic.topic_centroid_updated_at = now`
- Insert `topic_cluster_link` (`decision='created_new'`, `match_score=NULL or 1.0` depending on convention)
- Insert `topic_observation` (same as above)

---

## 2) Report generation: 2-call LLM pipeline

### Why 2 calls
- **Call 1 (Planner):** commits to structure (subthreads/outline), sets “depth intent”, and flags evidence gaps.
- **Call 2 (Writer):** writes one-shot following the plan, minimizing ramble and drift.

This structure improves stability without requiring tool use in MVP.

---

## 3) Call 1: Report Planner

### 3.1 MVP planner (little history, intent-driven)

#### Inputs
1) **User intent**
- `intent_mode` enum:
  - `quick_background` (5–10 min)
  - `research_briefing` (decision-oriented)
  - `brainstorm_directions` (novelty hunting)
  - `implementation_angle` (what to build/test)
- Optional: `user_intent_note`

2) **Cluster evidence pack**
- Optional: `cluster_observation.payload_json` (compact thematic summary if already available)
- **Top papers: practical decision rule**
  - Include **Top 3 papers** with full fields:
    - `paper_id`, `title`, `summary`, `keywords_json`, `url` (optional)
    - `rank_in_cluster`, `sim_to_centroid`
  - For remaining papers (optional):
    - include only `paper_id`, `title`, and a **1-line** summary/snippet
- Cluster metadata:
  - `size`, `cohesion` (if stored), period identifiers

3) **Existing reports metadata (if any)** — compressed only
For the primary topic (if already known), include last `N=2` report metadata:
- `title`, `summary`, `keywords_json`
- `covered_bullets_json`, `next_targets_json`
- `intent_mode`, `declared_level`, `created_at`

> MVP note: do not include full `report_md` in Call 1.

#### Outputs
Planner returns JSON:
- `plan_for_this_report`
  - `depth_mode_final`: `Onboard|Continue|Deepen|Restructure`
  - `declared_level_final`: `intro|intermediate|deep-dive`
  - `subthreads_final`: 2–4 `{name, paper_ids[]}` (binding structure)
  - `next_targets`: 3–8 bullets
  - `outline`: 6–12 bullets
  - `skip_or_defer`: 0–5 bullets (avoid repeats)
- `evidence_request_for_call_2` (no rerun; optional)
  - `sufficiency`: `sufficient|borderline|insufficient`
  - minimal requests (<=3 papers, <=2 history items)
  - fallback behavior if unavailable

#### Notes on “subthreads”
- Subthreads are a **report-structure decision**, so final subthreads should be produced by the **planner/writer** (strong model), not by a weaker summarizer.
- If you do compute subthreads earlier, treat them as **non-binding suggestions**.

---

### 3.2 Future planner (history assessment enabled)
When history becomes rich, planner additionally receives objective signals derived from DB:

- **Coverage count:** `n_reports(topic)` and/or `n_observations(topic)`
- **Recency:** days since last report/observation
- **Continuity:** consecutive periods the topic appears
- **Breadth proxy:** distinct sublabels/keywords across observations
- **Stability:** cosine statistics of topic centroid vs attached cluster centroids (all-time and recent window)

Planner then outputs:
- a `history_assessment` section (stable/drifting, what already covered, skip)
- `depth_mode_default` and reasons from history
- `depth_mode_final` after seeing current cluster evidence (may override default)

> Objective stability/cohesion is computed from embeddings; LLM interprets it, doesn’t recompute it from text.

---

## 4) Call 2: Report Writer

### 4.1 MVP writer (one-shot writing)

#### Inputs
- Planner JSON from Call 1
- Cluster evidence pack (same as planner)
- Optional supplements (if you choose to provide them), e.g.:
  - additional paper fields, extra report metadata (still bounded)

#### Outputs
Single JSON:
- `depth_context` (for storage)
  - `intent_mode`, `user_intent_note`
  - `depth_mode`, `declared_level`
  - `subthreads_json`, `covered_bullets_json`, `next_targets_json`, `skip_or_defer`
  - optional `evidence_gaps_json`, `cohesion_label/confidence`
- `report`
  - `title`, `summary`, `keywords_json`, `content_md`

#### Writing rules
- Use only provided evidence; cite papers as `[paper_id]`.
- Do not invent experiments/results; if missing, explicitly mark as “not provided”.

### 4.2 Future writer (history-aware writing)
Writer additionally uses:
- richer objective history signals (or selected excerpts) to avoid repetition and steer depth
- same subthread-driven structure and intent mode

---

## 5) Report storage strategy: DB vs local filesystem

### Option A: Store full report in DB (current schema)
Pros:
- Easy full-text search and retrieval
- Single source of truth

Cons:
- DB grows quickly
- Harder to manage file-like workflows and Git backup patterns

### Option B (recommended for client-side MVP): Store full report on local filesystem
DB stores:
- **metadata** (title/summary/keywords, intent, bullets, subthreads)
- **pointer** to full report file (path/url)
- optional hash for integrity

Local FS stores:
- full `report.md`

Recommended hybrid for migration simplicity:
- Keep `report.report_md` but allow it to store either:
  - full markdown, or
  - a small stub / excerpt
- Add `report_path` and related fields now; gradually shift “full content” to FS.

---

## 6) Write-back policy (MVP)

### MUST write (always)
- `report`:
  - provenance keys: `(source, period_start, period_end, embed_config_id, cluster_config_id, role, cluster_index)`
  - `created_at`, `llm_config_id` (if available)
  - **Either** `report_md` (full or stub) **or** `report_path` (preferred if moving to FS)
- `report_topic_link`: at least one row with `role='primary'`
- `topic_cluster_link` and `topic_observation` for the chosen cluster/topic decision

### SHOULD write (best effort; default if missing)
- `report.title` (fallback: derive from first H1 or “Topic report {cluster_index}”)
- `report.summary` (fallback: NULL or later backfill)
- `report.keywords_json` (fallback: `[]`)
- `report.intent_mode` (fallback: `research_briefing`)
- `report.declared_level` (fallback: `intermediate`)
- `report.covered_bullets_json`, `next_targets_json`, `subthreads_json` (fallback: `[]`)

### OPTIONAL
- `report.cohesion_label`, `cohesion_confidence`, `evidence_gaps_json`

### Topic centroid write-back
- `topic.topic_centroid_b64` may be NULL in early MVP.
- When present:
  - update geometry only on attachments (never canonical semantics)
  - keep `topic.embed_config_id` consistent

---

## 7) SQLite migration: ALTER statements (topic/report-related tables only)

> These are additive. SQLite does not support dropping columns via simple ALTER.

```sql
-- -----------------------------
-- MIGRATION: topic centroid + report metadata + report file pointer
-- (tables starting with topic / report only)
-- -----------------------------

-- Topic geometry centroid snapshot (optional now, enables stability later)
ALTER TABLE topic ADD COLUMN embed_config_id TEXT;
ALTER TABLE topic ADD COLUMN topic_centroid_b64 TEXT;           -- base64 float32 bytes
ALTER TABLE topic ADD COLUMN topic_centroid_updated_at TEXT;    -- ISO timestamp

-- Report compressed metadata (for history prompts without full text)
ALTER TABLE report ADD COLUMN title TEXT;
ALTER TABLE report ADD COLUMN summary TEXT;                     -- 80-120 words target
ALTER TABLE report ADD COLUMN keywords_json TEXT;               -- JSON list

ALTER TABLE report ADD COLUMN intent_mode TEXT;                 -- quick_background|research_briefing|brainstorm_directions|implementation_angle
ALTER TABLE report ADD COLUMN user_intent_note TEXT;
ALTER TABLE report ADD COLUMN declared_level TEXT;              -- intro|intermediate|deep-dive

ALTER TABLE report ADD COLUMN covered_bullets_json TEXT;        -- JSON list
ALTER TABLE report ADD COLUMN next_targets_json TEXT;           -- JSON list
ALTER TABLE report ADD COLUMN subthreads_json TEXT;             -- JSON list of {name, paper_ids:[...]}

-- Optional audit signals
ALTER TABLE report ADD COLUMN cohesion_label TEXT;              -- cohesive|mixed
ALTER TABLE report ADD COLUMN cohesion_confidence REAL;         -- 0..1
ALTER TABLE report ADD COLUMN evidence_gaps_json TEXT;          -- JSON list

-- Report storage pointer (for local filesystem storage)
ALTER TABLE report ADD COLUMN report_path TEXT;                 -- local path or file:// url
ALTER TABLE report ADD COLUMN report_sha256 TEXT;               -- optional integrity
ALTER TABLE report ADD COLUMN report_size_bytes INTEGER;        -- optional

-- Optional extra match score (future two-gate matching)
ALTER TABLE topic_cluster_link ADD COLUMN match_score_alt REAL;

-- Helpful indexes (topic/report only)
CREATE INDEX IF NOT EXISTS idx_report_created_at ON report(created_at);
CREATE INDEX IF NOT EXISTS idx_report_intent_mode ON report(intent_mode);
CREATE INDEX IF NOT EXISTS idx_report_declared_level ON report(declared_level);

CREATE INDEX IF NOT EXISTS idx_topic_observation_topic_period
  ON topic_observation(topic_id, period_start, period_end);

CREATE INDEX IF NOT EXISTS idx_topic_cluster_link_topic
  ON topic_cluster_link(topic_id);

CREATE INDEX IF NOT EXISTS idx_topic_cluster_link_match_score
  ON topic_cluster_link(match_score);
```

---

## 8) Open questions (tracked for later)
- Should topic centroid weight be stored as a field on `topic` (for incremental update), or computed from linked cluster sizes?
- Do you want a “gray zone” for similarity (log candidates) even if MVP always creates new topics in that zone?
- Whether to store any additional paper-enrichment cards (later: methods/experiments/limitations) as separate artifacts.

---
