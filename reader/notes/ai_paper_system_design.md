# AI Paper Curation System — Architecture v0

## Goal
Help a human track emerging AI research topics by:
- clustering recent papers
- assigning topics
- letting the human choose a cluster
- writing a depth-aware report

---

## Data Sources
Primary:
- Hugging Face `daily-papers` (supports daily/weekly/monthly)

Future optional:
- arXiv keyword expansion
- Corporate blogs / industry trackers

---

## Main Workflow (Stable Loop)

### 1) Fetch + Preprocess
For each new paper:
- title + abstract
- HF mini-summary (AI-provided)
- optional intro/contribution text (first N pages if scraped)
- embed text → vector

Store in DB table `papers`.

### 2) Cluster + Rank
- Cluster new embeddings (e.g., KMeans/HDBSCAN)
- Compute novelty score:
  - distance to cluster centroid
  - optional dense summary contrast
  - optional claimed-contribution score
- Rank papers within cluster

### 3) LLM Topic Enrichment
For each cluster:
- propose topic name
- describe human-understandable theme
- provide key papers + hooks

### 4) Human Cluster Choice
User selects a **cluster**, not individual paper.

### 5) Topic Matching
Compare cluster centroid to stored topics:
```
if distance < threshold:
    attach to existing topic (depth += 1)
else:
    create new topic (depth = 0)
```
Store:
- topic_id
- depth
- linked papers
- centroid snapshot

### 6) Depth-Aware Reporting
LLM reads:
- current cluster
- past reports (if topic exists)
- depth instructions

Output:
- new report (first iteration)
- OR deeper continuation (history-aware)

Store in DB `reports`.

---

## Topic Maintenance (Scheduled Task)
Not required for MVP, but design must support it.

Monthly or quarterly procedure:
1. Load **all** topic embeddings + papers
2. Re-cluster globally
3. Detect merge / split / rename candidates
4. LLM proposes restructuring
5. Human approves
6. DB update:
   - mark retired topics
   - spawn new topics when splitting

Schema must store:
- topic centroid embedding
- paper embeddings per topic
- status (active / retired / merged)

---

## Database Tables (MVP)
### `papers`
paper_id, title, abstract, source_url, embedding

### `topics`
topic_id, name, created_at, status, depth_level, centroid_embedding, descritpion

### `reports`
report_id, topic_id, month, report_text, summary_text, depth_snapshot

### `report_papers`
report_id, paper_id (join)

---

## Summarization Plan
### Lightweight summarization (MVP)
Use HF summarization pipeline on:
- abstract + first N pages (scraped)
Models suggested (small → large):
- `sshleifer/distilbart-cnn-12-6`
- `facebook/bart-large-cnn`
- `google/pegasus-cnn_dailymail` (longer summarization)

Dense summary feeds:
- clustering semantic boost
- novelty scoring

No full LLM needed in V0.

---

## Quick Prototype Idea
Run weekly/monthly HF pulls over 12 months:
- embed new papers
- cluster per period
- count clusters + size distribution
- visualize:
  - # clusters vs time
  - cluster drift (centroid distance)
Helps pick:
- best time granularity (weekly/monthly)
- thresholds for topic matching

---

## Future Extensions
- auto section selection (intro/method/contrib/experiments)
- arXiv enrichment pipeline
- integration with industry news
- vector DB retrieval for long-term memory
