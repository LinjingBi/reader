# TODO (Step 3–6 design notes)

This repo implements only Reader Step 1–2 for MVP:

- `memo fresh-paper` (write): atomic ingest of (snapshot + papers + best clustering)
- `memo get-best-run` (read): fetch best clustering for a snapshot to build the Step 3 LLM prompt

Planned next steps (not implemented):

## Step 3 (Reader): LLM enrichment (outside memo)
- Reader prompts an LLM to produce human-readable cluster cards (name, summary, labels).
- Only the user-selected card is persisted later as an observation.

## Step 4: User chooses a cluster; topic add vs create
New command candidates:
- `memo topic-observe --input <json|->`
  - Input: topic_id (optional), cluster_id, proposed_name/summary/labels, llm_config_id (optional).
  - Write: topic_observation, topic_cluster_link.
- `memo topic-create --input <json|->` (explicit create; no auto merge for MVP)

## Step 5: Report generation persistence
- `memo report-write --input <json|->`
  - Input: topic_id, period, report_md, llm_config_id.
  - Write: report, report_topic_link.

## Step 6: Topic evolution (merge/split/rename)
- `memo topic-event --input <json|->`
  - Input: event_type, affected_topic_ids, proposal_text, llm_config_id.
  - Write: topic_event, topic_lineage, topic canonical updates.

Notes:
- For MVP, prefer deterministic text IDs and idempotent upserts.
- Keep evolution as a separate workflow and do not run automatically.
