# Report Generation Validation Rules

This document lists all hard and soft validation rules for report generation outputs, indicating whether each is enforced in the **model** (Pydantic schema) or in **metrics** (validation functions).

---

## 1. Planner — LLMReportPlannerOutput

### Model constraints (report.py)


| Field                         | Constraint                         | Enforced in |
| ----------------------------- | ---------------------------------- | ----------- |
| plan.subthreads_final         | min_length=2, max_length=4         | model       |
| plan.next_targets             | min_length=3, max_length=8         | model       |
| plan.outline                  | min_length=6, max_length=12        | model       |
| plan.skip_or_defer            | max_length=5                       | model       |
| plan.depth_mode_final         | LLMReportPlannerDepthMode enum     | model       |
| plan.declared_level_final     | LLMReportPlannerDeclaredLevel enum | model       |
| plan.sufficiency              | LLMReportPlannerSufficiency enum   | model       |
| EvidenceGap.why               | required (non-empty)               | model       |
| EvidenceGap.priority          | Literal[1, 2, 3]                   | model       |
| EvidenceGap.paper_selectors   | List[PaperSelector]                | model       |
| EvidenceGap.history_selectors | List[HistoryReportSelector]        | model       |


### Hard rules (metrics)


| Rule                                 | Description                                                                                 | Enforced in |
| ------------------------------------ | ------------------------------------------------------------------------------------------- | ----------- |
| EvidenceGap target id                | Exactly one of paper_id or history_report_id set (XOR)                                      | metrics     |
| EvidenceGap selector exclusivity     | paper_id → paper_selectors non-empty, history_selectors empty; history_report_id → opposite | metrics     |
| Subthread name non-empty             | subthreads_final[].name not empty or whitespace-only                                        | metrics     |
| next_targets non-empty               | Items not empty or whitespace-only                                                          | metrics     |
| outline non-empty                    | Items not empty or whitespace-only                                                          | metrics     |
| skip_or_defer non-empty              | Items not empty or whitespace-only                                                          | metrics     |
| EvidenceGap.why non-empty            | Not empty or whitespace-only                                                                | metrics     |
| EvidenceGap.blocked_fields non-empty | Items not empty or whitespace-only                                                          | metrics     |
| Subthread names unique               | Unique (case-insensitive)                                                                   | metrics     |
| outline unique                       | Items unique                                                                                | metrics     |
| next_targets unique                  | Items unique                                                                                | metrics     |


### Soft rules (metrics)


| Rule                                       | Description                                      | Enforced in |
| ------------------------------------------ | ------------------------------------------------ | ----------- |
| EvidenceGap.why word count                 | 6–35 words                                       | metrics     |
| outline headings word count                | ≤18 words per item                               | metrics     |
| outline headings single line               | No newlines                                      | metrics     |
| outline headings no trailing period        | Must not end with `.`                            | metrics     |
| next_targets actionable word count         | 3–16 words                                       | metrics     |
| next_targets actionable single line        | No newlines                                      | metrics     |
| next_targets actionable no trailing period | Must not end with `.`                            | metrics     |
| next_targets no narrative phrasing         | Must not start with "In this report we will..."  | metrics     |
| skip_or_defer no overlap                   | Must not overlap next_targets (case-insensitive) | metrics     |
| Subthread name specific                    | Not generic placeholder (misc, others, etc.)     | metrics     |
| Subthread paper_id duplication             | No excessive reuse across subthreads             | metrics     |
| Sufficiency + evidence gaps minimal        | If sufficient, ≤3 evidence_gaps                  | metrics     |
| Sufficiency insufficient + non-empty       | If insufficient, evidence_gaps non-empty         | metrics     |
| EvidenceGap.blocked_fields known           | Must reference known SupportField                | metrics     |
| EvidenceGap target id unique               | Each paper_id/history_report_id at most once     | metrics     |
| outline distribution sanity                | No repetitive first-word pattern                 | metrics     |
| next_targets distribution sanity           | No repetitive first-word pattern                 | metrics     |


---

## 2. Writer Supply — ReportWriterSupplementOutput (W1)

### Model constraints (report.py)


| Field                                     | Constraint                  | Enforced in |
| ----------------------------------------- | --------------------------- | ----------- |
| supplements_requests                      | max_length=10 (W1-H5)       | model       |
| WriterSupplementRequest.why               | required                    | model       |
| WriterSupplementRequest.paper_selectors   | List[PaperSelector]         | model       |
| WriterSupplementRequest.history_selectors | List[HistoryReportSelector] | model       |


### Hard rules (metrics)


| Rule  | Description                                                                        | Enforced in |
| ----- | ---------------------------------------------------------------------------------- | ----------- |
| W1-H1 | Exactly one of paper_id or history_report_id set per request                       | metrics     |
| W1-H2 | Selector exclusivity matches target kind                                           | metrics     |
| W1-H3 | paper_id in available_paper_ids; history_report_id in available_history_report_ids | metrics     |
| W1-H4 | why non-empty; no empty selector strings                                           | metrics     |


### Soft rules (metrics)


| Rule  | Description                                     | Enforced in |
| ----- | ----------------------------------------------- | ----------- |
| W1-S1 | why phrased as question (? or question starter) | metrics     |
| W1-S2 | why 6–25 words                                  | metrics     |
| W1-S3 | Prefer 0–3 requests; penalize >3, strongly >6   | metrics     |
| W1-S5 | No duplicate (target_id, selectors) requests    | metrics     |


---

## 3. Writer Writing — ReportWriterSectionOutput (W2)

### Model constraints (report.py)


| Field        | Constraint                                     | Enforced in |
| ------------ | ---------------------------------------------- | ----------- |
| section_name | required                                       | model       |
| section_text | required                                       | model       |
| confidence   | required, List[Literal["high","medium","low"]] | model       |


### Hard rules (metrics)


| Rule  | Description                                                                  | Enforced in |
| ----- | ---------------------------------------------------------------------------- | ----------- |
| W2-H1 | section_name aligns with outline_item (substring/token overlap/equality)     | metrics     |
| W2-H2 | section_text > 80 chars after strip                                          | metrics     |
| W2-H3 | Every citation token in section_text must be in allowed_citations            | metrics     |
| W2-H4 | No raw unknown IDs (paper_id=, report_id=, P123, R7) outside citation tokens | metrics     |
| W2-H5 | confidence non-empty, all in {high,medium,low}, len==1                       | metrics     |


### Soft rules (metrics)


| Rule  | Description                                                | Enforced in |
| ----- | ---------------------------------------------------------- | ----------- |
| W2-S1 | ≥1 allowed citation per paragraph                          | metrics     |
| W2-S2 | No over-claiming language (SOTA, beats, outperforms, etc.) | metrics     |
| W2-S5 | Word count soft limit (500 words)                          | metrics     |


---

## 4. Writer Front Matter — ReportWriterFrontMatterOutput (W4 / D-H / D-S)

### Model constraints (report.py)


| Field    | Constraint                            | Enforced in |
| -------- | ------------------------------------- | ----------- |
| title    | min_length=5, max_length=120          | model       |
| summary  | min_length=40, max_length=1200        | model       |
| keywords | Set[str], min_length=5, max_length=12 | model       |


### Rules enforced in model


| Rule                                         | Description                                         | Enforced in    |
| -------------------------------------------- | --------------------------------------------------- | -------------- |
| W4-H1 Keywords uniqueness (case-insensitive) | model_validator rejects case-insensitive duplicates | model, metrics |


### Hard rules (metrics)


| Rule | Description                                                                                           | Enforced in |
| ---- | ----------------------------------------------------------------------------------------------------- | ----------- |
| D-H3 | Keywords count must be 5–12                                                                           | metrics     |
| D-H4 | Keywords must be case-insensitively unique                                                            | metrics     |
| D-H6 | Title must not be in banned set (technical report, research summary, weekly report, overview, report) | metrics     |


### Soft rules (metrics)


| Rule | Description                                                                                                                         | Enforced in |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| D-H1 | Title: 5–120 chars, non-empty after strip                                                                                           | metrics     |
| D-H2 | Summary: 40–1200 chars, non-empty after strip                                                                                       | metrics     |
| D-H5 | Each keyword: non-empty, ≤40 chars, ≥1 alphanumeric                                                                                 | metrics     |
| D-S1 | Title specific; avoid vague words (some, various, misc, general, thoughts, notes) and generic prefixes (A study of, An overview of) | metrics     |
| D-S2 | Title word count prefer 4–12                                                                                                        | metrics     |
| D-S3 | Summary prefer 3–8 sentences                                                                                                        | metrics     |
| D-S4 | Summary must avoid overclaiming/hype (state-of-the-art, breakthrough, proves, guarantees, solves)                                   | metrics     |
| D-S5 | Keywords: avoid generic (AI, ML, deep learning, paper, survey, method); prefer ≥2 multiword                                         | metrics     |


