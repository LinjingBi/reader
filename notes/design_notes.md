2026/01/13
Start with a fast baseline (all-MiniLM-L6-v2) to validate the pipeline.

Move to retrieval-grade embeddings that generally separate technical topics better:

Default choice: BAAI/bge-small-en-v1.5 (offline, free, CPU-friendly, strong semantic separation)

Runner-up: intfloat/e5-small-v2 (also offline/free, strong retrieval embeddings)

Keep keywords as an optional feature (mode A/B/C) and validate via ablation(check `clustering_metrics_reference.md` for more details):

A: title + summary

B: title + summary + top-N keywords

C: title + summary + all keywords
If keywords improve clustering metrics and interpretability, they are high-signal for your data.

Why this fits your constraints

Light and fast locally (no paid API required).

“Good enough” topical structure for a stronger LLM to later name and connect papers.
