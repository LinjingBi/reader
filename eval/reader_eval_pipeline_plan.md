# Evaluation pipeline implementation plan (P0 vs P-rich) for monthly clustering enrichment

This plan is intended for a code agent to implement. It defines a minimal, reproducible evaluation harness to compare:
- **P0 (simplest packing)**: title + HF summary/abstract + HF keywords
- **P-rich (rich packing)**: P0 + extracted highlights + extracted future-work/potential + optional bullet summarization

The output of the harness is a structured run log plus a human-readable report that helps you choose:
- the best **prompt template**
- whether **P-rich** is worth its added complexity/cost
- stable default generation parameters

---

## 1) Scope and inputs/outputs

### 1.1 Inputs
1. **Monthly clustering metadata** for each month (12 months), containing clusters and per-paper metadata:
   - cluster_index, size, cohesion
   - papers: paper_id, title, summary, keywords, url, rank_in_cluster, sim_to_centroid
2. **Prompt templates** (2–4 variants) for cluster enrichment (each returns JSON).
3. **Packing variants**:
   - `P0`
   - `P_RICH` (optional extraction stage, with fallback to P0 per paper)
4. **LLM config**:
   - provider/model name
   - temperature/top_p/max_tokens

### 1.2 Outputs
For each `(month, packing_variant, prompt_variant)` run:
- `model_output.json` (LLM output, raw)
- `judge_result.json` (scores + pass/fail checks + judge rationale)
- `rendered_cards.md` (human-readable cards for quick review)
- `run_metadata.json` (prompt hash, config, timestamps)

Aggregate outputs:
- `leaderboard.csv` (average scores, failure rates)
- `summary.md` (best winners, failure analysis, notable regressions)

---

## 2) Repository layout for eval harness

Recommended (inside your Reader repo, but separate from production pipeline):

```
reader/
  eval/
    README.md
    configs/
      eval.yaml
    data/
      monthly_clusters/
        2025-01.json
        ...
    prompts/
      enrich_light.txt
      enrich_medium.txt
      enrich_rich.txt
      llm_judge.txt
    packers/
      p0.py
      prich.py
    judges/
      heuristics.py
      llm_judge.py
      rubrics.md
    run_eval.py
    render_cards.py
    results/
      <timestamp>/
        runs/
        aggregate/
```

Notes:
- `eval/data/monthly_clusters/*.json` is your “validation dataset”.
- `eval/prompts/*.txt` are prompt templates (system + user blocks).
- `eval/packers/` builds the actual LLM input payload for P0/P-rich.
- `eval/judges/` contains both heuristic and LLM judge logic.

---

## 3) Config files

### 3.1 eval/configs/eval.yaml
Example (MVP):

```yaml
dataset:
  months: ["2025-01", "2025-02", "..."]
  input_dir: "eval/data/monthly_clusters"

packers:
  variants: ["P0", "P_RICH"]
  p_rich:
    top_papers_per_cluster: 3
    max_chars_per_paper: 3000
    enable_extraction: true
    extraction_fallback: "P0"

prompts:
  variants:
    - id: "light_json"
      path: "eval/prompts/enrich_light.txt"
    - id: "medium_json"
      path: "eval/prompts/enrich_medium.txt"

llm:
  provider: "openai"
  model: "gpt-4.1-mini"
  temperature: 0.2
  max_tokens: 2500

judge:
  use_llm_judge: true
  llm_judge_model: "gpt-4.1-mini"
  rubric_weights:
    schema_valid: 0.0   # gate
    citations_ok: 0.0   # gate
    coherence: 0.30
    specificity: 0.15
    selection_usefulness: 0.25
    reading_order: 0.15
    confidence_calibration: 0.10
    conciseness: 0.05
    highlights_quality: 0.20   # only for P_RICH; otherwise 0
```

---

## 4) Packing variants specification

### 4.1 P0 packer (baseline)
For each cluster, include top-N papers by `sim_to_centroid`.
Each paper block includes:
- paper_id, title, summary, keywords, url, rank_in_cluster, sim_to_centroid

Implementation:
- `eval/packers/p0.py` exports:
  - `build_input(payload: dict, cfg: dict) -> dict | str` (depending on your prompt style)

### 4.2 P-rich packer
Adds two optional fields per paper:
- `highlights`: list[str] (2–4 bullets)
- `future_work`: list[str] (1–3 bullets)

Constraints for MVP:
- Attempt extraction only for top `top_papers_per_cluster`.
- If extraction fails or text quality is low, set `highlights=[]`, `future_work=[]`.
- Cap total characters per paper (truncate).

Implementation:
- `eval/packers/prich.py` exports:
  - `build_input(...)` same signature as P0
  - calls `extract_highlights_and_future_work(paper)`

---

## 5) P-rich extraction + summarization candidates

You asked for candidates. Below are practical options, ordered from simplest to more capable.

### 5.1 Extraction sources (where to get text beyond HF summary)
1. **arXiv HTML (abs + “HTML version” when available)**  
   - Pros: easiest to fetch, no PDF parsing  
   - Cons: not always present; formatting varies

2. **arXiv PDF → text**  
   - Pros: always available  
   - Cons: parsing quality varies; section detection is noisy

3. **GROBID (PDF → structured TEI XML)**  
   - Pros: best-quality structure (sections, headings)  
   - Cons: adds a service dependency (local docker is common)

MVP recommendation:
- Start with (1) when available; fallback to (2).
- Add (3) only if you find PDF parsing quality unacceptable.

### 5.2 Highlight/future-work extraction methods
A) **Pure heuristic extraction (fastest MVP)**
- Highlights:
  - find sentences in Introduction containing cue phrases:
    - “In this paper, we …”, “We propose …”, “We present …”, “Our contributions …”
- Future work:
  - scan end sections for “future work”, “limitations”, “we plan to”, “could be extended”
- Then bulletize with simple rewriting (no LLM) OR keep as extracted sentences.

B) **Lightweight local summarization model (offline)**
If you want bullet compression but still avoid a paid LLM:
- DistilBART / BART-large-CNN (good general summarization, heavier)
- T5-small / T5-base (light, needs prompt-style prefix)
- PEGASUS variants (summarization strong, heavier)

C) **Small LLM (local) for bullet rewrite**
If you already run a small local model:
- Use it only to rewrite extracted sentences into bullets, not to “infer” content.

MVP recommendation:
- A) Heuristic extraction + optional rule-based bullet formatting first.
- Add B) only if heuristics produce too-long/too-messy bullets.

### 5.3 Bullet summarization policy (important)
Even in P-rich, avoid full-paper summarization at MVP:
- Summarize only extracted highlight/future-work snippets (small input).
- Keep each paper to <= 4 highlight bullets and <= 3 future-work bullets.

---

## 6) Prompt variants and how to test them

Create prompt templates as text files with placeholders. Each prompt file should contain:
- system instructions
- user instructions
- a placeholder token like `{{INPUT_JSON}}` or `{{INPUT_TEXT}}`

The runner replaces placeholders with packed input.

Prompt variants to include (MVP):
- `light_json`: outputs only topic name/one-liner/tags/confidence and representative papers
- `medium_json`: adds what/why/reading_order/search_query_seed
- `rich_json`: adds cluster_highlights and paper_highlights (only meaningful for P-rich packing)

---

## 7) Runner implementation (run_eval.py)

### 7.1 Core loop
For each month:
- load `monthly_clusters/<month>.json`
- for each `packing_variant`:
  - build input via packer
  - for each `prompt_variant`:
    - call LLM
    - store raw output
    - run judges
    - render markdown cards
    - append to aggregate table

### 7.2 Run identity
Create a stable run id:
- `run_id = "{month}__{packing}__{prompt}__{model}__t{temp}__{hash(prompt+input)}"`

Store:
- `prompt_hash`
- `input_hash`
- `llm_params`
- timestamp
- git commit (optional, recommended)

---

## 8) Judges

### 8.1 Heuristic judge (fast, always-on)
Implement in `eval/judges/heuristics.py`:

Pass/fail gates:
1. `schema_valid`: parse JSON, validate required keys/types
2. `citations_ok`:
   - every representative/reading_order url must exist in input urls
   - no new urls
3. `length_ok`:
   - max chars for fields (e.g., topic_name <= 80; one_liner <= 180)

Score features:
- `name_generic_penalty`: penalize names like “AI”, “LLM”, “Vision” unless supported by evidence_terms
- `tag_uniqueness`: fraction of tags unique across clusters

### 8.2 LLM judge (optional, higher quality)
Implement in `eval/judges/llm_judge.py`:

Prompt the judge with:
- cluster input (trimmed)
- model output card(s)
Ask for a JSON score breakdown:
- coherence
- specificity
- selection usefulness
- confidence calibration
- reading order
- conciseness
- highlights quality (only for P-rich)

Use **pairwise** mode later if you want more stable comparisons:
- present two outputs for the same cluster and ask which is better and why

---

## 9) Render cards for human review

Implement `eval/render_cards.py`:
- transform JSON output into a markdown file that looks like your UI
- include confidence and 1–2 representative paper links
- include cluster_index and cohesion for context
- include a “notes” section

This is how you will quickly eyeball whether the rubric matches your intuition.

---

## 10) Acceptance criteria for MVP

The eval pipeline is “done” when:
1. You can run: `python eval/run_eval.py --config eval/configs/eval.yaml`
2. It produces per-run artifacts and an aggregate leaderboard.
3. It can compare P0 vs P-rich across 12 months without manual intervention.
4. Failure modes are visible (schema/citation failures listed).

---

## 11) Suggested next steps (implementation order)

1. Implement P0 packer + one prompt template + heuristic judge.
2. Add aggregation + markdown rendering.
3. Add 2–3 prompt variants.
4. Add LLM judge (optional).
5. Add P-rich packer with heuristic extraction (no summarizer).
6. Add extraction fallback + quality filters.
7. Run the 12-month evaluation and pick defaults.
