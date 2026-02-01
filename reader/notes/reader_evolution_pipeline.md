
# Reader Evolution Pipeline — Design Specification

Version: v1.0 (Locked)
Status: Canonical
Audience: System / Research / Future-Self

---

## 0. Purpose

The **Reader Evolution Pipeline** governs how canonical topics evolve over time
(rename / merge / split) while preserving a clean separation between:

- **Archival monthly artifacts** (HF reports, unchosen clusters, alternative views)
- **Canonical reader knowledge** (what the user has actually understood and accepted)

This document defines *what is allowed*, *how it is justified*, and *how it is validated*.

---

## 1. Core Design Principles

1. **Canonical-first**
   - Evolution operates only on canonical topics.
   - Monthly data is contextual evidence, never a direct driver.

2. **Editorial ≠ Geometry**
   - Names and summaries are editorial.
   - Geometry is used only for validation and sanity checks.

3. **Human-in-the-loop by default**
   - LLM proposes.
   - Geometry checks.
   - Policy decides auto / review / reject.

4. **Cheap, explainable checks**
   - No heavy retraining or reclustering.
   - All checks must be interpretable.

---

## 2. Durable Evidence Model

Topic cohesion and evolution decisions rely on **durable evidence**, not raw text fields.

### Evidence Types

- **Observation cards**
  - User-chosen monthly cluster reports
- **Representative papers**
  - Papers explicitly attached via chosen observations
- **Reports (optional)**
  - Long-form summaries (used sparingly)

### Observation Embedding (MVP)

Each observation card is embedded as:

```
topic_card.name
+ one_liner
+ tags
+ top paper titles
```

Embeddings are recomputable on-demand; raw vectors need not be stored.

---

## 3. Topic Cohesion (Canonical Definition)

Topic cohesion is defined over **observation embeddings**, not topic text.

### Center Definition

The topic center is either:

- Mean of observation embeddings, or
- Medoid (most central observation)

### Cohesion Score

```
cohesion(T) = avg_sim(obs_i, center_T)
```

This score is **advisory**, never a hard gate.

---

## 4. Evolution Pipeline Overview

### Stage A — Semantic Proposal (LLM)

The LLM proposes an **operations plan** in structured JSON.

Each proposal must include:

- Proposed operation(s): rename / merge / split
- Explicit evidence links:
  - topic_ids
  - observation_ids
- Confidence score ∈ [0,1]
- Stated uncertainty / risks

No proposal is auto-applied without passing later stages.

---

### Stage B — Geometric Confirmation (Local)

Lightweight, deterministic checks to prevent nonsense.

#### 4.1 Topic Representation (MVP)

```
topic_repr(T) =
  embed(
    topic.name
  + topic.summary
  + top_labels
  + representative paper titles (optional)
  )
```

Used **only** for similarity checks.

---

### 4.2 Merge Checks

For proposed merge `{A, B -> T_new}`:

1. **Similarity Gate**
   ```
   sim(topic_repr(A), topic_repr(B)) >= τ_merge
   ```

2. **Cohesion Preservation**
   - `cohesion(T_new)` must not be catastrophically lower than:
     - `min(cohesion(A), cohesion(B))`

3. **Drift Guard**
   - `T_new` must not be closer to an unrelated topic `C`
     than it is to `A` or `B`

---

### 4.3 Split Checks

For proposed split `{A -> A1, A2, ...}`:

1. **Separability**
   ```
   avg_sim(within_groups) - avg_sim(across_groups) >= τ_split_gap
   ```

2. **Evidence Sufficiency**
   Each child topic must have at least:
   - `min_observations`, or
   - seed observations + recent reports

3. **Semantic Necessity**
   - Split must resolve conceptual overload, not cosmetic naming.

---

### 4.4 Rename Checks

Rename is primarily semantic.

Minimal validation only:

- Keyword / concept overlap between:
  - new name/summary
  - evidence snippets

No geometric checks required.

---

## 5. Stage C — Application Policy

Final decision combines **LLM confidence** and **geometric checks**.

### Auto-Apply

```
llm_confidence >= 0.8
AND passes all geometric checks
```

### Queue for Review

```
0.5 <= confidence < 0.8
OR checks are borderline
```

### Reject

- Fails hard checks
- Causes excessive drift
- Touches too many topics at once
- Lacks evidence links

---

## 6. Persistence Rules

- Monthly HF data:
  - Fully stored
  - Immutable / append-only
  - Never mutated by evolution

- Canonical topics:
  - Versioned
  - Evolution logged explicitly

Deprecated or split topics are **never deleted**.

---

## 7. Design Invariant (Non-Negotiable)

> Canonical topics represent the reader’s understanding —  
> not the model’s output space.

LLMs assist.
Geometry validates.
Humans decide.

---

End of document. Further implementation detail using chat: https://chatgpt.com/c/697ec67e-5f0c-83a2-adb6-30583c7ef1f3?ref=mini 
