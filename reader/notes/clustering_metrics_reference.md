# Clustering Metrics Reference (for Paper Embedding Clusters)

These are **internal clustering metrics**: they evaluate clustering quality **without ground-truth labels**. They mainly measure:

- **Compactness**: points in a cluster are close to each other  
- **Separation**: clusters are far from each other

This document explains the metrics used in the ablation script and how to interpret them for your setting (**~50 papers per batch**, typically **k = 3–5**).

---

## 1) Silhouette score (cosine)

### Definition (per point)
For a point *i*:

- **a(i)**: average distance from *i* to points in its own cluster  
- **b(i)**: minimum average distance from *i* to points in any other cluster (nearest neighboring cluster)

Silhouette:
\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

Overall silhouette is the mean of *s(i)* across all points.

### Range and meaning
- **+1**: point is much closer to its own cluster than others (excellent)
- **0**: point sits on the boundary between clusters
- **< 0**: point is closer to another cluster than its assigned one (bad)

### In our pipeline
We use cosine distance derived from embeddings:
- cosine similarity: \(\cos(x,y)\)
- cosine distance: \(1 - \cos(x,y)\)

### Practical interpretation (text embeddings)
Silhouette values for text clustering are often modest:
- **> 0.15**: usually decent separation
- **0.05–0.15**: weak but usable (common in diverse paper sets)
- **< 0.05**: likely mixed clusters or overly granular clustering

---

## 2) Davies–Bouldin Index (DBI)

### Intuition
DBI measures how much clusters **overlap**, using:
- within-cluster spread
- between-cluster separation

For each cluster *i*:
- \(S_i\): dispersion/spread of cluster *i*
- \(M_{ij}\): distance between centroids of clusters *i* and *j*

Compute:
\[
R_{ij} = \frac{S_i + S_j}{M_{ij}}
\]
\[
DB = \frac{1}{k}\sum_{i=1}^{k} \max_{j\neq i} R_{ij}
\]

### Meaning
- **Lower is better**
- High DBI means: “some cluster has a close neighbor relative to its spread” (overlapping clusters)

### Practical interpretation
- **< 1.0**: pretty good
- **1–2**: typical/OK for messy text clusters
- **> 2**: clusters overlap heavily

---

## 3) Calinski–Harabasz Score (CH)

### Intuition
CH is like an ANOVA ratio:
- between-cluster variance vs within-cluster variance

\[
CH = \frac{\text{between-cluster dispersion} / (k-1)}{\text{within-cluster dispersion} / (n-k)}
\]

### Meaning
- **Higher is better**
- Rewards clusters that are tight internally and far apart

### Caveat
CH can increase with more clusters. It is best used **comparatively** across a small set of candidate k values (e.g., **k = 3, 4, 5**), not as an absolute quality score.

---

## 4) Cluster cohesion: average cosine similarity to centroid

This is often the **most interpretable** metric for embedding-based paper clustering.

For a cluster *c*:

Centroid:
\[
\mu_c = \frac{1}{|c|}\sum_{x\in c} x
\]

Normalize centroid:
\[
\hat{\mu}_c = \frac{\mu_c}{\|\mu_c\|}
\]

Cohesion:
\[
\text{cohesion}(c) = \frac{1}{|c|}\sum_{x\in c} x^\top \hat{\mu}_c
\]

If embeddings are L2-normalized, \(x^\top \hat{\mu}_c\) is cosine similarity.

### Meaning
- **Higher = tighter cluster, less mixed**
- Often remains informative even when silhouette is low (common for overlapping research themes)

### Rough interpretation guide (varies by embed model)
- **0.35–0.55**: quite coherent for diverse papers
- **0.25–0.35**: acceptable
- **< 0.20**: likely “misc” bucket or mixed cluster

---

## 5) Worst cluster cohesion (minimum cohesion among clusters)

This answers:
**“Is any cluster really bad?”**

Even if average cohesion is OK, one cluster might become a “garbage bag.”

- **Higher is better**
- If worst cohesion is low (e.g., **< 0.20**), it usually indicates:
  - a miscellaneous cluster
  - wrong k for this batch
  - embeddings not separating topics well

---

## 6) Keyword entropy per cluster (optional)

This is a “topic focus” proxy using your per-paper keywords.

For a cluster, count keyword frequencies and compute Shannon entropy:

\[
H = -\sum_{t} p(t)\log p(t)
\]

- \(p(t)\): probability of keyword *t* within the cluster keyword multiset

### Meaning
- **Lower entropy**: vocabulary is concentrated → likely focused topic
- **Higher entropy**: vocabulary is scattered → cluster may be mixed

### Practical caveats
- If keywords are too generic, entropy can look low but be unhelpful.
- If keywords are rich and diverse, entropy can be high even for a good cluster.

Use entropy as a **secondary sanity signal**, not the primary decision-maker.

---

## How to use these metrics together (recommended)

A “good” run typically shows:
- silhouette: not terrible
- DBI: not huge
- avg cohesion: decent
- worst cohesion: not awful
- entropy: moderate/low (optional)

Common real-world pattern for diverse paper sets:
- silhouette close to 0 (themes overlap)
- cohesion still OK  
In that case, **trust cohesion more than silhouette**.

### Simple selection rule (works well in practice)
For each run (text-mode × k), pick the one that:
1. maximizes **avg cohesion**
2. subject to **worst cohesion** above a minimum threshold
3. tie-break using silhouette / DBI (and optionally entropy)

---

## Why k = 4 or 5 often works well here
With ~50 diverse ML papers, **k = 4 or 5** often provides a good tradeoff:
- enough clusters to separate major themes
- not so many that clusters become tiny and noisy

Operationally, **4–5 clusters is also a reasonable number of options for a human to choose from**, which matches your workflow requirement.
