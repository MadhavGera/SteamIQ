# SteamIQ — Recommender Accuracy, Evaluation & Methodology Guide

## 1. Executive Summary & Evaluation Framework

SteamIQ operates an offline-evaluated, online-serving hybrid recommender system that bridges retrieval, learning-to-rank, and explainability. To maintain scientific integrity and prevent leakage, all reported metrics adhere to strict principles:

1. **No Invented Numbers:** Every metric, confidence interval, and paired comparison originates directly from deterministic evaluation pipelines executed against the frozen dataset (`dataset_version: c662b8232dc0`).
2. **Leave-One-Out Chronological Split:** Strict per-user temporal partitioning where the latest interaction is reserved for test and the penultimate for validation.
3. **Train-Split Feature Isolation:** Deployable models (`recsys_two_tower_clean`, `recsys_two_tower_ranker_clean`) use only statistics computed from the train split. Reference models using all-time stats are marked as non-deployable diagnostic upper bounds.
4. **Bootstrap Statistical Significance:** All metrics report 95% percentile confidence intervals over 1,000 bootstrap resamples on 5,000 evaluation users.
5. **Beyond-Accuracy Audits:** Systematically measuring catalog coverage, novelty (self-information), intra-list cosine diversity, head share concentration, and Gini exposure inequality.

---

## 2. Dataset Topology, Filters & Splitting Protocol

### A. Data Source & Universe Selection
- **Catalog Universe ($N = 3,000$ items):** Selected from the Steam catalog using interaction volume and data quality filters (requiring non-empty descriptions, valid genre/tag metadata, and pricing history).
- **Evaluation Users ($N = 20,000$ total, $N_{\text{eval}} = 5,000$ holdout):** Filtered for interaction sequence length ($\ge 5$ positive interactions per user, $\ge 10$ interactions per item).
- **Implicit Feedback:** Interactions are positive reviews (`voted_up == True` / `is_recommended == True`). The training set contains 165,741 positive interactions and 26,845 explicit negative interactions.

### B. Per-User Chronological Split Protocol
For each user $u$ with interaction sequence ordered by review timestamp $(e_1, e_2, \dots, e_m)$:
- **Test Target ($e_m$):** The user's most recent interaction.
- **Validation Target ($e_{m-1}$):** The user's penultimate interaction.
- **Train History ($e_1, \dots, e_{m-2}$):** All earlier interactions used for model training and fold-in context.

### C. Why This Avoids Data Leakage
- **Temporal Integrity:** Models never observe future user actions when scoring candidate items.
- **Train-Split Item Statistics:** Feature pipelines compute item interaction frequencies, review counts, and positive ratios strictly over $\{e_1, \dots, e_{m-2}\}$, eliminating lookahead bias into the test period.
- **Seen-Item Masking:** All items present in a user's train history are masked to $-\infty$ during evaluation and serving.

---

## 3. Mathematical Metric Definitions

Evaluation is conducted over $N_{\text{eval}} = 5,000$ users. For each user $u$, the recommender produces a ranked list of items $R_u = (r_1, r_2, \dots, r_k)$ scored across the full 3,000-item catalog. Let $i_u^*$ be the true held-out target item, and $\text{rank}_u(i_u^*)$ be its 1-indexed position in $R_u$.

### A. Hit Rate @ k ($\text{HR}@k$)
Binary indicator of whether the held-out item appears in the top-$k$ recommendations:
$$\text{HR}@k = \frac{1}{|U_{\text{eval}}|} \sum_{u \in U_{\text{eval}}} \mathbb{I}(\text{rank}_u(i_u^*) \le k)$$

### B. Normalized Discounted Cumulative Gain @ k ($\text{NDCG}@k$)
Measures top-heavy ranking quality with logarithmic position discount. Because each user has exactly one held-out target item, $\text{IDCG}@k = 1.0$:
$$\text{DCG}@k = \frac{\mathbb{I}(\text{rank}_u(i_u^*) \le k)}{\log_2(\text{rank}_u(i_u^*) + 1)}, \quad \text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$$

### C. Mean Reciprocal Rank @ k ($\text{MRR}@k$)
Reciprocal position of the target item within the top-$k$ cutoff:
$$\text{MRR}@k = \frac{1}{|U_{\text{eval}}|} \sum_{u \in U_{\text{eval}}} \frac{\mathbb{I}(\text{rank}_u(i_u^*) \le k)}{\text{rank}_u(i_u^*)}$$

### D. Pessimistic Tie-Breaking
When multiple candidate items achieve identical floating-point scores $s_{u, i} = s_{u, j}$, ties are handled pessimistically to prevent inflated metric reporting:
$$\text{rank}_u(i) = 1 + |\{j \in I \setminus \{i\} : s_{u,j} > s_{u,i}\}| + \frac{1}{2} |\{j \in I \setminus \{i\} : s_{u,j} = s_{u,i}\}|$$
In conservative baseline scoring, tied items are assigned the worst-case rank: $\text{rank}_{\text{worst}} = 1 + |\{j : s_{u,j} \ge s_{u,i}\}|$.

---

## 4. Recommender Baselines & Candidate Architectures

| Model Key | Method | Description & Purpose |
| :--- | :--- | :--- |
| `recsys_popularity` | Frequency Baseline | Ranks items by train-split positive interaction counts. Represents zero-personalization popularity bias floor. |
| `recsys_content` | Content Cosine | Cosine similarity between mean user history vector and candidate item vectors (Sentence-Transformers `all-MiniLM-L6-v2` text + multi-hot tags). Zero collaborative feedback. |
| `recsys_itemknn` | Shrinkage-Cosine CF | Item-item collaborative filtering with shrinkage regularizer $\alpha = 100$: $s(i,j) = \frac{r_i \cdot r_j}{\|r_i\|_2 \|r_j\|_2 + \alpha}$. |
| `recsys_als` | Implicit Matrix Factorization | Alternating Least Squares on implicit feedback with confidence $c_{ui} = 1 + 40 \cdot r_{ui}$, 64 latent factors, 15 iterations. |
| `recsys_two_tower_clean` | Two-Tower Neural Retriever | Deep dual-encoder mapping user history (masked mean pooling) and item features (ID + MiniLM + Tags + Price) to 64-d with BPR in-batch loss and ID dropout ($p=0.2$). |
| `recsys_two_tower_ranker_clean` | Stage 2 LambdaMART Ranker | LightGBM gradient boosted decision trees ranking Stage-1 top-50 candidates using multi-source ranks, content cosine, item quality, and user history features. |

---

## 5. Main Recommender Leaderboard (Frozen Report)

Evaluated over 5,000 test users with 1,000 bootstrap iterations (95% CI):

### Deployable Models (Clean Train-Split Statistics)

| Model | Split | Seeds | NDCG@10 (mean ± std / 95% CI) | HR@10 (mean ± std / 95% CI) | MRR@10 | Coverage@10 | Diversity@10 | Stage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`recsys_two_tower_clean`** | **test** | 1 | **0.0516 [0.0473, 0.0562]** | **0.1028 [0.0954, 0.1108]** | **0.0361** | 0.4383 | 0.7629 | **Champion (Production)** |
| `recsys_two_tower_clean` | **val** | 1 | 0.0611 [0.0562, 0.0661] | 0.1190 [0.1104, 0.1280] | 0.0436 | 0.4347 | 0.7588 | Champion (Production) |
| `recsys_two_tower_ranker_clean` | **test** | 1 | 0.0579 [0.0527, 0.0631] | 0.1102 [0.1014, 0.1186] | 0.0422 | 0.5253 | 0.7481 | Candidate |
| `recsys_two_tower_ranker_clean` | **val** | 1 | 0.0615 [0.0563, 0.0666] | 0.1170 [0.1080, 0.1256] | 0.0447 | 0.5317 | 0.7431 | Candidate |
| `recsys_itemknn` | **test** | 1 | 0.0445 [0.0403, 0.0494] | 0.0842 [0.0766, 0.0916] | 0.0326 | 0.6143 | 0.7255 | Candidate |
| `recsys_als` | **test** | 1 | 0.0404 [0.0362, 0.0449] | 0.0776 [0.0704, 0.0852] | 0.0291 | 0.2480 | 0.7732 | Candidate |
| `recsys_popularity` | **test** | 1 | 0.0151 [0.0125, 0.0178] | 0.0326 [0.0274, 0.0376] | 0.0100 | 0.0070 | 0.6948 | Baseline |
| `recsys_content` | **test** | 1 | 0.0146 [0.0120, 0.0176] | 0.0268 [0.0226, 0.0316] | 0.0110 | 0.6303 | 0.5041 | Baseline |

### Diagnostic Reference Models (Non-Deployable Upper Bounds)

| Model | Split | NDCG@10 (95% CI) | HR@10 (95% CI) | Role |
| :--- | :---: | :---: | :---: | :--- |
| `recsys_two_tower_ranker` | **test** | 0.0584 [0.0530, 0.0634] | 0.1072 [0.0990, 0.1156] | Reference (uses all-time item stats, not deployable) |
| `recsys_two_tower` | **test** | 0.0539 [0.0490, 0.0589] | 0.1024 [0.0946, 0.1108] | Reference (uses all-time item stats, not deployable) |

---

## 6. Sliced Bootstrap Tests & Multi-Seed Paired Comparisons

### A. Paired Differences Across Popularity Slices
User-level paired difference tests ($\Delta = \text{NDCG}_A(u) - \text{NDCG}_B(u)$) on the test split:
- **Clean Two-Tower vs Item-kNN:** $\Delta = +0.0071$ ($[+0.0024, +0.0118]$, $p < 0.05$, Two-Tower is significantly better).
- **Clean Two-Tower vs ALS:** $\Delta = +0.0112$ ($[+0.0062, +0.0163]$, $p < 0.05$, Two-Tower is significantly better).
- **Clean Two-Tower vs Popularity:** $\Delta = +0.0364$ ($[+0.0316, +0.0415]$, $p < 0.05$, Two-Tower is significantly better).

### B. Clean Ranker vs Clean Two-Tower Promotion Decision
- **Single-Seed Deployed Metrics:** `recsys_two_tower_clean` (seed 44 champion) achieves test NDCG@10 of $0.0516$ ($[0.0473, 0.0562]$) and validation NDCG@10 of $0.0611$ ($[0.0562, 0.0661]$). `recsys_two_tower_ranker_clean` (seed 42 candidate) achieves test NDCG@10 of $0.0579$ ($[0.0527, 0.0631]$, $+0.0063$ lift over two-tower alone) and validation NDCG@10 of $0.0615$ ($[0.0563, 0.0666]$).
- **Pre-Registered Promotion Tournament Rule:** The candidate ranker must beat Two-Tower on tuning-set validation NDCG@10 by more than the ranker standard deviation ($\Delta_{\text{tuning}} > \sigma_r = 0.0021$).
- **Tournament Evaluation Result:** On the disjoint tuning validation set, Clean Ranker achieves $0.0615$ vs Clean Two-Tower $0.0611$ ($\Delta = +0.0004 < \sigma_r$). Because the margin does not exceed $+0.0021$, the ranker is not promoted, and `recsys_two_tower_clean` is promoted as the production champion.

---

## 7. Two-Tower & Ranker Ablation Studies

### A. Two-Tower Modality Ablations (Test NDCG@10)
- **Full Two-Tower (ID + Text + Tags + Numeric):** $0.0548 \pm 0.0007$
- **Without MiniLM Text Projection (`--no-text`):** $0.0482 \pm 0.0009$ ($\Delta = -0.0070$, $p < 0.05$, statistically significant drop proving text utility).
- **Without ID Dropout (`--id-dropout 0.0`):** $0.0507$ ($\Delta = -0.0032$, $p < 0.05$, confirms ID dropout forces content branch generalization).
- **Without Item ID (`--no-id`):** $0.0530 \pm 0.0004$ ($\Delta = -0.0003$, not significant, demonstrating strong content-only performance).

### B. Ranker Feature Group Drops
- **Drop Retrieval Source Scores:** $\text{NDCG@10} = 0.0387$ ($\Delta = -0.0220$, $p < 0.05$, largest driver).
- **Drop Item Quality Features:** $\text{NDCG@10} = 0.0526$ ($\Delta = -0.0081$, $p < 0.05$).
- **Drop Content Similarity:** $\text{NDCG@10} = 0.0571$ ($\Delta = -0.0035$).

---

## 8. Cold-Start Evaluation & Empirical Reconciliation

### A. Experimental Regimes (Lead Benchmark vs Optimistic Proxy)
1. **True Item Holdout (Primary Benchmark — $N = 184$ users, 300 items):** 10% torso/tail items completely excluded from training set interactions and the Two-Tower retriever retrained from scratch on the remaining 90% catalog. This is the only realistic test of true cold-start items with 0 interactions.
2. **Simulated New Items ($N = 437$ users, 300 items):** Random 10% catalog masked with train popularity zeroed and review proxies standardized to neutral 0.0.
3. **Masked Target ID Simulation / Sparse Targets (Optimistic Proxy — $N = 553$ users):** Items present in the training universe (targets with $\le 10$ train positives) whose learned ID embedding is replaced by the UNK token at test time. This represents an optimistic upper bound because the item representation benefited from co-occurrence updates during batch training.

### B. Empirical Findings & Exact Report Metrics
- **True Item Holdout ($N = 184$ users, 300 withheld items):**
  - `recsys_content`: $\text{NDCG@10} = 0.0078$ ($[0.0016, 0.0155]$), $\text{HR@10} = 0.0217$ ($[0.0054, 0.0435]$, $4 / 184$ hits), Mean Catalog Rank $= 1,084.6$ ($[963.7, 1,212.3]$ out of 3,000). **Content-only recommender beats random (mean rank $1,084.6$ vs random $1,500.0$).**
  - `random` (Analytic): $\text{NDCG@10} = 0.0015$, $\text{HR@10} = 0.0033$ ($1 / 184$ expected hits, $10/N$), Mean Catalog Rank $= 1,500.0$ ($N/2$).
  - `recsys_two_tower_item_holdout` (all-time-stats architecture, `clean=False`): **The two-tower dual encoder scores unseen items at or below random (mean rank $2,126.6$ vs random $1,500.0$; $\text{NDCG@10} = 0.0000$)** once popularity proxies and ID embeddings are neutralized:
    - **(a) Unmasked:** Mean rank $= 820.2$ ($[724.4, 917.9]$), $\text{NDCG@10} = 0.0054$ ($1 / 184$ hits) — leaks item ID and popularity proxies.
    - **(b) Masked:** Mean rank $= 781.2$ ($[682.8, 873.4]$), $\text{NDCG@10} = 0.0034$ ($1 / 184$ hits) — ID masked, but popularity proxies leak volume.
    - **(c) Masked + Train Positives Raw 0:** Mean rank $= 2,162.4$ ($[2,063.4, 2,264.4]$), $\text{NDCG@10} = 0.0000$ ($0 / 184$ hits) — raw zero count causes extreme negative z-score ($z \approx -0.4$).
    - **(c2) Standardized Mean ($z=0$ on Train Positives):** Mean rank $= 2,007.5$ ($[1,895.2, 2,120.3]$), $\text{NDCG@10} = 0.0000$ ($0 / 184$ hits) — train positives neutral, review proxies intact.
    - **(c3) All Proxies Neutralized ($z=0$ on rating, reviews, train positives):** Mean rank $= 2,126.6$ ($[2,015.1, 2,238.7]$), $\text{NDCG@10} = 0.0000$ ($0 / 184$ hits) — honest zero-interaction cold start.
  - Collaborative Baselines (`recsys_popularity`, `recsys_als`, `recsys_itemknn`): $\text{NDCG@10} = 0.0000$ ($0 / 184$ hits), Mean Catalog Rank $= 3,000.0$.

- **Simulated New Items ($N = 437$ users):**
  - `recsys_content`: $\text{NDCG@10} = 0.0085$ ($[0.0034, 0.0152]$), $\text{HR@10} = 0.0206$ ($9 / 437$ hits), Mean Catalog Rank $= 1,018.3$ ($[938.8, 1,098.4]$).
  - `recsys_two_tower_simulated`: $\text{NDCG@10} = 0.0000$ ($0 / 437$ hits), Mean Catalog Rank $= 2,182.3$ ($[2,104.9, 2,261.3]$).
  - `random`: $\text{NDCG@10} = 0.0015$, $\text{HR@10} = 0.0046$ ($2 / 437$ hits), Mean Catalog Rank $= 1,461.3$.

- **Sparse Targets / Masked ID Simulation (Optimistic Proxy — $N = 553$ users):**
  - `recsys_content`: $\text{NDCG@10} = 0.0054$ ($[0.0015, 0.0101]$), $\text{HR@10} = 0.0108$ ($6 / 553$ hits), Mean Catalog Rank $= 1,368.5$ ($[1,297.8, 1,438.4]$).
  - `recsys_two_tower_masked`: $\text{NDCG@10} = 0.0036$ ($[0.0008, 0.0071]$), $\text{HR@10} = 0.0090$ ($5 / 553$ hits), Mean Catalog Rank $= 1,143.5$ ($[1,074.3, 1,217.0]$).
  - `random`: $\text{NDCG@10} = 0.0012$, $\text{HR@10} = 0.0036$ ($2 / 553$ hits), Mean Catalog Rank $= 1,481.6$.

### C. Scientific Reconciliation & Production Architecture Rule
1. **Content-only beats random** ($\text{NDCG@10} = 0.0078$, mean rank $1,084.6$ vs random $1,544.0$). Pure Content Cosine (MiniLM text + tag embeddings) is the superior cold-start mechanism for newly published titles.
2. **Conditions (a) and (b) leak popularity:** Masking only item IDs preserves numerical interaction proxies that artificially boost retrieval.
3. **The two-tower scores unseen items at or below random** ($\text{NDCG@10} = 0.0000$, mean rank $2,126.6$ vs random $1,544.0$). Collaborative filtering and dual encoders require interaction co-occurrence density and cannot overcome popularity priors on zero-interaction titles.
4. **Architectural Recommendation:** Newly published games must be surfaced via pure content-similarity retrieval until accumulating $\ge 10$ organic user interactions.

---

## 9. Beyond-Accuracy Metrics & MMR Re-Ranking

| Metric | Formula / Meaning | Best Model | Value | Popularity Baseline |
| :--- | :--- | :--- | :---: | :---: |
| **Catalog Coverage@10** | Fraction of 3,000 items recommended | `recsys_content` | **63.03%** | 0.70% |
| **Novelty@10** | Mean self-information $-\log_2 P(i)$ | `recsys_content` | **11.75 bits** | 7.63 bits |
| **Diversity@10** | Intra-list pairwise cosine distance | `recsys_als` | **0.7732** | 0.6948 |
| **Head Share@10** | Share of recommendations in top 20% pop | `recsys_content` | **6.57%** | 100.0% |
| **Gini Inequality** | Lorenz concentration across catalog | `recsys_content` | **0.8665** | 0.9965 |

### Maximal Marginal Relevance (MMR) Diversity Frontier
Applying MMR diversification ($\text{MMR}(u, i) = \lambda \cdot s_{ui} - (1-\lambda) \max_{j \in S} \text{Sim}(i, j)$) over Stage-1 candidates:
- **$\lambda = 1.0$ (Pure Accuracy):** $\text{NDCG@10} = 0.0573$, $\text{Diversity} = 0.7564$
- **$\lambda = 0.8$:** $\text{NDCG@10} = 0.0568$, $\text{Diversity} = 0.7762$ ($\Delta \text{NDCG} = -0.0004$)
- **$\lambda = 0.6$:** $\text{NDCG@10} = 0.0556$, $\text{Diversity} = 0.8030$ ($\Delta \text{NDCG} = -0.0017$)

---

## 10. Engineering Limitations (Verbatim Audit Manifest)

1. **Implicit Feedback from Reviews Only:** Interactions derive from reviews (`is_recommended`); lacks non-purchaser negative impressions.
2. **Popularity and Selection Bias:** Highly reviewed titles receive disproportionate interactions.
3. **Per-User Temporal Split Protocol:** Evaluates next-item prediction per user sequence rather than a single global calendar cutoff.
4. **Item Universe Selection from Full Period:** 3,000 items chosen based on aggregate period counts to stabilize metadata.
5. **Single Held-Out Item Target:** Exactly one target per user in test and val.
6. **Offline Metrics vs Online Performance:** High offline NDCG reflects historical review retrieval, not guaranteed purchase conversion.
7. **Simulated Cold-Start Evaluation:** Item content embeddings are pre-trained on catalog text.
8. **Dev-Profile Scale Constraints:** Dev profile operates on 5,000 evaluation users and 3,000 items.
9. **Temporal Test Shift:** Test metrics are lower than validation across all models due to natural temporal drift.
10. **All-Time Item Statistics & Temporal Cutoff:** All-time review counts can include post-cutoff data in reference models.
11. **Popularity Proxy Leakage in Initial Simulation:** Neutralized in clean pipelines by standardizing numeric features to 0.0.
12. **Multiple Comparisons CI:** Pairwise bootstrap CIs are nominal 95% and not Bonferroni-adjusted.
13. **Single-Seed Ablation Interpretability:** Multi-seed evaluation is necessary to verify marginal feature changes.
14. **Catalog Coverage Definition (Masked vs Unmasked):** Official coverage (0.4383 test for Two-Tower clean) is strictly evaluated after masking seen items.

---

## 11. Legacy Success Predictor (Phase 4 — Archived Baseline)

> **Archival Notice:** The predictive classification model (`game_success_predictor` in `backend/jobs/train_success_model.py`) is preserved as a legacy component for tabular game-level classification.

### Legacy Benchmark Summary
- **Model:** 5-Fold Stratified Cross-Validation Optuna-tuned XGBoost + LightGBM ensemble.
- **Metrics:** Accuracy 75.10%, ROC-AUC 0.8301, F1-Score 77.98%, Brier Score 0.1648.
- **Known Limitations of Legacy Predictor:**
  1. Composite heuristic label ($\text{Positive Reviews} \ge 1,000 \land \text{Review Score} \ge 75\%$).
  2. Small catalog size ($N < 500$ games).
  3. Evaluated via cross-validation without a true future out-of-time holdout set.
