# Model Card: `recsys_two_tower_ranker_v2_trainstats`

**Family:** LambdaMART GBDT Re-ranker | **Stage:** `candidate` (Leak-Free Ranker Candidate)  
**Run ID:** `phase4_seed_44` | **Generated:** 2026-10-18  
**License:** Project Internal / MIT  

---

## 1. Model Overview & Architecture
`recsys_two_tower_ranker_v2_trainstats` (also referred to as `recsys_two_tower_ranker_clean`) is a LightGBM LambdaMART cross-encoder re-ranking model designed to re-order the top-100 candidates produced by the clean Stage-1 Two-Tower retriever and baseline candidate sources.
- **Candidate Generator:** Multi-source quota pool (`recsys_two_tower_clean` 120, `recsys_itemknn` 80, `recsys_als` 50, `recsys_content` 50).
- **Tabular Features:** Pairwise interaction features, cross-encoder cosine similarities, and train-only item statistics.
- **Objective:** `lambdarank` with NDCG evaluation metric.
- **TreeSHAP Integration:** Full post-hoc feature attribution computing local Shapley values per item recommendation.

---

## 2. Leakage Mitigation & Promotion Gate Evaluation
> [!NOTE]
> **Promotion Gate Failed — Two-Tower Retriever Retains Champion Status:**
> In early unconstrained iterations (`ranker_v1`), the model achieved inflated NDCG gains due to all-time review totals leaking future popularity.
> Once strictly retrained with train-only item statistics (`train_stats_only: true`), the paired bootstrap difference against the Stage-1 Two-Tower champion on 5,000 test users is **`+0.0019` (95% CI: `[-0.0024, +0.0062]`)**.
> Because the 95% confidence interval strictly includes zero, the pre-registered promotion condition ($\text{lo95} > 0$) is **not satisfied**.
> In accordance with SteamIQ empirical honesty principles, the Two-Tower model remains the headline production champion.

- **Integrity Flags:**
  - `clean: true` | `leak_free: true` | `deployable: false` (candidate status; gate failed)

---

## 3. Quantitative Performance & Empirical Evidence
Evaluated across 5,000 held-out evaluation users with 95% bootstrap confidence intervals (Champion Seed 44 and 5-seed sweep across seeds 42–46):

| Metric | Champion Validation (95% CI) | Champion Test (95% CI) | 5-Seed Validation (Mean ± Std) | 5-Seed Test (Mean ± Std) |
|---|:---:|:---:|:---:|:---:|
| **NDCG@10** | 0.0582 [0.0531, 0.0635] | **0.0532** [0.0480, 0.0587] | 0.0586 ± 0.0020 | 0.0517 ± 0.0020 |
| **Hit Rate@10** | 0.1098 [0.1014, 0.1190] | **0.1004** [0.0916, 0.1094] | 0.1105 ± 0.0035 | 0.0992 ± 0.0024 |
| **MRR@10** | 0.0428 [0.0385, 0.0473] | **0.0390** [0.0347, 0.0433] | 0.0430 ± 0.0019 | 0.0381 ± 0.0014 |
| **SNIPS-NDCG@10** | 0.0378 [0.0340, 0.0421] | **0.0311** [0.0274, 0.0350] | 0.0376 ± 0.0017 | 0.0307 ± 0.0011 |
| **Catalog Coverage@10** | 55.03% | **54.23%** | 51.55% | 53.60% |
| **Novelty@10** | 9.48 bits | **9.54 bits** | 9.36 bits | 9.50 bits |
| **Diversity@10** | 0.7450 | **0.7495** | 0.7511 | 0.7505 |

- **Paired Hypothesis Tests on Test Split (vs Two-Tower Champion):**
  - Overall Paired Difference: **`+0.0019`** (95% CI: `[-0.0024, +0.0062]`) — Not statistically significant.
  - Torso Slice (`pop_torso`): **`+0.0169`** (95% CI: `[+0.0091, +0.0246]`) — Statistically significant gain on mid-popularity items.
  - Tail Slice (`pop_tail`): **`+0.0051`** (95% CI: `[+0.0003, +0.0104]`) — Statistically significant gain on low-popularity items.

---

## 4. Hyperparameters
```json
{
  "num_leaves": 31,
  "learning_rate": 0.05,
  "n_estimators": 300,
  "min_child_samples": 50,
  "colsample_bytree": 0.8,
  "lambdarank_truncation_level": 20,
  "eval_at": [
    10
  ],
  "seed": 43,
  "n_jobs": 4,
  "max_candidates": 300,
  "source_run_ids": null,
  "train_stats_only": true
}
```

---

## 5. Latency Profile & Constraints
- Cold inference latency (`explain=False`): $P_{50} \approx 1.5\text{ ms}$, $P_{95} \approx 1.9\text{ ms}$.
- Explaining inference (`explain=True`, TreeSHAP): $P_{50} \approx 108\text{ ms}$, $P_{95} \approx 142\text{ ms}$.
