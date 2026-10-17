# Model Card: `recsys_two_tower_ranker_v2_trainstats`

**Family:** LambdaMART GBDT Re-ranker | **Stage:** `candidate` (Leak-Free Ranker Candidate)  
**Run ID:** `phase4_seed_43` | **Generated:** 2026-10-17  
**License:** Project Internal / MIT  

---

## 1. Model Overview & Architecture
`recsys_two_tower_ranker_v2_trainstats` is a LightGBM LambdaMART cross-encoder re-ranking model designed to re-order the top-100 candidates produced by the clean Stage-1 Two-Tower retriever.
- **Candidate Generator:** `recsys_two_tower_clean` ($K=100$ candidate pool).
- **Tabular Features:** Pairwise interaction features, cross-encoder cosine similarities, and train-only item statistics.
- **Objective:** `lambdarank` with NDCG evaluation metric.
- **TreeSHAP Integration:** Full post-hoc feature attribution computing local Shapley values per item recommendation.

---

## 2. Leakage Mitigation & Honest Findings
> [!NOTE]
> **No Evidence of Gain Over Two-Tower Retriever:**
> In early unconstrained iterations (`ranker_v1`), the model achieved inflated NDCG gains due to all-time review totals leaking future popularity.
> Once strictly retrained with train-only item statistics (`train_stats_only: true`), the paired bootstrap difference against the Stage-1 Two-Tower model on 5,000 users shows **no evidence of gain (CI includes 0)**; the all-time-stats gain is leakage-confounded.
> In accordance with SteamIQ empirical honesty principles, the Two-Tower model remains the headline production champion.

- **Integrity Flags:**
  - `clean: true` | `leak_free: true` | `deployable: false` (candidate status, not promoted to production champion)

---

## 3. Quantitative Performance & Empirical Evidence
Evaluated across 5,000 held-out evaluation users with 95% bootstrap confidence intervals:

| Metric | Validation Split (95% CI) | Test Split (95% CI) |
|---|:---:|:---:|
| **NDCG@10** | 0.0551 [0.0500, 0.0602] | 0.0485 [0.0440, 0.0534] |
| **Hit Rate@10** | 0.1026 [0.0942, 0.1116] | 0.0938 [0.0856, 0.1020] |
| **MRR@10** | 0.0407 [0.0363, 0.0454] | 0.0349 [0.0312, 0.0390] |
| **Catalog Coverage@10** | 44.97% | 43.03% |

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
