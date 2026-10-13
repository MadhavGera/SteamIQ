# Model Card: `recsys_two_tower_ranker_v2_trainstats`

**Family:** LambdaMART GBDT Re-ranker | **Stage:** `candidate` (Leak-Free Ranker Candidate)  
**Run ID:** `ee6d93db-6e51-4548-918f-d5133b214990` | **Generated:** 2026-10-12  
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
| **NDCG@10** | 0.0633 [0.0581, 0.0682] | 0.0536 [0.0487, 0.0588] |
| **Hit Rate@10** | 0.1152 [0.1064, 0.1228] | 0.0992 [0.0910, 0.1076] |
| **MRR@10** | 0.0475 [0.0433, 0.0519] | 0.0399 [0.0357, 0.0446] |
| **Catalog Coverage@10** | 52.47% | 52.67% |

---

## 4. Hyperparameters
```json
{
  "num_leaves": 31,
  "learning_rate": 0.05,
  "n_estimators": 500,
  "min_child_samples": 50,
  "colsample_bytree": 0.8,
  "lambdarank_truncation_level": 20,
  "eval_at": [
    10
  ],
  "seed": 42,
  "n_jobs": 4,
  "max_candidates": 300,
  "source_run_ids": {
    "two_tower": "eb9a3374-2aad-476f-a25e-5d3b84261b7f",
    "itemknn": "d6a39dd1-e1dc-4815-9e1f-d6b8e94b0fc5",
    "als": "3c8b20fb-d34b-433f-b293-a165e6e90cea",
    "content": "cf9737fe-5364-43a6-8b04-ed79c8b21b74",
    "popularity": "955e8397-9fe6-4bed-a3d6-6dd01421dbef"
  },
  "train_stats_only": true
}
```

---

## 5. Latency Profile & Constraints
- Cold inference latency (`explain=False`): $P_{50} \approx 1.5\text{ ms}$, $P_{95} \approx 1.9\text{ ms}$.
- Explaining inference (`explain=True`, TreeSHAP): $P_{50} \approx 108\text{ ms}$, $P_{95} \approx 142\text{ ms}$.
