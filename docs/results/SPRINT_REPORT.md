# SteamIQ: ML Evidence Hardening Sprint Report

## Summary
The ML Evidence Hardening Sprint is an end-to-end reliability and consistency overhaul of the SteamIQ recommender system and assistant pipelines. The sprint resolves known evaluation masking defects, audits and eliminates feature leakage across temporal boundaries, introduces quota-based candidate generation to solve content-retrieval starvation, logs multi-seed training curves for Model Lab visualization, performs rigorous 5-seed champion sweeps, establishes an authoritative assistant evaluation with a hand-written realistic holdout set, executes a single-run test evaluation, and implements the SNIPS exposure-debiased ranking metric.

### 10-Line Pipeline Architecture Note
1. **Data Ingestion & Chronological Splitting:** User interaction logs are temporally partitioned per user into `train.parquet`, `val.parquet`, and `test.parquet`, with implicit negative impressions stored in `negatives.parquet`.
2. **Standard Dataset Container:** `RecsysData` encapsulates Parquet splits, text embeddings (`item_text_emb.npy`), sparse tag matrices (`item_tags_sparse.npz`), and train-period item popularity (`item_train_pop.npy`).
3. **Stage 1 Retrieval Models:** Independent candidate generators (`recsys_popularity`, `recsys_content`, `recsys_itemknn`, `recsys_als`, `recsys_two_tower`) score the catalog given fold-in user histories.
4. **Hybrid Two-Tower Retriever:** User tower (masked history pool) and Item tower (text embedding + tags + numeric features + learned ID) trained via in-batch sampled softmax with logQ correction and false-negative masking.
5. **Quota-Based Candidate Generation:** `generate_candidates()` retrieves top-K items per source and merges them according to source-specific quotas (e.g., Two-Tower 120, ItemKNN 80, ALS 50, Content 50) into a deduplicated candidate union.
6. **Stage 2 Re-Ranking:** LightGBM LambdaMART ranker extracts 26 ranking features across candidate retrieval scores, catalog metadata, user interaction counts, and text/tag similarities.
7. **Leak-Free Offline Evaluation:** `evaluate_model()` ranks the catalog for each user against a single held-out target item, applying timestamp-aware seen-item masking (-inf) to all interactions prior to the target.
8. **Bootstrap Metrics & Decision Gating:** Metrics (`ndcg_at_k`, `hit_rate_at_k`, `mrr_at_k`, `catalog_coverage_at_k`, `novelty_at_k`, `diversity_at_k`, `snips_ndcg_at_k`) compute 95% bootstrap confidence intervals and paired difference hypothesis tests.
9. **Registry & Promotion Rule:** `retrain_gate` and `promote_champion` enforce that a challenger model is promoted over the champion only if its paired validation NDCG@10 difference 95% CI strictly excludes zero.
10. **Online Serving Parity:** `RecsysService` loads registered champion artifacts, executes candidate generation and live feature extraction, and serves recommendations matching offline ranking.

---

## Phase Log

### Phase 0: Baseline Snapshot & Safety
- **Branch:** Created and checked out `sprint/ml-evidence` from `feat/recsys`.
- **Baseline Test Suite Execution:** Ran `pytest backend/tests/ -v -m "not artifacts"`.
  - **Results:** 205 passed, 1 deselected, 28 warnings in 213.05s.
  - **Status:** 100% pass rate.
- **Data & Artifacts Verification:**
  - `backend/data/` verified present.
  - `backend/data/recsys/artifacts/` verified with 16 serialized model artifacts present (`recsys_two_tower_clean`, `recsys_two_tower_ranker_clean`, `recsys_itemknn`, `recsys_als`, `recsys_content`, `recsys_popularity`, etc.).
  - `backend/data/assistant/` verified present.
- **Pipeline Architecture Note:** Documented above.

---

## Blockers
*None currently encountered.*

---

## Final Numbers
*(To be populated as phases complete)*
