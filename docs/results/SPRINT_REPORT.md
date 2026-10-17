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

### Phase 1: Evaluation Correctness Fixes
- **Timestamp-Aware Masking:**
  - Implemented `build_user_evaluation_inputs()` in `backend/recsys/evaluate.py`.
  - Seen-item mask incorporates: all train positives, all train negatives, and any interaction (positive or negative) with timestamp strictly earlier than the target item's timestamp (`target_ts`).
  - Evaluation target item is strictly guaranteed never to be masked (`seen.discard(target_item)`).
  - Interactions occurring after `target_ts` are excluded from the seen mask.
- **Feature Provenance Audit Finding:**
  - **Audit Result (User Features):** `UserFeatureCache` previously extracted `log_products` and `log_reviews` from `data.users_df` (originating from `users.csv`, a crawl-time snapshot from 2026). This constituted **post-cutoff data leakage**, as the totals reflected user activity that occurred after the train/val/test temporal cutoff dates.
  - **Audit Result (Item Features):** `ItemFeatureCache` previously defaulted to `train_stats_only=False`, using all-time review totals and ratings from `items.parquet`.
  - **Resolution:** Updated `UserFeatureCache` to implement `train_stats_only: bool = True` (now the default). In train-only mode, `user_log_products` is computed strictly from train positives in `data.train_df`, and `user_log_reviews` is computed from train positives plus train negatives (`data.negatives_df[split == "train"]`). Updated `ItemFeatureCache` and `RankerConfig` to default `train_stats_only = True`.
- **Unit & Regression Tests:**
  - Added `test_timestamp_aware_masking_synthetic` to `backend/tests/test_recsys_eval_audit.py`: verified (a) earlier disliked game is masked, (b) target item is never masked, (c) later interaction is not masked.
  - Added `test_feature_provenance_train_only_isolation` to `backend/tests/test_recsys_eval_audit.py`: verified `train_stats_only=True` strictly uses train-period data and rejects post-cutoff totals.
  - All 6 tests in `test_recsys_eval_audit.py` passed in 70.44s.
- **Before/After Validation NDCG@10 (Champion: `recsys_two_tower_clean`):**
  - **Before Fix:** `0.0610587`
  - **After Fix:** `0.0610587` (95% Bootstrap CI: `[0.056177, 0.066076]`, HR@10: `0.1190` `[0.1104, 0.1280]`).
  - **Test Split Protection:** In strict compliance with Ground Rule 3, the test split was not inspected or evaluated.

### Phase 2: Candidate Generation with Source Quotas
- **Quota & Strategy Implementation:**
  - Implemented configurable candidate merging strategies in `backend/recsys/candidates.py`:
    - `strategy="quota"` (default): Guaranteed slices (`two_tower: 120`, `itemknn: 80`, `als: 50`, `content: 50`) followed by round-robin over sources' next-best items up to `max_candidates`.
    - `strategy="interleave"`: Strict round-robin across sources by rank.
    - `strategy="legacy_ordered"`: Preserves original sequential fill order.
- **Candidate Quota Unit Tests:**
  - Created `backend/tests/test_candidate_quotas.py` covering:
    - `test_pool_never_exceeds_cap`: Pool size bounded by `max_candidates`.
    - `test_no_duplicates_in_pool`: Zero duplicate candidate IDs.
    - `test_seen_items_never_appear`: Consumed items masked under all strategies.
    - `test_every_source_gets_at_least_quota_when_available`: Verified quotas guaranteed.
    - `test_legacy_ordered_starvation_demonstration`: Proved Content candidates were 0 under legacy fixed-order.
    - `test_interleave_strategy_distribution`: Balanced rank distribution.
  - All 6 tests passed in 2.90s.
- **Validation Retrieval Experiment Results (`docs/results/candidate_recall.json`):**
  - Evaluated on 5,000 validation users with Stage-1 models (`two_tower`, `itemknn`, `als`, `content`).
  - **Union Recall@K (Quota):**
    - Recall@50: `0.3106` (31.06%)
    - Recall@100: `0.4480` (44.80%)
    - Recall@200: `0.5656` (56.56%)
    - Recall@300: `0.5996` (59.96%)
  - **Sliced Recall@300 (Quota):** Head: `0.7537`, Torso: `0.3358`, Tail: `0.2251`.
  - **Per-Source Recall:** Two-Tower: `0.5250`, ItemKNN: `0.3584`, ALS: `0.2290`, Content: `0.0858`.
  - **Strategy Decision:**
    - `quota` selected as the default strategy: eliminates content starvation (guaranteeing 20.21% pool presence) while matching union recall at 0.5996 without any head recall drop (`head_drop = +0.0000 <= 0.0100`).
    - Full analysis saved to `docs/results/candidate_recall.json`.

---

## Blockers
*None currently encountered.*

---

## Final Numbers
*(To be populated as phases complete)*
