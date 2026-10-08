# 📜 SteamIQ Recommender System Pipeline Execution & Run Log

> **Pipeline Execution Run Record & Audit Trail**  
> **Profile:** `dev` (`RECSYS_PROFILE=dev`)  
> **Dataset Version:** `c662b8232dc0`  
> **Git Tag Reference:** `recsys-frozen` (commit `427f9a6`)  
> **Audit & Review Date:** `2026-10-08`  
> **Status:** ✅ **VERIFIED & AUDITED**

---

## 1. Provenance & Execution Model

To ensure reproducibility without mutating frozen benchmark results:
1. **Frozen Benchmark Reports (2026-10-07):** The core offline metrics (Leaderboard, Modality Ablations, Ranker Feature Drops, Beyond-Accuracy Diversity, and Limitations) originate from the frozen pipeline execution on dataset `c662b8232dc0` (git tag `recsys-frozen` at `427f9a6`). The canonical JSON payloads are persisted in `backend/data/recsys/reports/` (`leaderboard.json`, `ablations.json`, `beyond_accuracy.json`, `cold_start.json`, `limitations.json`).
2. **Real Verification Runs (2026-10-08):**
   - **Embedding Projection (`project_item_embeddings`):** Executed to generate 2D UMAP + KMeans clusters and populate the PostgreSQL `serving_rec_item_projection` database mart.
   - **Serving Parity & Latency SLA (`serve_check`):** Executed live via `python -m jobs.serve_check` to verify exact scoring parity ($< 10^{-5}$) across 500 users, ranker `explain=True` vs `explain=False` identical top-10 outputs, and CPU latency budgets.
   - **Cold-Start CI Bootstrap:** Recomputed exact mean catalog ranks and 95% bootstrap confidence intervals directly from the saved item rank arrays in `cold_start.json`.
   - **Automated Test Suite:** Executed live via `pytest tests/` (163 passed) and `ruff check`.

---

## 2. Environment & Package Versions

| Package / Runtime | Exact Version | Platform / Architecture |
| :--- | :--- | :--- |
| **Python** | `3.12.7 (tags/v3.12.7:0b05ead)` | Windows 11 x64 (AMD64) |
| **PyTorch (`torch`)** | `2.12.0+cpu` | CPU Execution (OpenMP enabled) |
| **Implicit (`implicit`)** | `0.7.3` | OpenBLAS Multi-threaded ALS |
| **LightGBM (`lightgbm`)** | `4.6.0` | C++ LambdaMART CPU Backend |
| **Sentence-Transformers** | `5.4.0` | MiniLM L6-v2 Embeddings |
| **Scikit-Learn (`scikit-learn`)** | `1.9.0` | NearestNeighbors, KMeans, Cosine |
| **NumPy (`numpy`)** | `2.4.6` | Vectorized Linear Algebra |
| **Pandas (`pandas`)** | `2.3.3` | Arrow / Parquet Serialization |
| **Node.js / Next.js** | `v18+` / `14.2.35` | App Router Production Bundle |

---

## 3. Pipeline Architecture & Execution Stages

| Stage # | Stage Name | Exact Command | Data Source / Artifact Path | Verification Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Data Audit** | `python -m jobs.audit_recsys_data` | `data/recsys/interim/audit_summary.json` | ✅ Frozen (2026-10-07) |
| **2** | **Data Preparation** | `python -m jobs.prepare_recsys_data` | `data/recsys/processed/*.parquet` | ✅ Frozen (2026-10-07) |
| **3** | **Baselines Training** | `python -m jobs.train_recsys_baselines` | `data/recsys/artifacts/recsys_*` | ✅ Frozen (2026-10-07) |
| **4** | **Two-Tower Neural Training** | `python -m jobs.train_two_tower` | `data/recsys/artifacts/recsys_two_tower_clean` | ✅ Frozen (2026-10-07) |
| **5** | **Stage-Two Ranker Training** | `python -m jobs.train_ranker` | `data/recsys/artifacts/recsys_two_tower_ranker_clean` | ✅ Frozen (2026-10-07) |
| **6** | **Scientific Experiments** | `python -m jobs.run_recsys_experiments` | `data/recsys/reports/*.json` | ✅ Frozen (2026-10-07) |
| **7** | **2D Embedding Projection** | `python -m jobs.project_item_embeddings` | DB table `serving_rec_item_projection` | ✅ Live Run Verified |
| **8** | **Reports Materialization** | `python -m jobs.generate_reports` | DB table `serving_rec_reports` | ✅ Live Run Verified |
| **9** | **Serving Parity & Latency** | `python -m jobs.serve_check` | `jobs/serve_check.py` | ✅ Live Run Verified |

---

## 4. Artifact & Database Run Registry (`model_runs`)

| Model / Experiment Run Name | Stage | Run ID (`model_runs.id`) | Key Validation Metric |
| :--- | :---: | :--- | :--- |
| `recsys_two_tower_clean` (**Production Champion**) | `production` | `9332aa72-a0fd-446a-937c-0def93dcd9c9` | Val NDCG@10: `0.0611` (Test: `0.0516`) |
| `recsys_two_tower_ranker_clean` (**Candidate**) | `candidate` | `36759388-26dd-453c-ae7d-994bb32c1611` | Val NDCG@10: `0.0645` (Test: `0.0596`) |
| `recsys_two_tower_ranker` (**Reference Benchmark**) | `candidate` | `4106d5c1-cbc4-4de3-a72c-ed95a84ba6da` | Val NDCG@10: `0.0621` (Test: `0.0584`) |
| `recsys_two_tower` (**Reference Benchmark**) | `candidate` | `b88c0acf-2bff-4c14-b99b-d61563478dad` | Val NDCG@10: `0.0610` (Test: `0.0539`) |
| `recsys_two_tower_item_holdout` | `candidate` | `0a99d05e-339b-4791-aecb-c764cac9807b` | Val NDCG@10: `0.0610` |
| `recsys_two_tower_ranker_without_train_popularity` | `candidate` | `e8e956d8-f0db-4516-8aab-7bca8f4743fa` | Val NDCG@10: `0.0614` |
| `recsys_two_tower_ranker_without_all_time_item_stats` | `candidate` | `62c287ed-0c0a-4c75-a51d-1746c012a7d8` | Val NDCG@10: `0.0619` |
| `recsys_two_tower_ranker_three_sources_only` | `candidate` | `fa77390b-8b92-4eff-8819-5bcaa1cb35ef` | Val NDCG@10: `0.0614` |
| `recsys_two_tower_ranker_without_item_quality` | `candidate` | `47dfaf22-1251-44b6-a37c-20097f06b988` | Val NDCG@10: `0.0584` |
| `recsys_two_tower_ranker_without_user_features` | `candidate` | `834fd602-0964-4c33-9992-a2053599f04e` | Val NDCG@10: `0.0666` |
| `recsys_two_tower_ranker_without_price_features` | `candidate` | `880a6c33-94f7-4f3a-854c-2081947a11f4` | Val NDCG@10: `0.0637` |
| `recsys_two_tower_ranker_without_content_sim` | `candidate` | `6924f320-0372-435f-9798-1cc045f94607` | Val NDCG@10: `0.0599` |
| `recsys_two_tower_ranker_without_source_scores` | `candidate` | `f09a40a1-2da3-4bbc-86ec-48d5a8b99f0b` | Val NDCG@10: `0.0345` |

---

## 5. Verification Checklist & Test Suites

- **Backend PyTest Suite:**
  - `pytest tests/ -v --tb=short`: **163 passed, 0 failed**.
- **Backend Code Quality & Linting:**
  - `ruff check .`: **All checks passed (0 errors)** on recsys code.
- **Frontend Quality Assurance:**
  - `scripts/check_honest_fallbacks.py`: **41 files scanned, 100% compliant** with `STEAMIQ-FE-0001`.
  - `scripts/check_no_hardcoded_metrics.py`: **43 files scanned, 0 hardcoded metric literals found**.
  - `npm run build`: **Compiled successfully, 8 static/dynamic routes generated**.
- **Online Inference Parity & Latency SLA (`serve_check.py`):**
  - Offline vs Online 500-User Scoring Parity: **Exact score parity ($< 10^{-5}$) and 0 order mismatches**.
  - Ranker Explain Parity: **`explain=True` vs `explain=False` generates identical top-10 ranked lists**.
  - CPU Latency Budget: **P95 $< 1,000\text{ ms}$ for all models with and without SHAP explanations**.
