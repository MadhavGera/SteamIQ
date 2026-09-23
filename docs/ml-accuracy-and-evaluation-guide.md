# SteamIQ — ML Accuracy Scores, Metrics & Evaluation Guide

## Executive Summary

**Yes, SteamIQ currently tracks and enforces ML accuracy scores and evaluation metrics directly in the project.** 

In SteamIQ, accuracy is structured as a multi-tier verification framework spanning:
1. **Predictive ML (Success Classifier)** — 5-Fold Stratified Cross-Validation tracking **Accuracy (75.1%)**, **ROC-AUC (0.8301)**, **F1-Score (0.7798)**, and **Brier Loss (0.1648)** logged in **MLflow** and persisted in the `model_runs` table.
2. **Automated Promotion Guardrails** — A strict promotion floor (`ROC-AUC >= 0.55`, default target `> 0.80`) ensuring candidate models are only promoted to `production` if they pass statistical validation.
3. **Data Integrity & Anti-Leakage Guardrails** — Strict temporal cutoffs, self-inclusion filters, and label leakage unit test assertions.
4. **NLP & Embedding Quality Checks** — Sentiment discordance checks, topic coherence, and vector similarity ranking metrics (MRR / Hit@K).

---

## 1. What ML Accuracy Scores Exist Right Now?

All predictive training runs ([`backend/jobs/train_success_model.py`](backend/jobs/train_success_model.py)) run **5-Fold Stratified Cross-Validation** comparing 4 model architectures and record granular out-of-fold (OOF) and cross-validation statistics.

### Active Production Model Benchmark (`game_success_predictor`)

| Metric | Current Production Value | Cross-Validation (5-Fold Mean ± Std) | Purpose & Interpretation |
|---|---|---|---|
| **Accuracy** | **75.10%** (`0.7510`) | `0.7510 ± 0.0850` | Overall fraction of correct hit/flop classifications. |
| **ROC-AUC** | **0.8301** (`0.8301`) | `0.8409 ± 0.0684` | Model's ability to discriminate successful games from unsuccessful across all classification thresholds. (Industry standard for ranking/classification). |
| **F1-Score** | **77.98%** (`0.7798`) | `0.7723 ± 0.0958` | Harmonic mean of Precision and Recall, balancing false hits vs. missed hits. |
| **Brier Score** | **0.1648** (`0.1648`) | `0.1649 ± 0.0266` | Probability calibration error (lower is better; 0.0 is perfect calibration, 0.25 is random guessing). |

### Multi-Model Baseline Progression Tracked in MLflow:
1. **Baseline 1 — Logistic Regression** (with `StandardScaler`): Baseline linear benchmark (`ROC-AUC ~ 0.70`).
2. **Baseline 2 — Random Forest** (`n_estimators=100`, `max_depth=6`): Non-linear tree ensemble baseline (`ROC-AUC ~ 0.80`).
3. **Baseline 3 — XGBoost Baseline** (`n_estimators=100`, `max_depth=4`, `lr=0.08`): Gradient boosted decision trees (`ROC-AUC ~ 0.82`).
4. **Champion Candidate — XGBoost + LightGBM Ensemble (Optuna Tuned)**: Multi-trial Bayesian hyperparameter search tuning learning rates, tree depth, leaf counts, and blend weight $\alpha$ (`ROC-AUC: 0.8301–0.8409`).

---

## 2. Where Are Accuracy Scores Stored & Tracked?

### A. Database Audit Table: `model_runs`
Every training execution records full audit metadata in PostgreSQL:
```sql
SELECT id, model_name, stage, dataset_version, metrics, created_at 
FROM model_runs 
ORDER BY created_at DESC;
```
- **`stage`**: State lifecycle (`candidate` → `staging` → `production` → `archived`).
- **`metrics`**: JSONB storing `roc_auc`, `accuracy`, `f1_score`, `brier_score`, and fold standard deviations.
- **`hyperparameters`**: JSONB storing all model parameters and Optuna blend weights.
- **`artifact_path`**: URI link to the MLflow run and SHAP explainability artifacts.

### B. Prediction Traceability: `serving_predictions`
Every row in `serving_predictions` contains:
- `model_run_id` foreign key referencing the exact `model_runs` record that generated the score.
- `prediction_value` (success probability 0.00–1.00).
- `shap_feature_attributions` (JSON breakdown showing why the model predicted the score).

### C. MLflow Tracking: `backend/mlruns/`
- Every run is tracked under the experiment `steamiq_success_predictor`.
- Logs all trial metrics per fold, model artifacts, and global `champion_shap_summary.json`.

---

## 3. Comprehensive Accuracy & Quality Checks We Can Do

Here is the complete matrix of accuracy and quality verification checks categorized by subsystem:

### Suite A: Predictive ML Accuracy Checks (Classification & Calibration)

| Check Name | Technique | Formula / Threshold | Current Status |
|---|---|---|---|
| **Stratified K-Fold CV** | Out-of-fold cross validation with class-balanced splits | 5-Fold (`StratifiedKFold(n_splits=5)`) | ✅ Implemented |
| **ROC-AUC Score** | Area under Receiver Operating Characteristic curve | Target: `> 0.80` (Floor: `0.55`) | ✅ Implemented |
| **Precision-Recall AUC (PR-AUC)** | Area under Precision-Recall curve | Essential for imbalanced hit datasets | 🔲 Recommended |
| **Brier Score Calibration** | Mean squared probability error | $BS = \frac{1}{N}\sum (f_t - o_t)^2 < 0.20$ | ✅ Implemented |
| **Calibration Curve / ECE** | Expected Calibration Error across decile confidence bins | $\|acc(B_m) - conf(B_m)\| < 0.05$ | 🔲 Recommended |
| **Confusion Matrix Audit** | False Positive Rate (FPR) vs False Negative Rate (FNR) | Cost-weighted penalty | 🔲 Recommended |
| **SHAP Attribution Non-Degeneracy** | Verify TreeExplainer outputs non-zero, distinct contributions per game | $\sum \|\phi_i\| > 0.01$ | ✅ Implemented |

---

## 4. Integrity & Data-Leakage Guardrails (Automated Test Suite)

Accuracy scores are meaningless if data leaks into training. SteamIQ implements automated architectural tests in [`backend/tests/test_phase4_ml.py`](backend/tests/test_phase4_ml.py):

| Guardrail Test | What it Enforces | File / Implementation |
|---|---|---|
| **`test_no_label_leakage_in_feature_matrix`** | Asserts that target columns (`positive_reviews`, `review_score`, `target_success_score`, `sentiment_score`, etc.) never enter feature matrix $X$. | [`backend/tests/test_phase4_ml.py:183`](backend/tests/test_phase4_ml.py) |
| **`test_cutoff_date_boundary_isolation`** | Enforces temporal cutoff: reviews created after `feature_cutoff_date` are mathematically discarded to prevent lookahead bias. | [`backend/tests/test_phase4_ml.py:105`](backend/tests/test_phase4_ml.py) |
| **`test_aggregate_features_exclude_target_game_self_inclusion`** | Asserts developer/publisher aggregates exclude the game itself (prevents circular self-validation). | [`backend/tests/test_phase4_ml.py:213`](backend/tests/test_phase4_ml.py) |
| **`test_degenerate_label_guardrail`** | Aborts training if positive class ratio is $<10\%$ or $>90\%$ to prevent training on degenerate distributions. | [`backend/jobs/train_success_model.py:255`](backend/jobs/train_success_model.py) |
| **`test_promotion_floor_barrier`** | Blocks candidate models from reaching `production` if `ROC-AUC < 0.55`. | [`backend/jobs/train_success_model.py:566`](backend/jobs/train_success_model.py) |

---

## 5. NLP & Review Intelligence Accuracy Checks (Phase 2)

| Check Name | Description | How to Measure | Status |
|---|---|---|---|
| **Sentiment vs Steam Vote Discordance** | Measures how often positive sentiment corresponds to `voted_up=True` vs `voted_up=False`. | $\text{Discordance Rate} = \frac{N(\text{PosSentiment} \land \text{ThumbDown}) + N(\text{NegSentiment} \land \text{ThumbUp})}{N_{\text{total}}} < 12\%$ | 🔲 Ready to add |
| **Complaint Precision / Recall** | Evaluates rule-based and zero-shot regex taxonomies against gold review samples. | $F_1 = \frac{2 \cdot P \cdot R}{P + R} \ge 0.85$ | 🔲 Ready to add |
| **Topic Coherence ($C_v$)** | Assesses semantic consistency of discovered BERTopic / TF-IDF clusters. | Normalized Pointwise Mutual Information (NPMI) | 🔲 Ready to add |

---

## 6. Vector Similarity & Embedding Accuracy Checks (Phase 3)

| Check Name | Description | How to Measure | Status |
|---|---|---|---|
| **Genre / Tag Jaccard Agreement** | Checks if top-5 nearest neighbors in embedding space share primary & secondary genres. | $J(A, B) = \frac{\|G_A \cap G_B\|}{\|G_A \cup G_B\|} \ge 0.60$ | 🔲 Ready to add |
| **Mean Reciprocal Rank (MRR)** | Validates that known sequel/franchise pairs appear in top-3 recommendations (e.g. *Hollow Knight* → *Silksong* / *Ori*). | $MRR = \frac{1}{\|Q\|}\sum \frac{1}{\text{rank}_i} \ge 0.75$ | 🔲 Ready to add |
| **Embedding Drift Check** | Measures cosine distance shift when re-embedding game catalog versions. | $\Delta \cos(\vec{v}_{t}, \vec{v}_{t+1}) < 0.05$ | 🔲 Ready to add |

---

## 7. How to Execute Accuracy Checks & Training Locally

### 1. Run the Full ML Test Suite
```bash
# Run all Phase 4 ML integrity and accuracy unit tests
make test
# Or specifically:
cd backend && pytest tests/test_phase4_ml.py -v
```

### 2. Re-Train and Evaluate Models with Optuna Tuning
```bash
# Build tabular features from latest data
make build-features

# Train baseline models + run 15 Optuna ensemble trials + evaluate promotion floor
make train-model TRIALS=15
```

### 3. Inspect Accuracy Metrics in DB
```bash
docker exec steamiq_backend python -c "
from sqlalchemy import create_engine, text
import json
engine = create_engine('postgresql://steamiq:changeme@db:5432/steamiq')
with engine.connect() as conn:
    rows = conn.execute(text(\"\"\"
        SELECT stage, dataset_version, metrics 
        FROM model_runs 
        WHERE model_name='game_success_predictor' 
        ORDER BY created_at DESC LIMIT 5
    \"\"\")).fetchall()
    for r in rows:
        print(f'Stage: {r[0]:<12} Version: {r[1]} Metrics: {json.dumps(r[2])}')
"
```
