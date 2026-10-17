# Model Card: `assistant_intent_main`

**Family:** MiniLM Sentence Embedding + Temperature Scaled Classifier | **Stage:** `production` (Ask SteamIQ Headline)  
**Run ID:** `dacc7c5601d1` | **Generated:** 2026-10-17  
**License:** Project Internal / MIT  

---

## 1. Model Overview & Architecture
`assistant_intent_main` is the intent classification and query routing model for Ask SteamIQ. It classifies natural language questions into 12 domain intents plus an explicit out-of-scope class, strictly routing to deterministic database queries.
- **Encoder:** `all-MiniLM-L6-v2` dense sentence embeddings (384 dimensions).
- **Classification Head:** Multi-class Logistic Regression with L2 regularization.
- **Uncertainty Calibration:** Multi-class Temperature Scaling fitted on a held-out calibration split.
- **Selective Abstention:** Threshold $\tau$ calibrated on validation queries to achieve high precision. Queries with confidence $< \tau$ abstain or fall back to extractive passage retrieval.
- **Zero Free-Text Generation Invariant:** Never invokes a generative LLM; answers are constructed via pre-validated SQL execution against materialized tables (`mart_*`).

---

## 2. Intended Use & Dual-Mode Scope
- **Player Mode:** Answers game overviews, review sentiment summaries, top complaints, loved features, and personalized recommendations.
- **Developer Mode:** Answers market opportunities, success probability forecasts, price sensitivity, and update impacts.
- **Mode Refusal Guard:** Developer-only intents requested in Player mode are honestly refused with explicit guidance.

---

## 3. Quantitative Performance & Empirical Evidence
Evaluated on gold domain queries and realistic hand-written holdout queries:

| Metric | Value | Baseline Comparison |
|---|:---:|---|
| **Gold Test Accuracy** | 0.9251 | vs TF-IDF Baseline: 0.7954 |
| **Gold Macro-F1** | 0.9218 | vs Keyword Baseline: 0.4670 |
| **Realistic Holdout Accuracy** | 0.8623 | vs TF-IDF Baseline: 0.7964 |
| **Realistic Holdout Macro-F1** | 0.8606 | 167 hand-written queries |
| **Expected Calibration Error (ECE)** | 0.0231 | Temperature Scaling Calibrated |
| **Selective Accuracy at $\tau$** | 0.9251 | Calibrated threshold $\tau=0.10$ |
| **Extractive Recall@3** | 0.9538 | Review passage retrieval fallback |

---

## 4. Limitations & Safety
1. **No Generative Text:** The assistant cannot engage in open-ended chitchat or generate fabricated claims.
2. **Catalog Bound:** Cannot answer questions about games not present in the indexed catalog.
3. **Extractive Fallback:** Questions not mapped to structured domain intents fall back to unedited quoted snippets from Steam player reviews.
