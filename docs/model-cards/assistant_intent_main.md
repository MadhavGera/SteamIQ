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
Evaluated on gold domain queries (n=347) and synthetic paraphrase holdout queries (n=167):

| Metric | Value | Baseline Comparison / Context |
|---|:---:|---|
| **Gold Test Accuracy** | 0.9251 [0.8963, 0.9510] | vs TF-IDF: 0.7954 (+0.1297), Keyword: 0.4524 (+0.4726) |
| **Gold Macro-F1** | 0.9218 | vs TF-IDF: 0.7973, Keyword: 0.4670 |
| **Synthetic Holdout Accuracy** | 0.8623 [0.8144, 0.9162] | vs TF-IDF: 0.7964 (+0.0659), Keyword: 0.4072 (+0.4551) |
| **Synthetic Holdout Selective Acc (@$\tau$)** | 0.9085 | Coverage: 91.62% at calibrated $\tau = 0.630$ |
| **Gold Test Selective Acc (@$\tau$)** | 0.9592 | Coverage: 91.93% (Abstention: 8.07%) at $\tau = 0.630$ |
| **Expected Calibration Error (ECE)** | 0.0231 | Temperature Scaling Calibrated ($T = 0.183$) |
| **Extractive Recall@1** | 0.7385 [0.6308, 0.8462] | vs TF-IDF: 0.4154 (+0.3231) |
| **Extractive Recall@3 / Recall@5** | 0.9538 / 1.0000 | Review passage retrieval fallback (MRR: 0.8536) |
| **Test-Train Overlap: Exact Match** | 0 / 347 (0.00%) | Zero memorized strings |
| **Test-Train Overlap: Template Match** | 0 / 347 (0.00%) | Zero memorized templates |
| **Non-Overlapping Subset Accuracy** | 0.9217 (92.17%) | 332 / 347 queries (95.68% of test set) |

---

## 4. Limitations, Safety & Evaluation Honesty
1. **Holdout Dataset Lineage:** The 167 holdout queries are synthetic paraphrases constructed by the pipeline author. A human holdout dataset scaffold is established at `backend/assistant/data/holdout_queries_human.jsonl` (empty scaffold, requires $\ge 50$ human-authored queries before evaluation).
2. **Selective Abstention:** Threshold $\tau = 0.630$ is tuned on the validation split for target precision $\ge 0.99$ (coverage 96.7% on val). At test time, abstaining on lowest-confidence inputs boosts selective accuracy to 95.92%.
3. **No Generative Text:** The assistant cannot engage in open-ended chitchat or generate hallucinated claims.
4. **Catalog Bound:** Cannot answer questions about games not present in the indexed catalog.
5. **Extractive Fallback:** Questions not mapped to structured domain intents fall back to unedited quoted snippets from Steam player reviews.
