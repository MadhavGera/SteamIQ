# Model Card: `recsys_sasrec`

**Family:** Sequential Transformer Recommender | **Stage:** `challenger` (Exploratory Challenger)  
**Run ID:** `phase_f_seed_43` | **Generated:** 2026-10-18  
**License:** Project Internal / MIT  

---

## 1. Model Overview & Architecture
`recsys_sasrec` is a lightweight Self-Attentive Sequential Recommendation (SASRec) model designed to capture chronological item transitions and next-item preferences over user interaction sequences.
- **Backbone:** 2-layer TransformerEncoder, 2 attention heads, hidden dimension $d=64$, feedforward dimension 256, dropout 0.2, GELU activation.
- **Sequence Processing:** Right-aligned interaction sequences ($L=50$) with causal attention masking and learned positional embeddings.
- **Trainable Parameters:** 295,361 parameters.
- **Loss:** In-batch sampled softmax loss with popularity logQ correction over item dot products.
- **Online Scoring:** MIPS dot-product scoring against precomputed L2-normalized catalog item embeddings.

---

## 2. Intended Use & Role
- **Primary Role:** Evaluated as an exploratory sequential challenger and an optional 5th candidate generation source within Stage 1 quota merging.
- **Promotion Status:** Not promoted to primary production retriever because dense semantic content embeddings in Two-Tower strongly outperform pure sequential transition modeling on short user interaction sequences (median sequence length = 7 items).

---

## 3. Quantitative Performance & Empirical Evidence
Evaluated across 5,000 held-out evaluation users with 95% bootstrap confidence intervals (Champion Seed 43 and 3-seed sweep across seeds 42–44):

| Metric | Champion Validation (95% CI) | Champion Test (95% CI) | 3-Seed Validation (Mean ± Std) | 3-Seed Test (Mean ± Std) |
|---|:---:|:---:|:---:|:---:|
| **NDCG@10** | 0.0216 [0.0187, 0.0246] | **0.0165** [0.0139, 0.0192] | 0.0217 ± 0.0005 | 0.0172 ± 0.0007 |
| **Hit Rate@10** | 0.0446 [0.0388, 0.0504] | **0.0344** [0.0296, 0.0394] | 0.0453 ± 0.0006 | 0.0374 ± 0.0026 |
| **MRR@10** | 0.0148 [0.0124, 0.0172] | **0.0111** [0.0090, 0.0133] | 0.0147 ± 0.0003 | 0.0117 ± 0.0006 |
| **SNIPS-NDCG@10** | 0.0062 [0.0053, 0.0071] | **0.0040** [0.0034, 0.0048] | 0.0062 ± 0.0001 | 0.0043 ± 0.0003 |
| **Catalog Coverage@10** | 3.40% | **3.50%** | 2.96% | 3.20% |
| **Novelty@10** | 7.72 bits | **7.73 bits** | 7.73 bits | 7.73 bits |
| **Diversity@10** | 0.7095 | **0.7099** | 0.7120 | 0.7110 |

- **Paired Comparison vs Two-Tower Champion (Seed 44):**
  - Validation Paired Diff: **`-0.0393`** (95% CI: `[-0.0450, -0.0340]`)
  - Test Paired Diff: **`-0.0348`** (95% CI: `[-0.0399, -0.0299]`)
  - **Verdict:** Two-Tower outperforms SASRec by $+211\%$ relative NDCG@10 on test split ($p < 0.001$).

---

## 4. Serving Latency & Candidate Union Recall
- **Measured CPU Inference Latency (1,000 requests on Intel Tiger Lake):**
  - Median ($P_{50}$): **1.07 ms**
  - 90th Percentile ($P_{90}$): **1.42 ms**
  - 95th Percentile ($P_{95}$): **1.71 ms** (SLA target $< 50\text{ ms}$)
  - 99th Percentile ($P_{99}$): **2.51 ms**
- **Candidate Pool Recall Lift (1,000 users, Top-100 candidates):**
  - Two-Tower alone Recall@100: **41.80%**
  - SASRec alone Recall@100: **26.70%**
  - Union (Two-Tower + SASRec, 200 candidates): **48.70%** (**+16.5% relative recall lift**)

---

## 5. Limitations & Caveats
1. **Short History Sparsity:** Sequential self-attention mechanisms require longer history sequences to build rich contextual representations; gaming libraries with $<5$ interactions provide limited chronological signals.
2. **Catalog Concentration:** Coverage is restricted to active game sequences (3.5% catalog coverage@10), concentrating top recommendations on frequently transitioned titles.
