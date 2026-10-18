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
Evaluated across 5,000 held-out evaluation users with 95% bootstrap confidence intervals (Champion Seed 43 and 3-seed sweep across seeds 42–44 following Round 4 token alignment and causal mask bugfix):

| Metric | Champion Validation (95% CI) | Champion Test (95% CI) | 3-Seed Validation (Mean ± Std) | 3-Seed Test (Mean ± Std) |
|---|:---:|:---:|:---:|:---:|
| **NDCG@10** | 0.0488 [0.0443, 0.0535] | **0.0420** [0.0379, 0.0465] | 0.0491 ± 0.0007 | 0.0424 ± 0.0004 |
| **Hit Rate@10** | 0.0974 [0.0888, 0.1056] | **0.0824** [0.0750, 0.0900] | 0.0988 ± 0.0016 | 0.0833 ± 0.0010 |
| **MRR@10** | 0.0341 [0.0304, 0.0382] | **0.0297** [0.0263, 0.0336] | 0.0342 ± 0.0005 | 0.0301 ± 0.0003 |
| **SNIPS-NDCG@10** | 0.0202 [0.0181, 0.0223] | **0.0143** [0.0127, 0.0159] | 0.0206 ± 0.0003 | 0.0145 ± 0.0002 |
| **Catalog Coverage@10** | 17.90% | **18.13%** | 18.21% ± 0.35% | 18.20% ± 0.09% |
| **Novelty@10** | 8.57 bits | **8.64 bits** | 8.58 ± 0.01 bits | 8.64 ± 0.02 bits |
| **Diversity@10** | 0.7460 | **0.7485** | 0.7491 ± 0.0032 | 0.7517 ± 0.0031 |

- **Paired Comparison vs Two-Tower Champion (Seed 44):**
  - Validation Paired Diff: **`-0.0123`** (95% CI: `[-0.0172, -0.0078]`)
  - Test Paired Diff: **`-0.0093`** (95% CI: `[-0.0142, -0.0044]`)
  - **Verdict:** Two-Tower retains a statistically significant lead over SASRec due to rich multimodal feature blending (text + tags + continuous signals), but the gap is dramatically narrower than in earlier rounds (-0.0093 vs -0.0348 previously) following token alignment correction.

---

## 4. Serving Latency & Candidate Union Recall
- **Measured CPU Inference Latency (1,000 requests on Intel Tiger Lake):**
  - Median ($P_{50}$): **2.29 ms**
  - 90th Percentile ($P_{90}$): **2.96 ms**
  - 95th Percentile ($P_{95}$): **3.23 ms** (SLA target $< 50\text{ ms}$)
  - 99th Percentile ($P_{99}$): **4.05 ms**
- **Candidate Pool Recall Lift (1,000 users, Top-100 candidates):**
  - Two-Tower alone Recall@100: **41.80%**
  - SASRec alone Recall@100: **38.40%**
  - Union (Two-Tower + SASRec, 200 candidates): **49.70%** (**+18.9% relative recall lift**)

---

## 5. Limitations & Caveats
1. **Multimodal Representation Advantage:** Pure sequential self-attention over item IDs learns high-quality transition patterns (reaching NDCG@10 = 0.0541 on short histories), but does not fully replace multimodal tag and description embeddings on sparse catalogs.
2. **Complementary Stage 1 Retrieval Role:** While Two-Tower serves as the primary retriever, SASRec provides strong complementary coverage: blending SASRec top-100 candidates lifts candidate pool recall from 41.8% to 49.7% (+18.9% relative lift).
