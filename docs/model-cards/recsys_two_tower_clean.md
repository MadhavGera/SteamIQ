# Model Card: `recsys_two_tower_clean`

**Family:** Two-Tower Retrieval | **Stage:** `production` (Headline Champion)  
**Run ID:** `phase4_seed_44` | **Generated:** 2026-10-18  
**License:** Project Internal / MIT  

---

## 1. Model Overview & Architecture
`recsys_two_tower_clean` is the headline production retrieval model for SteamIQ. It maps user library histories and game items into a shared 64-dimensional semantic embedding space.
- **Query Tower:** Self-attention pooling (`pool='attn'`) over historical item representations with ID dropout (`id_dropout=0.25`) to prevent overfitting to high-volume user items.
- **Candidate Tower:** Deep content projections combining item IDs, MiniLM-L6 text representations, sparse tag projections, and log-transformed item statistics.
- **Loss:** In-batch sampled softmax loss with temperature scaling over cosine distances and logQ popularity correction.
- **Serving Invariant:** Bounded in-memory inference with exact MIPS top-k ranking ($P_{95} < 30\text{ ms}$ CPU budget).

---

## 2. Intended Use & Target Users
- **Primary Use:** Powers personalized recommendations across Player Mode (discovery, genre affinity) and Developer Mode (competitive benchmarks, audience overlap).
- **Out-of-Scope Uses:** Does not perform real-time price optimization or bidding. Cannot recommend un-indexed games not present in the catalog snapshot.

---

## 3. Training Data & Leakage Guarantees
- **Dataset Version:** `c662b8232dc0`
- **Temporal Splitting:** Strict chronological cutoffs per user (train $\rightarrow$ val $\rightarrow$ test).
- **Leak-Free Guarantee:**
  - `clean: true` | `leak_free: true` | `deployable: true`
  - Item popularity features strictly computed from the training split (`log_train_pop`). Zero future interaction exposure.
  - Seen-item masking enforced at inference time ($-10^9$ score on previously interacted items).

---

## 4. Quantitative Performance & Empirical Evidence
Evaluated across 5,000 held-out evaluation users with 95% bootstrap confidence intervals (Champion Seed 44 and 5-seed sweep across seeds 42–46):

| Metric | Champion Validation (95% CI) | Champion Test (95% CI) | 5-Seed Validation (Mean ± Std) | 5-Seed Test (Mean ± Std) |
|---|:---:|:---:|:---:|:---:|
| **NDCG@10** | 0.0611 [0.0557, 0.0662] | **0.0513** [0.0467, 0.0563] | 0.0613 ± 0.0010 | 0.0528 ± 0.0019 |
| **Hit Rate@10** | 0.1190 [0.1104, 0.1280] | **0.1016** [0.0936, 0.1102] | 0.1204 ± 0.0026 | 0.1030 ± 0.0030 |
| **MRR@10** | 0.0436 [0.0391, 0.0481] | **0.0362** [0.0322, 0.0403] | 0.0434 ± 0.0011 | 0.0368 ± 0.0009 |
| **SNIPS-NDCG@10** | 0.0338 [0.0305, 0.0373] | **0.0260** [0.0229, 0.0291] | 0.0342 ± 0.0014 | 0.0264 ± 0.0009 |
| **Catalog Coverage@10** | 43.47% | **43.70%** | 41.89% | 43.10% |
| **Novelty@10** | 9.25 bits | **9.34 bits** | 9.27 bits | 9.32 bits |
| **Diversity@10** | 0.7588 | **0.7627** | 0.7603 | 0.7618 |

- **Comparison vs Item-kNN Baseline:**
  - Validation Paired Diff: **`+0.0058`** (95% CI: `[+0.0008, +0.0111]`) — Statistically significant win ($p < 0.05$).
  - Test Paired Diff: **`+0.0080`** (95% CI: `[+0.0033, +0.0131]`) — Statistically significant win ($p < 0.001$).

---

## 5. Hyperparameters
```json
{
  "hidden_dim": 256,
  "output_dim": 64,
  "pool": "attn",
  "id_dropout": 0.25,
  "proxy_dropout": 0.25,
  "no_pop_proxies": true,
  "numeric_dim": 6,
  "use_id": true,
  "use_text": true,
  "use_tags": true,
  "use_numeric": true,
  "max_history": 50
}
```

---

## 6. Limitations & Known Caveats
1. **Catalog Snapshot:** Recommendation space is restricted to games crawled in the data ingestion pipeline; newly released titles require re-indexing.
2. **Crawl-Time Price Limitation:** Prices reflect snapshot crawl values rather than dynamic Steam sales.
3. **Cold-Start Users:** Users with $<3$ interactions fall back to popularity and genre-conditioned priors.
