# Model Card: `recsys_two_tower_clean`

**Family:** Two-Tower Retrieval | **Stage:** `production` (Headline Champion)  
**Run ID:** `phase4_seed_42` | **Generated:** 2026-10-17  
**License:** Project Internal / MIT  

---

## 1. Model Overview & Architecture
`recsys_two_tower_clean` is the headline production retrieval model for SteamIQ. It maps user library histories and game items into a shared 64-dimensional semantic embedding space.
- **Query Tower:** Self-attention pooling (`pool='attn'`) over historical item representations with ID dropout (`id_dropout=0.25`) to prevent overfitting to high-volume user items.
- **Candidate Tower:** Deep content projections combining item IDs, MiniLM-L6 text representations, sparse tag projections, and log-transformed item statistics.
- **Loss:** In-batch sampled softmax loss with temperature scaling over cosine distances.
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
Evaluated across 5,000 held-out evaluation users with 95% bootstrap confidence intervals:

| Metric | Validation Split (95% CI) | Test Split (95% CI) |
|---|:---:|:---:|
| **NDCG@10** | 0.0474 [0.0427, 0.0519] | 0.0415 [0.0376, 0.0457] |
| **Hit Rate@10** | 0.0932 [0.0850, 0.1012] | 0.0828 [0.0756, 0.0902] |
| **MRR@10** | 0.0337 [0.0297, 0.0375] | 0.0291 [0.0257, 0.0327] |
| **Catalog Coverage@10** | 12.93% | 12.67% |
| **Novelty@10** | 8.66 bits | 8.71 bits |
| **Diversity@10** | 0.7547 | 0.7571 |

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
