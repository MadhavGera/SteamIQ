# Model Card: `recsys_two_tower_clean`

**Family:** Two-Tower Retrieval | **Stage:** `production` (Headline Champion)  
**Run ID:** `eb9a3374-2aad-476f-a25e-5d3b84261b7f` | **Generated:** 2026-10-17  
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
| **NDCG@10** | 0.0611 [0.0562, 0.0661] | 0.0516 [0.0473, 0.0562] |
| **Hit Rate@10** | 0.1190 [0.1104, 0.1280] | 0.1028 [0.0954, 0.1108] |
| **MRR@10** | 0.0436 [0.0394, 0.0478] | 0.0361 [0.0324, 0.0400] |
| **Catalog Coverage@10** | 43.47% | 43.83% |
| **Novelty@10** | 9.25 bits | 9.34 bits |
| **Diversity@10** | 0.7588 | 0.7629 |

---

## 5. Hyperparameters
```json
{
  "hidden_dim": 256,
  "output_dim": 64,
  "pool": "attn",
  "id_dropout": 0.25,
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
