# SteamIQ Recommender System — Project Review Q&A

This document contains 25 essential questions and answers prepared for the technical architecture review. Every answer is grounded directly in code, database schemas, ADRs, or generated evaluation reports.

---

### Q1: Why use a two-stage retrieve-and-rerank architecture?
**Answer:** A two-stage pipeline separates candidate retrieval speed from ranking expressiveness. In Stage 1, fast multi-source retrievers (Two-Tower dual encoders, Item-kNN, ALS, and popularity) retrieve the Top-50 candidate pool in $\sim 2\text{ ms}$ over the entire 3,000-item catalog. In Stage 2, a LightGBM LambdaMART ranker uses 33 clean cross-features (pairwise cosine similarities, price ratios, review quality, and multi-source ranks) to score and re-rank the candidates with high precision.  
**Evidence:** [`backend/recsys/ranker.py`](backend/recsys/ranker.py), [`backend/jobs/train_ranker.py`](backend/jobs/train_ranker.py), [`docs/steamiq-0005-online-recommender-inference.md`](docs/steamiq-0005-online-recommender-inference.md).

---

### Q2: Why use in-batch softmax with logQ frequency correction in the Two-Tower model?
**Answer:** In-batch negative sampling allows efficient training without computing full catalog partition functions. However, popular items are sampled as in-batch negatives proportionally to their interaction frequency $p_j$. Without correction, the model over-penalizes popular items. Subtracting $\log p_j$ from logits ($s_{ij} - \log p_j$) removes this sampling artifact and yields unbiased maximum likelihood estimation.  
**Evidence:** [`backend/recsys/models/two_tower.py`](backend/recsys/models/two_tower.py), [`backend/jobs/train_two_tower.py`](backend/jobs/train_two_tower.py).

---

### Q3: What does ID dropout do in the Two-Tower item tower?
**Answer:** During training, item ID embeddings are randomly dropped to the unknown token with probability $p = 0.25$. This forces the neural network to avoid relying exclusively on memorized item IDs and instead learn robust representations from semantic text embeddings (`all-MiniLM-L6-v2`), multi-hot tags, and pricing. In ablations, removing ID dropout drops NDCG@10 from $0.0548 \pm 0.0007$ to $0.0507$ ($\Delta = -0.0032$, $p < 0.05$).  
**Evidence:** [`backend/data/recsys/reports/ablations.json`](backend/data/recsys/reports/ablations.json), [`backend/recsys/models/two_tower.py`](backend/recsys/models/two_tower.py).

---

### Q4: Why use a per-user chronological leave-one-out split rather than random or global time splits?
**Answer:** Random user-item splits leak future interactions into the training matrix and inflate metrics. A global calendar cutoff discards users who were only active in earlier periods. The per-user chronological split orders each user's positive reviews by timestamp: the latest interaction is the **test** target, the second-to-latest is the **validation** target, and all earlier events form the **training** sequence. This accurately benchmarks next-item fold-in recommendation without temporal lookahead.  
**Evidence:** [`backend/recsys/data.py`](backend/recsys/data.py), [`backend/jobs/prepare_recsys_data.py`](backend/jobs/prepare_recsys_data.py).

---

### Q5: Why is pessimistic tie handling required in ranking evaluations?
**Answer:** When multiple candidate items receive identical predicted scores (common in popularity or sparse content baselines), optimistic tie-breaking (assigning the top rank) artificially inflates NDCG and Hit Rate. SteamIQ applies pessimistic tie handling: tied candidates receive the average or worst rank among the tied set ($\text{rank}_{\text{worst}} = 1 + |\{j : s_{u,j} \ge s_{u,i}\}|$), guaranteeing honest benchmark evaluations.  
**Evidence:** [`backend/recsys/evaluate.py`](backend/recsys/evaluate.py), [`backend/tests/test_recsys_eval_audit.py`](backend/tests/test_recsys_eval_audit.py).

---

### Q6: Why compute 1,000-iteration bootstrap confidence intervals and paired user-level differences?
**Answer:** Point metric estimates do not reveal whether differences between model versions are real or stochastic noise. We compute non-parametric percentile 95% confidence intervals over 1,000 bootstrap resamples on the 5,000 evaluation users. For head-to-head comparisons, we compute user-level paired differences $\Delta_u = \text{NDCG}_A(u) - \text{NDCG}_B(u)$; a difference is declared statistically significant only if the 95% CI excludes zero.  
**Evidence:** [`backend/recsys/evaluate.py`](backend/recsys/evaluate.py), [`backend/data/recsys/reports/leaderboard.md`](backend/data/recsys/reports/leaderboard.md).

---

### Q7: How is item cold start evaluated, and what do the true holdout vs masking studies prove?
**Answer:** Cold-start items lack interaction histories. SteamIQ distinguishes between primary prospective evidence and optimistic proxy simulations:
1. **Primary Benchmark: True Item Holdout ($N = 184$ users, 300 items withheld):** 10% torso/tail items are completely excluded from training set interactions and models retrained from scratch on 90% of the catalog. Two-Tower achieves $\text{NDCG@10} = 0.0000$ ($0 / 184$ hits) with a mean catalog rank of $2,126.6$ ($[2,015.1, 2,238.7]$), proving that content representations alone without collaborative interaction density cannot break into the top 10. Pure Content Cosine achieves $\text{NDCG@10} = 0.0078$ ($4 / 184$ hits) with a mean rank of $1,084.6$.
2. **Optimistic Proxy Simulation: Masked Target ID ($N = 553$ users, target $\le 10$ train positives):** When target item ID is masked at inference on a model trained with the full catalog (using ID dropout $p=0.25$), Two-Tower achieves $\text{NDCG@10} = 0.0036$ ($5 / 553$ hits) with a mean catalog rank of $1,143.5$ ($[1,074.3, 1,217.0]$), showing that pre-trained text and tag embeddings provide a viable semantic prior.  
**Evidence:** [`backend/data/recsys/reports/cold_start.json`](backend/data/recsys/reports/cold_start.json), [`docs/ml-accuracy-and-evaluation-guide.md`](docs/ml-accuracy-and-evaluation-guide.md).

---

### Q8: What does popularity bias look like in the empirical results?
**Answer:** In the beyond-accuracy report, the Popularity baseline exhibits extreme exposure concentration: 100% of its recommendations come from the top 20% most popular games (`head_share_at_10: 1.0`), with a Gini inequality of $0.9965$ and catalog coverage of just $0.70\%$. In contrast, `recsys_two_tower_clean` achieves $43.83\%$ catalog coverage, Gini of $0.9299$, and intra-list diversity of $0.7629$.  
**Evidence:** [`backend/data/recsys/reports/beyond_accuracy.json`](backend/data/recsys/reports/beyond_accuracy.json), [`backend/data/recsys/reports/leaderboard.md`](backend/data/recsys/reports/leaderboard.md).

---

### Q9: Why is Implicit ALS a strong collaborative baseline?
**Answer:** Alternating Least Squares on implicit feedback treats unobserved user-item pairs as negative signals with low confidence, while positive reviews scale confidence as $c_{ui} = 1 + \alpha r_{ui}$ ($\alpha=40$). It achieves $\text{HR@10} = 0.0776$ and $\text{NDCG@10} = 0.0404$ on the test split, outperforming content cosine ($0.0146$) and popularity ($0.0151$).  
**Evidence:** [`backend/recsys/models/als.py`](backend/recsys/models/als.py), [`backend/data/recsys/reports/leaderboard.md`](backend/data/recsys/reports/leaderboard.md).

---

### Q10: How are recommendation explanations generated in serving?
**Answer:** For the Two-Tower model, explainability is generated via cosine nearest-neighbor anchor retrieval against the user's seed games ("Because you liked: [Game Title]"). For the LightGBM Ranker, TreeSHAP computes exact per-feature attributions for the top-scoring candidate. For baselines without attribution capabilities (ALS, Popularity), no explanation UI is rendered.  
**Evidence:** [`backend/recsys/inference.py`](backend/recsys/inference.py), [`frontend/components/recsys/RecsysResultCard.tsx`](frontend/components/recsys/RecsysResultCard.tsx).

---

### Q11: How is the ADR 0005 Golden Rule exception justified for online inference?
**Answer:** The architecture rule strictly prohibits route handlers from executing expensive DB queries or on-the-fly training. ADR 0005 defines an explicit exception for `RecommenderService`: it loads fixed-size 64-d item vectors into RAM on startup and computes user history mean vectors and matrix dot products ($O(N \times 64)$) entirely in-memory. Latency is $<3\text{ ms}$ for 3,000 items, and responses include strict provenance metadata.  
**Evidence:** [`docs/steamiq-0005-online-recommender-inference.md`](docs/steamiq-0005-online-recommender-inference.md), [`backend/recsys/inference.py`](backend/recsys/inference.py).

---

### Q12: What would you do with real online feedback in a live deployment?
**Answer:** We would log impressions, click-through rates (CTR), purchase conversions, and playtime hours. Using these logs, we would deploy an online contextual multi-armed bandit (LinUCB / Thompson Sampling) or dynamic $\epsilon$-greedy exploration to continuously balance exploitation of high-conversion titles with exploration of tail catalog games.  
**Evidence:** [`docs/steamiq-0005-online-recommender-inference.md`](docs/steamiq-0005-online-recommender-inference.md).

---

### Q13: What was the all-time-stats leakage, and how was it discovered and resolved?
**Answer:** During initial experiments, the Stage-2 ranker used all-time total review counts and positive review percentages computed through the entire dataset. This leaked future engagement into past split targets. We resolved this by building clean feature pipelines (`recsys_two_tower_clean` and `recsys_two_tower_ranker_clean`) that compute popularity and review counts strictly on the training partition. All-time models are retained solely as non-deployable diagnostic reference benchmarks.  
**Evidence:** [`backend/jobs/train_ranker.py`](backend/jobs/train_ranker.py), [`backend/data/recsys/reports/leaderboard.md`](backend/data/recsys/reports/leaderboard.md).

---

### Q14: What is the pre-registered model promotion rule, and why is `recsys_two_tower_clean` the champion?
**Answer:** Under the pre-registered tournament rule, the Clean Ranker must win over Clean Two-Tower on tuning-set validation NDCG@10 by more than its standard deviation: $\Delta_{\text{tuning}} > \max(\sigma_{\text{ranker}}, \sigma_{\text{two\_tower}})$. In multi-seed tuning evaluation, the Clean Ranker achieved $0.0621 \pm 0.0022$ vs Clean Two-Tower $0.0615 \pm 0.0007$ ($\Delta = +0.0006$, threshold $= 0.0022$). Because the difference did not exceed the decision threshold, `recsys_two_tower_clean` won the tournament and was promoted to `production`.  
**Evidence:** [`backend/jobs/promote_recsys.py`](backend/jobs/promote_recsys.py), [`backend/recsys/registry.py`](backend/recsys/registry.py), [`backend/data/recsys/reports/leaderboard.md`](backend/data/recsys/reports/leaderboard.md).

---

### Q15: What causes metric variance across multi-seed training runs?
**Answer:** When evaluating PyTorch Two-Tower and LightGBM across multiple random seeds (`[42, 43, 44, 45, 46]`), minor metric variance occurs due to stochastic in-batch negative sampling, mini-batch shuffling order, and gradient descent trajectories. Reporting multi-seed mean $\pm$ standard deviation ($0.0541 \pm 0.0007$ over 3 seeds for Two-Tower clean, $0.0500 \pm 0.0022$ over 5 seeds for Ranker clean) ensures we report true expected performance rather than an overfitted single lucky seed.  
**Evidence:** [`backend/data/recsys/reports/leaderboard.md`](backend/data/recsys/reports/leaderboard.md), [`backend/recsys/evaluate.py`](backend/recsys/evaluate.py).

---

### Q16: What is the Stage-2 Ranker candidate training miss rate and serving architecture?
**Answer:** During Stage-2 LambdaMART training, candidates are pooled from Stage-1 retrievers up to $k=300$. For $\sim 39\%$ of training instances, the user's ground-truth target item falls outside the top-300 Stage-1 candidate pool, placing an upper ceiling on ranker training recall. In online serving, `RecommenderService` retrieves Top-50 candidates from Two-Tower and applies LightGBM re-ranking. Models operating under online serving approximations expose `is_preview: true` with a UI badge and tooltip explaining that serving-path retrieval differs from offline full-catalog test numbers.  
**Evidence:** [`backend/jobs/train_ranker.py`](backend/jobs/train_ranker.py), [`backend/api/recsys.py`](backend/api/recsys.py), [`frontend/components/recsys/RecsysModelSelector.tsx`](frontend/components/recsys/RecsysModelSelector.tsx).

---

### Q17: Why do non-game software items (like Wallpaper Engine) exist in the universe?
**Answer:** The dataset is derived from Steam store catalogs where utility software (e.g. Wallpaper Engine, Soundpad) have high positive interaction volumes. Unless filtered out by strict application-type metadata, they enter the 3,000-item universe. Their multi-hot tags and embeddings accurately cluster them into software/utility neighborhoods.  
**Evidence:** [`backend/jobs/audit_recsys_data.py`](backend/jobs/audit_recsys_data.py), [`backend/data/recsys/reports/dataset.json`](backend/data/recsys/reports/dataset.json).

---

### Q18: How is Item-kNN formulated and regularized?
**Answer:** Item-kNN computes item-item collaborative similarity from the user-item interaction matrix using cosine similarity with shrinkage regularizer $\alpha = 100$: $s(i, j) = \frac{r_i \cdot r_j}{\|r_i\|_2 \|r_j\|_2 + \alpha}$ with top-100 neighbors ($k=100$) retained per item. Shrinkage penalizes item pairs that co-occur rarely by chance, stabilizing neighbor similarity scores against noise.  
**Evidence:** [`backend/recsys/models/itemknn.py`](backend/recsys/models/itemknn.py).

---

### Q19: How are the 64-dimensional Two-Tower embeddings projected to 2D for the Embedding Map?
**Answer:** The offline job `jobs.project_item_embeddings` extracts the 64-d item vector table from the production run, executes UMAP (`n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42`) to produce $(x, y)$ coordinates, fits KMeans ($k=20$), and labels each cluster with its 2 most distinctive tags by lift score. Results are stored in `serving_rec_item_projection`.  
**Evidence:** [`backend/jobs/project_item_embeddings.py`](backend/jobs/project_item_embeddings.py), [`frontend/app/embedding-map/page.tsx`](frontend/app/embedding-map/page.tsx).

---

### Q20: Why are already-consumed items masked to $-\infty$ during evaluation and serving?
**Answer:** In user-facing recommender systems, users do not want recommendations for games they have already purchased or reviewed. Masking seen items to $-\infty$ ensures that top-$k$ lists consist exclusively of novel discoveries.  
**Evidence:** [`backend/recsys/evaluate.py`](backend/recsys/evaluate.py), [`backend/recsys/inference.py`](backend/recsys/inference.py).

---

### Q21: How is catalog coverage measured, and why does unmasked coverage look misleadingly higher?
**Answer:** Official catalog coverage@10 ($43.83\%$ test for Two-Tower clean) is measured after masking user history. If computed without masking, coverage appears artificially higher ($70.83\%$) because models repetitively recommend already-consumed items from the user's history across different users. Masked evaluation is the only valid measure for new recommendation reach.  
**Evidence:** [`backend/data/recsys/reports/limitations.json`](backend/data/recsys/reports/limitations.json), [`backend/recsys/evaluate.py`](backend/recsys/evaluate.py).

---

### Q22: What is the empirical impact of removing Sentence-Transformers MiniLM text embeddings?
**Answer:** In Two-Tower ablations, dropping the 384-d text projection branch (`--no-text`) causes test NDCG@10 to fall from $0.0548 \pm 0.0007$ to $0.0482 \pm 0.0009$ ($\Delta = -0.0070$ $[-0.0109, -0.0032]$, $p < 0.05$). This confirms that deep semantic text embeddings provide critical ranking signal.  
**Evidence:** [`backend/data/recsys/reports/ablations.json`](backend/data/recsys/reports/ablations.json), [`backend/data/recsys/reports/leaderboard.md`](backend/data/recsys/reports/leaderboard.md).

---

### Q23: What is the role of Maximal Marginal Relevance (MMR) re-ranking in production?
**Answer:** MMR balances relevance against intra-list redundancy: $\text{MMR}(u, i) = \lambda \cdot s(u, i) - (1 - \lambda) \max_{j \in S} \text{Sim}(i, j)$. Tuning $\lambda = 0.8$ increases list diversity from $0.7564$ to $0.7762$ with a negligible accuracy drop of $\Delta \text{NDCG} = -0.0004$ ($[-0.0014, +0.0005]$, not significant), preventing homogeneous top-10 lists.  
**Evidence:** [`backend/data/recsys/reports/beyond_accuracy.json`](backend/data/recsys/reports/beyond_accuracy.json), [`backend/recsys/evaluate.py`](backend/recsys/evaluate.py).

---

### Q24: How are unknown app IDs handled when a user enters games outside the 3,000-item universe?
**Answer:** If a user submits an app ID not present in `serving_rec_items`, `RecommenderService` filters it out, includes the unknown ID in `unknown_app_ids`, sets a warning message, and proceeds with inference over all recognized items. The frontend displays an alert banner listing the omitted IDs.  
**Evidence:** [`backend/recsys/inference.py`](backend/recsys/inference.py), [`frontend/app/page.tsx`](frontend/app/page.tsx).

---

### Q25: What are the seven primary known limitations of the dataset and recommender system?
**Answer:**
1. **Implicit Feedback from Reviews Only:** Interactions derive from reviews (`is_recommended=true`); non-purchaser negative impressions are absent.
2. **Popularity and Selection Bias:** Highly reviewed titles receive disproportionate interactions.
3. **Per-User Temporal Split Protocol:** Evaluates next-item prediction per user sequence rather than a single global calendar cutoff.
4. **Item Universe Selection from Full Period:** 3,000 items chosen based on aggregate period counts to stabilize metadata.
5. **Single Held-Out Item Target:** Exactly one target per user in test and val (Hit Rate equals Recall).
6. **Offline Metrics vs Online Performance:** High offline NDCG reflects historical review retrieval, not guaranteed purchase conversion.
7. **Simulated Cold-Start Evaluation:** Item content embeddings are pre-trained on catalog text, meaning offline simulations benchmark semantic alignment rather than live prospective adoption.  
**Evidence:** [`backend/data/recsys/reports/limitations.json`](backend/data/recsys/reports/limitations.json), [`docs/ml-accuracy-and-evaluation-guide.md`](docs/ml-accuracy-and-evaluation-guide.md).
