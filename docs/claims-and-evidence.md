# SteamIQ Recommender: Claims and Evidence

One-page reference for the review. Every quantitative claim maps directly to a reproducible, authoritative evidence file in `docs/results/`.

---

## Authoritative Claims & Evidence Mapping

| # | Claim | Evidence File | Key Numbers & Statistical Verdict |
|---|-------|---------------|-----------------------------------|
| 1 | **Clean Two-Tower beats all baselines on both splits** | [`leaderboard.json`](results/leaderboard.json) | Champion (Seed 44) Test NDCG@10 **0.0513** [0.0467, 0.0563] vs Item-kNN 0.0433 (Paired Diff: **+0.0080** [+0.0033, +0.0131], $p < 0.001$), ALS 0.0356, Pop 0.0151. 5-Seed Test: **0.0528 ± 0.0019**; Val: **0.0613 ± 0.0010** vs Item-kNN 0.0553 (+0.0058 [+0.0008, +0.0111], $p < 0.05$). |
| 2 | **Ranker promotion gate failed with clean train-only stats** | [`leaderboard.json`](results/leaderboard.json) | Ranker Test NDCG@10 **0.0532** [0.0480, 0.0587] vs Two-Tower 0.0513. Paired diff **+0.0019** [−0.0024, +0.0062] — CI strictly spans 0 ($\text{lo95} > 0$ is False). Two-Tower remains headline production champion. |
| 3 | **Two-Tower architectural ablations prove modality value** | [`ablations.json`](results/ablations.json) | Evaluated on validation split across 3 seeds (42, 43, 44) at full 15-epoch budget: text and tag projections provide statistically significant gains ($p < 0.05$): MiniLM text (**-0.0091** [-0.0117, -0.0067], sig), tags (**-0.0023** [-0.0043, -0.0005], sig), standardized numeric (**-0.0004** [-0.0023, +0.0015], ns), item ID (**+0.0005** [-0.0018, +0.0027], ns). |
| 4 | **Ranker ablations confirm feature contributions** | [`ranker_ablations.json`](results/ranker_ablations.json) | Evaluated on validation split (Seed 44, 35 features): removing source retrieval scores (**-0.0033** [-0.0058, -0.0007]), price features (**-0.0036** [-0.0062, -0.0010]), and item quality (**-0.0055** [-0.0081, -0.0029]) causes statistically significant validation degradations ($p < 0.05$). |
| 5 | **SASRec challenger delivers strong sequential retrieval and candidate recall lift** | [`sasrec_benchmark.json`](results/sasrec_benchmark.json) | Champion (Seed 42) Test NDCG@10 **0.0569** [0.0517, 0.0617] (3-Seed Test **0.0545 ± 0.0025**; 3-Seed Val **0.0609 ± 0.0016** vs Two-Tower 0.0611, paired diff: **-0.0009** [-0.0056, +0.0034]). Measured CPU latency $P_{95} = \mathbf{2.70\text{ ms}}$ (< 50 ms SLA; unified benchmark $P_{95} = \mathbf{3.78\text{ ms}}$). Quota merge candidate recall lift = **+22.5% relative** (41.8% $\to$ 51.2%). Following bugfix of token alignment and causal masking, re-selection sweep selected $d=128$, dropout=0.2, no-text configuration. |
| 6 | **Fair Cold-Start Multi-Cohort Evaluation yields WEAK evidence** | [`cold_start.json`](results/cold_start.json) | 4 mutually disjoint cohorts (1 tuning + 3 eval, 800 withheld items with zero training interactions); zero-only routing variant achieves Val NDCG@10 **0.0407** [0.0239, 0.0577] on cold targets (Group a), exceeding cold-only random (0.0228); warm-target users (Group b) suffer degradation ($\Delta = \mathbf{-0.0044}$ [-0.0050, -0.0038]). Graded **WEAK** by mechanical rule due to warm-target damage; old inflated over-random claim purged. |
| 7 | **Ask SteamIQ selective prediction achieves high precision** | [`assistant_metrics.json`](results/assistant_metrics.json) | Selective accuracy rises from 92.51% (raw) to **95.92%** at $\tau = 0.630$ (91.93% coverage, 8.07% abstention); Temperature scaling ($T=0.183$) achieves ECE **0.0231**; generalization audit proves 95.68% non-overlapping queries with **92.17%** accuracy. |
| 8 | **Serving reproduces offline evaluation within noise** | [`evidence/serve_check.log`](evidence/serve_check.log) | Val −0.0012 [−0.0033, +0.0010]; Test +0.0007 [−0.0012, +0.0024]; 0 order mismatches across stage-1 models. |
| 9 | **Statistical drift detectors show zero FPR on null resamples** | [`evidence/drift_detector_power.log`](evidence/drift_detector_power.log) | Null FPR **0.0000** for PSI/JSD, Power **0.82–1.00** on moderate shifts. |
| 10 | **Test split evaluation is locked and protected** | [`.test_eval_lock`](results/.test_eval_lock) | Test split evaluated exactly once under locked manifest; reruns strictly prevented without documented audit reason. |

---

## Rehearsal Q&A

### Why didn't the Ranker beat the Two-Tower Champion?
In early unconstrained iterations (`ranker_v1`), the ranker achieved inflated NDCG gains due to all-time review totals leaking future popularity. Once strictly retrained with train-only item statistics (`train_stats_only: true`), the paired bootstrap difference against the Two-Tower champion on 5,000 test users is **`+0.0019`** with a 95% CI of **`[-0.0024, +0.0062]`** that includes zero. Under the pre-registered promotion gate, the ranker failed to achieve statistically significant superiority, so `recsys_two_tower_clean` remains the production serving champion.

### Did budget truncation cause the Two-Tower regression in Phase 4?
Yes, confirmed in Phase A and resolved in Phase B. In Phase 4, the sweep script capped training at 8 epochs and 60,000 subsampled pairs/epoch with patience 3, starving the model (>5.15x fewer gradient steps). Training loss was still dropping and had not converged. Restoring the full budget (15 epochs on all 145,741 next-item pairs per epoch derived from 165,741 train interactions across 20,000 users, total ~2.18M exposure pairs, patience 5) restored Val NDCG@10 to **0.0613 ± 0.0010** (Seed 44 = **0.0611**), statistically significantly beating Item-kNN (**+0.0058** [+0.0008, +0.0111], $p < 0.05$).

### How were multi-seed reporting inconsistencies resolved?
Under Issue I2, `backend/recsys/aggregation.py` was introduced. It computes true sample means and standard deviations across all metrics for neural sweeps (5 seeds for Two-Tower and Ranker, 3 seeds for SASRec). Deterministic baselines (Item-kNN) explicitly report `"n/a"` standard deviations (`0.0433 (n/a)`) instead of misleading `0.0000`. An automated guard (`verify_aggregation_integrity`) strictly rejects any payload where multi-seed means identically match single-seed champion values across varying metrics.

### Is cold start solved?
No. In Round 3 and Round 4, we corrected the cold-start benchmark by partitioning items into 4 mutually disjoint cohorts (1 tuning cohort of 200 items + 3 eval cohorts of 200 items each, total 800 withheld items with zero training interactions) and evaluating against a fair cold-only random expectation rather than a catalog-wide random baseline. In Round 4, `route_zero_only` routing surfaces cold targets with Val NDCG@10 = **0.0407 [0.0239, 0.0577]**, beating cold-only random (0.0228). However, on warm-target users (Group b), routing introduces statistically significant damage ($\Delta = \mathbf{-0.0044}$ `[-0.0050, -0.0038]`, $p < 0.05$). The mechanical grading rule classifies the evidence as **WEAK**, and the earlier inflated over-random claim is refuted and purged.

---

## Known Limitations

1. **Crawl-Time Price Snapshots** — `price_final` and `discount` reflect a single Steam store crawl, not dynamic prices at each user's interaction time.
2. **All-Period Item Universe** — The 3,000-item universe was selected using all-period review counts (design trade-off for metadata stability).
3. **Synthetic Paraphrase Assistant Holdout** — The 167 queries in `holdout_queries.jsonl` are synthetic developer paraphrases, not real external user queries. A contributor scaffold (`holdout_queries_human.jsonl`) is established with an automated $\ge 50$ query threshold before human metrics are reported.
4. **Implicit Feedback Only** — User histories derive from positive reviews; non-purchaser negative impressions and un-reviewed purchases are absent.
5. **Offline Simulation Boundaries** — True item holdout withheld games completely from training, but still utilized descriptions and tags curated over the game's lifecycle. Prospective live A/B evaluation is required for live adoption tracking.
