# SteamIQ Recommender: Claims and Evidence

One-page reference for the review. Every quantitative claim maps directly to a reproducible, authoritative evidence file in `docs/results/`.

---

## Authoritative Claims & Evidence Mapping

| # | Claim | Evidence File | Key Numbers & Statistical Verdict |
|---|-------|---------------|-----------------------------------|
| 1 | **Clean Two-Tower beats all baselines on both splits** | [`leaderboard.json`](results/leaderboard.json) | Champion (Seed 44) Test NDCG@10 **0.0513** [0.0467, 0.0563] vs Item-kNN 0.0433 (Paired Diff: **+0.0080** [+0.0033, +0.0131], $p < 0.001$), ALS 0.0356, Pop 0.0151. 5-Seed Test: **0.0528 ± 0.0019**; Val: **0.0613 ± 0.0010** vs Item-kNN 0.0553 (+0.0058 [+0.0008, +0.0111], $p < 0.05$). |
| 2 | **Ranker promotion gate failed with clean train-only stats** | [`leaderboard.json`](results/leaderboard.json) | Ranker Test NDCG@10 **0.0532** [0.0480, 0.0587] vs Two-Tower 0.0513. Paired diff **+0.0019** [−0.0024, +0.0062] — CI strictly spans 0 ($\text{lo95} > 0$ is False). Two-Tower remains headline production champion. |
| 3 | **Two-Tower architectural ablations prove modality value** | [`ablations.json`](results/ablations.json) | All modalities statistically significant ($p < 0.05$ across 3 seeds): MiniLM text (**-0.0259** [-0.0290, -0.0228]), tags (**-0.0148**), numeric (**-0.0146**), item ID (**-0.0132**). |
| 4 | **Ranker ablations confirm feature contributions** | [`ranker_ablations.json`](results/ranker_ablations.json) | Removing source retrieval scores (**-0.0033** [-0.0058, -0.0007]), price features (**-0.0036**), and item quality (**-0.0055**) causes statistically significant test degradations ($p < 0.05$). |
| 5 | **SASRec challenger underperforms Two-Tower on short histories** | [`sasrec_benchmark.json`](results/sasrec_benchmark.json) | Test NDCG@10 **0.0165** [0.0139, 0.0192] (Paired diff vs TT: **-0.0348** [-0.0399, -0.0299]). Measured CPU latency $P_{95} = \mathbf{1.71\text{ ms}}$ (< 50 ms SLA). Quota merge candidate recall lift = **+16.5% relative** (41.8% $\to$ 48.7%). |
| 6 | **Calibrated Routed Hybrid achieves STRONG cold-start evidence** | [`cold_start.json`](results/cold_start.json) | 3 distinct cohorts (779 unique withheld items, 601 validation users); Calibrated Routed Hybrid achieves Val NDCG@10 **0.0335** [0.0198, 0.0475], **22.2x over random** (0.0015); $\text{lo95} > 10\times\text{random}$ satisfies **STRONG** evidence status. |
| 7 | **Ask SteamIQ selective prediction achieves high precision** | [`assistant_metrics.json`](results/assistant_metrics.json) | Selective accuracy rises from 92.51% (raw) to **95.92%** at $\tau = 0.630$ (91.93% coverage, 8.07% abstention); Temperature scaling ($T=0.183$) achieves ECE **0.0231**; generalization audit proves 95.68% non-overlapping queries with **92.17%** accuracy. |
| 8 | **Serving reproduces offline evaluation within noise** | [`evidence/serve_check.log`](evidence/serve_check.log) | Val −0.0012 [−0.0033, +0.0010]; Test +0.0007 [−0.0012, +0.0024]; 0 order mismatches across stage-1 models. |
| 9 | **Statistical drift detectors show zero FPR on null resamples** | [`evidence/drift_detector_power.log`](evidence/drift_detector_power.log) | Null FPR **0.0000** for PSI/JSD, Power **0.82–1.00** on moderate shifts. |
| 10 | **Test split evaluation is locked and protected** | [`.test_eval_lock`](results/.test_eval_lock) | Test split evaluated exactly once under locked manifest; reruns strictly prevented without documented audit reason. |

---

## Rehearsal Q&A

### Why didn't the Ranker beat the Two-Tower Champion?
In early unconstrained iterations (`ranker_v1`), the ranker achieved inflated NDCG gains due to all-time review totals leaking future popularity. Once strictly retrained with train-only item statistics (`train_stats_only: true`), the paired bootstrap difference against the Two-Tower champion on 5,000 test users is **`+0.0019`** with a 95% CI of **`[-0.0024, +0.0062]`** that includes zero. Under the pre-registered promotion gate, the ranker failed to achieve statistically significant superiority, so `recsys_two_tower_clean` remains the production serving champion.

### Did budget truncation cause the Two-Tower regression in Phase 4?
Yes, confirmed in Phase A and resolved in Phase B. In Phase 4, the sweep script capped training at 8 epochs and 60,000 subsampled pairs/epoch with patience 3, starving the model (>5.15x fewer gradient steps). Training loss was still dropping and had not converged. Restoring the full budget (15 epochs, ~164,749 pairs/epoch $\approx$ 2.47M exposure pairs, patience 5) restored Val NDCG@10 to **0.0613 ± 0.0010** (Seed 44 = **0.0611**), statistically significantly beating Item-kNN (**+0.0058** [+0.0008, +0.0111], $p < 0.05$).

### How were multi-seed reporting inconsistencies resolved?
Under Issue I2, `backend/recsys/aggregation.py` was introduced. It computes true sample means and standard deviations across all metrics for neural sweeps (5 seeds for Two-Tower and Ranker, 3 seeds for SASRec). Deterministic baselines (Item-kNN) explicitly report `"n/a"` standard deviations (`0.0433 (n/a)`) instead of misleading `0.0000`. An automated guard (`verify_aggregation_integrity`) strictly rejects any payload where multi-seed means identically match single-seed champion values across varying metrics.

### Is cold start solved?
We do not claim cold start is solved. However, Phase E established **STRONG** empirical evidence: across 3 distinct cohorts (779 unique withheld items that had zero training interactions), the Calibrated Routed Hybrid achieves pooled Val NDCG@10 = **0.0335 [0.0198, 0.0475]**, which is **22.2x over the analytic random baseline** (0.0015). The lower 95% bound (0.0198) strictly exceeds 10x random, satisfying the rigorous threshold for strong evidence.

---

## Known Limitations

1. **Crawl-Time Price Snapshots** — `price_final` and `discount` reflect a single Steam store crawl, not dynamic prices at each user's interaction time.
2. **All-Period Item Universe** — The 3,000-item universe was selected using all-period review counts (design trade-off for metadata stability).
3. **Synthetic Paraphrase Assistant Holdout** — The 167 queries in `holdout_queries.jsonl` are synthetic developer paraphrases, not real external user queries. A contributor scaffold (`holdout_queries_human.jsonl`) is established with an automated $\ge 50$ query threshold before human metrics are reported.
4. **Implicit Feedback Only** — User histories derive from positive reviews; non-purchaser negative impressions and un-reviewed purchases are absent.
5. **Offline Simulation Boundaries** — True item holdout withheld games completely from training, but still utilized descriptions and tags curated over the game's lifecycle. Prospective live A/B evaluation is required for live adoption tracking.
