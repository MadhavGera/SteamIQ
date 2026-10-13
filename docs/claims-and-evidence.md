# SteamIQ Recommender: Claims and Evidence

One-page reference for the review. Every quantitative claim maps to a reproducible evidence file.

---

## Claims

| # | Claim | Evidence File | Key Numbers |
|---|-------|---------------|-------------|
| 1 | Clean two-tower beats all baselines with paired CIs | [`leaderboard.json`](../results/leaderboard.json) | Test NDCG@10 **0.0516** [0.0473, 0.0562] vs Item-kNN 0.0445, ALS 0.0404, Popularity 0.0151 |
| 2 | Ranker shows no gain with train-only item stats | [`leaderboard.json`](../results/leaderboard.json) | Paired diff +0.0021 [−0.0026, +0.0065] — CI includes 0 |
| 3 | All-time-stats ranker gain is leakage-confounded | [`leaderboard.json`](../results/leaderboard.json) | +0.0111 (0.0627 vs 0.0516); relies on post-cutoff review aggregates |
| 4 | Serving reproduces offline evaluation within noise | [`evidence/serve_check.log`](../evidence/serve_check.log) Part B | Val −0.0012 [−0.0033, +0.0010]; Test +0.0007 [−0.0012, +0.0024] |
| 5 | Content-only beats random on cold-start items | [README cold-start section](../../README.md) | Mean rank ~1,085 vs ~1,500 (random); NDCG@10 0.0078 |
| 6 | All-time two-tower fails cold start once proxies neutralized | [README cold-start section](../../README.md) | NDCG@10 = 0.0000, 0/184 hits, mean rank 2,127 |
| 7 | Online scoring matches offline scoring (stage-1 models) | [`evidence/serve_check.log`](../evidence/serve_check.log) Part A | Max diff 0.00e+00, 0/500 order mismatches, all 5 models |
| 8 | Ranker actively re-ranks (not pass-through) | [`evidence/serve_check.log`](../evidence/serve_check.log) Part A.2 | `service == ranker.score() ≠ two_tower` for 3 users |
| 9 | Ask SteamIQ MiniLM intent router outperforms baselines | [`evidence/assistant_train.log`](../evidence/assistant_train.log) | Accuracy **0.9222** [0.8933, 0.9481] vs TF-IDF 0.8098, Keyword 0.4524 |
| 10 | Ask SteamIQ Temperature Scaling calibration achieves low ECE | [`evidence/assistant_train.log`](../evidence/assistant_train.log) | ECE **0.0298**, Out-of-scope AUROC **0.9955**, Selective accuracy **0.9222** |
| 11 | Ask SteamIQ online serving matches offline with low latency | [`evidence/assistant_serve_check.log`](../evidence/assistant_serve_check.log) | 100% parity across 200 queries (max diff 4.99e-5), P95 **18.74 ms** (< 300 ms SLA) |
| 12 | Statistical drift detectors show zero FPR on null resamples | [`evidence/drift_detector_power.log`](../evidence/drift_detector_power.log) | Null FPR **0.0000** for PSI/JSD, Power **0.82-1.00** on moderate shifts |

---

## Rehearsal Q&A

### Why doesn't the ranker beat the two-tower?

Its earlier gain came from all-time review statistics, which include post-cutoff information. With train-only statistics the gain is +0.0021 with a 95% CI of [−0.0026, +0.0065] that includes 0. This is a single-seed comparison, so a small true effect isn't ruled out — but we have no evidence of one.

### Why one seed?

The multi-seed runs were not saved, so we report only what we can reproduce. The single-seed result has bootstrap CIs around every metric to quantify sampling uncertainty.

### Is cold start solved?

No. Content-only beats random on held-out items (mean rank ~1,085 vs ~1,500 out of 3,000). The all-time-stats two-tower scored at or below random once popularity proxies were neutralised (mean rank 2,127). The clean architecture was never tested on a true item holdout. Offline simulations zero out interaction IDs but still rely on descriptions and tags curated over the game's lifecycle.

---

## Known Limitations

1. **Single seed** — multi-seed runs were not saved; bootstrap CIs quantify sampling uncertainty but not training variance.
2. **Crawl-time price snapshots** — `price_final` and `discount` reflect a single Steam store crawl, not the price at each user's interaction time.
3. **All-period item universe** — the 3,000-item universe was selected using all-period review counts (design trade-off for metadata stability).
4. **Exploratory test split** — the test split was inspected repeatedly during development; the ranker's test gain is exploratory and requires prospective A/B evaluation.
5. **Implicit feedback only** — interactions derive from positive reviews; non-purchaser negative impressions are absent.
6. **Simulated cold start** — offline simulations benchmark semantic alignment, not live prospective adoption.
