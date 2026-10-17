# 🏆 SteamIQ Recommender System Evaluation & Evidence Report

- **Dataset Version:** `c662b8232dc0`
- **Git Commit:** `e2708d5932b015a628b7a98e42e00d02bec8b5b3`
- **Evaluation Users:** 5,000
- **Random Seed:** 42
- **Generated At:** 2026-10-17T18:37:25.962224+00:00

---

## 1. Main Recommender Leaderboard (Deployable Models)

Multi-seed mean ± std reported for deployable neural models; 95% bootstrap percentile CIs over 5,000 users for baselines.

| Model | Split | Seeds | NDCG@10 (mean ± std / 95% CI) | SNIPS-NDCG@10 | HR@10 (mean ± std / 95% CI) | MRR@10 | Coverage@10 | Diversity@10 | Stage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`recsys_two_tower_clean`** | **test** | 1 | **0.0415 [0.0376, 0.0457]** | **0.0156 [0.0135, 0.0178]** | **0.0828 [0.0756, 0.0902]** | **0.0291** | 0.1267 | 0.7571 | `champion` |
| `recsys_two_tower_clean` | **val** | 1 | 0.0474 [0.0427, 0.0519] | 0.0197 [0.0175, 0.0218] | 0.0932 [0.0850, 0.1012] | 0.0337 | 0.1293 | 0.7547 | `champion` |
| `recsys_two_tower_ranker_clean` (stage-1 two-tower used all-time stats, not leak-free) | **test** | 1 | 0.0485 [0.0440, 0.0534] | 0.0253 [0.0223, 0.0286] | 0.0938 [0.0856, 0.1020] | 0.0349 | 0.4303 | 0.7609 | `production` |
| `recsys_two_tower_ranker_clean` (stage-1 two-tower used all-time stats, not leak-free) | **val** | 1 | 0.0551 [0.0500, 0.0602] | 0.0336 [0.0300, 0.0376] | 0.1026 [0.0942, 0.1116] | 0.0407 | 0.4497 | 0.7569 | `production` |
| `recsys_itemknn` | **test** | 1 | 0.0433 [0.0389, 0.0480] | 0.0216 [0.0188, 0.0247] | 0.0826 [0.0748, 0.0900] | 0.0315 | 0.6150 | 0.7255 | `candidate` |
| `recsys_als` | **test** | 1 | 0.0356 [0.0315, 0.0396] | 0.0160 [0.0141, 0.0179] | 0.0724 [0.0654, 0.0796] | 0.0246 | 0.3233 | 0.7772 | `candidate` |
| `recsys_popularity` | **test** | 1 | 0.0151 [0.0125, 0.0178] | 0.0035 [0.0029, 0.0042] | 0.0326 [0.0274, 0.0376] | 0.0100 | 0.0070 | 0.6948 | `candidate` |
| `recsys_content` | **test** | 1 | 0.0141 [0.0116, 0.0171] | 0.0106 [0.0085, 0.0131] | 0.0258 [0.0216, 0.0304] | 0.0106 | 0.6337 | 0.5046 | `candidate` |

### Appendix: Single-Seed Runs & Diagnostic Reference Models

| Model | Split | HR@10 (95% CI) | HR@20 (95% CI) | NDCG@10 (95% CI) | SNIPS-NDCG@10 | NDCG@20 (95% CI) | MRR@10 | Coverage@10 | Diversity@10 | Role / Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `recsys_two_tower_clean` | **val** | 0.0932 [0.0850, 0.1012] | 0.1490 [0.1388, 0.1586] | 0.0474 [0.0427, 0.0519] | 0.0197 | 0.0615 [0.0565, 0.0660] | 0.0337 | 0.1293 | 0.7547 | champion |
| `recsys_two_tower_clean` | **test** | 0.0828 [0.0756, 0.0902] | 0.1308 [0.1220, 0.1404] | 0.0415 [0.0376, 0.0457] | 0.0156 | 0.0537 [0.0493, 0.0581] | 0.0291 | 0.1267 | 0.7571 | champion |
| `recsys_two_tower_ranker_clean` | **val** | 0.1026 [0.0942, 0.1116] | 0.1596 [0.1494, 0.1698] | 0.0551 [0.0500, 0.0602] | 0.0336 | 0.0694 [0.0642, 0.0749] | 0.0407 | 0.4497 | 0.7569 | production |
| `recsys_two_tower_ranker_clean` | **test** | 0.0938 [0.0856, 0.1020] | 0.1434 [0.1340, 0.1534] | 0.0485 [0.0440, 0.0534] | 0.0253 | 0.0610 [0.0565, 0.0659] | 0.0349 | 0.4303 | 0.7609 | production |
| `recsys_two_tower_ranker_v2_trainstats` | **val** | 0.1026 [0.0942, 0.1116] | 0.1596 [0.1494, 0.1698] | 0.0551 [0.0500, 0.0602] | 0.0336 | 0.0694 [0.0642, 0.0749] | 0.0407 | 0.4497 | 0.7569 | Reference (all-time stats) |
| `recsys_two_tower_ranker_v2_trainstats` | **test** | 0.0938 [0.0856, 0.1020] | 0.1434 [0.1340, 0.1534] | 0.0485 [0.0440, 0.0534] | 0.0253 | 0.0610 [0.0565, 0.0659] | 0.0349 | 0.4303 | 0.7609 | Reference (all-time stats) |
| `recsys_itemknn` | **val** | 0.1014 [0.0934, 0.1096] | 0.1548 [0.1452, 0.1640] | 0.0553 [0.0505, 0.0600] | 0.0333 | 0.0687 [0.0637, 0.0735] | 0.0413 | 0.6697 | 0.7241 | candidate |
| `recsys_itemknn` | **test** | 0.0826 [0.0748, 0.0900] | 0.1246 [0.1152, 0.1346] | 0.0433 [0.0389, 0.0480] | 0.0216 | 0.0539 [0.0490, 0.0588] | 0.0315 | 0.6150 | 0.7255 | candidate |
| `recsys_als` | **val** | 0.0770 [0.0698, 0.0840] | 0.1248 [0.1160, 0.1338] | 0.0390 [0.0348, 0.0430] | 0.0206 | 0.0510 [0.0467, 0.0554] | 0.0276 | 0.3287 | 0.7762 | candidate |
| `recsys_als` | **test** | 0.0724 [0.0654, 0.0796] | 0.1126 [0.1040, 0.1216] | 0.0356 [0.0315, 0.0396] | 0.0160 | 0.0457 [0.0417, 0.0501] | 0.0246 | 0.3233 | 0.7772 | candidate |
| `recsys_popularity` | **val** | 0.0414 [0.0362, 0.0476] | 0.0732 [0.0662, 0.0806] | 0.0200 [0.0173, 0.0232] | 0.0055 | 0.0280 [0.0249, 0.0313] | 0.0136 | 0.0070 | 0.6947 | candidate |
| `recsys_popularity` | **test** | 0.0326 [0.0274, 0.0376] | 0.0614 [0.0548, 0.0680] | 0.0151 [0.0125, 0.0178] | 0.0035 | 0.0225 [0.0196, 0.0254] | 0.0100 | 0.0070 | 0.6948 | candidate |
| `recsys_content` | **val** | 0.0318 [0.0272, 0.0368] | 0.0478 [0.0422, 0.0532] | 0.0182 [0.0152, 0.0215] | 0.0170 | 0.0222 [0.0191, 0.0254] | 0.0141 | 0.6937 | 0.5086 | candidate |
| `recsys_content` | **test** | 0.0258 [0.0216, 0.0304] | 0.0394 [0.0340, 0.0452] | 0.0141 [0.0116, 0.0171] | 0.0106 | 0.0176 [0.0149, 0.0207] | 0.0106 | 0.6337 | 0.5046 | candidate |
| `recsys_two_tower` | **val** | 0.1002 [0.0924, 0.1086] | 0.1594 [0.1488, 0.1700] | 0.0532 [0.0482, 0.0582] | 0.0266 | 0.0682 [0.0630, 0.0734] | 0.0391 | 0.4483 | 0.7677 | Reference (all-time stats) |
| `recsys_two_tower` | **test** | 0.1002 [0.0924, 0.1086] | 0.1594 [0.1488, 0.1700] | 0.0532 [0.0482, 0.0582] | 0.0266 | 0.0682 [0.0630, 0.0734] | 0.0391 | 0.4483 | 0.7677 | Reference (all-time stats) |
| `recsys_two_tower_ranker` | **val** | 0.0938 [0.0858, 0.1018] | 0.1514 [0.1408, 0.1614] | 0.0484 [0.0437, 0.0530] | 0.0290 | 0.0628 [0.0581, 0.0678] | 0.0348 | 0.5727 | 0.7383 | Reference (all-time stats) |
| `recsys_two_tower_ranker` | **test** | 0.0938 [0.0858, 0.1018] | 0.1514 [0.1408, 0.1614] | 0.0484 [0.0437, 0.0530] | 0.0290 | 0.0628 [0.0581, 0.0678] | 0.0348 | 0.5727 | 0.7383 | Reference (all-time stats) |

---

## 2. Two-Tower Architectural Ablations (Multi-Seed Test)

| Variant | Description | Seeds | Test NDCG@10 (mean ± std) | Test HR@10 (mean ± std) | Tail NDCG@10 (mean ± std) | Paired Diff vs Full (95% CI) | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `full` | Full Two-Tower Model (ID + Text + Tags + Numeric + Attn) | 42,43,44 | 0.0548 ± 0.0007 | 0.1053 ± 0.0021 | 0.0029 ± 0.0005 | +0.0000 [+0.0000, +0.0000] | NO |
| `no_text` | Without MiniLM Text Projection (--no-text) | 42,43,44 | 0.0482 ± 0.0009 | 0.0945 ± 0.0035 | 0.0013 ± 0.0003 | -0.0070 [-0.0109, -0.0032] | YES (p<0.05) |
| `no_tags` | Without Tag Multi-Hot Projection (--no-tags) | 42 | 0.0528 (n=1) | 0.1002 (n=1) | 0.0020 (n=1) | -0.0010 [-0.0041, +0.0017] | NO |
| `no_numeric` | Without Standardized Numeric Projection (--no-numeric) | 42 | 0.0505 (n=1) | 0.0992 (n=1) | 0.0022 (n=1) | -0.0034 [-0.0069, +0.0002] | NO |
| `no_id` | Without Item ID Embedding (--no-id) | 42,43,44 | 0.0530 ± 0.0004 | 0.1044 ± 0.0014 | 0.0030 ± 0.0012 | -0.0003 [-0.0037, +0.0029] | NO |
| `pool_mean` | Masked Mean History Pooling (--pool mean) | 42 | 0.0542 (n=1) | 0.1030 (n=1) | 0.0008 (n=1) | +0.0003 [-0.0029, +0.0039] | NO |
| `id_dropout_0` | Without ID Dropout (--id-dropout 0.0) | 42 | 0.0507 (n=1) | 0.0996 (n=1) | 0.0048 (n=1) | -0.0032 [-0.0064, -0.0002] | YES (p<0.05) |

---

## 3. Learning-to-Rank Feature-Group Ablations

| Ablated Feature Group | Description | Val NDCG@10 | Val Diff vs Full | Test NDCG@10 | Test Diff vs Full (95% CI) | Significant on Test? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `without_source_scores` | Without Multi-Source Retrieval Scores/Ranks | 0.0345 | -0.0292 | 0.0387 | -0.0220 [-0.0271, -0.0164] | YES (p<0.05) |
| `without_content_sim` | Without Content Similarity Features (Cosine & Jaccard) | 0.0599 | -0.0038 | 0.0571 | -0.0035 [-0.0078, +0.0017] | NO |
| `without_price_features` | Without Price & Discount Features | 0.0637 | +0.0000 | 0.0589 | -0.0018 [-0.0036, +0.0041] | NO |
| `without_user_features` | Without User Profile & Hours Features | 0.0666 | +0.0030 | 0.0642 | +0.0036 [-0.0003, +0.0082] | NO |
| `without_item_quality` | Without Item Quality & Popularity Features | 0.0584 | -0.0052 | 0.0526 | -0.0081 [-0.0055, +0.0036] | NO |
| `three_sources_only` | Stage 1 with 3 Sources Only (Without Item-kNN) | 0.0614 | -0.0023 | 0.0581 | -0.0026 [-0.0058, +0.0025] | NO |
| `without_all_time_item_stats` | Without All-Time Item Stats (Drops log_user_reviews & positive_ratio only) | 0.0619 | -0.0018 | 0.0562 | -0.0044 [-0.0063, +0.0020] | NO |
| `without_train_popularity` | Without Train Popularity Features (Drops log_train_pop & popularity_rank_pct only) | 0.0614 | -0.0023 | 0.0559 | -0.0047 [-0.0050, +0.0038] | NO |

---

## 4. Sliced Paired Bootstrap Significance Tests

| Comparison | Split | Slice Bucket | N Users | Model A NDCG | Model B NDCG | Paired Diff (95% CI) | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Clean Ranker vs Clean Two-Tower | **test** | `overall` | 5000 | 0.0485 | 0.0415 | +0.0069 [+0.0024, +0.0119] | **better** |
| Ranker vs Two-Tower | **test** | `pop_head` | 2992 | 0.0715 | 0.0686 | +0.0029 [-0.0052, +0.0102] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `pop_torso` | 1034 | 0.0228 | 0.0005 | +0.0223 [+0.0147, +0.0301] | **better** |
| Ranker vs Two-Tower | **test** | `pop_tail` | 974 | 0.0052 | 0.0021 | +0.0030 [-0.0009, +0.0075] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `hist_3_4` | 2069 | 0.0526 | 0.0484 | +0.0042 [-0.0032, +0.0128] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `hist_5_9` | 1744 | 0.0414 | 0.0356 | +0.0057 [-0.0021, +0.0132] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `hist_10_plus` | 1187 | 0.0518 | 0.0383 | +0.0134 [+0.0033, +0.0235] | **better** |

---

## 5. Cold-Start Evaluation Studies

### Part A: Sparse Targets Subgroup (<= 10 Train Positives, N = 553)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Note |
| :--- | :---: | :---: | :---: | :--- |
| `random` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Baseline |
| `popularity` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Baseline |
| `content` | 0.0108 [0.0036, 0.0199] | 0.0054 [0.0013, 0.0104] | 0.0037 | Baseline |
| `als` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Baseline |
| `two_tower_intact` | 0.0090 [0.0018, 0.0181] | 0.0037 [0.0007, 0.0073] | 0.0021 | Intact Model |
| `two_tower_masked` | 0.0108 [0.0036, 0.0199] | 0.0033 [0.0011, 0.0061] | 0.0013 | Simulated Masked Target ID |

### Part B: Simulated 10% New Items (N = 437 users, 300 new items)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Category |
| :--- | :---: | :---: | :---: | :--- |
| `random` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Reference Baseline |
| `popularity` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | CF Baseline (Zeroed) |
| `als` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | CF Baseline (Zeroed) |
| `itemknn` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | CF Baseline (Zeroed) |
| `content` | 0.0206 [0.0092, 0.0366] | 0.0085 [0.0034, 0.0152] | 0.0049 | Content Hybrid |
| `two_tower_masked` | 0.0503 [0.0297, 0.0709] | 0.0230 [0.0132, 0.0335] | 0.0149 | Content Hybrid |

### Part C: True Item Holdout (10% Torso/Tail Retrained, N = 184 users, 300 held-out items)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Training Interactions |
| :--- | :---: | :---: | :---: | :--- |
| `random` | 0.0033 [0.0033, 0.0033] | 0.0015 [0.0015, 0.0015] | 0.0010 | Zero Train Overlap |
| `content` | 0.0217 [0.0054, 0.0435] | 0.0078 [0.0016, 0.0155] | 0.0037 | Zero Train Overlap |
| `two_tower_unmasked` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `two_tower_masked` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `two_tower_masked_train_pos_raw_0` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `two_tower_masked_standardized_z0` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `two_tower_masked_all_proxies_neutralized` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `routed_hybrid` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Zero Train Overlap |

---

## 6. Beyond-Accuracy Metrics & MMR Diversity Re-Ranking (k = 10)

| Model | Catalog Coverage@10 | Novelty@10 (bits) | Diversity@10 | Head Share@10 (1%) | Gini Exposure@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recsys_two_tower_ranker_clean` | 0.4303 | 9.23 | 0.7609 | 0.2763 | 0.0000 |
| `recsys_two_tower_clean` | 0.1267 | 8.71 | 0.7571 | 0.4375 | 0.0000 |
| `recsys_itemknn` | 0.6150 | 9.31 | 0.7255 | 0.3437 | 0.0000 |
| `recsys_als` | 0.3233 | 9.27 | 0.7772 | 0.1958 | 0.0000 |
| `recsys_popularity` | 0.0070 | 7.63 | 0.6948 | 1.0000 | 0.0000 |
| `recsys_content` | 0.6337 | 11.75 | 0.5046 | 0.0657 | 0.0000 |

### MMR Diversity Trade-Off over Ranker Top-50 Candidates

| Lambda Parameter (λ) | NDCG@10 (95% CI) | Diversity@10 | NDCG Diff vs λ=1.0 (95% CI) |
| :---: | :---: | :---: | :---: |
| **λ = 1.0** | 0.0491 [0.0446, 0.0541] | 0.7609 | +0.0000 [+0.0000, +0.0000] |
| **λ = 0.8** | 0.0489 [0.0444, 0.0538] | 0.7798 | -0.0002 [-0.0011, +0.0009] |
| **λ = 0.6** | 0.0481 [0.0437, 0.0530] | 0.8022 | -0.0010 [-0.0024, +0.0005] |
