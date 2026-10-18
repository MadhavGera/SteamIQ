# 🏆 SteamIQ Recommender System Evaluation & Evidence Report

- **Dataset Version:** `c662b8232dc0`
- **Git Commit:** `3304ee921892a9db818f11e294209cf1958372de`
- **Evaluation Users:** 5,000
- **Random Seed:** 42
- **Generated At:** 2026-10-18T10:05:26.422957+00:00

---

## 1. Main Recommender Leaderboard (Deployable Models)

Multi-seed mean ± std reported for deployable neural models; 95% bootstrap percentile CIs over 5,000 users for baselines.

| Model | Split | Seeds | NDCG@10 (mean ± std / 95% CI) | SNIPS-NDCG@10 | HR@10 (mean ± std / 95% CI) | MRR@10 | Coverage@10 | Diversity@10 | Stage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`recsys_two_tower_clean`** | **test** | 1 | **0.0513 [0.0467, 0.0563]** | **0.0260 [0.0229, 0.0291]** | **0.1016 [0.0936, 0.1102]** | **0.0362** | 0.4370 | 0.7627 | `champion` |
| `recsys_two_tower_clean` | **val** | 1 | 0.0611 [0.0557, 0.0662] | 0.0338 [0.0305, 0.0373] | 0.1190 [0.1104, 0.1280] | 0.0436 | 0.4347 | 0.7588 | `champion` |
| `recsys_two_tower_ranker_clean` | **test** | 1 | 0.0532 [0.0480, 0.0587] | 0.0311 [0.0274, 0.0350] | 0.1004 [0.0916, 0.1094] | 0.0390 | 0.5423 | 0.7495 | `candidate` |
| `recsys_two_tower_ranker_clean` | **val** | 1 | 0.0582 [0.0531, 0.0635] | 0.0378 [0.0340, 0.0421] | 0.1098 [0.1014, 0.1190] | 0.0428 | 0.5503 | 0.7450 | `candidate` |
| `recsys_sasrec` | **test** | 1 | 0.0420 [0.0379, 0.0465] | 0.0143 [0.0127, 0.0159] | 0.0824 [0.0750, 0.0900] | 0.0297 | 0.1813 | 0.7485 | `challenger` |
| `recsys_itemknn` | **test** | 1 | 0.0433 [0.0389, 0.0480] | 0.0216 [0.0188, 0.0247] | 0.0826 [0.0748, 0.0900] | 0.0315 | 0.6150 | 0.7255 | `candidate` |
| `recsys_als` | **test** | 1 | 0.0356 [0.0315, 0.0396] | 0.0160 [0.0141, 0.0179] | 0.0724 [0.0654, 0.0796] | 0.0246 | 0.3233 | 0.7772 | `candidate` |
| `recsys_popularity` | **test** | 1 | 0.0151 [0.0125, 0.0178] | 0.0035 [0.0029, 0.0042] | 0.0326 [0.0274, 0.0376] | 0.0100 | 0.0070 | 0.6948 | `candidate` |
| `recsys_content` | **test** | 1 | 0.0141 [0.0116, 0.0171] | 0.0106 [0.0085, 0.0131] | 0.0258 [0.0216, 0.0304] | 0.0106 | 0.6337 | 0.5046 | `candidate` |

### Appendix: Single-Seed Runs & Diagnostic Reference Models

| Model | Split | HR@10 (95% CI) | HR@20 (95% CI) | NDCG@10 (95% CI) | SNIPS-NDCG@10 | NDCG@20 (95% CI) | MRR@10 | Coverage@10 | Diversity@10 | Role / Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `recsys_two_tower_clean` | **val** | 0.1190 [0.1104, 0.1280] | 0.1872 [0.1766, 0.1986] | 0.0611 [0.0557, 0.0662] | 0.0338 | 0.0781 [0.0728, 0.0835] | 0.0436 | 0.4347 | 0.7588 | champion |
| `recsys_two_tower_clean` | **test** | 0.1016 [0.0936, 0.1102] | 0.1598 [0.1496, 0.1698] | 0.0513 [0.0467, 0.0563] | 0.0260 | 0.0659 [0.0610, 0.0709] | 0.0362 | 0.4370 | 0.7627 | champion |
| `recsys_two_tower_ranker_clean` | **val** | 0.1098 [0.1014, 0.1190] | 0.1686 [0.1586, 0.1798] | 0.0582 [0.0531, 0.0635] | 0.0378 | 0.0731 [0.0679, 0.0788] | 0.0428 | 0.5503 | 0.7450 | candidate |
| `recsys_two_tower_ranker_clean` | **test** | 0.1004 [0.0916, 0.1094] | 0.1528 [0.1422, 0.1630] | 0.0532 [0.0480, 0.0587] | 0.0311 | 0.0664 [0.0610, 0.0721] | 0.0390 | 0.5423 | 0.7495 | candidate |
| `recsys_two_tower_ranker_v2_trainstats` | **val** | 0.1098 [0.1014, 0.1190] | 0.1686 [0.1586, 0.1798] | 0.0582 [0.0531, 0.0635] | 0.0378 | 0.0731 [0.0679, 0.0788] | 0.0428 | 0.5503 | 0.7450 | Reference (all-time stats) |
| `recsys_two_tower_ranker_v2_trainstats` | **test** | 0.1004 [0.0916, 0.1094] | 0.1528 [0.1422, 0.1630] | 0.0532 [0.0480, 0.0587] | 0.0311 | 0.0664 [0.0610, 0.0721] | 0.0390 | 0.5423 | 0.7495 | Reference (all-time stats) |
| `recsys_itemknn` | **val** | 0.1014 [0.0934, 0.1096] | 0.1548 [0.1452, 0.1640] | 0.0553 [0.0505, 0.0600] | 0.0333 | 0.0687 [0.0637, 0.0735] | 0.0413 | 0.6697 | 0.7241 | candidate |
| `recsys_itemknn` | **test** | 0.0826 [0.0748, 0.0900] | 0.1246 [0.1152, 0.1346] | 0.0433 [0.0389, 0.0480] | 0.0216 | 0.0539 [0.0490, 0.0588] | 0.0315 | 0.6150 | 0.7255 | candidate |
| `recsys_als` | **val** | 0.0770 [0.0698, 0.0840] | 0.1248 [0.1160, 0.1338] | 0.0390 [0.0348, 0.0430] | 0.0206 | 0.0510 [0.0467, 0.0554] | 0.0276 | 0.3287 | 0.7762 | candidate |
| `recsys_als` | **test** | 0.0724 [0.0654, 0.0796] | 0.1126 [0.1040, 0.1216] | 0.0356 [0.0315, 0.0396] | 0.0160 | 0.0457 [0.0417, 0.0501] | 0.0246 | 0.3233 | 0.7772 | candidate |
| `recsys_popularity` | **val** | 0.0414 [0.0362, 0.0476] | 0.0732 [0.0662, 0.0806] | 0.0200 [0.0173, 0.0232] | 0.0055 | 0.0280 [0.0249, 0.0313] | 0.0136 | 0.0070 | 0.6947 | candidate |
| `recsys_popularity` | **test** | 0.0326 [0.0274, 0.0376] | 0.0614 [0.0548, 0.0680] | 0.0151 [0.0125, 0.0178] | 0.0035 | 0.0225 [0.0196, 0.0254] | 0.0100 | 0.0070 | 0.6948 | candidate |
| `recsys_content` | **val** | 0.0318 [0.0272, 0.0368] | 0.0478 [0.0422, 0.0532] | 0.0182 [0.0152, 0.0215] | 0.0170 | 0.0222 [0.0191, 0.0254] | 0.0141 | 0.6937 | 0.5086 | candidate |
| `recsys_content` | **test** | 0.0258 [0.0216, 0.0304] | 0.0394 [0.0340, 0.0452] | 0.0141 [0.0116, 0.0171] | 0.0106 | 0.0176 [0.0149, 0.0207] | 0.0106 | 0.6337 | 0.5046 | candidate |
| `recsys_sasrec` | **val** | 0.0974 [0.0888, 0.1056] | 0.1586 [0.1486, 0.1686] | 0.0488 [0.0443, 0.0535] | 0.0202 | 0.0642 [0.0595, 0.0690] | 0.0341 | 0.1790 | 0.7460 | challenger |
| `recsys_sasrec` | **test** | 0.0824 [0.0750, 0.0900] | 0.1296 [0.1206, 0.1384] | 0.0420 [0.0379, 0.0465] | 0.0143 | 0.0539 [0.0496, 0.0583] | 0.0297 | 0.1813 | 0.7485 | challenger |
| `recsys_two_tower` | **val** | 0.1002 [0.0924, 0.1086] | 0.1594 [0.1488, 0.1700] | 0.0532 [0.0482, 0.0582] | 0.0266 | 0.0682 [0.0630, 0.0734] | 0.0391 | 0.4483 | 0.7677 | Reference (all-time stats) |
| `recsys_two_tower` | **test** | 0.1002 [0.0924, 0.1086] | 0.1594 [0.1488, 0.1700] | 0.0532 [0.0482, 0.0582] | 0.0266 | 0.0682 [0.0630, 0.0734] | 0.0391 | 0.4483 | 0.7677 | Reference (all-time stats) |
| `recsys_two_tower_ranker` | **val** | 0.0938 [0.0858, 0.1018] | 0.1514 [0.1408, 0.1614] | 0.0484 [0.0437, 0.0530] | 0.0290 | 0.0628 [0.0581, 0.0678] | 0.0348 | 0.5727 | 0.7383 | Reference (all-time stats) |
| `recsys_two_tower_ranker` | **test** | 0.0938 [0.0858, 0.1018] | 0.1514 [0.1408, 0.1614] | 0.0484 [0.0437, 0.0530] | 0.0290 | 0.0628 [0.0581, 0.0678] | 0.0348 | 0.5727 | 0.7383 | Reference (all-time stats) |

---

## 2. Two-Tower Architectural Ablations (Multi-Seed Test)

| Variant | Description | Seeds | Test NDCG@10 (mean ± std) | Test HR@10 (mean ± std) | Tail NDCG@10 (mean ± std) | Paired Diff vs Full (95% CI) | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `full` | Full Two-Tower Model (ID + Text + Tags + Numeric + Attn) | 42,43,44 | 0.0615 ± 0.0011 | 0.1195 ± 0.0012 | 0.0062 ± 0.0011 | +0.0000 [+0.0000, +0.0000] | NO |
| `no_text` | Without MiniLM Text Projection (--no-text) | 42,43,44 | 0.0524 ± 0.0015 | 0.1042 ± 0.0023 | 0.0010 ± 0.0001 | -0.0091 [-0.0117, -0.0067] | YES (p<0.05) |
| `no_tags` | Without Tag Multi-Hot Projection (--no-tags) | 42,43,44 | 0.0592 ± 0.0005 | 0.1160 ± 0.0010 | 0.0047 ± 0.0003 | -0.0023 [-0.0043, -0.0005] | YES (p<0.05) |
| `no_numeric` | Without Standardized Numeric Projection (--no-numeric) | 42,43,44 | 0.0611 ± 0.0021 | 0.1175 ± 0.0049 | 0.0034 ± 0.0010 | -0.0004 [-0.0023, +0.0015] | NO |
| `no_id` | Without Item ID Embedding (--no-id) | 42,43,44 | 0.0620 ± 0.0004 | 0.1207 ± 0.0006 | 0.0046 ± 0.0012 | +0.0005 [-0.0018, +0.0027] | NO |

---

## 3. Learning-to-Rank Feature-Group Ablations

| Ablated Feature Group | Description | Val NDCG@10 | Val Diff vs Full | Test NDCG@10 | Test Diff vs Full (95% CI) | Significant on Test? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `without_source_scores` | Without Multi-Source Retrieval Scores/Ranks | 0.0548 | -0.0033 | 0.0548 | -0.0033 [-0.0058, -0.0007] | YES (p<0.05) |
| `without_content_sim` | Without Content Similarity Features (Cosine & Jaccard) | 0.0558 | -0.0022 | 0.0558 | -0.0022 [-0.0048, +0.0004] | NO |
| `without_price_features` | Without Price & Discount Features | 0.0545 | -0.0036 | 0.0545 | -0.0036 [-0.0062, -0.0010] | YES (p<0.05) |
| `without_user_features` | Without User Profile & Hours Features | 0.0631 | +0.0051 | 0.0631 | +0.0051 [+0.0025, +0.0076] | NO |
| `without_item_quality` | Without Item Quality & Popularity Features | 0.0525 | -0.0055 | 0.0525 | -0.0055 [-0.0081, -0.0029] | YES (p<0.05) |
| `without_train_popularity` | Without Train Popularity Features | 0.0579 | -0.0001 | 0.0579 | -0.0001 [-0.0028, +0.0025] | NO |

---

## 4. Sliced Paired Bootstrap Significance Tests

| Comparison | Split | Slice Bucket | N Users | Model A NDCG | Model B NDCG | Paired Diff (95% CI) | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Clean Ranker vs Clean Two-Tower | **test** | `overall` | 5000 | 0.0532 | 0.0513 | +0.0019 [-0.0024, +0.0062] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `pop_head` | 2992 | 0.0732 | 0.0776 | -0.0044 [-0.0115, +0.0023] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `pop_torso` | 1034 | 0.0365 | 0.0196 | +0.0169 [+0.0091, +0.0246] | **better** |
| Ranker vs Two-Tower | **test** | `pop_tail` | 974 | 0.0095 | 0.0044 | +0.0051 [+0.0003, +0.0104] | **better** |
| Ranker vs Two-Tower | **test** | `hist_3_4` | 2069 | 0.0588 | 0.0597 | -0.0009 [-0.0074, +0.0059] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `hist_5_9` | 1744 | 0.0452 | 0.0440 | +0.0013 [-0.0058, +0.0085] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `hist_10_plus` | 1187 | 0.0552 | 0.0476 | +0.0076 [-0.0020, +0.0162] | **no significant difference** |

---

## 5. Cold-Start Evaluation Studies

### Part A: Sparse Targets Subgroup (<= 10 Train Positives, N = 553)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Note |
| :--- | :---: | :---: | :---: | :--- |
| `random` | 0.0033 [0.0033, 0.0033] | 0.0015 [0.0015, 0.0015] | 0.0010 | Baseline |
| `popularity` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Baseline |
| `content` | 0.0108 [0.0036, 0.0199] | 0.0054 [0.0013, 0.0104] | 0.0037 | Baseline |
| `als` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Baseline |
| `two_tower_intact` | 0.0108 [0.0036, 0.0199] | 0.0043 [0.0012, 0.0083] | 0.0023 | Intact Model |
| `two_tower_masked` | 0.0163 [0.0072, 0.0272] | 0.0066 [0.0023, 0.0118] | 0.0039 | Simulated Masked Target ID |

### Part B: Simulated 10% New Items (N = 437 users, 300 new items)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Category |
| :--- | :---: | :---: | :---: | :--- |
| `random` | 0.0033 [0.0033, 0.0033] | 0.0015 [0.0015, 0.0015] | 0.0010 | Reference Baseline |
| `popularity` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | CF Baseline (Zeroed) |
| `als` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | CF Baseline (Zeroed) |
| `content` | 0.0206 [0.0092, 0.0366] | 0.0085 [0.0034, 0.0152] | 0.0049 | Content Hybrid |
| `two_tower_masked` | 0.0481 [0.0297, 0.0686] | 0.0227 [0.0132, 0.0332] | 0.0150 | Content Hybrid |

### Part C: True Item Holdout (10% Torso/Tail Retrained, N = 5000 users, 0 held-out items)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Training Interactions |
| :--- | :---: | :---: | :---: | :--- |
| `cold_only_random` | 0.0000 [0.0000, 0.0000] | 0.0228 [0.0000, 0.0000] | 0.0000 | Zero Train Overlap |
| `catalog_random` | 0.0033 [0.0000, 0.0000] | 0.0015 [0.0000, 0.0000] | 0.0000 | Zero Train Overlap |
| `random` | 0.0033 [0.0000, 0.0000] | 0.0015 [0.0000, 0.0000] | 0.0000 | Zero Train Overlap |
| `content` | 0.0505 [0.0288, 0.0721] | 0.0313 [0.0183, 0.0464] | 0.0000 | Zero Train Overlap |
| `two_tower` | 0.0264 [0.0120, 0.0409] | 0.0131 [0.0057, 0.0221] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `routed_hybrid` | 0.0553 [0.0337, 0.0793] | 0.0407 [0.0239, 0.0577] | 0.0000 | Zero Train Overlap |
| `routed_hybrid_zero_only` | 0.0553 [0.0337, 0.0793] | 0.0407 [0.0239, 0.0577] | 0.0000 | Zero Train Overlap |
| `boost_cold_equal` | 0.3702 [0.3245, 0.4183] | 0.2182 [0.1864, 0.2491] | 0.0000 | Zero Train Overlap |
| `boost_cold_capped` | 0.3341 [0.2884, 0.3798] | 0.2011 [0.1697, 0.2318] | 0.0000 | Zero Train Overlap |

---

## 6. Beyond-Accuracy Metrics & MMR Diversity Re-Ranking (k = 10)

| Model | Catalog Coverage@10 | Novelty@10 (bits) | Diversity@10 | Head Share@10 (1%) | Gini Exposure@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recsys_two_tower_ranker_clean` | 0.5423 | 9.54 | 0.7495 | 0.2352 | 0.0000 |
| `recsys_two_tower_clean` | 0.4370 | 9.34 | 0.7627 | 0.2643 | 0.0000 |
| `recsys_itemknn` | 0.6150 | 9.31 | 0.7255 | 0.3437 | 0.0000 |
| `recsys_als` | 0.3233 | 9.27 | 0.7772 | 0.1958 | 0.0000 |
| `recsys_popularity` | 0.0070 | 7.63 | 0.6948 | 1.0000 | 0.0000 |
| `recsys_content` | 0.6337 | 11.75 | 0.5046 | 0.0657 | 0.0000 |
| `recsys_sasrec` | 0.0350 | 7.73 | 0.7099 | 0.9624 | 0.0000 |

### MMR Diversity Trade-Off over Ranker Top-50 Candidates

| Lambda Parameter (λ) | NDCG@10 (95% CI) | Diversity@10 | NDCG Diff vs λ=1.0 (95% CI) |
| :---: | :---: | :---: | :---: |
| **λ = 1.0** | 0.0535 [0.0487, 0.0586] | 0.7494 | +0.0000 [+0.0000, +0.0000] |
| **λ = 0.8** | 0.0537 [0.0490, 0.0590] | 0.7723 | +0.0002 [-0.0007, +0.0012] |
| **λ = 0.6** | 0.0531 [0.0485, 0.0583] | 0.8017 | -0.0004 [-0.0020, +0.0011] |
