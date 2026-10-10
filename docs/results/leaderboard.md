# 🏆 SteamIQ Recommender System Evaluation & Evidence Report

- **Dataset Version:** `c662b8232dc0`
- **Git Commit:** `9e9b5b8c7d6e1661e967feb25426b0b87fa11166`
- **Evaluation Users:** 5,000
- **Random Seed:** 42
- **Generated At:** 2026-10-07T21:30:15.166205+00:00

---

## 1. Main Recommender Leaderboard (Deployable Models)

Multi-seed mean ± std reported for deployable neural models; 95% bootstrap percentile CIs over 5,000 users for baselines.

| Model | Split | Seeds | NDCG@10 (mean ± std / 95% CI) | HR@10 (mean ± std / 95% CI) | MRR@10 | Coverage@10 | Diversity@10 | Stage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`recsys_two_tower_clean`** | **test** | 3 | **0.0516 [0.0473, 0.0562]** | **0.1028 [0.0954, 0.1108]** | **0.0361** | 0.4383 | 0.7629 | **Champion (Production)** |
| `recsys_two_tower_clean` | **val** | 3 | 0.0611 [0.0562, 0.0661] | 0.1190 [0.1104, 0.1280] | 0.0436 | 0.4347 | 0.7588 | Champion |
| `recsys_two_tower_ranker_clean` | **test** | 5 | 0.0579 [0.0527, 0.0631] | 0.1102 [0.1014, 0.1186] | 0.0422 | 0.5253 | 0.7481 | Candidate |
| `recsys_two_tower_ranker_clean` | **val** | 5 | 0.0615 [0.0563, 0.0666] | 0.1170 [0.1080, 0.1256] | 0.0447 | 0.5317 | 0.7431 | Candidate |
| `recsys_itemknn` | **test** | 1 | 0.0445 [0.0403, 0.0494] | 0.0842 [0.0766, 0.0916] | 0.0326 | 0.6143 | 0.7255 | Candidate |
| `recsys_als` | **test** | 1 | 0.0404 [0.0362, 0.0449] | 0.0776 [0.0704, 0.0852] | 0.0291 | 0.2480 | 0.7732 | Candidate |
| `recsys_popularity` | **test** | 1 | 0.0151 [0.0125, 0.0178] | 0.0326 [0.0274, 0.0376] | 0.0100 | 0.0070 | 0.6948 | Candidate |
| `recsys_content` | **test** | 1 | 0.0146 [0.0120, 0.0176] | 0.0268 [0.0226, 0.0316] | 0.0110 | 0.6303 | 0.5041 | Candidate |

### Appendix: Single-Seed Runs & Diagnostic Reference Models

| Model | Split | HR@10 (95% CI) | HR@20 (95% CI) | NDCG@10 (95% CI) | NDCG@20 (95% CI) | MRR@10 | Coverage@10 | Diversity@10 | Role / Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `recsys_two_tower_clean` | **val** | 0.1190 [0.1104, 0.1280] | 0.1872 [0.1764, 0.1974] | 0.0611 [0.0562, 0.0661] | 0.0781 [0.0729, 0.0832] | 0.0436 | 0.4347 | 0.7588 | champion |
| `recsys_two_tower_clean` | **test** | 0.1028 [0.0954, 0.1108] | 0.1604 [0.1506, 0.1702] | 0.0516 [0.0473, 0.0562] | 0.0660 [0.0616, 0.0708] | 0.0361 | 0.4383 | 0.7629 | champion |
| `recsys_two_tower_ranker_clean` | **val** | 0.1170 [0.1080, 0.1256] | 0.1812 [0.1708, 0.1918] | 0.0615 [0.0563, 0.0666] | 0.0776 [0.0723, 0.0827] | 0.0447 | 0.5317 | 0.7431 | candidate |
| `recsys_two_tower_ranker_clean` | **test** | 0.1102 [0.1014, 0.1186] | 0.1632 [0.1536, 0.1734] | 0.0579 [0.0527, 0.0631] | 0.0714 [0.0661, 0.0769] | 0.0422 | 0.5253 | 0.7481 | candidate |
| `recsys_two_tower_ranker` | **val** | 0.1148 [0.1060, 0.1234] | 0.1756 [0.1652, 0.1856] | 0.0621 [0.0570, 0.0675] | 0.0773 [0.0719, 0.0828] | 0.0462 | 0.4697 | 0.7599 | Reference (all-time stats) |
| `recsys_two_tower_ranker` | **test** | 0.1072 [0.0990, 0.1156] | 0.1602 [0.1500, 0.1708] | 0.0584 [0.0530, 0.0634] | 0.0716 [0.0661, 0.0770] | 0.0436 | 0.4557 | 0.7642 | Reference (all-time stats) |
| `recsys_two_tower` | **val** | 0.1182 [0.1098, 0.1274] | 0.1866 [0.1758, 0.1964] | 0.0610 [0.0563, 0.0662] | 0.0783 [0.0733, 0.0835] | 0.0438 | 0.4547 | 0.7637 | Reference (all-time stats) |
| `recsys_two_tower` | **test** | 0.1024 [0.0946, 0.1108] | 0.1612 [0.1506, 0.1714] | 0.0539 [0.0490, 0.0589] | 0.0688 [0.0636, 0.0739] | 0.0392 | 0.4500 | 0.7678 | Reference (all-time stats) |
| `recsys_itemknn` | **val** | 0.1014 [0.0934, 0.1096] | 0.1548 [0.1452, 0.1640] | 0.0553 [0.0505, 0.0600] | 0.0687 [0.0637, 0.0735] | 0.0413 | 0.6697 | 0.7241 | candidate |
| `recsys_itemknn` | **test** | 0.0842 [0.0766, 0.0916] | 0.1266 [0.1174, 0.1366] | 0.0445 [0.0403, 0.0494] | 0.0552 [0.0504, 0.0602] | 0.0326 | 0.6143 | 0.7255 | candidate |
| `recsys_als` | **val** | 0.0864 [0.0786, 0.0940] | 0.1370 [0.1276, 0.1466] | 0.0451 [0.0407, 0.0492] | 0.0578 [0.0534, 0.0624] | 0.0326 | 0.2487 | 0.7725 | candidate |
| `recsys_als` | **test** | 0.0776 [0.0704, 0.0852] | 0.1220 [0.1124, 0.1314] | 0.0404 [0.0362, 0.0449] | 0.0515 [0.0469, 0.0561] | 0.0291 | 0.2480 | 0.7732 | candidate |
| `recsys_popularity` | **val** | 0.0414 [0.0362, 0.0476] | 0.0732 [0.0662, 0.0806] | 0.0200 [0.0173, 0.0232] | 0.0280 [0.0249, 0.0313] | 0.0136 | 0.0070 | 0.6947 | candidate |
| `recsys_popularity` | **test** | 0.0326 [0.0274, 0.0376] | 0.0612 [0.0546, 0.0678] | 0.0151 [0.0125, 0.0178] | 0.0225 [0.0196, 0.0254] | 0.0100 | 0.0070 | 0.6948 | candidate |
| `recsys_content` | **val** | 0.0318 [0.0272, 0.0368] | 0.0478 [0.0422, 0.0532] | 0.0182 [0.0152, 0.0215] | 0.0222 [0.0191, 0.0254] | 0.0141 | 0.6937 | 0.5086 | candidate |
| `recsys_content` | **test** | 0.0268 [0.0226, 0.0316] | 0.0406 [0.0352, 0.0464] | 0.0146 [0.0120, 0.0176] | 0.0181 [0.0154, 0.0211] | 0.0110 | 0.6303 | 0.5041 | candidate |

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
| Clean Ranker vs Clean Two-Tower | **val** | `overall` | 5000 | 0.0615 | 0.0611 | +0.0004 [-0.0041, +0.0049] | **no significant difference** |
| Clean Ranker vs Clean Two-Tower | **test** | `overall` | 5000 | 0.0579 | 0.0516 | +0.0063 [+0.0020, +0.0108] | **better** |
| Ranker vs Two-Tower | **val** | `pop_head` | 3342 | 0.0819 | 0.0850 | -0.0031 [-0.0098, +0.0040] | **no significant difference** |
| Ranker vs Two-Tower | **val** | `pop_torso` | 956 | 0.0330 | 0.0163 | +0.0166 [+0.0084, +0.0256] | **better** |
| Ranker vs Two-Tower | **val** | `pop_tail` | 702 | 0.0076 | 0.0075 | +0.0001 [-0.0055, +0.0060] | **no significant difference** |
| Ranker vs Two-Tower | **val** | `hist_3_4` | 2069 | 0.0647 | 0.0625 | +0.0023 [-0.0059, +0.0098] | **no significant difference** |
| Ranker vs Two-Tower | **val** | `hist_5_9` | 1744 | 0.0697 | 0.0699 | -0.0002 [-0.0092, +0.0081] | **no significant difference** |
| Ranker vs Two-Tower | **val** | `hist_10_plus` | 1187 | 0.0465 | 0.0453 | +0.0012 [-0.0084, +0.0096] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `pop_head` | 2989 | 0.0838 | 0.0813 | +0.0025 [-0.0045, +0.0097] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `pop_torso` | 1015 | 0.0323 | 0.0225 | +0.0098 [+0.0001, +0.0201] | **better** |
| Ranker vs Two-Tower | **test** | `pop_tail` | 996 | 0.0087 | 0.0036 | +0.0051 [+0.0007, +0.0096] | **better** |
| Ranker vs Two-Tower | **test** | `hist_3_4` | 1239 | 0.0586 | 0.0568 | +0.0018 [-0.0074, +0.0128] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `hist_5_9` | 2378 | 0.0561 | 0.0536 | +0.0025 [-0.0046, +0.0094] | **no significant difference** |
| Ranker vs Two-Tower | **test** | `hist_10_plus` | 1383 | 0.0620 | 0.0516 | +0.0104 [+0.0024, +0.0186] | **better** |
| Ranker vs Item-kNN | **val** | `pop_head` | 3342 | 0.0819 | 0.0716 | +0.0104 [+0.0034, +0.0179] | **better** |
| Ranker vs Item-kNN | **val** | `pop_torso` | 956 | 0.0330 | 0.0307 | +0.0023 [-0.0075, +0.0114] | **no significant difference** |
| Ranker vs Item-kNN | **val** | `pop_tail` | 702 | 0.0076 | 0.0113 | -0.0037 [-0.0095, +0.0021] | **no significant difference** |
| Ranker vs Item-kNN | **val** | `hist_3_4` | 2069 | 0.0647 | 0.0618 | +0.0030 [-0.0040, +0.0105] | **no significant difference** |
| Ranker vs Item-kNN | **val** | `hist_5_9` | 1744 | 0.0697 | 0.0588 | +0.0108 [+0.0012, +0.0211] | **better** |
| Ranker vs Item-kNN | **val** | `hist_10_plus` | 1187 | 0.0465 | 0.0387 | +0.0077 [-0.0028, +0.0177] | **no significant difference** |
| Ranker vs Item-kNN | **test** | `pop_head` | 2989 | 0.0838 | 0.0662 | +0.0175 [+0.0100, +0.0253] | **better** |
| Ranker vs Item-kNN | **test** | `pop_torso` | 1015 | 0.0323 | 0.0174 | +0.0148 [+0.0060, +0.0238] | **better** |
| Ranker vs Item-kNN | **test** | `pop_tail` | 996 | 0.0087 | 0.0070 | +0.0017 [-0.0017, +0.0055] | **no significant difference** |
| Ranker vs Item-kNN | **test** | `hist_3_4` | 1239 | 0.0586 | 0.0546 | +0.0040 [-0.0057, +0.0136] | **no significant difference** |
| Ranker vs Item-kNN | **test** | `hist_5_9` | 2378 | 0.0561 | 0.0464 | +0.0098 [+0.0020, +0.0170] | **better** |
| Ranker vs Item-kNN | **test** | `hist_10_plus` | 1383 | 0.0620 | 0.0323 | +0.0297 [+0.0201, +0.0401] | **better** |
| Two-Tower vs Item-kNN | **val** | `pop_head` | 3342 | 0.0850 | 0.0716 | +0.0134 [+0.0062, +0.0213] | **better** |
| Two-Tower vs Item-kNN | **val** | `pop_torso` | 956 | 0.0163 | 0.0307 | -0.0143 [-0.0223, -0.0066] | **worse** |
| Two-Tower vs Item-kNN | **val** | `pop_tail` | 702 | 0.0075 | 0.0113 | -0.0038 [-0.0104, +0.0025] | **no significant difference** |
| Two-Tower vs Item-kNN | **val** | `hist_3_4` | 2069 | 0.0625 | 0.0618 | +0.0007 [-0.0078, +0.0091] | **no significant difference** |
| Two-Tower vs Item-kNN | **val** | `hist_5_9` | 1744 | 0.0699 | 0.0588 | +0.0111 [+0.0013, +0.0224] | **better** |
| Two-Tower vs Item-kNN | **val** | `hist_10_plus` | 1187 | 0.0453 | 0.0387 | +0.0066 [-0.0030, +0.0161] | **no significant difference** |
| Two-Tower vs Item-kNN | **test** | `pop_head` | 2989 | 0.0813 | 0.0662 | +0.0150 [+0.0073, +0.0231] | **better** |
| Two-Tower vs Item-kNN | **test** | `pop_torso` | 1015 | 0.0225 | 0.0174 | +0.0050 [-0.0022, +0.0124] | **no significant difference** |
| Two-Tower vs Item-kNN | **test** | `pop_tail` | 996 | 0.0036 | 0.0070 | -0.0034 [-0.0078, +0.0011] | **no significant difference** |
| Two-Tower vs Item-kNN | **test** | `hist_3_4` | 1239 | 0.0568 | 0.0546 | +0.0022 [-0.0090, +0.0132] | **no significant difference** |
| Two-Tower vs Item-kNN | **test** | `hist_5_9` | 2378 | 0.0536 | 0.0464 | +0.0072 [+0.0001, +0.0147] | **better** |
| Two-Tower vs Item-kNN | **test** | `hist_10_plus` | 1383 | 0.0516 | 0.0323 | +0.0193 [+0.0099, +0.0288] | **better** |

---

## 5. Cold-Start Evaluation Studies

### Part A: Sparse Targets Subgroup (<= 10 Train Positives, N = 553)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Note |
| :--- | :---: | :---: | :---: | :--- |
| `random` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Baseline |
| `popularity` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Baseline |
| `content` | 0.0108 [0.0036, 0.0199] | 0.0054 [0.0013, 0.0104] | 0.0037 | Baseline |
| `als` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Baseline |
| `two_tower_intact` | 0.0090 [0.0018, 0.0181] | 0.0037 [0.0008, 0.0074] | 0.0021 | Intact Model |
| `two_tower_masked (reference: all-time stats)` | 0.0090 [0.0018, 0.0181] | 0.0036 [0.0009, 0.0072] | 0.0020 | Simulated Masked Target ID |

### Part B: Simulated 10% New Items (N = 437 users, 300 new items)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Category |
| :--- | :---: | :---: | :---: | :--- |
| `random` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Reference Baseline |
| `popularity` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | CF Baseline (Zeroed) |
| `als` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | CF Baseline (Zeroed) |
| `itemknn` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | CF Baseline (Zeroed) |
| `content` | 0.0206 [0.0092, 0.0366] | 0.0085 [0.0034, 0.0152] | 0.0049 | Content Hybrid |
| `two_tower_masked (reference: all-time stats)` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Content Hybrid |

### Part C: True Item Holdout (10% Torso/Tail Retrained, N = 184 users, 300 held-out items)

| Model | HR@10 (95% CI) | NDCG@10 (95% CI) | MRR@10 | Training Interactions |
| :--- | :---: | :---: | :---: | :--- |
| `random` | 0.0033 [0.0033, 0.0033] | 0.0015 [0.0015, 0.0015] | 0.0010 | Zero Train Overlap |
| `content` | 0.0217 [0.0054, 0.0435] | 0.0078 [0.0016, 0.0155] | 0.0037 | Zero Train Overlap |
| `two_tower_unmasked` | 0.0054 [0.0000, 0.0163] | 0.0054 [0.0000, 0.0163] | 0.0054 | Retrained (0 Interactions for Held-out Items) |
| `two_tower_masked` | 0.0054 [0.0000, 0.0163] | 0.0034 [0.0000, 0.0103] | 0.0027 | Retrained (0 Interactions for Held-out Items) |
| `two_tower_masked_train_pos_raw_0` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `two_tower_masked_standardized_z0` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `two_tower_masked_all_proxies_neutralized` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |
| `two_tower (all-time-stats architecture, clean=False)` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | Retrained (0 Interactions for Held-out Items) |

---

## 6. Beyond-Accuracy Metrics & MMR Diversity Re-Ranking (k = 10)

| Model | Catalog Coverage@10 | Novelty@10 (bits) | Diversity@10 | Head Share@10 (1%) | Gini Exposure@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recsys_two_tower_ranker` | 0.4557 | 9.46 | 0.7642 | 0.2140 | 0.9118 |
| `recsys_two_tower` | 0.4500 | 9.39 | 0.7678 | 0.2491 | 0.9062 |
| `recsys_itemknn` | 0.6143 | 9.31 | 0.7255 | 0.3447 | 0.8931 |
| `recsys_als` | 0.2480 | 9.13 | 0.7732 | 0.2194 | 0.9133 |
| `recsys_popularity` | 0.0070 | 7.63 | 0.6948 | 1.0000 | 0.9965 |
| `recsys_content` | 0.6303 | 11.75 | 0.5041 | 0.0657 | 0.8665 |

### MMR Diversity Trade-Off over Ranker Top-50 Candidates

| Lambda Parameter (λ) | NDCG@10 (95% CI) | Diversity@10 | NDCG Diff vs λ=1.0 (95% CI) |
| :---: | :---: | :---: | :---: |
| **λ = 1.0** | 0.0584 [0.0530, 0.0634] | 0.7642 | +0.0000 [+0.0000, +0.0000] |
| **λ = 0.8** | 0.0582 [0.0530, 0.0633] | 0.7812 | -0.0002 [-0.0010, +0.0005] |
| **λ = 0.6** | 0.0570 [0.0518, 0.0623] | 0.8035 | -0.0014 [-0.0027, -0.0001] |
