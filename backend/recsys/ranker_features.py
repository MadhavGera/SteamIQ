"""
Feature extraction pipeline for the Learning-to-Rank recommender.

Computes 26 tabular ranking features per (user, candidate_item) pair:
  1. Multi-source retrieval signals: scores, ranks (1..300, 301 if absent), and presence flags for TT, Item-kNN, ALS, Content.
  2. Item catalog features: log popularity, popularity percentile, age, positive ratio, log reviews, price, discount, OS flags, description flag.
  3. User-Item interaction features (derived strictly from fold-in history):
     - price_gap, abs_price_gap (vs history median)
     - tag_jaccard (candidate tags vs user's top-20 history tags)
     - content_max_cos, content_mean_cos (MiniLM cosine vs history items)
     - cos_to_last_item (MiniLM cosine vs most recent history item)
     - release_year_gap (vs history median)
  4. User profile features: history length, log products, log reviews, mean log interaction hours.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.candidates import CandidateInfo, CandidateResult
from recsys.data import RecsysData

logger = logging.getLogger("steamiq.recsys.ranker_features")

FEATURE_COLUMNS: list[str] = [
    # Source scores and ranks
    "tt_score",
    "tt_rank",
    "in_tt",
    "itemknn_score",
    "itemknn_rank",
    "in_itemknn",
    "als_score",
    "als_rank",
    "in_als",
    "content_score",
    "content_rank",
    "in_content",
    # Item features
    "log_train_pop",
    "popularity_rank_pct",
    "age_years",
    "positive_ratio",
    "log_user_reviews",
    "price_final",
    "discount",
    "win",
    "mac",
    "linux",
    "steam_deck",
    "has_description",
    # User-item features
    "price_gap",
    "abs_price_gap",
    "tag_jaccard",
    "content_max_cos",
    "content_mean_cos",
    "cos_to_last_item",
    "release_year_gap",
    # User features
    "history_len",
    "log_products",
    "log_reviews",
    "mean_log_hours",
]


class ItemFeatureCache:
    """Pre-extracted and vectorized item catalog features for fast batch feature extraction."""

    def __init__(self, data: RecsysData, train_stats_only: bool = False) -> None:
        n_items = data.n_items
        self.n_items = n_items
        self.train_stats_only = train_stats_only

        # 1. Popularity features
        if data.item_train_pop is not None and len(data.item_train_pop) == n_items:
            pop = np.asarray(data.item_train_pop, dtype=np.float32)
        else:
            pop = np.zeros(n_items, dtype=np.float32)

        self.log_train_pop = np.log1p(pop)
        if len(pop) > 0 and pop.max() > pop.min():
            ranks = pd.Series(pop).rank(pct=True).to_numpy(dtype=np.float32)
            self.popularity_rank_pct = ranks
        else:
            self.popularity_rank_pct = np.zeros(n_items, dtype=np.float32)

        # 2. Item metadata features
        self.price_final = np.zeros(n_items, dtype=np.float32)
        self.release_year = np.full(n_items, 2020.0, dtype=np.float32)
        self.age_years = np.full(n_items, 5.0, dtype=np.float32)
        self.positive_ratio = np.full(n_items, 80.0, dtype=np.float32)
        self.log_user_reviews = np.zeros(n_items, dtype=np.float32)
        self.discount = np.zeros(n_items, dtype=np.float32)
        self.win = np.ones(n_items, dtype=np.float32)
        self.mac = np.zeros(n_items, dtype=np.float32)
        self.linux = np.zeros(n_items, dtype=np.float32)
        self.steam_deck = np.zeros(n_items, dtype=np.float32)
        self.has_description = np.zeros(n_items, dtype=np.float32)

        # Derive train-only positive_ratio and log_review_count from train + train negatives if requested
        if train_stats_only and data.train_df is not None:
            pos_count = np.bincount(data.train_df["item_idx"], minlength=n_items)
            neg_count = np.zeros(n_items, dtype=np.int64)
            if data.negatives_df is not None and "split" in data.negatives_df.columns:
                neg_train = data.negatives_df[data.negatives_df["split"] == "train"]
                neg_count = np.bincount(neg_train["item_idx"], minlength=n_items)
            tot_reviews = pos_count + neg_count
            has_reviews = tot_reviews > 0
            self.positive_ratio[has_reviews] = (pos_count[has_reviews] / tot_reviews[has_reviews]) * 100.0
            self.log_user_reviews = np.log1p(tot_reviews.astype(np.float32))

        if data.items_df is not None and not data.items_df.empty:
            df = data.items_df
            idx_arr = df["item_idx"].to_numpy() if "item_idx" in df.columns else np.arange(len(df))

            for col in ["price_final", "price"]:
                if col in df.columns:
                    self.price_final[idx_arr] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
                    break

            for col in ["release_year", "year"]:
                if col in df.columns:
                    years = pd.to_numeric(df[col], errors="coerce").fillna(2020.0).to_numpy()
                    self.release_year[idx_arr] = years
                    self.age_years[idx_arr] = np.maximum(0.0, 2026.0 - years)
                    break

            if not train_stats_only:
                for col in ["positive_ratio", "pos_ratio", "rating"]:
                    if col in df.columns:
                        self.positive_ratio[idx_arr] = pd.to_numeric(df[col], errors="coerce").fillna(80.0).to_numpy()
                        break

                for col in ["user_reviews", "reviews", "num_reviews"]:
                    if col in df.columns:
                        revs = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
                        self.log_user_reviews[idx_arr] = np.log1p(np.maximum(0.0, revs))
                        break

            for col in ["discount", "discount_pct"]:
                if col in df.columns:
                    self.discount[idx_arr] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
                    break

            for os_name, target_arr in [
                ("win", self.win),
                ("mac", self.mac),
                ("linux", self.linux),
                ("steam_deck", self.steam_deck),
            ]:
                if os_name in df.columns:
                    target_arr[idx_arr] = df[os_name].astype(bool).astype(np.float32).to_numpy()

            if "has_description" in df.columns:
                self.has_description[idx_arr] = df["has_description"].astype(bool).astype(np.float32).to_numpy()
            elif "description" in df.columns:
                has_d = df["description"].fillna("").astype(str).str.len() > 10
                self.has_description[idx_arr] = has_d.astype(np.float32).to_numpy()

        # 3. Normalized text embeddings for cosine similarity
        if data.item_text_emb is not None and len(data.item_text_emb) == n_items:
            emb = np.asarray(data.item_text_emb, dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.text_emb_norm = emb / norms
            # Update has_description from embedding norm if not already set
            nonzero_emb = (norms.squeeze() > 1e-4).astype(np.float32)
            if self.has_description.sum() == 0:
                self.has_description = nonzero_emb
        else:
            self.text_emb_norm = np.zeros((n_items, 384), dtype=np.float32)

        # 4. Item tags sparse CSR matrix and set lookups
        if data.item_tags_sparse is not None:
            self.tags_csr: sp.csr_matrix = data.item_tags_sparse.tocsr()
        else:
            self.tags_csr = sp.csr_matrix((n_items, 300), dtype=np.float32)

        self.item_tag_sets: list[set[int]] = []
        for i in range(n_items):
            row_tags = self.tags_csr[i].indices
            self.item_tag_sets.append(set(int(t) for t in row_tags))


class UserFeatureCache:
    """Pre-extracted user profile stats and interaction hours."""

    def __init__(self, data: RecsysData) -> None:
        self.user_log_products: dict[int, float] = {}
        self.user_log_reviews: dict[int, float] = {}
        self.user_item_hours: dict[tuple[int, int], float] = {}
        self.median_log_products: float = 4.9836
        self.median_log_reviews: float = 2.3026
        self.median_log_hours: float = 3.2809

        if data.users_df is not None and not data.users_df.empty:
            u_df = data.users_df
            all_prods = []
            all_revs = []
            for r in u_df.to_dict(orient="records"):
                u_idx = int(r.get("user_idx", -1))
                if u_idx >= 0:
                    prods = float(r.get("products", 0.0) or 0.0)
                    revs = float(r.get("reviews", 0.0) or 0.0)
                    lp = float(np.log1p(max(0.0, prods)))
                    lr = float(np.log1p(max(0.0, revs)))
                    self.user_log_products[u_idx] = lp
                    self.user_log_reviews[u_idx] = lr
                    all_prods.append(lp)
                    all_revs.append(lr)
            if all_prods:
                self.median_log_products = float(np.median(all_prods))
            if all_revs:
                self.median_log_reviews = float(np.median(all_revs))

        if data.train_df is not None and not data.train_df.empty and "hours" in data.train_df.columns:
            all_hrs = []
            for u, it, h in zip(
                data.train_df["user_idx"].to_numpy(),
                data.train_df["item_idx"].to_numpy(),
                data.train_df["hours"].to_numpy(),
                strict=True,
            ):
                h_val = float(h or 0.0)
                self.user_item_hours[(int(u), int(it))] = h_val
                all_hrs.append(np.log1p(max(0.0, h_val)))
            if all_hrs:
                self.median_log_hours = float(np.median(all_hrs))


def build_features(
    histories: list[np.ndarray],
    candidates: CandidateResult | list[list[int]],
    data: RecsysData,
    user_indices: list[int] | np.ndarray | None = None,
    candidate_metadata: list[list[CandidateInfo]] | None = None,
    item_cache: ItemFeatureCache | None = None,
    user_cache: UserFeatureCache | None = None,
) -> pd.DataFrame:
    """
    Builds the 26 ranking features for all (user, candidate_item) pairs.

    CRITICAL: Features are derived STRICTLY from the user's history items and catalog metadata.
              The target item is never used in feature computation.

    Args:
        histories: List of user fold-in histories (item_idx arrays).
        candidates: CandidateResult or list of candidate item lists per user.
        data: RecsysData artifact bundle.
        user_indices: Optional array of explicit user IDs (defaults to 0..N-1).
        candidate_metadata: Optional per-candidate source scores/ranks (if not passed in CandidateResult).
        item_cache: Optional cached ItemFeatureCache.
        user_cache: Optional cached UserFeatureCache.

    Returns:
        pd.DataFrame containing feature columns plus ['user_idx', 'item_idx', 'group_id'].
    """
    if isinstance(candidates, CandidateResult):
        user_cands = candidates.user_candidates
        c_meta = candidates.candidate_metadata
    else:
        user_cands = candidates
        c_meta = candidate_metadata

    n_users = len(histories)
    u_ids = list(range(n_users)) if user_indices is None else [int(u) for u in user_indices]

    if item_cache is None:
        item_cache = ItemFeatureCache(data)
    if user_cache is None:
        user_cache = UserFeatureCache(data)

    rows: list[dict[str, Any]] = []

    for u_i, (u_id, hist) in enumerate(zip(u_ids, histories, strict=True)):
        c_items = user_cands[u_i]
        meta_list = c_meta[u_i] if c_meta and u_i < len(c_meta) else None

        # 1. Compute user history stats
        hist_arr = np.asarray(hist, dtype=np.int64)
        valid_hist = hist_arr[(hist_arr >= 0) & (hist_arr < item_cache.n_items)]
        hist_len = len(valid_hist)

        # Median price & release year across history
        if hist_len > 0:
            hist_prices = item_cache.price_final[valid_hist]
            hist_years = item_cache.release_year[valid_hist]
            med_price = float(np.median(hist_prices))
            med_year = float(np.median(hist_years))

            # User top 20 tags across history
            hist_tag_counts: dict[int, int] = {}
            for h_it in valid_hist:
                for t in item_cache.item_tag_sets[h_it]:
                    hist_tag_counts[t] = hist_tag_counts.get(t, 0) + 1

            if hist_tag_counts:
                sorted_tags = sorted(hist_tag_counts.items(), key=lambda x: x[1], reverse=True)
                user_top20_tags = set(t for t, _ in sorted_tags[:20])
            else:
                user_top20_tags = set()

            # Normalized history embeddings for cosine similarities: [L, 384]
            hist_embs = item_cache.text_emb_norm[valid_hist]
            last_emb = item_cache.text_emb_norm[valid_hist[-1]]

            # Mean log hours (uses training interactions if known user, or median for anonymous)
            if u_id is not None and u_id in user_cache.user_log_products:
                hours_list = [user_cache.user_item_hours.get((u_id, int(h_it)), 0.0) for h_it in valid_hist]
                mean_log_hrs = float(np.mean([np.log1p(max(0.0, h)) for h in hours_list]))
            else:
                mean_log_hrs = user_cache.median_log_hours
        else:
            med_price = 0.0
            med_year = 2020.0
            user_top20_tags = set()
            hist_embs = np.empty((0, 384), dtype=np.float32)
            last_emb = np.zeros(384, dtype=np.float32)
            mean_log_hrs = user_cache.median_log_hours

        u_log_prods = user_cache.user_log_products.get(u_id, user_cache.median_log_products) if u_id is not None else user_cache.median_log_products
        u_log_revs = user_cache.user_log_reviews.get(u_id, user_cache.median_log_reviews) if u_id is not None else user_cache.median_log_reviews

        for c_idx, cand_item in enumerate(c_items):
            c_int = int(cand_item)
            if c_int < 0 or c_int >= item_cache.n_items:
                continue

            info: CandidateInfo | None = meta_list[c_idx] if meta_list and c_idx < len(meta_list) else None

            # Source signals
            tt_score = info.tt_score if info else 0.0
            tt_rank = info.tt_rank if info else 301
            in_tt = int(info.in_tt) if info else 0

            knn_score = info.itemknn_score if info else 0.0
            knn_rank = info.itemknn_rank if info else 301
            in_knn = int(info.in_itemknn) if info else 0

            als_score = info.als_score if info else 0.0
            als_rank = info.als_rank if info else 301
            in_als = int(info.in_als) if info else 0

            cnt_score = info.content_score if info else 0.0
            cnt_rank = info.content_rank if info else 301
            in_cnt = int(info.in_content) if info else 0

            # Item static features
            c_price = float(item_cache.price_final[c_int])
            c_year = float(item_cache.release_year[c_int])
            c_age = float(item_cache.age_years[c_int])
            c_pop_log = float(item_cache.log_train_pop[c_int])
            c_pop_pct = float(item_cache.popularity_rank_pct[c_int])
            c_pos_ratio = float(item_cache.positive_ratio[c_int])
            c_log_revs = float(item_cache.log_user_reviews[c_int])
            c_disc = float(item_cache.discount[c_int])
            c_win = float(item_cache.win[c_int])
            c_mac = float(item_cache.mac[c_int])
            c_linux = float(item_cache.linux[c_int])
            c_deck = float(item_cache.steam_deck[c_int])
            c_has_desc = float(item_cache.has_description[c_int])

            # User-Item relative features
            if hist_len > 0:
                price_gap = c_price - med_price
                abs_price_gap = abs(price_gap)
                release_year_gap = c_year - med_year

                # Tag Jaccard vs top 20 history tags
                c_tags = item_cache.item_tag_sets[c_int]
                union_tags = c_tags.union(user_top20_tags)
                if union_tags:
                    tag_jaccard = float(len(c_tags.intersection(user_top20_tags)) / len(union_tags))
                else:
                    tag_jaccard = 0.0

                # Content cosine similarities: [L]
                c_emb = item_cache.text_emb_norm[c_int]
                cos_sims = np.dot(hist_embs, c_emb)
                content_max_cos = float(np.max(cos_sims))
                content_mean_cos = float(np.mean(cos_sims))
                cos_to_last = float(np.dot(last_emb, c_emb))
            else:
                price_gap = 0.0
                abs_price_gap = 0.0
                release_year_gap = 0.0
                tag_jaccard = 0.0
                content_max_cos = 0.0
                content_mean_cos = 0.0
                cos_to_last = 0.0

            rows.append({
                "user_idx": u_id,
                "item_idx": c_int,
                "group_id": u_i,
                # Source features
                "tt_score": tt_score,
                "tt_rank": tt_rank,
                "in_tt": in_tt,
                "itemknn_score": knn_score,
                "itemknn_rank": knn_rank,
                "in_itemknn": in_knn,
                "als_score": als_score,
                "als_rank": als_rank,
                "in_als": in_als,
                "content_score": cnt_score,
                "content_rank": cnt_rank,
                "in_content": in_cnt,
                # Item features
                "log_train_pop": c_pop_log,
                "popularity_rank_pct": c_pop_pct,
                "age_years": c_age,
                "positive_ratio": c_pos_ratio,
                "log_user_reviews": c_log_revs,
                "price_final": c_price,
                "discount": c_disc,
                "win": c_win,
                "mac": c_mac,
                "linux": c_linux,
                "steam_deck": c_deck,
                "has_description": c_has_desc,
                # User-item features
                "price_gap": price_gap,
                "abs_price_gap": abs_price_gap,
                "tag_jaccard": tag_jaccard,
                "content_max_cos": content_max_cos,
                "content_mean_cos": content_mean_cos,
                "cos_to_last_item": cos_to_last,
                "release_year_gap": release_year_gap,
                # User features
                "history_len": hist_len,
                "log_products": u_log_prods,
                "log_reviews": u_log_revs,
                "mean_log_hours": mean_log_hrs,
            })

    if not rows:
        empty_cols = ["user_idx", "item_idx", "group_id"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=empty_cols)

    return pd.DataFrame(rows)
