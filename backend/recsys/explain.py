"""
Explanation engine for two-stage recommender recommendations.

Implements:
  1. Global SHAP feature importances using shap.TreeExplainer on sample ranking rows.
  2. Per-recommendation local explanation synthesis:
     - 'because_you_liked': top 2 history items with highest cosine similarity to candidate.
     - 'reasons': top 3 features by positive SHAP contribution mapped to human-readable templates.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from recsys.data import RecsysData

logger = logging.getLogger("steamiq.recsys.explain")

FEATURE_TEMPLATES: dict[str, str] = {
    # Item features
    "age_years": "Release era aligns with games you liked/reviewed",
    "price_final": "Matches your preferred game pricing tier",
    "discount": "Currently on discount in the Steam store",
    # User-item features
    "price_gap": "Priced close to games you usually like/review",
    "abs_price_gap": "Priced similarly to titles you liked/reviewed",
    "tag_jaccard": "Shares tags with games you liked/reviewed",
    "content_max_cos": "Very similar themes and description to games you liked/reviewed",
    "content_mean_cos": "Strong thematic match with games you liked/reviewed",
    "cos_to_last_item": "Similar in style to games you liked/reviewed",
    "release_year_gap": "Released in a similar era to games you liked/reviewed",
}

FEATURE_CONCEPTS: dict[str, str] = {
    "age_years": "era",
    "release_year_gap": "era",
    "price_final": "price",
    "price_gap": "price",
    "abs_price_gap": "price",
    "discount": "discount",
    "tag_jaccard": "tags",
    "content_max_cos": "content",
    "content_mean_cos": "content",
    "cos_to_last_item": "content",
}


def global_shap(
    ranker: Any,
    X_sample: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    max_sample_size: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Computes mean absolute SHAP value per feature using shap.TreeExplainer.

    Args:
        ranker: Trained LightGBM ranker or Booster.
        X_sample: Feature matrix (DataFrame or 2D array).
        feature_names: Optional list of feature names.
        max_sample_size: Maximum rows to sample for SHAP computation.
        seed: Random seed for sampling.

    Returns:
        Dictionary mapping feature_name -> mean_abs_shap sorted descending.
    """
    if isinstance(X_sample, pd.DataFrame):
        names = feature_names or list(X_sample.columns)
        if len(X_sample) > max_sample_size:
            sampled_df = X_sample.sample(n=max_sample_size, random_state=seed)
        else:
            sampled_df = X_sample
        X_arr = sampled_df.to_numpy(dtype=np.float32)
    else:
        names = feature_names or [f"feature_{i}" for i in range(X_sample.shape[1])]
        if len(X_sample) > max_sample_size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_sample), size=max_sample_size, replace=False)
            X_arr = np.asarray(X_sample[idx], dtype=np.float32)
        else:
            X_arr = np.asarray(X_sample, dtype=np.float32)

    try:
        # Extract underlying Booster if LGBMModel wrapper
        booster = getattr(ranker, "booster_", ranker)
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_arr)

        if isinstance(shap_values, list):
            shap_mat = np.asarray(shap_values[0], dtype=np.float32)
        else:
            shap_mat = np.asarray(shap_values, dtype=np.float32)

        mean_abs_shap = np.mean(np.abs(shap_mat), axis=0)
        shap_dict: dict[str, float] = {}
        for name, val in zip(names, mean_abs_shap, strict=False):
            shap_dict[name] = float(val)

        # Sort descending by mean absolute SHAP
        return dict(sorted(shap_dict.items(), key=lambda x: x[1], reverse=True))
    except Exception as exc:
        logger.warning("SHAP calculation fallback: %s", exc)
        # Fallback to feature importances if SHAP fails
        importances = getattr(ranker, "feature_importances_", None)
        if importances is not None:
            return {
                name: float(imp)
                for name, imp in sorted(zip(names, importances, strict=False), key=lambda x: x[1], reverse=True)
            }
        return {name: 0.0 for name in names}


def explain_recommendation(
    item_idx: int,
    user_history: list[int] | np.ndarray,
    shap_row: dict[str, float] | np.ndarray,
    data: RecsysData,
    feature_names: list[str] | None = None,
    two_tower_item_vecs: np.ndarray | None = None,
    top_n_anchors: int = 2,
    top_n_reasons: int = 3,
    min_similarity: float = 0.2,
) -> dict[str, Any]:
    """
    Synthesizes human-readable explanations for a recommended candidate item.

    Args:
        item_idx: Recommended candidate item index.
        user_history: List or array of item_idx in user's history.
        shap_row: SHAP values for this specific recommendation row (dict or array).
        data: RecsysData bundle holding catalog and embeddings.
        feature_names: Feature names corresponding to shap_row array indices.
        two_tower_item_vecs: Optional precomputed L2-normalized Two-Tower item vectors.
        top_n_anchors: Number of 'because you liked' anchor games to return (default: 2).
        top_n_reasons: Number of human-readable explanation reasons to return (default: 3).
        min_similarity: Cosine similarity cutoff for valid anchor games (default: 0.2).

    Returns:
        Dictionary holding item_idx, game_name, because_you_liked list, and reasons list.
    """
    # 1. Resolve item name
    item_name = f"Game #{item_idx}"
    if data.items_df is not None and not data.items_df.empty:
        df = data.items_df
        if "name" in df.columns:
            name_matches = df[df["item_idx"] == item_idx]["name"]
            if not name_matches.empty:
                item_name = str(name_matches.iloc[0])

    # 2. Compute 'because_you_liked' anchor games via cosine similarity
    hist_arr = np.asarray(user_history, dtype=np.int64)
    valid_hist = hist_arr[(hist_arr >= 0) & (hist_arr < data.n_items)]

    because_you_liked: list[dict[str, Any]] = []
    if len(valid_hist) > 0:
        if two_tower_item_vecs is not None and item_idx < len(two_tower_item_vecs):
            c_vec = two_tower_item_vecs[item_idx]
            h_vecs = two_tower_item_vecs[valid_hist]
            sims = np.dot(h_vecs, c_vec)
        elif data.item_text_emb is not None and item_idx < len(data.item_text_emb):
            c_vec = data.item_text_emb[item_idx]
            c_norm = np.linalg.norm(c_vec) or 1.0
            h_vecs = data.item_text_emb[valid_hist]
            h_norms = np.linalg.norm(h_vecs, axis=1, keepdims=True)
            h_norms[h_norms == 0] = 1.0
            sims = np.dot(h_vecs / h_norms, c_vec / c_norm)
        else:
            sims = np.ones(len(valid_hist), dtype=np.float32)

        # Top k anchors
        k_anchors = min(top_n_anchors, len(valid_hist))
        top_anchor_idx = np.argsort(-sims)[:k_anchors]

        for a_pos in top_anchor_idx:
            sim_val = float(sims[a_pos])
            if sim_val < min_similarity:
                continue
            anchor_item = int(valid_hist[a_pos])
            a_name = f"Game #{anchor_item}"
            if data.items_df is not None and not data.items_df.empty and "name" in data.items_df.columns:
                matches = data.items_df[data.items_df["item_idx"] == anchor_item]["name"]
                if not matches.empty:
                    a_name = str(matches.iloc[0])

            because_you_liked.append({
                "item_idx": anchor_item,
                "name": a_name,
                "similarity": sim_val,
            })

    # 3. Map top positive SHAP features to human-readable explanation reasons (deduped by concept)
    if isinstance(shap_row, dict):
        shap_items = list(shap_row.items())
    elif feature_names is not None:
        shap_items = list(zip(feature_names, shap_row, strict=False))
    else:
        shap_items = [(f"feature_{i}", float(val)) for i, val in enumerate(shap_row)]

    # Filter for positive contributions (features that pushed score higher)
    pos_shap = [(feat, float(val)) for feat, val in shap_items if val > 0]
    pos_shap.sort(key=lambda x: x[1], reverse=True)

    reasons: list[str] = []
    seen_concepts: set[str] = set()

    for feat, _val in pos_shap:
        if feat not in FEATURE_TEMPLATES:
            continue
        concept = FEATURE_CONCEPTS.get(feat, feat)
        if concept in seen_concepts:
            continue
        seen_concepts.add(concept)
        template = FEATURE_TEMPLATES[feat]
        reasons.append(template)
        if len(reasons) >= top_n_reasons:
            break

    # Fallback if no positive SHAP feature was found
    if not reasons:
        reasons.append("Matches your overall library play preferences")

    return {
        "item_idx": item_idx,
        "name": item_name,
        "because_you_liked": because_you_liked,
        "reasons": reasons,
    }
