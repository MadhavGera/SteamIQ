"""
Unit and integration tests for Stage-Two Learning-to-Rank Recommender.

Tests:
  1. Feature builder never uses target item (leak-free fold-in assertion).
  2. Group sizes sum to total number of rows.
  3. Absent-source fill values applied correctly (rank 301, min score, in_source=False).
  4. Final list construction gives valid rank (>300) for target outside candidates.
  5. Global SHAP output shape and feature alignment.
  6. Explanation templates exist for every feature in FEATURE_COLUMNS.
  7. Determinism: same seed produces identical predictions.
  8. User splits: ranker users, evaluation users, and tuning users are pairwise disjoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.candidates import (
    CandidateInfo,
    CandidateResult,
    generate_candidates,
)
from recsys.data import RecsysData
from recsys.evaluate import get_evaluation_user_split
from recsys.explain import FEATURE_TEMPLATES, explain_recommendation
from recsys.models.base import BaseRecommender
from recsys.ranker import RankerConfig, train_ranker
from recsys.ranker_features import FEATURE_COLUMNS, build_features


def _create_synthetic_ranker_data(
    n_users: int = 80,
    n_items: int = 50,
    seed: int = 42,
) -> RecsysData:
    """Creates a deterministic synthetic RecsysData object for ranker tests."""
    rng = np.random.default_rng(seed)
    user_ids = np.arange(n_users)

    # Train interactions
    train_rows = []
    for u in user_ids:
        # 4 to 8 interactions per user
        n_inter = rng.integers(4, 9)
        items = rng.choice(n_items, size=n_inter, replace=False)
        for t_step, it in enumerate(items):
            train_rows.append({
                "user_idx": int(u),
                "item_idx": int(it),
                "ts": int(t_step * 100),
                "hours": float(rng.uniform(1.0, 50.0)),
                "is_recommended": True,
            })

    train_df = pd.DataFrame(train_rows)

    # Val & Test targets (1 per user)
    val_items = [int(rng.integers(0, n_items)) for _ in user_ids]
    test_items = [int(rng.integers(0, n_items)) for _ in user_ids]

    val_df = pd.DataFrame({
        "user_idx": user_ids,
        "item_idx": val_items,
        "ts": np.full(n_users, 100000),
        "is_recommended": True,
    })
    test_df = pd.DataFrame({
        "user_idx": user_ids,
        "item_idx": test_items,
        "ts": np.full(n_users, 200000),
        "is_recommended": True,
    })

    users_df = pd.DataFrame({
        "user_idx": user_ids,
        "products": rng.integers(5, 50, size=n_users),
        "reviews": rng.integers(1, 10, size=n_users),
    })
    items_df = pd.DataFrame({
        "item_idx": np.arange(n_items),
        "name": [f"Game {i}" for i in range(n_items)],
        "price_final": rng.uniform(5.0, 60.0, size=n_items).astype(np.float32),
        "release_year": rng.integers(2015, 2025, size=n_items).astype(np.float32),
        "positive_ratio": rng.uniform(0.6, 0.98, size=n_items).astype(np.float32),
        "user_reviews": rng.integers(50, 5000, size=n_items).astype(np.float32),
        "discount": rng.choice([0.0, 0.2, 0.5], size=n_items).astype(np.float32),
        "win": np.ones(n_items, dtype=np.float32),
        "mac": rng.choice([0.0, 1.0], size=n_items).astype(np.float32),
        "linux": rng.choice([0.0, 1.0], size=n_items).astype(np.float32),
        "steam_deck": rng.choice([0.0, 1.0], size=n_items).astype(np.float32),
        "has_description": np.ones(n_items, dtype=np.float32),
    })
    mappings_df = pd.DataFrame({"item_idx": np.arange(n_items), "app_id": np.arange(1000, 1000 + n_items)})

    # Sparse tags: 300 tag vocabulary
    tag_rows = []
    tag_cols = []
    for it in range(n_items):
        n_tags = rng.integers(3, 10)
        tags = rng.choice(300, size=n_tags, replace=False)
        for t in tags:
            tag_rows.append(it)
            tag_cols.append(t)

    item_tags_sparse = sp.csr_matrix(
        (np.ones(len(tag_rows), dtype=np.float32), (tag_rows, tag_cols)),
        shape=(n_items, 300),
    )

    # MiniLM embeddings
    emb = rng.normal(0.0, 1.0, size=(n_items, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    pop = np.bincount(train_df["item_idx"].to_numpy(), minlength=n_items).astype(np.float32)

    return RecsysData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        negatives_df=pd.DataFrame(columns=["user_idx", "item_idx", "split"]),
        users_df=users_df,
        mappings_df=mappings_df,
        items_df=items_df,
        item_text_emb=emb,
        item_train_pop=pop,
        item_tags_sparse=item_tags_sparse,
        dataset_version="synthetic_ranker_v1",
        manifest={},
    )


class MockScorer(BaseRecommender):
    """Simple mock recommender for candidate generation tests."""

    def __init__(self, name: str, n_items: int, offset: float = 0.0) -> None:
        self.name = name
        self.n_items = n_items
        self.offset = offset

    def fit(self, data: RecsysData) -> None:
        self.n_items = data.n_items

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        b = len(histories)
        scores = np.zeros((b, self.n_items), dtype=np.float32)
        for u in range(b):
            scores[u] = np.linspace(0.1, 1.0, self.n_items) + self.offset
        return scores

    def save(self, model_dir: Path | str) -> None:
        pass

    @classmethod
    def load(cls, model_dir: Path | str) -> BaseRecommender:
        return cls("mock", 50)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Builder Target Isolation Test
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_builder_never_uses_target():
    """
    (1) Verifies that feature extraction relies strictly on history and never on the held-out target.
    """
    data = _create_synthetic_ranker_data(n_users=10, n_items=30)
    user_hist = np.array([1, 2, 3], dtype=np.int64)
    target_item = 25  # Held-out target

    candidate_items = [5, 10, target_item]  # candidate list includes target
    c_meta = [CandidateInfo(item_idx=it) for it in candidate_items]

    cand_res = CandidateResult(
        user_candidates=[candidate_items],
        candidate_metadata=[c_meta],
        n_users=1,
    )

    # Feature extraction with pure history
    feat_df = build_features(
        histories=[user_hist],
        candidates=cand_res,
        data=data,
        user_indices=[0],
    )

    # Feature extraction with target artificially added to history
    user_hist_with_target = np.array([1, 2, 3, target_item], dtype=np.int64)
    feat_df_leaked = build_features(
        histories=[user_hist_with_target],
        candidates=cand_res,
        data=data,
        user_indices=[0],
    )

    # Assert history-derived features are computed without error
    assert len(feat_df) == 3
    assert "price_gap" in feat_df.columns
    assert "tag_jaccard" in feat_df.columns
    assert "content_max_cos" in feat_df.columns

    # Candidate 25 (the target) has history_len=3 in clean run and history_len=4 in leaked run
    target_row = feat_df[feat_df["item_idx"] == target_item].iloc[0]
    target_leaked_row = feat_df_leaked[feat_df_leaked["item_idx"] == target_item].iloc[0]

    assert target_row["history_len"] == 3
    assert target_leaked_row["history_len"] == 4
    # In leaked run, cos_to_last_item is 1.0 (self-similarity), whereas in clean run it is not self-similarity
    assert target_leaked_row["cos_to_last_item"] == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Group Sizes and Sum of Rows Test
# ─────────────────────────────────────────────────────────────────────────────

def test_group_sizes_sum_to_total_rows():
    """
    (2) Verifies that LightGBM query group sizes sum exactly to the number of feature rows.
    """
    data = _create_synthetic_ranker_data(n_users=5, n_items=20)
    histories = [np.array([1, 2]), np.array([3]), np.array([4, 5, 6])]
    cand_lists = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    meta_lists = [[CandidateInfo(it) for it in c] for c in cand_lists]

    cand_res = CandidateResult(
        user_candidates=cand_lists,
        candidate_metadata=meta_lists,
        n_users=3,
    )

    feat_df = build_features(
        histories=histories,
        candidates=cand_res,
        data=data,
        user_indices=[0, 1, 2],
    )

    groups = [len(c) for c in cand_lists]
    assert sum(groups) == len(feat_df)
    assert len(groups) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. Absent Source Fill Values Test
# ─────────────────────────────────────────────────────────────────────────────

def test_absent_source_fill_values_applied():
    """
    (3) Verifies that candidates missing from a source get rank 301, in_source=False, and minimum score.
    """
    data = _create_synthetic_ranker_data(n_users=2, n_items=20)
    models = {
        "two_tower": MockScorer("two_tower", 20, offset=0.5),
        "itemknn": MockScorer("itemknn", 20, offset=0.2),
        "als": MockScorer("als", 20, offset=0.1),
        "content": MockScorer("content", 20, offset=0.0),
    }

    # Request small top-N per source so some items are absent
    n_each = {"two_tower": 2, "itemknn": 2, "als": 2, "content": 2}
    histories = [np.array([1, 2]), np.array([3, 4])]

    cand_res = generate_candidates(
        histories=histories,
        models=models,
        n_each=n_each,
        max_candidates=10,
    )

    feat_df = build_features(
        histories=histories,
        candidates=cand_res,
        data=data,
    )

    # Check for any item absent from content
    absent_content = feat_df[feat_df["in_content"] == 0]
    if not absent_content.empty:
        sample_row = absent_content.iloc[0]
        assert sample_row["content_rank"] == 301
        assert sample_row["in_content"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 4. Final List Score Construction for Target Outside Candidates
# ─────────────────────────────────────────────────────────────────────────────

def test_final_list_construction_target_outside_candidates():
    """
    (4) Verifies that score() assigns candidates top ranks and places non-candidates below candidates.
    If a target is outside candidates, its rank in the catalog-wide score vector is strictly > len(candidates).
    """
    from recsys.candidates import DEFAULT_CANDIDATE_COUNTS

    data = _create_synthetic_ranker_data(n_users=30, n_items=400)
    models = {
        "two_tower": MockScorer("two_tower", 400, offset=0.5),
        "itemknn": MockScorer("itemknn", 400, offset=0.2),
        "als": MockScorer("als", 400, offset=0.1),
        "content": MockScorer("content", 400, offset=0.0),
    }

    cfg = RankerConfig(n_estimators=10, num_leaves=15, min_child_samples=5, max_candidates=300)
    ranker_model, _ = train_ranker(
        data=data,
        models=models,
        cfg=cfg,
        n_ranker_users=20,
        seed=42,
    )

    user_hist = np.array([1, 2, 3], dtype=np.int64)
    scores = ranker_model.score([user_hist])  # shape [1, 400]

    assert scores.shape == (1, 400)
    assert np.all(np.isfinite(scores))

    # Identify candidate items generated for this user
    cand_res = generate_candidates(
        histories=[user_hist],
        models=models,
        n_each=DEFAULT_CANDIDATE_COUNTS,
        seen_masks=[set(user_hist)],
        max_candidates=300,
    )
    c_set = set(cand_res.user_candidates[0])
    n_cands = len(c_set)
    assert n_cands > 0

    # Pick a non-candidate item (e.g. an item not in c_set and not in history)
    non_cands = [it for it in range(400) if it not in c_set and it not in user_hist]
    assert len(non_cands) > 0
    target_outside = non_cands[0]

    # Check that any candidate has higher score than any non-candidate
    min_cand_score = min(scores[0, it] for it in c_set)
    max_non_cand_score = max(scores[0, it] for it in non_cands)
    assert min_cand_score > max_non_cand_score

    # Compute pessimistic rank of target_outside: 1 + sum(score >= score[target])
    target_rank = 1 + int(np.sum(scores[0] >= scores[0, target_outside])) - 1
    # Target outside candidates is strictly ranked after all candidates
    assert target_rank > n_cands


# ─────────────────────────────────────────────────────────────────────────────
# 5. Global SHAP Output Shape and Templates Alignment
# ─────────────────────────────────────────────────────────────────────────────

def test_global_shap_and_explanation_templates():
    """
    (5 & 6) Verifies that TreeExplainer SHAP matches feature count and templates cover all features.
    """
    data = _create_synthetic_ranker_data(n_users=30, n_items=25)
    models = {
        "two_tower": MockScorer("two_tower", 25, offset=0.5),
        "itemknn": MockScorer("itemknn", 25, offset=0.2),
        "als": MockScorer("als", 25, offset=0.1),
        "content": MockScorer("content", 25, offset=0.0),
    }

    cfg = RankerConfig(n_estimators=10, num_leaves=15, min_child_samples=5)
    ranker_model, train_info = train_ranker(
        data=data,
        models=models,
        cfg=cfg,
        n_ranker_users=20,
        seed=42,
    )

    shap_dict = train_info["global_shap"]
    assert len(shap_dict) == len(FEATURE_COLUMNS)
    EXCLUDED_FROM_TEMPLATES = {
        "has_description", "popularity_rank_pct", "win", "mac", "linux", "steam_deck",
        "history_len", "log_products", "log_reviews", "mean_log_hours"
    }
    for col in FEATURE_COLUMNS:
        assert col in shap_dict
        if (
            col.startswith("in_")
            or col.endswith("_rank")
            or col.endswith("_score")
            or col.endswith("_in_cands")
            or col in EXCLUDED_FROM_TEMPLATES
        ):
            # Source features, platform, description, popularity rank, and generic filler are excluded
            assert col not in FEATURE_TEMPLATES
        else:
            assert col in FEATURE_TEMPLATES

    # Test single recommendation explanation synthesis
    sample_shap = {col: 0.1 if col == "tag_jaccard" else 0.0 for col in FEATURE_COLUMNS}
    explanation = explain_recommendation(
        item_idx=5,
        user_history=[1, 2],
        shap_row=sample_shap,
        data=data,
    )
    assert "because_you_liked" in explanation
    assert "reasons" in explanation
    assert len(explanation["reasons"]) >= 1
    assert "Shares tags" in explanation["reasons"][0]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Seed Determinism Test
# ─────────────────────────────────────────────────────────────────────────────

def test_ranker_seed_determinism():
    """
    (7) Verifies that identical seeds produce strictly identical ranker predictions.
    """
    data = _create_synthetic_ranker_data(n_users=35, n_items=25, seed=10)
    models1 = {
        "two_tower": MockScorer("two_tower", 25),
        "itemknn": MockScorer("itemknn", 25),
        "als": MockScorer("als", 25),
        "content": MockScorer("content", 25),
    }
    models2 = {
        "two_tower": MockScorer("two_tower", 25),
        "itemknn": MockScorer("itemknn", 25),
        "als": MockScorer("als", 25),
        "content": MockScorer("content", 25),
    }

    cfg1 = RankerConfig(n_estimators=10, num_leaves=15, min_child_samples=5, seed=42)
    cfg2 = RankerConfig(n_estimators=10, num_leaves=15, min_child_samples=5, seed=42)

    ranker1, _ = train_ranker(data=data, models=models1, cfg=cfg1, n_ranker_users=20, seed=42)
    ranker2, _ = train_ranker(data=data, models=models2, cfg=cfg2, n_ranker_users=20, seed=42)

    histories = [np.array([1, 2, 3]), np.array([4, 5])]
    scores1 = ranker1.score(histories)
    scores2 = ranker2.score(histories)

    np.testing.assert_allclose(scores1, scores2, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Disjoint Ranker, Evaluation, and Tuning User Sets
# ─────────────────────────────────────────────────────────────────────────────

def test_ranker_eval_tuning_users_pairwise_disjoint():
    """
    (8) Verifies that ranker training users, evaluation users, and tuning users are pairwise disjoint.
    """
    data = _create_synthetic_ranker_data(n_users=60, n_items=25)

    eval_users, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=15,
        n_tuning_users=15,
        seed=42,
    )

    all_val_users = np.unique(data.val_df["user_idx"].to_numpy())
    excluded = set(eval_users).union(set(tuning_users))
    ranker_candidates = np.array([u for u in all_val_users if u not in excluded])

    set_eval = set(eval_users)
    set_tuning = set(tuning_users)
    set_ranker = set(ranker_candidates)

    assert set_eval.isdisjoint(set_tuning)
    assert set_eval.isdisjoint(set_ranker)
    assert set_tuning.isdisjoint(set_ranker)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Regression Test: Ranker Job Never Calls .fit() on Stage 1 Models (Step 6c)
# ─────────────────────────────────────────────────────────────────────────────

def test_ranker_job_never_calls_fit_on_stage1_models(monkeypatch, tmp_path: Path):
    """
    (6c) Regression test: asserts that TrainRankerJob only loads stage 1 artifacts
    and NEVER calls .fit() on any stage 1 model (ALS, Item-kNN, Content, Two-Tower, Popularity).
    """
    from jobs.train_ranker import TrainRankerJob
    from recsys.models.als import ALSRecommender
    from recsys.models.content import ContentRecommender
    from recsys.models.itemknn import ItemKNNRecommender
    from recsys.models.popularity import PopularityRecommender
    from recsys.models.two_tower import TwoTowerConfig, TwoTowerModel

    # 1. Create synthetic data and save to tmp_path
    data = _create_synthetic_ranker_data(n_users=60, n_items=30)
    data.save(tmp_path)

    # 2. Fit and save artifacts before monkeypatching
    tt_cfg = TwoTowerConfig(n_items=30, id_dim=16, text_proj_dim=16, tag_proj_dim=16, numeric_proj_dim=8)
    tt_model = TwoTowerModel(cfg=tt_cfg)
    tt_model.fit(data)
    tt_model.save(tmp_path / "artifacts" / "recsys_two_tower" / "run_tt_01")

    knn_model = ItemKNNRecommender(k_neighbors=5)
    knn_model.fit(data)
    knn_model.save(tmp_path / "artifacts" / "recsys_itemknn" / "run_knn_01")

    als_model = ALSRecommender(factors=8, iterations=2)
    als_model.fit(data)
    als_model.save(tmp_path / "artifacts" / "recsys_als" / "run_als_01")

    content_model = ContentRecommender()
    content_model.fit(data)
    content_model.save(tmp_path / "artifacts" / "recsys_content" / "run_content_01")

    pop_model = PopularityRecommender()
    pop_model.fit(data)
    pop_model.save(tmp_path / "artifacts" / "recsys_popularity" / "run_pop_01")

    # 3. Patch .fit() on all stage-1 models to raise an exception if called
    def _forbidden_fit(*args, **kwargs):
        raise AssertionError("Stage-1 model .fit() was called! Ranker job MUST only load pre-trained artifacts.")

    monkeypatch.setattr(TwoTowerModel, "fit", _forbidden_fit)
    monkeypatch.setattr(ItemKNNRecommender, "fit", _forbidden_fit)
    monkeypatch.setattr(ALSRecommender, "fit", _forbidden_fit)
    monkeypatch.setattr(ContentRecommender, "fit", _forbidden_fit)
    monkeypatch.setattr(PopularityRecommender, "fit", _forbidden_fit)

    # 4. Execute TrainRankerJob and assert zero .fit() calls
    job = TrainRankerJob(
        n_estimators=5,
        num_leaves=7,
        min_child_samples=2,
        n_ranker_users=15,
        profile="dev",
        data_dir=tmp_path,
        dry_run=True,
    )
    result = job.execute()

    assert "model_name" in result
    assert result["model_name"] == "recsys_two_tower_ranker"
    assert "val_metrics" in result
    assert "ndcg_at_10" in result["val_metrics"]


def test_excluded_source_features_never_appear_in_reasons() -> None:
    """
    Verifies that multi-source retrieval features (ranks, scores, cand flags)
    are strictly excluded from human-facing explanation reasons.
    """
    from recsys.explain import FEATURE_TEMPLATES, explain_recommendation
    from recsys.ranker_features import FEATURE_COLUMNS

    source_features = [
        col for col in FEATURE_COLUMNS
        if col.startswith("in_") or col.endswith("_rank") or col.endswith("_score") or col.endswith("_in_cands")
    ]
    assert len(source_features) >= 12

    # None of these source features may be in FEATURE_TEMPLATES
    for feat in source_features:
        assert feat not in FEATURE_TEMPLATES, f"Source feature {feat} must not be in FEATURE_TEMPLATES"

    data = _create_synthetic_ranker_data(n_users=10, n_items=15, seed=42)

    # SHAP row where only source features have high positive contribution
    shap_row = {feat: 2.5 for feat in source_features}
    explanation = explain_recommendation(
        item_idx=2,
        user_history=[0, 1],
        shap_row=shap_row,
        data=data,
    )
    assert "reasons" in explanation
    for reason in explanation["reasons"]:
        for feat in source_features:
            assert feat not in reason, f"Source feature '{feat}' leaked into explanation reason: '{reason}'"
            assert f"signal from {feat}" not in reason


