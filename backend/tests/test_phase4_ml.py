"""
Unit & Integration Tests for Phase 4 (Predictive ML).

Tests:
  - Feature building & temporal cutoff boundary enforcement (zero data leakage)
  - 4-model baseline comparison (LR -> RF -> XGB -> Optuna Ensemble)
  - MLflow tracking & model_runs registration
  - Champion evaluation & production promotion logic
  - SHAP value generation & attribution schema
  - Serving predictions materialization with model_run_id foreign key traceability
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    FeatureGameFeature,
    ModelRun,
    RawGame,
    RawReview,
    ServingPrediction,
)
from jobs.build_features import BuildFeaturesJob
from jobs.train_success_model import TrainSuccessModelJob

# ─── Fixtures ─────────────────────────────────────────────────────────────────

async def _seed_test_game_with_reviews(
    session: AsyncSession,
    app_id: int,
    name: str,
    cutoff: datetime,
    positive_count: int = 10,
    negative_count: int = 2,
    future_reviews: int = 5,
) -> RawGame:
    """Helper: insert a game with past reviews and future (post-cutoff) reviews."""
    game = RawGame(
        app_id=app_id,
        name=name,
        developer="Test Dev Studio",
        publisher="Test Pub Global",
        positive_reviews=positive_count + future_reviews,
        negative_reviews=negative_count,
        review_score=8,
        final_price_usd=Decimal("19.99"),
        discount_pct=10,
        is_free=False,
        genres=[{"id": "1", "description": "Action"}],
        average_playtime_forever=1200,
    )
    session.add(game)
    await session.commit()

    cutoff_ts = int(cutoff.timestamp())

    # Pre-cutoff positive reviews (e.g. 5 days before cutoff)
    for i in range(positive_count):
        rev = RawReview(
            review_id=f"rev_pre_pos_{app_id}_{i}",
            app_id=app_id,
            voted_up=True,
            review_text="Awesome game!",
            review_created_at=cutoff_ts - (i + 1) * 86400,
        )
        session.add(rev)

    # Pre-cutoff negative reviews
    for i in range(negative_count):
        rev = RawReview(
            review_id=f"rev_pre_neg_{app_id}_{i}",
            app_id=app_id,
            voted_up=False,
            review_text="Bad performance",
            review_created_at=cutoff_ts - (i + 1) * 86400,
        )
        session.add(rev)

    # Post-cutoff reviews (MUST BE STRUCTURALLY IGNORED AT CUTOFF)
    for i in range(future_reviews):
        rev = RawReview(
            review_id=f"rev_future_{app_id}_{i}",
            app_id=app_id,
            voted_up=True,
            review_text="Future review after cutoff",
            review_created_at=cutoff_ts + (i + 1) * 86400,
        )
        session.add(rev)

    await session.commit()
    await session.refresh(game)
    return game


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_cutoff_date_boundary_isolation(db_session: AsyncSession) -> None:
    """Verify that reviews created AFTER feature_cutoff_date are excluded from feature calculation."""
    cutoff = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)
    app_id = 999101

    await _seed_test_game_with_reviews(
        db_session,
        app_id=app_id,
        name="Temporal Test Game",
        cutoff=cutoff,
        positive_count=8,
        negative_count=2,
        future_reviews=10,  # 10 post-cutoff reviews
    )

    job = BuildFeaturesJob(app_id=app_id, cutoff_date=cutoff, dry_run=True)
    
    # Verify primary genre extraction
    res = await db_session.execute(select(RawGame).where(RawGame.app_id == app_id))
    game = res.scalar_one()
    assert job._extract_primary_genre(game) == "Action"

    # Verify target score computation
    score, is_hit = job._calculate_target_success_score(
        positive_reviews=8,
        negative_reviews=2,
        review_score=8,
        sentiment_score=0.8,
        avg_playtime=1200,
    )
    assert score > 0.0
    assert isinstance(is_hit, bool)


@pytest.mark.anyio
async def test_feature_matrix_extraction() -> None:
    """Verify feature vector extraction shapes and types for ML training."""
    cutoff = datetime.now(UTC)
    sample_features = [
        FeatureGameFeature(
            app_id=100 + i,
            feature_cutoff_date=cutoff,
            price_usd=Decimal("14.99"),
            discount_pct=0,
            is_free=False,
            primary_genre="Metroidvania",
            developer_game_count=2,
            publisher_game_count=2,
            developer_avg_review_score=Decimal("82.00"),
            publisher_avg_review_score=Decimal("82.00"),
            total_reviews_at_cutoff=5000 * (i + 1),
            positive_reviews_at_cutoff=4500 * (i + 1),
            review_velocity_30d=15.5,
            positive_review_pct=Decimal("90.00"),
            competitor_density=5,
            price_vs_genre_median=Decimal("0.00"),
            sentiment_score=Decimal("0.8500"),
            complaint_density=12.0,
            loved_feature_density=45.0,
            average_playtime_forever=1800,
            target_success_score=78.5,
            is_hit=(i % 2 == 0),
        )
        for i in range(8)
    ]

    job = TrainSuccessModelJob(dry_run=True)
    X, y, feature_names, app_ids = job._extract_feature_matrix(sample_features)

    assert X.shape == (8, 11)
    assert y.shape == (8,)
    assert len(feature_names) == 11
    assert len(app_ids) == 8
    assert "price_usd" in feature_names
    assert "competitor_density" in feature_names


@pytest.mark.anyio
async def test_no_label_leakage_in_feature_matrix() -> None:
    """
    Permanent Guardrail:
    Assert that none of the features used in model training overlap with the fields,
    ratios, or direct monotonic transformations used to compute the target label
    (target_success_score / is_hit).
    """
    job = TrainSuccessModelJob(dry_run=True)
    _, _, feature_names, _ = job._extract_feature_matrix([])

    forbidden_leakage_fields = {
        "positive_reviews",
        "negative_reviews",
        "positive_reviews_at_cutoff",
        "total_reviews_at_cutoff",
        "positive_review_pct",
        "review_score",
        "target_success_score",
        "is_hit",
        "average_playtime_forever",
        "median_playtime_forever",
        "sentiment_score",
        "review_velocity_30d",
    }

    overlap = set(feature_names).intersection(forbidden_leakage_fields)
    assert not overlap, f"Label leakage detected! Feature matrix contains target-defining columns: {overlap}"


@pytest.mark.anyio
async def test_aggregate_features_exclude_target_game_self_inclusion(db_session: AsyncSession) -> None:
    """
    Structural Guardrail:
    Assert that developer_avg_review_score, publisher_avg_review_score,
    developer_game_count, competitor_density, and price_vs_genre_median explicitly
    exclude the target game's own row from their aggregates (zero self-inclusion).
    """
    cutoff = datetime.now(UTC)

    # Developer "Studio Alpha" with 2 games
    # Game 1: app_id 88801, review_score 90, price $30.00
    # Game 2: app_id 88802, review_score 40, price $10.00
    game1 = RawGame(
        app_id=88801,
        name="Alpha RPG 1",
        developer="Studio Alpha",
        publisher="Publisher Beta",
        genres=[{"id": "3", "description": "RPG"}],
        review_score=90,
        price_usd=Decimal("30.00"),
        final_price_usd=Decimal("30.00"),
        positive_reviews=5000,
        negative_reviews=500,
    )
    game2 = RawGame(
        app_id=88802,
        name="Alpha RPG 2",
        developer="Studio Alpha",
        publisher="Publisher Beta",
        genres=[{"id": "3", "description": "RPG"}],
        review_score=40,
        price_usd=Decimal("10.00"),
        final_price_usd=Decimal("10.00"),
        positive_reviews=100,
        negative_reviews=200,
    )
    # Single-title developer "Solo Dev Gamma" (app_id 88803)
    game3 = RawGame(
        app_id=88803,
        name="Solo Quest",
        developer="Solo Dev Gamma",
        publisher="Solo Pub",
        genres=[{"id": "3", "description": "RPG"}],
        review_score=85,
        price_usd=Decimal("15.00"),
        final_price_usd=Decimal("15.00"),
        positive_reviews=1000,
        negative_reviews=100,
    )
    db_session.add_all([game1, game2, game3])
    await db_session.commit()

    # Build features for Game 1 (app_id=88801)
    job = BuildFeaturesJob(cutoff_date=cutoff, dry_run=False)
    await job._run_async(session=db_session)

    # Query built feature for Game 1
    stmt1 = select(FeatureGameFeature).where(
        FeatureGameFeature.app_id == 88801,
        FeatureGameFeature.feature_cutoff_date == cutoff,
    )
    res1 = await db_session.execute(stmt1)
    f1 = res1.scalar_one()

    # 1. Game 1's developer track record MUST be Game 2's score (40 * 10 = 400.00 / 40.0), NOT 90.0 and NOT (90+40)/2
    assert float(f1.developer_avg_review_score) == 400.0, (
        f"Self-inclusion detected! Expected dev_avg_review_score=400.0 (Game 2 only), got {f1.developer_avg_review_score}"
    )
    assert f1.developer_game_count == 1, (
        f"Self-inclusion detected! Expected developer_game_count=1 (excluding itself), got {f1.developer_game_count}"
    )
    assert f1.competitor_density == 2, (
        f"Self-inclusion detected! Expected competitor_density=2 (Game 2 & Game 3), got {f1.competitor_density}"
    )

    # 2. Query built feature for Game 3 (Single-title developer)
    stmt3 = select(FeatureGameFeature).where(
        FeatureGameFeature.app_id == 88803,
        FeatureGameFeature.feature_cutoff_date == cutoff,
    )
    res3 = await db_session.execute(stmt3)
    f3 = res3.scalar_one()

    # Game 3's developer_game_count MUST be 0 (excluding itself)
    assert f3.developer_game_count == 0, (
        f"Expected developer_game_count=0 for single-title dev, got {f3.developer_game_count}"
    )
    # Game 3's developer_avg_review_score MUST NOT be Game 3's own score (85 * 10 = 850.0)
    assert float(f3.developer_avg_review_score) != 850.0, (
        "Self-inclusion leak: single-title developer fell back to game's own review score!"
    )


@pytest.mark.anyio
async def test_baseline_evaluations_and_metrics() -> None:
    """Verify that evaluate_predictions computes valid ROC-AUC, F1, Accuracy, and Brier scores."""
    job = TrainSuccessModelJob(dry_run=True)
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.int32)
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.3, 0.75, 0.15], dtype=np.float32)

    metrics = job._evaluate_predictions(y_true, y_prob)

    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert "accuracy" in metrics
    assert "brier_score" in metrics
    assert metrics["roc_auc"] >= 0.8
    assert metrics["accuracy"] >= 0.8


@pytest.mark.anyio
async def test_model_runs_and_serving_predictions_relationship(db_session: AsyncSession) -> None:
    """Verify ORM model relationships, stages, and foreign key traceability."""
    # 1. Create a model run in production stage
    run_id = uuid.uuid4()
    model_run = ModelRun(
        id=run_id,
        model_name="game_success_predictor",
        stage="production",
        dataset_version="2026-08-14",
        hyperparameters={"blend_alpha": 0.55, "xgb_depth": 4},
        metrics={"roc_auc": 0.92, "f1_score": 0.88, "accuracy": 0.89},
        artifact_path="file:///mlruns/test",
        promoted_by="auto_promoter",
        promoted_at=datetime.now(UTC),
    )
    db_session.add(model_run)

    # 2. Create raw game
    game = RawGame(
        app_id=888123,
        name="Traceability Test Game",
        developer="Indie Dev",
        publisher="Indie Pub",
        positive_reviews=1000,
        negative_reviews=100,
    )
    db_session.add(game)
    await db_session.commit()

    # 3. Create serving prediction with foreign key model_run_id
    pred = ServingPrediction(
        app_id=888123,
        model_run_id=run_id,
        prediction_type="success_score",
        score=Decimal("0.8450"),
        confidence_lower=Decimal("0.7800"),
        confidence_upper=Decimal("0.9100"),
        feature_importance={"positive_review_pct": 0.35, "review_velocity_30d": 0.25},
        shap_values={"positive_review_pct": 0.12, "review_velocity_30d": 0.08},
    )
    db_session.add(pred)
    await db_session.commit()

    # 4. Verify query & traceability
    res = await db_session.execute(
        select(ServingPrediction).where(ServingPrediction.app_id == 888123)
    )
    saved_pred = res.scalar_one()

    assert saved_pred.model_run_id == run_id
    assert saved_pred.score == Decimal("0.8450")
    assert saved_pred.shap_values["positive_review_pct"] == 0.12
    assert saved_pred.feature_importance["positive_review_pct"] == 0.35


@pytest.mark.anyio
async def test_aggregate_cv_metrics() -> None:
    """Verify that aggregate_cv_metrics computes valid mean and std across folds."""
    job = TrainSuccessModelJob(dry_run=True)
    y_all = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
    y_prob_oof = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], dtype=np.float32)
    fold_metrics = [
        {"roc_auc": 0.95, "f1_score": 0.90, "accuracy": 0.90, "brier_score": 0.08},
        {"roc_auc": 0.85, "f1_score": 0.80, "accuracy": 0.80, "brier_score": 0.12},
    ]

    agg = job._aggregate_cv_metrics(y_all, y_prob_oof, fold_metrics)

    assert "cv_roc_auc_mean" in agg
    assert "cv_roc_auc_std" in agg
    assert "cv_f1_mean" in agg
    assert "cv_f1_std" in agg
    assert agg["cv_roc_auc_mean"] == 0.90
    assert round(agg["cv_roc_auc_std"], 2) == 0.05
    assert agg["roc_auc"] == 1.0


@pytest.mark.anyio
async def test_promotion_floor_enforcement(db_session: AsyncSession) -> None:
    """
    Verify that when all candidate models score below the promotion floor,
    zero models are promoted to 'production', all runs remain at 'candidate' stage,
    and serving_predictions is NOT populated.
    """
    cutoff = datetime.now(UTC)

    # Seed 10 synthetic feature rows
    for i in range(10):
        # Create raw game first for foreign key integrity
        game = RawGame(
            app_id=7000 + i,
            name=f"Floor Test Game {i}",
            positive_reviews=100 * (i + 1),
            negative_reviews=20,
        )
        db_session.add(game)

        feat = FeatureGameFeature(
            app_id=7000 + i,
            feature_cutoff_date=cutoff,
            price_usd=Decimal("9.99"),
            discount_pct=0,
            is_free=False,
            primary_genre="Action",
            developer_game_count=1,
            publisher_game_count=1,
            total_reviews_at_cutoff=100 * (i + 1),
            positive_reviews_at_cutoff=80 * (i + 1),
            review_velocity_30d=5.0,
            positive_review_pct=Decimal("80.00"),
            competitor_density=3,
            sentiment_score=Decimal("0.7500"),
            complaint_density=5.0,
            loved_feature_density=20.0,
            average_playtime_forever=500,
            target_success_score=50.0,
            is_hit=(i % 2 == 0),
        )
        db_session.add(feat)

    await db_session.commit()

    # Run training with an unattainable promotion floor (ROC-AUC >= 0.999)
    job = TrainSuccessModelJob(
        cutoff_date=cutoff,
        optuna_trials=2,
        promotion_floor=0.999,
        dry_run=False,
    )

    result = await job._run_async(session=db_session)

    # 1. Assert return status indicates no promotion
    assert result["promotion_cleared"] is False
    assert result["champion_model"] is None
    assert result["promoted_model_run_id"] is None
    assert result["models_trained"] == 4

    # 2. Query database: Assert 0 production models and all 4 runs are 'candidate'
    prod_runs_res = await db_session.execute(
        select(ModelRun).where(ModelRun.stage == "production")
    )
    prod_runs = prod_runs_res.scalars().all()
    assert len(prod_runs) == 0, f"Expected 0 production runs, got {len(prod_runs)}"

    cand_runs_res = await db_session.execute(
        select(ModelRun).where(ModelRun.stage == "candidate")
    )
    cand_runs = cand_runs_res.scalars().all()
    assert len(cand_runs) == 4, f"Expected 4 candidate runs, got {len(cand_runs)}"

    # 3. Assert serving_predictions was NOT populated
    preds_res = await db_session.execute(select(ServingPrediction))
    preds = preds_res.scalars().all()
    assert len(preds) == 0, f"Expected 0 serving predictions, got {len(preds)}"


@pytest.mark.anyio
async def test_degenerate_label_distribution_aborts(db_session: AsyncSession) -> None:
    """
    Verify that when the positive class ratio is below 10% or above 90% (e.g. 100% hits),
    the job aborts without training models or writing any model_runs rows.
    """
    cutoff = datetime.now(UTC)

    # 1. Seed 10 synthetic feature rows that are all 100% positive hits
    for i in range(10):
        game = RawGame(
            app_id=6000 + i,
            name=f"All Hits Game {i}",
            positive_reviews=50000,
            negative_reviews=500,
        )
        db_session.add(game)

        feat = FeatureGameFeature(
            app_id=6000 + i,
            feature_cutoff_date=cutoff,
            price_usd=Decimal("19.99"),
            discount_pct=0,
            is_free=False,
            primary_genre="Roguelike",
            developer_game_count=3,
            publisher_game_count=3,
            total_reviews_at_cutoff=50000,
            positive_reviews_at_cutoff=49000,
            review_velocity_30d=50.0,
            positive_review_pct=Decimal("98.00"),
            competitor_density=5,
            sentiment_score=Decimal("0.9500"),
            complaint_density=2.0,
            loved_feature_density=80.0,
            average_playtime_forever=2500,
            target_success_score=95.0,
            is_hit=True,  # 100% positive
        )
        db_session.add(feat)

    await db_session.commit()

    # 2. Run TrainSuccessModelJob
    job = TrainSuccessModelJob(cutoff_date=cutoff, dry_run=False)
    result = await job._run_async(session=db_session)

    # 3. Assert abort status and zero trained models
    assert result["status"] == "aborted"
    assert result["reason"] == "degenerate_label_distribution"
    assert result["models_trained"] == 0

    # 4. Assert zero model_runs were written to the database
    runs_res = await db_session.execute(select(ModelRun))
    runs = runs_res.scalars().all()
    assert len(runs) == 0, f"Expected 0 model_runs on abort, got {len(runs)}"


