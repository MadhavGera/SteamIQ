"""
Unit tests for Recommender System Fix-Up Tasks (Item-kNN Labeling, Ranker Determinism, Promotion Demotion).

Offline tests that execute with zero network access and tiny synthetic data.
Rule 9: 100% offline, tiny synthetic fixtures.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
import pytest
from lightgbm import LGBMRanker
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.models import Base, ModelRun
from recsys.models.itemknn import ItemKNNRecommender
from recsys.registry import promote_champion


def test_itemknn_description_label() -> None:
    """ItemKNN docstrings and descriptions must strictly follow the required specification."""
    doc = ItemKNNRecommender.__doc__ or ""
    assert "BM25" not in doc, "ItemKNN docstring must not mention BM25"
    assert "item-item collaborative filtering with shrinkage cosine similarity, top-k neighbors per item (k=100)" in doc


def test_lightgbm_ranker_determinism() -> None:
    """Two LGBMRanker fits on the same synthetic data and seed must produce identical predictions."""
    rng = np.random.default_rng(42)
    n_queries = 25
    group_size = 8
    n_samples = n_queries * group_size
    n_features = 6

    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for q in range(n_queries):
        # assign 1 positive item per query
        pos_idx = q * group_size + rng.integers(0, group_size)
        y[pos_idx] = 1.0
    groups = [group_size] * n_queries

    seed = 42

    # Fit 1
    ranker1 = LGBMRanker(
        objective="lambdarank",
        eval_at=[5, 10],
        num_leaves=15,
        learning_rate=0.05,
        n_estimators=30,
        min_child_samples=5,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_row_wise=True,
    )
    ranker1.fit(X, y, group=groups)
    preds1 = ranker1.predict(X)

    # Fit 2
    ranker2 = LGBMRanker(
        objective="lambdarank",
        eval_at=[5, 10],
        num_leaves=15,
        learning_rate=0.05,
        n_estimators=30,
        min_child_samples=5,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_row_wise=True,
    )
    ranker2.fit(X, y, group=groups)
    preds2 = ranker2.predict(X)

    # Predictions must be bitwise or strictly identical
    np.testing.assert_allclose(preds1, preds2, atol=1e-7, err_msg="LGBMRanker predictions differ across identical seed runs")


@pytest.mark.asyncio
async def test_promote_champion_demotes_other_production_recsys_runs() -> None:
    """Promoting a winner in recsys must demote any existing production recsys run to candidate."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        # 1. Insert existing production run (recsys_popularity)
        run_pop = ModelRun(
            id=uuid.uuid4(),
            model_name="recsys_popularity",
            stage="production",
            dataset_version="v1_test",
            hyperparameters={"family": "popularity"},
            metrics={"val_ndcg_at_10": 0.020},
            artifact_path="/tmp/pop",
            created_at=datetime.now(UTC),
        )
        session.add(run_pop)

        # 2. Insert candidate run (recsys_two_tower_clean)
        run_tt = ModelRun(
            id=uuid.uuid4(),
            model_name="recsys_two_tower_clean",
            stage="candidate",
            dataset_version="v1_test",
            hyperparameters={"family": "two_tower"},
            metrics={"val_ndcg_at_10": 0.061},
            artifact_path="/tmp/tt",
            created_at=datetime.now(UTC),
        )
        session.add(run_tt)
        await session.commit()

        # 3. Promote two-tower clean
        promoted = await promote_champion(
            family="two_tower",
            metric_key="val_ndcg_at_10",
            session=session,
        )
        assert promoted is not None
        assert promoted.model_name == "recsys_two_tower_clean"
        assert promoted.stage == "production"

        # 4. Query all runs to verify exactly ONE production run remains
        stmt = select(ModelRun).order_by(ModelRun.created_at.desc())
        res = await session.execute(stmt)
        all_runs = res.scalars().all()

        prod_runs = [r for r in all_runs if r.stage == "production"]
        assert len(prod_runs) == 1, f"Expected exactly 1 production run, found {len(prod_runs)}"
        assert prod_runs[0].model_name == "recsys_two_tower_clean"

        # 5. Verify the old popularity run was demoted to 'candidate'
        stmt_pop = select(ModelRun).where(ModelRun.model_name == "recsys_popularity")
        res_pop = await session.execute(stmt_pop)
        pop_db = res_pop.scalar_one()
        assert pop_db.stage == "candidate", f"Expected recsys_popularity to be candidate, got {pop_db.stage}"

    await engine.dispose()
