"""
Model Registry integration for Recommender Systems.

Integrates:
  1. MLflow tracking under experiment "steamiq_recommender" using local file-store.
  2. PostgreSQL `model_runs` table persistence with stage lifecycle management.

Safety Guardrail:
  All recsys model names MUST start with "recsys_".
  Archive and promotion operations are strictly partitioned by the "recsys_" prefix
  to ensure existing Phase 4 predictive models (e.g. game_success_predictor) are never altered.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ModelRun

logger = logging.getLogger("steamiq.recsys.registry")

EXPERIMENT_NAME = "steamiq_recommender"
RECSYS_PREFIX = "recsys_"


def _setup_mlflow() -> str:
    """
    Initializes MLflow tracking directory and sets the recommender experiment.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    # Resolve to root mlruns folder
    mlruns_dir = Path(settings.resolved_recsys_data_dir).parents[1] / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlruns_dir.as_uri()


def _get_git_commit() -> str:
    """Returns the current git HEAD commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).resolve().parent),
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _get_evaluate_sha1() -> str:
    """Returns SHA1 hex digest of evaluate.py to ensure evaluation code provenance."""
    try:
        eval_path = Path(__file__).resolve().parent / "evaluate.py"
        if eval_path.exists():
            return hashlib.sha1(eval_path.read_bytes()).hexdigest()
    except Exception:
        pass
    return "unknown"


def _sanitize_metric_key(key: str) -> str:
    """
    Ensures metric keys are compliant with MLflow naming rules (no '@', valid chars).
    """
    return key.replace("@", "_at_").replace("+", "_plus_").replace("-", "_")


async def log_recsys_run(
    model_name: str,
    family: str,
    dataset_version: str,
    hyperparameters: dict[str, Any] | None = None,
    val_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    extra: dict[str, Any] | None = None,
    session: AsyncSession | None = None,
) -> tuple[ModelRun, str]:
    """
    Logs a recommender training run to MLflow and inserts a candidate ModelRun row into the database.

    Args:
        model_name: Unique model identifier, MUST start with "recsys_".
        family: Model algorithmic family (e.g., 'popularity', 'item_knn', 'lightfm', 'bpr', 'sasrec', 'hybrid').
        dataset_version: Hex hash or version tag of dataset used.
        hyperparameters: Model hyperparameter dictionary.
        val_metrics: Validation metrics dictionary or EvaluationResult.
        test_metrics: Optional test metrics dictionary or EvaluationResult.
        artifact_dir: Path to directory containing serialized model weights/artifacts.
        extra: Additional metadata (e.g. seed, runtime_sec, profile).
        session: Optional SQLAlchemy AsyncSession.

    Returns:
        (ModelRun record, MLflow run_id)
    """
    if not model_name.startswith(RECSYS_PREFIX):
        raise ValueError(
            f"Invalid model_name '{model_name}'. All recsys models MUST start with '{RECSYS_PREFIX}'"
        )

    _setup_mlflow()
    hyperparameters = hyperparameters or {}
    extra = dict(extra) if extra else {}

    # Step 7: Record git commit and evaluate.py provenance SHA1
    git_commit = _get_git_commit()
    evaluate_sha1 = _get_evaluate_sha1()
    extra["git_commit"] = git_commit
    extra["evaluate_sha1"] = evaluate_sha1

    # Extract scalar metrics dicts
    flat_val = val_metrics if isinstance(val_metrics, dict) else (val_metrics.metrics if val_metrics else {})
    flat_test = test_metrics if isinstance(test_metrics, dict) else (test_metrics.metrics if test_metrics else {})

    combined_metrics: dict[str, float] = {}
    mlflow_metrics: dict[str, float] = {}

    for k, v in flat_val.items():
        if isinstance(v, int | float) and not np_isnan(v):
            clean_key = _sanitize_metric_key(k)
            full_key = clean_key if clean_key.startswith("val_") else f"val_{clean_key}"
            combined_metrics[full_key] = float(v)
            mlflow_metrics[full_key] = float(v)

    for k, v in flat_test.items():
        if isinstance(v, int | float) and not np_isnan(v):
            clean_key = _sanitize_metric_key(k)
            full_key = clean_key if clean_key.startswith("test_") else f"test_{clean_key}"
            combined_metrics[full_key] = float(v)
            mlflow_metrics[full_key] = float(v)


    # 1. Log to MLflow
    with mlflow.start_run(run_name=model_name) as run:
        mlflow_run_id = run.info.run_id

        # Log params
        mlflow_params = {
            "family": family,
            "dataset_version": dataset_version,
            **{f"hp_{k}": str(v)[:250] for k, v in hyperparameters.items()},
            **{f"extra_{k}": str(v)[:250] for k, v in extra.items()},
        }
        mlflow.log_params(mlflow_params)

        # Log metrics
        if mlflow_metrics:
            mlflow.log_metrics(mlflow_metrics)

        # Log tags
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("family", family)
        mlflow.set_tag("dataset_version", dataset_version)
        if artifact_dir:
            mlflow.set_tag("artifact_dir", str(artifact_dir))

        artifact_uri = run.info.artifact_uri

    # 2. Persist to PostgreSQL model_runs
    merged_hp = {
        "family": family,
        **hyperparameters,
        **extra,
    }

    run_record = ModelRun(
        id=uuid.uuid4(),
        model_name=model_name,
        stage="candidate",
        dataset_version=dataset_version,
        hyperparameters=merged_hp,
        metrics=combined_metrics,
        artifact_path=str(artifact_dir) if artifact_dir else artifact_uri,
        promoted_by=None,
        promoted_at=None,
        created_at=datetime.now(UTC),
    )


    if session is not None:
        session.add(run_record)
        await session.commit()
    else:
        engine = create_async_engine(settings.async_database_url, echo=False)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as sess:
            sess.add(run_record)
            await sess.commit()
        await engine.dispose()

    logger.info(
        "Logged recsys run to MLflow (%s) and model_runs (%s) for %s (val_ndcg@10=%.4f)",
        mlflow_run_id,
        run_record.id,
        model_name,
        combined_metrics.get("val_ndcg_at_10", 0.0),
    )

    return run_record, mlflow_run_id


async def promote_champion(
    family: str | None = None,
    metric_key: str = "val_ndcg_at_10",
    session: AsyncSession | None = None,
) -> ModelRun | None:
    """
    Evaluates candidate runs and promotes the best performing run to 'production'.
    Archives older 'production' runs of the same model_name.

    Guards:
      - Filters exclusively by model_name starting with 'recsys_'.
      - Never touches non-recsys models (e.g., game_success_predictor).

    Args:
        family: Optional family filter (e.g. 'popularity'). If None, considers all recsys candidates.
        metric_key: Metric used to determine champion (default: 'val_ndcg_at_10').
        session: Optional SQLAlchemy AsyncSession.

    Returns:
        Promoted ModelRun instance, or None if no candidate exists.
    """
    if session is not None:
        return await _promote_champion_with_session(family, metric_key, session)

    engine = create_async_engine(settings.async_database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as sess:
        result = await _promote_champion_with_session(family, metric_key, sess)
    await engine.dispose()
    return result


async def _promote_champion_with_session(
    family: str | None,
    metric_key: str,
    session: AsyncSession,
) -> ModelRun | None:
    # 1. Fetch all candidate runs with recsys_ prefix
    stmt = (
        select(ModelRun)
        .where(
            ModelRun.model_name.like(f"{RECSYS_PREFIX}%"),
            ModelRun.stage.in_(["candidate", "production"]),
        )
        .order_by(ModelRun.created_at.desc())
    )
    res = await session.execute(stmt)
    runs = res.scalars().all()

    if not runs:
        logger.warning("No recsys model runs found for champion promotion.")
        return None

    # Filter by family if requested
    if family is not None:
        matched_runs = []
        for r in runs:
            hp = r.hyperparameters or {}
            if hp.get("family") == family or r.model_name == f"{RECSYS_PREFIX}{family}":
                matched_runs.append(r)
        runs = matched_runs

    if not runs:
        logger.warning("No recsys model runs found for family '%s'.", family)
        return None

    # Find champion run with best validation metric
    def score_run(r: ModelRun) -> float:
        m = r.metrics or {}
        val = m.get(metric_key, 0.0)
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    champion = max(runs, key=score_run)
    champion_score = score_run(champion)

    # Archive previous production runs for this specific model_name
    archive_stmt = (
        update(ModelRun)
        .where(
            ModelRun.model_name == champion.model_name,
            ModelRun.model_name.like(f"{RECSYS_PREFIX}%"),  # safety guard
            ModelRun.stage == "production",
            ModelRun.id != champion.id,
        )
        .values(stage="archived")
    )
    await session.execute(archive_stmt)

    # Promote champion
    champion.stage = "production"
    champion.promoted_by = "auto_promoter"
    champion.promoted_at = datetime.now(UTC)

    await session.commit()
    logger.info(
        "🏆 Promoted champion model %s (UUID: %s) to 'production' with %s=%.4f",
        champion.model_name,
        champion.id,
        metric_key,
        champion_score,
    )
    return champion


async def get_latest_recsys_runs(
    dataset_version: str | None = None,
    session: AsyncSession | None = None,
) -> list[ModelRun]:
    """
    Fetches the latest ModelRun record for each unique (model_name, dataset_version)
    where model_name starts with 'recsys_'.

    Args:
        dataset_version: Optional dataset version filter.
        session: Optional SQLAlchemy AsyncSession.

    Returns:
        List of latest ModelRun instances.
    """
    if session is not None:
        return await _get_latest_recsys_runs_with_session(dataset_version, session)

    engine = create_async_engine(settings.async_database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as sess:
        result = await _get_latest_recsys_runs_with_session(dataset_version, sess)
    await engine.dispose()
    return result


async def _get_latest_recsys_runs_with_session(
    dataset_version: str | None,
    session: AsyncSession,
) -> list[ModelRun]:
    stmt = (
        select(ModelRun)
        .where(ModelRun.model_name.like(f"{RECSYS_PREFIX}%"))
        .order_by(ModelRun.created_at.desc())
    )
    if dataset_version is not None:
        stmt = stmt.where(ModelRun.dataset_version == dataset_version)

    res = await session.execute(stmt)
    all_runs = res.scalars().all()

    seen_keys: set[tuple[str, str]] = set()
    latest_runs: list[ModelRun] = []

    for run in all_runs:
        key = (run.model_name, run.dataset_version)
        if key not in seen_keys:
            seen_keys.add(key)
            latest_runs.append(run)

    return latest_runs


def get_latest_recsys_runs_sync(*args: Any, **kwargs: Any) -> list[ModelRun]:
    """Synchronous convenience wrapper for get_latest_recsys_runs."""
    return asyncio.run(get_latest_recsys_runs(*args, **kwargs))


def np_isnan(val: Any) -> bool:
    try:
        import numpy as np
        return bool(np.isnan(val))
    except Exception:
        return False


def log_recsys_run_sync(*args: Any, **kwargs: Any) -> tuple[ModelRun, str]:
    """Synchronous convenience wrapper for log_recsys_run."""
    return asyncio.run(log_recsys_run(*args, **kwargs))


def promote_champion_sync(*args: Any, **kwargs: Any) -> ModelRun | None:
    """Synchronous convenience wrapper for promote_champion."""
    return asyncio.run(promote_champion(*args, **kwargs))

