"""
Predictive ML Training & Champion Promotion Pipeline — Phase 4.

Trains tabular success prediction models with baseline comparison:
  Logistic Regression -> Random Forest -> XGBoost -> Optuna-Tuned XGBoost+LightGBM Ensemble

Features:
  - Full MLflow experiment tracking from run 1
  - Multi-model evaluation recorded into model_runs table
  - Automated evaluate_champion() promotion to 'production' stage
  - SHAP explainability (global importance + per-game local attributions)
  - Materializes serving_predictions with model_run_id foreign keys

Usage:
  python -m jobs.train_success_model
  python -m jobs.train_success_model --trials 20
  python -m jobs.train_success_model --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import FeatureGameFeature, ModelRun, ServingPrediction
from jobs.base_job import BaseJob

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAME = "game_success_predictor"
EXPERIMENT_NAME = "steamiq_success_predictor"
PROMOTION_FLOOR_ROC_AUC: float = 0.55


class TrainSuccessModelJob(BaseJob):
    """
    Trains tabular models comparing baselines against Optuna-tuned ensemble,
    records runs to MLflow and model_runs table, evaluates champion against a promotion floor,
    and materializes predictions with SHAP explainability.
    """

    job_name = "train_success_model"

    def __init__(
        self,
        *,
        cutoff_date: datetime | None = None,
        optuna_trials: int = 15,
        promotion_floor: float = PROMOTION_FLOOR_ROC_AUC,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.cutoff_date = cutoff_date
        self.optuna_trials = optuna_trials
        self.promotion_floor = promotion_floor
        self._mlflow_configured = False

    def _setup_mlflow(self) -> None:
        """Initialize MLflow tracking directory and experiment."""
        if not self._mlflow_configured:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            os.environ["GIT_PYTHON_REFRESH"] = "quiet"
            mlflow_dir = self.resolve_path("..", "mlruns").resolve()
            mlflow_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(mlflow_dir.as_uri())
            mlflow.set_experiment(EXPERIMENT_NAME)
            self._mlflow_configured = True
            self.logger.info("MLflow tracking initialized at %s", mlflow_dir)

    def _extract_feature_matrix(
        self, features: list[FeatureGameFeature]
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        """
        Extract numerical feature matrix X, binary target vector y, feature names, and app_ids.
        """
        feature_names = [
            "price_usd",
            "discount_pct",
            "is_free",
            "developer_game_count",
            "publisher_game_count",
            "developer_avg_review_score",
            "publisher_avg_review_score",
            "competitor_density",
            "price_vs_genre_median",
            "complaint_density",
            "loved_feature_density",
        ]

        X_rows = []
        y_rows = []
        app_ids = []

        for f in features:
            app_ids.append(f.app_id)
            # Binary target: is_hit or composite success > 60
            y_val = 1 if (f.is_hit or (f.target_success_score and f.target_success_score >= 60.0)) else 0
            y_rows.append(y_val)

            row = [
                float(f.price_usd) if f.price_usd is not None else 14.99,
                float(f.discount_pct),
                1.0 if f.is_free else 0.0,
                float(f.developer_game_count),
                float(f.publisher_game_count),
                float(f.developer_avg_review_score) if f.developer_avg_review_score is not None else 75.0,
                float(f.publisher_avg_review_score) if f.publisher_avg_review_score is not None else 75.0,
                float(f.competitor_density),
                float(f.price_vs_genre_median) if f.price_vs_genre_median is not None else 0.0,
                float(f.complaint_density),
                float(f.loved_feature_density),
            ]
            X_rows.append(row)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)
        return X, y, feature_names, app_ids

    def _evaluate_fold_predictions(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
        """Compute single-fold evaluation metrics."""
        y_pred = (y_prob >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = 0.5

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        brier = float(brier_score_loss(y_true, y_prob))

        return {
            "roc_auc": round(auc, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
        }

    def _aggregate_cv_metrics(
        self,
        y_true_all: np.ndarray,
        y_prob_oof: np.ndarray,
        fold_metrics_list: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate out-of-fold and per-fold cross-validation metrics (mean ± std)."""
        oof_metrics = self._evaluate_fold_predictions(y_true_all, y_prob_oof)

        aucs = [m["roc_auc"] for m in fold_metrics_list]
        f1s = [m["f1_score"] for m in fold_metrics_list]
        accs = [m["accuracy"] for m in fold_metrics_list]
        briers = [m["brier_score"] for m in fold_metrics_list]

        return {
            "roc_auc": oof_metrics["roc_auc"],
            "f1_score": oof_metrics["f1_score"],
            "accuracy": oof_metrics["accuracy"],
            "brier_score": oof_metrics["brier_score"],
            "cv_roc_auc_mean": round(float(np.mean(aucs)), 4),
            "cv_roc_auc_std": round(float(np.std(aucs)), 4),
            "cv_f1_mean": round(float(np.mean(f1s)), 4),
            "cv_f1_std": round(float(np.std(f1s)), 4),
            "cv_accuracy_mean": round(float(np.mean(accs)), 4),
            "cv_accuracy_std": round(float(np.std(accs)), 4),
            "cv_brier_mean": round(float(np.mean(briers)), 4),
            "cv_brier_std": round(float(np.std(briers)), 4),
        }

    def _evaluate_predictions(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
        """Backward-compatible helper for single prediction evaluations."""
        return self._evaluate_fold_predictions(y_true, y_prob)

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        self._setup_mlflow()
        if session is not None:
            return await self._run_pipeline_with_session(session)

        engine = create_async_engine(settings.async_database_url, echo=False)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as sess:
            result = await self._run_pipeline_with_session(sess)
        await engine.dispose()
        return result

    async def _run_pipeline_with_session(self, session: AsyncSession) -> dict[str, Any]:
        # 1. Fetch feature dataset
        if self.cutoff_date is not None:
            feat_stmt = select(FeatureGameFeature).where(
                FeatureGameFeature.feature_cutoff_date == self.cutoff_date
            )
            res = await session.execute(feat_stmt)
            features = res.scalars().all()
        else:
            # Select latest cutoff date partition
            latest_cutoff_res = await session.execute(
                select(FeatureGameFeature.feature_cutoff_date)
                .order_by(FeatureGameFeature.feature_cutoff_date.desc())
                .limit(1)
            )
            latest_cutoff = latest_cutoff_res.scalar_one_or_none()
            if latest_cutoff is not None:
                feat_stmt = select(FeatureGameFeature).where(
                    FeatureGameFeature.feature_cutoff_date == latest_cutoff
                )
                res = await session.execute(feat_stmt)
                features = res.scalars().all()
            else:
                features = []

        if not features:
            self.logger.warning("No game features found to train on. Run build_features first.")
            return {"status": "skipped", "reason": "no_features"}

        dataset_version = (
            features[0].feature_cutoff_date.strftime("%Y-%m-%d")
            if features
            else datetime.now(UTC).strftime("%Y-%m-%d")
        )

        # 2. Extract feature matrix X, y
        X, y, feature_names, app_ids = self._extract_feature_matrix(features)
        pos_ratio = float(np.mean(y)) if len(y) > 0 else 0.0
        self.logger.info(
            "Training dataset loaded: %d samples, %d features, positive class ratio: %.2f%%",
            X.shape[0],
            X.shape[1],
            pos_ratio * 100.0,
        )

        # Hard failure on degenerate label distribution (<10% or >90% positive class)
        if pos_ratio < 0.10 or pos_ratio > 0.90:
            self.logger.error(
                "❌ Degenerate label distribution: positive class ratio is %.2f%% (must be between 10%% and 90%%). "
                "Aborting model training without writing model_runs.",
                pos_ratio * 100.0,
            )
            return {
                "status": "aborted",
                "reason": "degenerate_label_distribution",
                "pos_ratio": pos_ratio,
                "models_trained": 0,
                "champion_model": None,
                "promoted_model_run_id": None,
            }

        # 5-Fold Cross Validation Setup (scaled by class representation)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        n_splits = min(5, max(2, min(n_pos, n_neg)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        trained_models_info: list[dict[str, Any]] = []

        # ─────────────────────────────────────────────────────────────────
        # 1. Baseline 1 — Logistic Regression
        # ─────────────────────────────────────────────────────────────────
        with mlflow.start_run(run_name="baseline_logistic_regression") as lr_run:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr_probs = np.zeros(len(y), dtype=np.float32)
            fold_metrics: list[dict[str, float]] = []
            for train_idx, val_idx in cv.split(X_scaled, y):
                lr_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr_clf.fit(X_scaled[train_idx], y[train_idx])
                val_probs = lr_clf.predict_proba(X_scaled[val_idx])[:, 1]
                lr_probs[val_idx] = val_probs
                fold_metrics.append(self._evaluate_fold_predictions(y[val_idx], val_probs))

            lr_metrics = self._aggregate_cv_metrics(y, lr_probs, fold_metrics)
            mlflow.log_params({"model_type": "LogisticRegression", "C": 1.0, "scaler": "StandardScaler"})
            mlflow.log_metrics(lr_metrics)

            # Fit final model on full dataset
            final_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            final_lr.fit(X_scaled, y)

            trained_models_info.append({
                "name": "logistic_regression",
                "model_type": "LogisticRegression",
                "metrics": lr_metrics,
                "hyperparameters": {"C": 1.0, "penalty": "l2"},
                "mlflow_run_id": lr_run.info.run_id,
                "artifact_uri": lr_run.info.artifact_uri,
                "model_obj": final_lr,
                "scaler": scaler,
                "probs": lr_probs,
            })
            self.logger.info(
                "Baseline 1 (Logistic Regression) Metrics: OOF ROC-AUC=%.4f (CV: %.4f ± %.4f), F1=%.4f, Acc=%.4f",
                lr_metrics["roc_auc"],
                lr_metrics["cv_roc_auc_mean"],
                lr_metrics["cv_roc_auc_std"],
                lr_metrics["f1_score"],
                lr_metrics["accuracy"],
            )

        # ─────────────────────────────────────────────────────────────────
        # 2. Baseline 2 — Random Forest
        # ─────────────────────────────────────────────────────────────────
        with mlflow.start_run(run_name="baseline_random_forest") as rf_run:
            rf_probs = np.zeros(len(y), dtype=np.float32)
            fold_metrics = []
            for train_idx, val_idx in cv.split(X, y):
                rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
                rf_clf.fit(X[train_idx], y[train_idx])
                val_probs = rf_clf.predict_proba(X[val_idx])[:, 1]
                rf_probs[val_idx] = val_probs
                fold_metrics.append(self._evaluate_fold_predictions(y[val_idx], val_probs))

            rf_metrics = self._aggregate_cv_metrics(y, rf_probs, fold_metrics)
            mlflow.log_params({"model_type": "RandomForest", "n_estimators": 100, "max_depth": 6})
            mlflow.log_metrics(rf_metrics)

            final_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
            final_rf.fit(X, y)

            trained_models_info.append({
                "name": "random_forest",
                "model_type": "RandomForest",
                "metrics": rf_metrics,
                "hyperparameters": {"n_estimators": 100, "max_depth": 6},
                "mlflow_run_id": rf_run.info.run_id,
                "artifact_uri": rf_run.info.artifact_uri,
                "model_obj": final_rf,
                "scaler": None,
                "probs": rf_probs,
            })
            self.logger.info(
                "Baseline 2 (Random Forest) Metrics: OOF ROC-AUC=%.4f (CV: %.4f ± %.4f), F1=%.4f, Acc=%.4f",
                rf_metrics["roc_auc"],
                rf_metrics["cv_roc_auc_mean"],
                rf_metrics["cv_roc_auc_std"],
                rf_metrics["f1_score"],
                rf_metrics["accuracy"],
            )

        # ─────────────────────────────────────────────────────────────────
        # 3. Baseline 3 — XGBoost
        # ─────────────────────────────────────────────────────────────────
        with mlflow.start_run(run_name="baseline_xgboost") as xgb_run:
            xgb_probs = np.zeros(len(y), dtype=np.float32)
            fold_metrics = []
            for train_idx, val_idx in cv.split(X, y):
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.08,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=42,
                )
                xgb_clf.fit(X[train_idx], y[train_idx])
                val_probs = xgb_clf.predict_proba(X[val_idx])[:, 1]
                xgb_probs[val_idx] = val_probs
                fold_metrics.append(self._evaluate_fold_predictions(y[val_idx], val_probs))

            xgb_metrics = self._aggregate_cv_metrics(y, xgb_probs, fold_metrics)
            mlflow.log_params({
                "model_type": "XGBoost",
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.08,
            })
            mlflow.log_metrics(xgb_metrics)

            final_xgb = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
                random_state=42,
            )
            final_xgb.fit(X, y)

            trained_models_info.append({
                "name": "xgboost_baseline",
                "model_type": "XGBoost",
                "metrics": xgb_metrics,
                "hyperparameters": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08},
                "mlflow_run_id": xgb_run.info.run_id,
                "artifact_uri": xgb_run.info.artifact_uri,
                "model_obj": final_xgb,
                "scaler": None,
                "probs": xgb_probs,
            })
            self.logger.info(
                "Baseline 3 (XGBoost) Metrics: OOF ROC-AUC=%.4f (CV: %.4f ± %.4f), F1=%.4f, Acc=%.4f",
                xgb_metrics["roc_auc"],
                xgb_metrics["cv_roc_auc_mean"],
                xgb_metrics["cv_roc_auc_std"],
                xgb_metrics["f1_score"],
                xgb_metrics["accuracy"],
            )

        # ─────────────────────────────────────────────────────────────────
        # 4. Champion Candidate — XGBoost + LightGBM Ensemble (Optuna Tuned)
        # ─────────────────────────────────────────────────────────────────
        self.logger.info("Tuning XGBoost + LightGBM Ensemble with Optuna (%d trials)...", self.optuna_trials)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.2, log=True)
            xgb_depth = trial.suggest_int("xgb_depth", 3, 7)
            xgb_n_est = trial.suggest_int("xgb_n_est", 50, 150, step=25)
            lgb_lr = trial.suggest_float("lgb_lr", 0.01, 0.2, log=True)
            lgb_leaves = trial.suggest_int("lgb_leaves", 15, 63)
            lgb_n_est = trial.suggest_int("lgb_n_est", 50, 150, step=25)
            alpha = trial.suggest_float("blend_alpha", 0.3, 0.7)

            blend_probs = np.zeros(len(y), dtype=np.float32)
            for train_idx, val_idx in cv.split(X, y):
                model_x = xgb.XGBClassifier(
                    n_estimators=xgb_n_est,
                    max_depth=xgb_depth,
                    learning_rate=xgb_lr,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=42,
                )
                model_x.fit(X[train_idx], y[train_idx])
                p_x = model_x.predict_proba(X[val_idx])[:, 1]

                model_l = lgb.LGBMClassifier(
                    n_estimators=lgb_n_est,
                    num_leaves=lgb_leaves,
                    learning_rate=lgb_lr,
                    n_jobs=1,
                    verbose=-1,
                    random_state=42,
                )
                model_l.fit(X[train_idx], y[train_idx])
                p_l = model_l.predict_proba(X[val_idx])[:, 1]

                blend_probs[val_idx] = alpha * p_x + (1.0 - alpha) * p_l

            try:
                return float(roc_auc_score(y, blend_probs))
            except ValueError:
                return float(accuracy_score(y, (blend_probs >= 0.5).astype(int)))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
        best_params = study.best_params
        self.logger.info("Optuna best trial params: %s (ROC-AUC: %.4f)", best_params, study.best_value)

        with mlflow.start_run(run_name="ensemble_xgboost_lightgbm_optuna") as ens_run:
            alpha = best_params["blend_alpha"]
            ens_probs = np.zeros(len(y), dtype=np.float32)
            fold_metrics = []

            for train_idx, val_idx in cv.split(X, y):
                model_x = xgb.XGBClassifier(
                    n_estimators=best_params["xgb_n_est"],
                    max_depth=best_params["xgb_depth"],
                    learning_rate=best_params["xgb_lr"],
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=42,
                )
                model_x.fit(X[train_idx], y[train_idx])
                p_x = model_x.predict_proba(X[val_idx])[:, 1]

                model_l = lgb.LGBMClassifier(
                    n_estimators=best_params["lgb_n_est"],
                    num_leaves=best_params["lgb_leaves"],
                    learning_rate=best_params["lgb_lr"],
                    n_jobs=1,
                    verbose=-1,
                    random_state=42,
                )
                model_l.fit(X[train_idx], y[train_idx])
                p_l = model_l.predict_proba(X[val_idx])[:, 1]

                val_blend = alpha * p_x + (1.0 - alpha) * p_l
                ens_probs[val_idx] = val_blend
                fold_metrics.append(self._evaluate_fold_predictions(y[val_idx], val_blend))

            ens_metrics = self._aggregate_cv_metrics(y, ens_probs, fold_metrics)
            mlflow.log_params({"model_type": "XGBoost_LightGBM_Ensemble", **best_params})
            mlflow.log_metrics(ens_metrics)

            # Fit final ensemble components on full dataset
            final_xgb_ens = xgb.XGBClassifier(
                n_estimators=best_params["xgb_n_est"],
                max_depth=best_params["xgb_depth"],
                learning_rate=best_params["xgb_lr"],
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
                random_state=42,
            )
            final_xgb_ens.fit(X, y)

            final_lgb_ens = lgb.LGBMClassifier(
                n_estimators=best_params["lgb_n_est"],
                num_leaves=best_params["lgb_leaves"],
                learning_rate=best_params["lgb_lr"],
                n_jobs=1,
                verbose=-1,
                random_state=42,
            )
            final_lgb_ens.fit(X, y)

            trained_models_info.append({
                "name": "ensemble_xgboost_lightgbm",
                "model_type": "XGBoost_LightGBM_Ensemble",
                "metrics": ens_metrics,
                "hyperparameters": best_params,
                "mlflow_run_id": ens_run.info.run_id,
                "artifact_uri": ens_run.info.artifact_uri,
                "model_obj": (final_xgb_ens, final_lgb_ens, alpha),
                "scaler": None,
                "probs": ens_probs,
            })
            self.logger.info(
                "Ensemble (XGBoost + LightGBM) Metrics: OOF ROC-AUC=%.4f (CV: %.4f ± %.4f), F1=%.4f, Acc=%.4f",
                ens_metrics["roc_auc"],
                ens_metrics["cv_roc_auc_mean"],
                ens_metrics["cv_roc_auc_std"],
                ens_metrics["f1_score"],
                ens_metrics["accuracy"],
            )

        # ─────────────────────────────────────────────────────────────────
        # 5. Model Runs Registration & evaluate_champion() Promotion Logic
        # ─────────────────────────────────────────────────────────────────
        # Determine candidate with highest ROC-AUC, fallback to F1
        champion_candidate = max(
            trained_models_info,
            key=lambda m: (m["metrics"]["roc_auc"], m["metrics"]["f1_score"]),
        )

        candidate_auc = champion_candidate["metrics"]["roc_auc"]
        candidate_auc_mean = champion_candidate["metrics"]["cv_roc_auc_mean"]
        candidate_auc_std = champion_candidate["metrics"]["cv_roc_auc_std"]
        clears_floor = bool(candidate_auc >= self.promotion_floor)

        if clears_floor:
            self.logger.info(
                "🏆 Champion Model Qualified for Promotion: '%s' (OOF ROC-AUC: %.4f, CV Mean: %.4f ± %.4f >= Floor %.2f, F1: %.4f)",
                champion_candidate["name"],
                candidate_auc,
                candidate_auc_mean,
                candidate_auc_std,
                self.promotion_floor,
                champion_candidate["metrics"]["f1_score"],
            )
        else:
            self.logger.warning(
                "⚠️ No model candidate cleared the promotion floor (Best ROC-AUC: %.4f [CV: %.4f ± %.4f] < Floor: %.2f). "
                "All %d candidate models will remain at 'candidate' stage. Zero production models promoted; "
                "serving_predictions will NOT be populated.",
                candidate_auc,
                candidate_auc_mean,
                candidate_auc_std,
                self.promotion_floor,
                len(trained_models_info),
            )

        # Compute SHAP Values if champion cleared floor, or fallback for artifact logging
        self.logger.info("Computing SHAP explanations...")
        model_obj = champion_candidate["model_obj"]
        scaler = champion_candidate.get("scaler")
        X_input = scaler.transform(X) if scaler is not None else X

        try:
            if champion_candidate.get("model_type") == "LogisticRegression":
                explainer = shap.LinearExplainer(model_obj, X_input)
                shap_values_raw = explainer.shap_values(X_input)
            elif champion_candidate["name"] == "ensemble_xgboost_lightgbm":
                primary_tree = model_obj[0]
                explainer = shap.TreeExplainer(primary_tree)
                shap_values_raw = explainer.shap_values(X)
            elif isinstance(model_obj, RandomForestClassifier | xgb.XGBClassifier | lgb.LGBMClassifier):
                explainer = shap.TreeExplainer(model_obj)
                shap_values_raw = explainer.shap_values(X)
            else:
                primary_tree = trained_models_info[2]["model_obj"]
                explainer = shap.TreeExplainer(primary_tree)
                shap_values_raw = explainer.shap_values(X)
        except Exception as e:
            self.logger.warning("Primary SHAP explainer failed (%s), using XGBoost TreeExplainer", e)
            primary_tree = trained_models_info[2]["model_obj"]
            explainer = shap.TreeExplainer(primary_tree)
            shap_values_raw = explainer.shap_values(X)

        # Handle binary classifier output shapes
        if isinstance(shap_values_raw, list) and len(shap_values_raw) > 1:
            shap_matrix = shap_values_raw[1]
        elif isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
            shap_matrix = shap_values_raw[:, :, 1]
        else:
            shap_matrix = shap_values_raw

        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        global_feature_importance = {
            feature_names[i]: round(float(mean_abs_shap[i]), 4)
            for i in np.argsort(-mean_abs_shap)
        }

        # Save SHAP artifact in MLflow
        shap_artifact_path = self.resolve_path("..", "mlruns", "champion_shap_summary.json")
        with open(shap_artifact_path, "w", encoding="utf-8") as f:
            json.dump(global_feature_importance, f, indent=2)
        try:
            with mlflow.start_run(run_id=champion_candidate["mlflow_run_id"]):
                mlflow.log_artifact(str(shap_artifact_path))
        except Exception as exc:
            self.logger.warning("Could not log SHAP artifact to MLflow run (%s)", exc)

        # Database persistence: Write model_runs rows and promote champion if floor cleared
        promoted_run_id = None
        if not self.dry_run:
            if clears_floor:
                # Archive existing production models
                await session.execute(
                    update(ModelRun)
                    .where(ModelRun.model_name == MODEL_NAME, ModelRun.stage == "production")
                    .values(stage="archived")
                )

            # Insert all model runs
            for model_info in trained_models_info:
                is_champion = clears_floor and (model_info["name"] == champion_candidate["name"])
                stage = "production" if is_champion else "candidate"
                run_uuid = uuid.uuid4()

                run_record = ModelRun(
                    id=run_uuid,
                    model_name=MODEL_NAME,
                    stage=stage,
                    dataset_version=dataset_version,
                    hyperparameters=model_info["hyperparameters"],
                    metrics=model_info["metrics"],
                    artifact_path=model_info["artifact_uri"],
                    promoted_by="auto_promoter" if is_champion else None,
                    promoted_at=datetime.now(UTC) if is_champion else None,
                )
                session.add(run_record)

                if is_champion:
                    promoted_run_id = run_uuid

            await session.commit()
            self.logger.info(
                "Persisted %d model_runs. Promoted run UUID: %s (stage: %s)",
                len(trained_models_info),
                promoted_run_id,
                "production" if clears_floor else "none (all candidate)",
            )

            # ─────────────────────────────────────────────────────────────
            # 6. Materialize serving_predictions ONLY IF a champion was promoted
            # ─────────────────────────────────────────────────────────────
            if clears_floor and promoted_run_id is not None:
                # Clear previous predictions for this prediction_type
                await session.execute(
                    delete(ServingPrediction).where(ServingPrediction.prediction_type == "success_score")
                )

                seen_app_ids: set[int] = set()
                predictions_count = 0
                for idx, app_id in enumerate(app_ids):
                    if app_id in seen_app_ids:
                        continue
                    seen_app_ids.add(app_id)

                    prob = float(champion_candidate["probs"][idx])
                    se = np.sqrt(max(1e-5, prob * (1.0 - prob) / 30.0))
                    ci_lower = max(0.0, prob - 1.96 * se)
                    ci_upper = min(1.0, prob + 1.96 * se)

                    local_shap = {
                        feature_names[j]: round(float(shap_matrix[idx, j]), 4)
                        for j in range(len(feature_names))
                    }
                    sorted_local_shap = dict(
                        sorted(local_shap.items(), key=lambda x: -abs(x[1]))[:6]
                    )

                    serving_row = ServingPrediction(
                        app_id=app_id,
                        model_run_id=promoted_run_id,
                        prediction_type="success_score",
                        score=Decimal(str(round(prob, 4))),
                        confidence_lower=Decimal(str(round(ci_lower, 4))),
                        confidence_upper=Decimal(str(round(ci_upper, 4))),
                        feature_importance=global_feature_importance,
                        shap_values=sorted_local_shap,
                    )
                    session.add(serving_row)
                    predictions_count += 1

                await session.commit()
                self.logger.info(
                    "Populated serving_predictions: %d game predictions stored (Model Run: %s)",
                    predictions_count,
                    promoted_run_id,
                )
            else:
                self.logger.info("Serving predictions unpopulated because no model cleared promotion floor.")

        return {
            "models_trained": len(trained_models_info),
            "champion_model": champion_candidate["name"] if clears_floor else None,
            "champion_metrics": champion_candidate["metrics"] if clears_floor else None,
            "promoted_model_run_id": str(promoted_run_id) if promoted_run_id else None,
            "promotion_cleared": clears_floor,
            "promotion_floor": self.promotion_floor,
            "dataset_version": dataset_version,
        }

    def run(self) -> dict[str, Any]:
        return asyncio.run(self._run_async())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train success predictor models, compare baselines, and promote champion (Phase 4)"
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Feature cutoff date in ISO format (e.g. 2026-08-01T00:00:00Z)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of Optuna hyperparameter optimization trials (default: 15)",
    )
    parser.add_argument(
        "--promotion-floor",
        type=float,
        default=PROMOTION_FLOOR_ROC_AUC,
        help=f"Minimum ROC-AUC threshold required to promote to production (default: {PROMOTION_FLOOR_ROC_AUC})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Train and log without DB persistence")
    args = parser.parse_args()

    cutoff = None
    if args.cutoff_date:
        cutoff = datetime.fromisoformat(args.cutoff_date.replace("Z", "+00:00"))

    job = TrainSuccessModelJob(
        cutoff_date=cutoff,
        optuna_trials=args.trials,
        promotion_floor=args.promotion_floor,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()

