"""
JSON Schema and Pydantic validation for Recommender System Report Payloads.

Ensures:
  1. Strict structural typing for all report types (leaderboard, ablations, ranker_ablations,
     slices, cold_start, beyond_accuracy, shap_global, training_curves, dataset, limitations).
  2. Complete prohibition of NaN, Infinity, -Infinity from serialized report payloads.
  3. Consistent metadata schema (dataset_version, git_commit, n_eval_users, seed, generated_at).

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReportMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_key: str
    dataset_version: str
    git_commit: str = "unknown"
    n_eval_users: int
    seed: int
    generated_at: str


class LeaderboardModelEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str
    family: str
    stage: str = "candidate"
    model_run_id: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training_time_sec: float | None = None
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None = None
    val_slices: dict[str, dict[str, float]] | None = None
    test_slices: dict[str, dict[str, float]] | None = None


class LeaderboardPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    models: list[LeaderboardModelEntry]


class TwoTowerAblationVariant(BaseModel):
    model_config = ConfigDict(extra="ignore")

    variant: str
    description: str
    seeds: list[int]
    test_ndcg_at_10_mean: float
    test_ndcg_at_10_std: float
    test_hr_at_10_mean: float
    test_hr_at_10_std: float
    tail_ndcg_at_10_mean: float
    tail_ndcg_at_10_std: float
    paired_ndcg_diff_vs_full: float
    paired_ndcg_diff_lo95: float
    paired_ndcg_diff_hi95: float
    significant_vs_full: bool


class AblationsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    variants: list[TwoTowerAblationVariant]


class RankerAblationEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ablated_group: str
    description: str
    val_ndcg_at_10: float
    val_hr_at_10: float
    val_ndcg_diff_vs_full: float
    val_ndcg_diff_lo95: float
    val_ndcg_diff_hi95: float
    test_ndcg_at_10: float
    test_hr_at_10: float
    test_ndcg_diff_vs_full: float
    test_ndcg_diff_lo95: float
    test_ndcg_diff_hi95: float
    is_significant_test: bool


class RankerAblationsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    full_ranker_features_count: int
    ablations: list[RankerAblationEntry]


class SliceComparisonEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    comparison: str
    split: str
    slice_name: str
    n_users: int
    model_a_ndcg: float
    model_b_ndcg: float
    paired_diff: float
    lo95: float
    hi95: float
    verdict: str  # 'better', 'worse', 'no significant difference'


class SlicesPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    slice_definitions: dict[str, str]
    comparisons: list[SliceComparisonEntry]


class ColdStartModelMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hit_rate_at_10: float
    hit_rate_at_10_lo95: float
    hit_rate_at_10_hi95: float
    ndcg_at_10: float
    ndcg_at_10_lo95: float
    ndcg_at_10_hi95: float
    mrr_at_10: float


class ColdStartPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    sparse_targets_study: dict[str, Any]
    simulated_new_items_study: dict[str, Any]
    item_holdout_study: dict[str, Any] | None = None


class BeyondAccuracyModelEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str
    catalog_coverage_at_10: float
    novelty_at_10: float
    diversity_at_10: float
    head_share_at_10: float
    gini_coefficient_at_10: float


class MMRTradeoffEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lambda_param: float
    ndcg_at_10: float
    ndcg_at_10_lo95: float
    ndcg_at_10_hi95: float
    diversity_at_10: float
    paired_ndcg_diff_vs_lambda1: float
    paired_ndcg_diff_lo95: float
    paired_ndcg_diff_hi95: float


class BeyondAccuracyPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    metrics_at_10: list[BeyondAccuracyModelEntry]
    mmr_reranking: list[MMRTradeoffEntry]


class GlobalShapPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    global_shap: dict[str, float]
    feature_importance_gain: dict[str, float]
    feature_importance_split: dict[str, float]


class TrainingCurvesPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    two_tower_curves: dict[str, Any] | list[dict[str, Any]]
    ranker_curves: dict[str, Any] | list[dict[str, Any]] | None = None


class DatasetReportPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    manifest: dict[str, Any]
    split_protocol: str


class LimitationsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ReportMetadata
    limitations: list[dict[str, str]]


REPORT_MODEL_MAP: dict[str, type[BaseModel]] = {
    "leaderboard": LeaderboardPayload,
    "ablations": AblationsPayload,
    "ranker_ablations": RankerAblationsPayload,
    "slices": SlicesPayload,
    "cold_start": ColdStartPayload,
    "beyond_accuracy": BeyondAccuracyPayload,
    "shap_global": GlobalShapPayload,
    "training_curves": TrainingCurvesPayload,
    "dataset": DatasetReportPayload,
    "limitations": LimitationsPayload,
}


def check_no_nan_inf(obj: Any, path: str = "") -> None:
    """
    Recursively validates that no float NaN, Infinity, or -Infinity exist in obj.
    Raises ValueError if found.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Found invalid float value '{obj}' at path '{path}'")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_no_nan_inf(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list | tuple):
        for idx, item in enumerate(obj):
            check_no_nan_inf(item, f"{path}[{idx}]")


def validate_report_payload(report_type: str, payload: dict[str, Any]) -> BaseModel:
    """
    Validates report payload against Pydantic schema and verifies zero NaN/inf values.

    Args:
        report_type: Identifier key (e.g. 'leaderboard', 'ablations', etc.)
        payload: Dictionary payload to validate.

    Returns:
        Validated Pydantic model instance.
    """
    # 1. NaN and Inf check
    check_no_nan_inf(payload)

    # 2. Pydantic schema validation
    if report_type not in REPORT_MODEL_MAP:
        raise KeyError(f"Unknown report_type '{report_type}'. Expected one of {list(REPORT_MODEL_MAP.keys())}")

    model_cls = REPORT_MODEL_MAP[report_type]
    return model_cls.model_validate(payload)
