"""
Multi-Seed Aggregation & Integrity Verification Module (Issue I2 Resolution).

Enforces:
1. Each seed's own model evaluation is stored in a `per_seed` list (seed, all metrics, CI).
2. Shared function `aggregate_per_seed_metrics` computes mean and std across all metrics.
3. Deterministic baselines report `deterministic: True` with std as 'n/a' instead of zero.
4. Champion entry is separate and references a specific seed.
5. Strict artifact-level guard rejecting suspicious copying of champion metrics across multi-seed rows.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("steamiq.recsys.aggregation")


def aggregate_per_seed_metrics(
    per_seed: list[dict[str, Any]],
    deterministic: bool = False,
    metrics_keys: list[str] | None = None,
) -> dict[str, Any]:
    """
    Computes mean and std across a list of per-seed evaluation dictionaries.

    Args:
        per_seed: List of dicts, each containing 'seed' and metric key-values.
        deterministic: If True, std is reported as 'n/a' instead of 0.0.
        metrics_keys: Optional explicit keys to aggregate. If None, discovers all numeric keys.

    Returns:
        dict with:
            - deterministic: bool
            - n_seeds: int
            - seeds: list of seed IDs
            - means: dict of metric_name -> mean_value
            - stds: dict of metric_name -> std_value (or 'n/a')
            - summary: dict of metric_name -> {'mean': float, 'std': float|'n/a', 'formatted': str}
    """
    if not per_seed:
        return {
            "deterministic": deterministic,
            "n_seeds": 0,
            "seeds": [],
            "means": {},
            "stds": {},
            "summary": {},
        }

    seeds = [int(s["seed"]) for s in per_seed if "seed" in s]
    n_seeds = len(per_seed)

    # Discover numeric metric keys if not provided
    if metrics_keys is None:
        cand_keys = set()
        for s in per_seed:
            for k, v in s.items():
                if k not in ("seed", "training_time_sec", "val_eval", "test_eval") and isinstance(
                    v, int | float | np.floating | np.integer
                ):
                    cand_keys.add(k)
        metrics_keys = sorted(cand_keys)

    means: dict[str, float] = {}
    stds: dict[str, Any] = {}
    summary: dict[str, dict[str, Any]] = {}

    for k in metrics_keys:
        vals = [float(s[k]) for s in per_seed if k in s and s[k] is not None]
        if not vals:
            continue
        m = float(np.mean(vals))
        means[k] = m
        if deterministic or n_seeds <= 1:
            s_val: Any = "n/a"
            fmt = f"{m:.4f} (n/a)"
        else:
            s_val = float(np.std(vals))
            fmt = f"{m:.4f} ± {s_val:.4f}"
        stds[k] = s_val
        summary[k] = {
            "mean": m,
            "std": s_val,
            "formatted": fmt,
        }

    return {
        "deterministic": deterministic,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "means": means,
        "stds": stds,
        "summary": summary,
    }


def verify_aggregation_integrity(
    per_seed: list[dict[str, Any]],
    champion_entry: dict[str, Any],
    aggregate_result: dict[str, Any],
    tolerance: float = 1e-6,
) -> None:
    """
    Integrity verification guard (Issue I2):
    Fails if a multi-seed mean equals the champion value across all metrics
    while per-seed values actually differ, or if aggregate math is inconsistent.
    """
    if not per_seed or len(per_seed) < 2:
        return

    is_deterministic = aggregate_result.get("deterministic", False)
    if is_deterministic:
        return

    means = aggregate_result.get("means", {})
    stds = aggregate_result.get("stds", {})

    # Check 1: Math consistency
    for k, mean_val in means.items():
        vals = [float(s[k]) for s in per_seed if k in s and s[k] is not None]
        if vals:
            expected_mean = float(np.mean(vals))
            if abs(mean_val - expected_mean) > tolerance:
                raise ValueError(
                    f"Aggregation math error for metric '{k}': expected mean {expected_mean}, got {mean_val}"
                )
            if stds.get(k) != "n/a":
                expected_std = float(np.std(vals))
                if abs(float(stds[k]) - expected_std) > tolerance:
                    raise ValueError(
                        f"Aggregation math error for metric '{k}': expected std {expected_std}, got {stds[k]}"
                    )

    # Check 2: Copy-paste detection
    keys_with_variance = []
    for k in means:
        vals = [float(s[k]) for s in per_seed if k in s and s[k] is not None]
        if len(vals) >= 2 and np.std(vals) > 1e-5:
            keys_with_variance.append(k)

    if keys_with_variance:
        all_equal_champion = True
        for k in keys_with_variance:
            champ_val = champion_entry.get(k)
            if champ_val is not None and abs(float(champ_val) - float(means[k])) > 1e-5:
                all_equal_champion = False
                break
        if all_equal_champion:
            raise ValueError(
                "Suspicious aggregation detected (Issue I2): Multi-seed aggregate means "
                "identically match champion values on all varying metrics while per-seed values differ! "
                "Do not copy champion metrics into multi-seed rows."
            )


def compute_consistent_ablation_diff(
    variant_user_metrics_per_seed: list[np.ndarray],
    full_user_metrics_per_seed: list[np.ndarray],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Computes mathematically consistent paired differences for multi-seed ablation studies (Issue I3).

    Rule:
    1. Per-seed, per-user paired difference: diff_{s, u} = variant_{s, u} - full_{s, u}
       (evaluated on the identical users for each seed).
    2. Exact sample mean difference: mean(diff) = mean(variant) - mean(full).
       Sign of reported difference is mathematically guaranteed to match sign of (variant_mean - full_mean).
    3. Bootstrap 95% confidence interval is computed over the pooled user paired differences.
    """
    if len(variant_user_metrics_per_seed) != len(full_user_metrics_per_seed):
        raise ValueError(
            f"Seed count mismatch: variant has {len(variant_user_metrics_per_seed)} seeds, "
            f"full has {len(full_user_metrics_per_seed)} seeds"
        )
    if not variant_user_metrics_per_seed:
        return {
            "mean_diff": 0.0,
            "lo95": 0.0,
            "hi95": 0.0,
            "ci": [0.0, 0.0],
            "significant_loss": False,
            "significant_gain": False,
            "variant_mean": 0.0,
            "full_mean": 0.0,
            "variant_std": 0.0,
            "full_std": 0.0,
        }

    n_seeds = len(variant_user_metrics_per_seed)
    var_means = []
    full_means = []
    seed_paired_diffs = []

    for s_idx in range(n_seeds):
        v_arr = np.asarray(variant_user_metrics_per_seed[s_idx], dtype=np.float64)
        f_arr = np.asarray(full_user_metrics_per_seed[s_idx], dtype=np.float64)
        if len(v_arr) != len(f_arr):
            raise ValueError(
                f"User count mismatch at seed index {s_idx}: variant has {len(v_arr)}, full has {len(f_arr)}"
            )
        var_means.append(float(np.mean(v_arr)))
        full_means.append(float(np.mean(f_arr)))
        seed_paired_diffs.append(v_arr - f_arr)

    variant_mean = float(np.mean(var_means))
    full_mean = float(np.mean(full_means))
    variant_std = float(np.std(var_means)) if n_seeds > 1 else 0.0
    full_std = float(np.std(full_means)) if n_seeds > 1 else 0.0

    pooled_diff = np.concatenate(seed_paired_diffs)
    mean_diff = float(np.mean(pooled_diff))

    # Bootstrap CI over pooled differences
    rng = np.random.default_rng(seed)
    n = len(pooled_diff)
    if n > 1:
        boot_idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = np.mean(pooled_diff[boot_idx], axis=1)
        lo95 = float(np.percentile(boot_means, 2.5))
        hi95 = float(np.percentile(boot_means, 97.5))
    else:
        lo95 = mean_diff
        hi95 = mean_diff

    # Assert mathematical consistency
    expected_diff = variant_mean - full_mean
    if abs(mean_diff - expected_diff) > 1e-6:
        raise ValueError(
            f"Mathematical inconsistency in ablation difference: mean_diff={mean_diff}, "
            f"variant_mean - full_mean={expected_diff}"
        )

    return {
        "mean_diff": mean_diff,
        "lo95": lo95,
        "hi95": hi95,
        "ci": [lo95, hi95],
        "significant_loss": bool(hi95 < 0.0),
        "significant_gain": bool(lo95 > 0.0),
        "variant_mean": variant_mean,
        "full_mean": full_mean,
        "variant_std": variant_std,
        "full_std": full_std,
    }
