"""
Statistical Drift Detection Engine — Phase 7.

Algorithms:
  1. Population Stability Index (PSI)
  2. Two-sample Kolmogorov-Smirnov (KS) test
  3. Jensen-Shannon Distance (JSD)

Monitored distributions:
  (a) Item numeric features (price_final, positive_ratio, log_review_count)
  (b) Liked-library length distribution
  (c) Recommendation score distribution
  (d) Top-k popularity concentration and catalog coverage
  (e) Assistant intent distribution and mean confidence
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger("steamiq.mlops.drift")


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current samples.
      PSI < 0.10: No significant distribution change (PASS)
      0.10 <= PSI < 0.20: Moderate distribution shift (WARN)
      PSI >= 0.20: Significant distribution shift (ALERT)
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    # Create quantile bin edges from reference distribution
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref, percentiles)
    # Ensure strictly increasing edges to avoid empty degenerate bins
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    ref_pct = (ref_counts / len(ref)) + epsilon
    curr_pct = (curr_counts / len(curr)) + epsilon

    # Normalize after epsilon smoothing
    ref_pct = ref_pct / np.sum(ref_pct)
    curr_pct = curr_pct / np.sum(curr_pct)

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


def calculate_ks(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """
    Computes two-sample Kolmogorov-Smirnov test statistic and p-value.
    Returns: (ks_stat, p_value)
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(ref, curr)
    return float(res.statistic), float(res.pvalue)


def calculate_js_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Computes Jensen-Shannon distance between two categorical probability distributions.
    Bounded in [0.0, 1.0].
    """
    p_arr = np.asarray(p, dtype=float) + 1e-9
    q_arr = np.asarray(q, dtype=float) + 1e-9

    p_norm = p_arr / np.sum(p_arr)
    q_norm = q_arr / np.sum(q_arr)

    return float(jensenshannon(p_norm, q_norm))


def evaluate_distribution_drift(
    reference: np.ndarray,
    current: np.ndarray,
    psi_warn: float = 0.10,
    psi_alert: float = 0.20,
    ks_alpha: float = 0.01,
) -> dict[str, Any]:
    """
    Evaluates drift across multiple metrics for continuous distributions.
    """
    psi = calculate_psi(reference, current)
    ks_stat, p_val = calculate_ks(reference, current)

    if psi >= psi_alert or p_val < (ks_alpha / 10.0):
        status = "ALERT"
    elif psi >= psi_warn or p_val < ks_alpha:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "psi": round(psi, 4),
        "ks_stat": round(ks_stat, 4),
        "ks_pvalue": round(p_val, 5),
        "ref_samples": len(reference),
        "curr_samples": len(current),
    }


def evaluate_categorical_drift(
    reference_counts: dict[str, int],
    current_counts: dict[str, int],
    jsd_warn: float = 0.15,
    jsd_alert: float = 0.30,
) -> dict[str, Any]:
    """
    Evaluates drift for discrete categorical distributions (e.g. intent shares).
    """
    all_keys = sorted(set(reference_counts.keys()) | set(current_counts.keys()))
    if not all_keys:
        return {"status": "PASS", "js_distance": 0.0}

    ref_vec = np.array([reference_counts.get(k, 0) for k in all_keys], dtype=float)
    curr_vec = np.array([current_counts.get(k, 0) for k in all_keys], dtype=float)

    jsd = calculate_js_distance(ref_vec, curr_vec)

    if jsd >= jsd_alert:
        status = "ALERT"
    elif jsd >= jsd_warn:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "js_distance": round(jsd, 4),
        "categories_evaluated": len(all_keys),
    }
