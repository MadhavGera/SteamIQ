"""
Probability Calibration and Selective Abstention for Ask SteamIQ — Phase 6.

Key requirements:
  - Temperature scaling fit on held-out calibration split.
  - Abstention threshold tau chosen strictly on VALIDATION split (never on test)
    to achieve target precision >= 0.95.
  - Metrics: ECE, reliability bins, coverage-vs-accuracy curve, selective accuracy at tau,
    out-of-scope detection AUROC and FPR@95%TPR.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score


class TemperatureCalibrator:
    """
    Multi-class Temperature Scaling calibrator.
    Scales logits z by temperature T > 0: p_cal = softmax(z / T).
    Preserves top-1 rank order while producing well-calibrated confidence scores.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature: float = float(temperature)
        self.is_fitted: bool = False

    def fit(self, probs: np.ndarray, true_indices: np.ndarray) -> TemperatureCalibrator:
        """
        Fits temperature T by minimizing negative log-likelihood on calibration split.
        probs: [N, n_classes] uncalibrated probabilities.
        true_indices: [N] integer class indices.
        """
        if len(probs) == 0:
            self.temperature = 1.0
            self.is_fitted = True
            return self

        # Convert probabilities to pseudo-logits
        clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
        logits = np.log(clipped)
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        n = len(true_indices)

        def nll_loss(t_arr: np.ndarray) -> float:
            t = max(float(t_arr[0]), 1e-3)
            scaled = logits / t
            # Stable log-sum-exp
            max_s = np.max(scaled, axis=1, keepdims=True)
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
            log_p = scaled - lse
            # NLL on true class
            return -float(np.mean(log_p[np.arange(n), true_indices]))

        res = minimize(nll_loss, x0=np.array([1.0]), bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0
        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Applies temperature scaling to [N, n_classes] probabilities."""
        if len(probs) == 0:
            return probs
        if abs(self.temperature - 1.0) < 1e-5:
            return probs

        clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
        logits = np.log(clipped) / self.temperature
        max_l = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "temperature.json").write_text(
            json.dumps({"temperature": self.temperature}, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> TemperatureCalibrator:
        p = Path(path)
        data = json.loads((p / "temperature.json").read_text(encoding="utf-8"))
        cal = cls(temperature=data["temperature"])
        cal.is_fitted = True
        return cal


def compute_ece(
    probs: np.ndarray,
    true_indices: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and returns reliability diagram bins.
    """
    if len(probs) == 0:
        return 0.0, []

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == true_indices).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_data: list[dict[str, Any]] = []

    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
            bins_data.append({
                "bin_lower": round(float(bin_lower), 3),
                "bin_upper": round(float(bin_upper), 3),
                "bin_center": round(float((bin_lower + bin_upper) / 2.0), 3),
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            })
        else:
            bins_data.append({
                "bin_lower": round(float(bin_lower), 3),
                "bin_upper": round(float(bin_upper), 3),
                "bin_center": round(float((bin_lower + bin_upper) / 2.0), 3),
                "count": 0,
                "accuracy": 0.0,
                "confidence": round(float((bin_lower + bin_upper) / 2.0), 4),
            })

    return float(ece), bins_data


def select_abstention_threshold(
    val_probs: np.ndarray,
    val_true_indices: np.ndarray,
    target_precision: float = 0.95,
    min_coverage: float = 0.40,
) -> float:
    """
    Selects abstention threshold tau on the VALIDATION split.
    Finds the lowest tau that achieves precision >= target_precision while keeping coverage >= min_coverage.
    """
    if len(val_probs) == 0:
        return 0.50

    confidences = np.max(val_probs, axis=1)
    predictions = np.argmax(val_probs, axis=1)
    correct = (predictions == val_true_indices).astype(float)

    thresholds = np.linspace(0.10, 0.98, 89)
    best_tau = 0.50
    best_tau_candidate = None

    for tau in thresholds:
        accepted = confidences >= tau
        coverage = float(np.mean(accepted))
        if coverage < min_coverage:
            continue
        if np.sum(accepted) == 0:
            continue

        precision = float(np.mean(correct[accepted]))
        if precision >= target_precision:
            best_tau_candidate = tau
            break

    if best_tau_candidate is not None:
        best_tau = float(best_tau_candidate)
    else:
        # Fallback to threshold that maximizes selective F1
        f1_scores = []
        for tau in thresholds:
            accepted = confidences >= tau
            cov = float(np.mean(accepted))
            prec = float(np.mean(correct[accepted])) if np.sum(accepted) > 0 else 0.0
            f1 = (2 * prec * cov) / (prec + cov) if (prec + cov) > 0 else 0.0
            f1_scores.append((f1, tau))
        f1_scores.sort(reverse=True)
        best_tau = float(f1_scores[0][1])

    return round(float(best_tau), 3)


def evaluate_selective_prediction(
    probs: np.ndarray,
    true_indices: np.ndarray,
    tau: float,
) -> dict[str, float]:
    """
    Evaluates selective prediction at threshold tau.
    Returns:
      coverage: fraction of predictions accepted (conf >= tau).
      selective_accuracy: accuracy on accepted predictions.
      abstention_rate: 1 - coverage.
      overall_accuracy: standard unconstrained accuracy.
    """
    if len(probs) == 0:
        return {
            "coverage": 0.0,
            "selective_accuracy": 0.0,
            "abstention_rate": 1.0,
            "overall_accuracy": 0.0,
        }

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == true_indices).astype(float)

    accepted = confidences >= tau
    coverage = float(np.mean(accepted))
    abstention_rate = float(1.0 - coverage)
    selective_acc = float(np.mean(correct[accepted])) if np.sum(accepted) > 0 else 0.0
    overall_acc = float(np.mean(correct))

    return {
        "coverage": round(coverage, 4),
        "selective_accuracy": round(selective_acc, 4),
        "abstention_rate": round(abstention_rate, 4),
        "overall_accuracy": round(overall_acc, 4),
    }


def compute_coverage_accuracy_curve(
    probs: np.ndarray,
    true_indices: np.ndarray,
    n_points: int = 25,
) -> list[dict[str, Any]]:
    """
    Computes coverage-vs-accuracy curve over a range of confidence thresholds.
    """
    if len(probs) == 0:
        return []

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == true_indices).astype(float)

    thresholds = np.linspace(0.0, 0.98, n_points)
    curve = []

    for tau in thresholds:
        accepted = confidences >= tau
        cov = float(np.mean(accepted))
        acc = float(np.mean(correct[accepted])) if np.sum(accepted) > 0 else 1.0
        curve.append({
            "threshold": round(float(tau), 3),
            "coverage": round(cov, 4),
            "selective_accuracy": round(acc, 4),
        })

    return curve


def evaluate_out_of_scope_detection(
    probs: np.ndarray,
    classes: list[str],
    true_labels: list[str],
    oos_intent: str = "out_of_scope",
) -> dict[str, float]:
    """
    Evaluates detection of Out-Of-Scope queries using the calibrated out_of_scope probability.
    Computes AUROC and FPR @ 95% TPR.
    """
    if len(probs) == 0 or oos_intent not in classes:
        return {"auroc": 0.5, "fpr_at_95_tpr": 1.0}

    oos_idx = classes.index(oos_intent)
    oos_scores = probs[:, oos_idx]
    binary_ground_truth = np.array([1 if lab == oos_intent else 0 for lab in true_labels], dtype=int)

    n_pos = int(np.sum(binary_ground_truth == 1))
    n_neg = int(np.sum(binary_ground_truth == 0))

    if n_pos == 0 or n_neg == 0:
        return {"auroc": 1.0, "fpr_at_95_tpr": 0.0}

    auroc = float(roc_auc_score(binary_ground_truth, oos_scores))

    # FPR @ 95% TPR
    pos_scores = oos_scores[binary_ground_truth == 1]
    neg_scores = oos_scores[binary_ground_truth == 0]

    tpr95_threshold = float(np.percentile(pos_scores, 5.0))  # 95% of positive scores are >= this
    fpr = float(np.mean(neg_scores >= tpr95_threshold))

    return {
        "auroc": round(auroc, 4),
        "fpr_at_95_tpr": round(fpr, 4),
    }
