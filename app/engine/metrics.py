"""Evaluation metrics covering the three things the UI must demonstrate.

1. **Wet-day frequency** -- the zero point-mass (occurrence stage).
2. **Quantile changes** -- how each quantile of the corrected distribution moves
   relative to raw and observed (amount stage, including the QM vs QDM difference).
3. **Extreme tail** -- upper quantiles and the partial-duration exceedance rate.

Plus scalar accuracy measures used to *select* a method on the validation block:
a Brier score for the wet/dry decision and MAE/RMSE on wet-day amounts.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

WET_THRESHOLD = 0.0


def wet_frequency(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    return float(np.mean(values > WET_THRESHOLD)) if values.size else 0.0


def dry_frequency(values: np.ndarray) -> float:
    return 1.0 - wet_frequency(values)


def quantile_curve(values: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    """Empirical quantiles (type 7); zeros included so the wet threshold shows."""
    values = np.asarray(values, float)
    if values.size == 0:
        return np.full(len(probs), np.nan)
    return np.quantile(values, np.asarray(probs, float), method="linear")


def quantile_change_table(raw, corrected, obs,
                          probs=(0.5, 0.75, 0.9, 0.95, 0.99)) -> Dict:
    """Per-quantile values and the change preserved by each method."""
    probs = list(probs)
    q_raw = quantile_curve(raw, probs)
    q_corr = quantile_curve(corrected, probs)
    q_obs = quantile_curve(obs, probs)
    return {
        "probs": probs,
        "raw": q_raw.tolist(),
        "corrected": q_corr.tolist(),
        "obs": q_obs.tolist(),
        # How much of the raw->future signal remains after correction.
        "delta_raw_vs_obs": (q_raw - q_obs).tolist(),
        "delta_corr_vs_obs": (q_corr - q_obs).tolist(),
    }


def extreme_tail_summary(values: np.ndarray,
                         quantiles=(0.95, 0.99, 0.995),
                         exceed_threshold_quantile: float = 0.99) -> Dict:
    """Upper-tail diagnostics: high quantiles + partial-duration exceedance."""
    values = np.asarray(values, float)
    qs = quantile_curve(values, list(quantiles))
    if values.size:
        threshold = float(np.quantile(values, exceed_threshold_quantile))
        n_exceed = int(np.sum(values > threshold))
        max_v = float(np.max(values))
    else:
        threshold, n_exceed, max_v = np.nan, 0, np.nan
    return {
        "quantiles": list(quantiles),
        "values": qs.tolist(),
        "exceedance_threshold": threshold,
        "n_exceedance": n_exceed,
        "exceedance_rate": float(n_exceed / values.size) if values.size else 0.0,
        "max": max_v,
    }


# --------------------------------------------------------------------------- #
# Scalar losses for method selection on the validation block
# --------------------------------------------------------------------------- #
def brier_score(wet_prob_or_flag, observed_wet) -> float:
    """Brier score for the occurrence decision (lower is better)."""
    p = np.asarray(wet_prob_or_flag, float)
    y = (np.asarray(observed_wet, float) > WET_THRESHOLD).astype(float)
    return float(np.mean((p - y) ** 2))


def mae(pred, obs) -> float:
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    return float(np.mean(np.abs(pred - obs)))


def rmse(pred, obs) -> float:
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def wet_amount_mae(pred, obs) -> float:
    """MAE restricted to days that are wet in either series (amount skill)."""
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    mask = (pred > WET_THRESHOLD) | (obs > WET_THRESHOLD)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(pred[mask] - obs[mask])))


def _safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, float)
    return float(np.mean(values)) if values.size else default


def full_summary(raw, corrected, obs,
                 tail_quantiles=(0.95, 0.99, 0.995)) -> Dict:
    """Bundle all diagnostics for the dashboard and the evaluation view."""
    raw = np.asarray(raw, float)
    corrected = np.asarray(corrected, float)
    obs = np.asarray(obs, float)
    return {
        "wet_frequency": {
            "raw": wet_frequency(raw),
            "corrected": wet_frequency(corrected),
            "obs": wet_frequency(obs),
        },
        "quantile_change": quantile_change_table(raw, corrected, obs),
        "tail_raw": extreme_tail_summary(raw, tail_quantiles),
        "tail_corrected": extreme_tail_summary(corrected, tail_quantiles),
        "tail_obs": extreme_tail_summary(obs, tail_quantiles),
        "amount_mae": {
            "raw": wet_amount_mae(raw, obs),
            "corrected": wet_amount_mae(corrected, obs),
        },
        "rmse": {
            "raw": rmse(raw, obs),
            "corrected": rmse(corrected, obs),
        },
    }
