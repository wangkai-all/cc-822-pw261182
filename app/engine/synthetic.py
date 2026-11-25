"""Synthetic data generator with *known* bias and *known* trend.

The generator exists so the whole chain can be verified against ground truth
without any meteorological platform.  A latent "truth" process is drawn first; the
observed series is that truth (clean), and every forecast series is produced by a
documented bias transform of the truth.  Two independent, controllable effects are
built in:

* **Known bias** (constant in time) -- a wet/drizzle bias (some dry days become
  light rain) plus a multiplicative intensity bias with log-normal scatter.
* **Known trend** (in the truth) -- wet-day log-odds and wet-amount mean both drift
  linearly with year.  The future block therefore genuinely differs from history,
  which is exactly what plain QM erases and QDM/QDM-occurrence is meant to retain.

The future truth is returned for *verification only* and is never passed to a fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    history_years: int = 18
    future_years: int = 6
    start_year: int = 2000
    # Seasonal base wet-day probability and mean wet amount (mm).
    base_wet_prob: Tuple[float, ...] = (0.25, 0.30, 0.35, 0.30)
    base_wet_mean: Tuple[float, ...] = (4.0, 6.0, 9.0, 7.0)
    gamma_shape: float = 0.9
    # Known trend: log-odds per year and log-intensity per year.
    wet_logodds_trend: float = 0.03       # +3% log-odds / yr
    intensity_log_trend: float = 0.02     # +2% mean amount / yr
    # Known forecast bias.
    drizzle_prob: float = 0.18            # dry truth -> light forecast rain
    drizzle_scale: float = 0.6
    intensity_bias: float = 1.25          # forecast = 1.25 * truth (wet)
    bias_scatter_sigma: float = 0.25      # log-normal scatter
    seed: int = 20261125


_SEASON_MONTHS = ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))


def _season_index(month: int) -> int:
    for i, months in enumerate(_SEASON_MONTHS):
        if month in months:
            return i
    return 0


def _date_frame(start_year: int, n_years: int) -> pd.DataFrame:
    dates = pd.date_range(f"{start_year}-01-01",
                          periods=n_years * 365, freq="D")
    return pd.DataFrame({"date": dates,
                         "year": dates.year,
                         "month": dates.month,
                         "dayofyear": dates.day_of_year})


def _draw_truth(df: pd.DataFrame, cfg: SyntheticConfig,
                rng: np.random.Generator) -> np.ndarray:
    n = len(df)
    season_idx = df["month"].map(_season_index).to_numpy()
    year0 = df["year"].min()
    yr = (df["year"] - year0).to_numpy(dtype=float)

    base_p = np.array(cfg.base_wet_prob)[season_idx]
    base_mu = np.array(cfg.base_wet_mean)[season_idx]
    logit_p = np.log(base_p / (1 - base_p)) + cfg.wet_logodds_trend * yr
    p_wet = 1.0 / (1.0 + np.exp(-logit_p))
    wet = rng.random(n) < p_wet

    mu = base_mu * np.exp(cfg.intensity_log_trend * yr)
    # Gamma with given shape; scale = mean / shape.
    amounts = rng.gamma(shape=cfg.gamma_shape,
                        scale=mu / cfg.gamma_shape)
    truth = np.where(wet, amounts, 0.0)
    return truth


def _apply_forecast_bias(truth: np.ndarray, cfg: SyntheticConfig,
                         rng: np.random.Generator) -> np.ndarray:
    """The documented, time-constant bias transform: truth -> forecast."""
    n = truth.size
    out = np.zeros(n)
    wet = truth > 0.0
    # Wet truth: multiplicative intensity bias with log-normal scatter.
    scatter = rng.lognormal(mean=0.0, sigma=cfg.bias_scatter_sigma,
                            size=n)
    out[wet] = cfg.intensity_bias * truth[wet] * scatter[wet]
    # Dry truth: drizzle contamination (the wet bias).
    dry = ~wet
    drizzle = rng.random(n) < cfg.drizzle_prob
    drizzle_draw = rng.gamma(shape=0.6,
                             scale=cfg.drizzle_scale / 0.6, size=n)
    out[dry & drizzle] = drizzle_draw[dry & drizzle]
    return np.maximum(out, 0.0)


def generate_synthetic_dataset(
        cfg: SyntheticConfig | None = None) -> Dict[str, pd.DataFrame]:
    """Return history (hindcast+obs) and future (raw forecast + hidden truth)."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    hist = _date_frame(cfg.start_year, cfg.history_years)
    fut_start = cfg.start_year + cfg.history_years
    fut = _date_frame(fut_start, cfg.future_years)

    # Independent truth draws, but a continuous trend across the whole timeline.
    all_df = pd.concat([hist, fut], ignore_index=True)
    truth_all = _draw_truth(all_df, cfg, rng)
    n_hist = len(hist)

    hist = hist.copy()
    fut = fut.copy()
    hist["obs"] = truth_all[:n_hist]
    # Hindcast = biased history truth.
    hist["hindcast"] = _apply_forecast_bias(truth_all[:n_hist], cfg, rng)
    fut["forecast"] = _apply_forecast_bias(truth_all[n_hist:], cfg, rng)
    # Verification-only truth; the pipeline must not fit on this column.
    fut["future_truth_hidden"] = truth_all[n_hist:]

    return {
        "history": hist,
        "future": fut,
        "config": cfg,
    }


def dataset_manifest(result: Dict) -> Dict:
    """Human-readable statement of the known effects, shown on the page."""
    cfg: SyntheticConfig = result["config"]
    hist, fut = result["history"], result["future"]
    return {
        "history_years": [int(hist["year"].min()), int(hist["year"].max())],
        "future_years": [int(fut["year"].min()), int(fut["year"].max())],
        "known_bias": {
            "drizzle_prob": cfg.drizzle_prob,
            "intensity_multiplier": cfg.intensity_bias,
            "lognormal_scatter_sigma": cfg.bias_scatter_sigma,
        },
        "known_trend": {
            "wet_logodds_per_year": cfg.wet_logodds_trend,
            "intensity_log_per_year": cfg.intensity_log_trend,
        },
        "history_wet_freq_obs": float(np.mean(hist["obs"] > 0)),
        "history_wet_freq_hindcast": float(np.mean(hist["hindcast"] > 0)),
    }
