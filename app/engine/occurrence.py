"""Stage 1 -- precipitation *occurrence* correction (the zero point-mass).

Two calibration philosophies mirror the amount stage:

* ``plain``  -- quantile-mapping style: force the future wet-day frequency onto the
  observed historical frequency ``f_o``.  This erases any frequency trend.
* ``trend``  -- quantile-delta-mapping style: keep the *relative change* in wet-day
  odds between hindcast and future, applying it to the observed odds.  A genuine
  trend in wet-day frequency therefore survives correction.

Two realisations of each target frequency are offered:

* :data:`OCC_DETERMINISTIC` -- a midpoint-rank threshold.  The ``k`` largest forecast
  values are wet, everything else dry; fully reproducible, no randomness.
* :data:`OCC_STOCHASTIC` -- randomised quantile mapping for zero-inflated data.
  Each day receives a uniform draw inside its empirical CDF step
  ``[F(x-), F(x)]`` (Dunn & Smyth randomised quantile).  The seed, the complete
  bit-generator state and every draw are stored on the calibration so the result is
  exactly reproducible.  Frequency change is *never* achieved by silently dropping
  points without a persisted random state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

OCC_DETERMINISTIC = "deterministic"
OCC_STOCHASTIC = "stochastic"

OCC_PLAIN = "plain"      # match observed historical wet frequency
OCC_TREND = "trend"      # preserve the odds-ratio change signal

_EPS = 1e-9


def _wet_indicator(values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values > threshold).astype(np.int8)


@dataclass
class OccurrenceFit:
    """Everything learned from the historical hindcast/obs pair."""

    f_hindcast: float          # historical hindcast wet frequency
    f_obs: float               # historical observed wet frequency
    n_history: int
    method: str

    def target_frequency(self, f_future: float) -> float:
        """Wet frequency to impose on a future block with raw frequency ``f_future``."""
        f_o = np.clip(self.f_obs, _EPS, 1.0 - _EPS)
        if self.method == OCC_PLAIN:
            return float(self.f_obs)

        # Trend-preserving: work on log-odds (multiplicative odds correction).
        f_h = np.clip(self.f_hindcast, _EPS, 1.0 - _EPS)
        f_r = np.clip(f_future, _EPS, 1.0 - _EPS)
        log_odds_o = np.log(f_o / (1.0 - f_o))
        log_odds_h = np.log(f_h / (1.0 - f_h))
        log_odds_r = np.log(f_r / (1.0 - f_r))
        # Calibration shift on log-odds, applied to the future projected odds.
        log_odds_corr = log_odds_o + (log_odds_r - log_odds_h)
        odds_corr = np.exp(log_odds_corr)
        return float(odds_corr / (1.0 + odds_corr))


@dataclass
class OccurrenceResult:
    wet: np.ndarray                       # boolean decisions
    target_frequency: float
    realized_frequency: float
    threshold_value: float
    randomized_uniforms: Optional[np.ndarray]
    rng_state: Optional[dict]
    seed: Optional[int]
    mode: str
    method: str
    # Per-day diagnostics for the trace view.
    empirical_cdf: np.ndarray            # F(x) for each day
    empirical_cdf_left: np.ndarray       # F(x-) for each day
    records: List[Dict] = field(default_factory=list)


class OccurrenceCorrector:
    """Fit historical occurrence bias and apply it to a future block."""

    def __init__(self, method: str = OCC_TREND):
        if method not in (OCC_PLAIN, OCC_TREND):
            raise ValueError(f"unknown occurrence method {method!r}")
        self.method = method
        self.fit_: Optional[OccurrenceFit] = None

    # ------------------------------------------------------------------ #
    def fit(self, hindcast: np.ndarray, obs: np.ndarray) -> "OccurrenceCorrector":
        h = _wet_indicator(hindcast)
        o = _wet_indicator(obs)
        self.fit_ = OccurrenceFit(
            f_hindcast=float(np.mean(h)),
            f_obs=float(np.mean(o)),
            n_history=h.size,
            method=self.method,
        )
        return self

    # ------------------------------------------------------------------ #
    @staticmethod
    def _empirical_steps(values: np.ndarray):
        """Return ``(F(x), F(x-))`` type-7-ish empirical CDF incl. zero mass.

        Uses rank-based positions so ties share the same step and the left limit
        marks the start of the tie block -- exactly the interval a randomised
        quantile draw lives in.
        """
        n = values.size
        order = np.argsort(values, kind="mergesort")
        sorted_vals = values[order]
        # Rank of each position (1-based), averaged over ties.
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            avg_rank = 0.5 * (i + 1 + j + 1)
            ranks[i:j + 1] = avg_rank
            i = j + 1
        f_at = np.empty(n)
        f_left = np.empty(n)
        f_at[order] = ranks / n
        # Left limit: rank of the start of the tie block, minus one, over n.
        left_rank = np.empty(n)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            left_rank[i:j + 1] = i  # number of values strictly below
            i = j + 1
        f_left[order] = left_rank / n
        return f_at, f_left

    def apply(self, future: np.ndarray, mode: str = OCC_DETERMINISTIC,
              seed: Optional[int] = None,
              fixed_uniforms: Optional[np.ndarray] = None) -> OccurrenceResult:
        if self.fit_ is None:
            raise RuntimeError("OccurrenceCorrector must be fitted before apply")
        if mode not in (OCC_DETERMINISTIC, OCC_STOCHASTIC):
            raise ValueError(f"unknown occurrence mode {mode!r}")
        future = np.asarray(future, dtype=float)
        n = future.size
        f_future = float(np.mean(_wet_indicator(future)))
        f_target = self.fit_.target_frequency(f_future)

        f_at, f_left = self._empirical_steps(future)

        rng_state = None
        seed_used = None
        if fixed_uniforms is not None:
            u = np.asarray(fixed_uniforms, dtype=float)
        elif mode == OCC_STOCHASTIC:
            seed_used = int(np.random.SeedSequence().entropy) if seed is None \
                else int(seed)
            rng = np.random.default_rng(seed_used)
            u = rng.random(n)
            rng_state = rng.bit_generator.state
        else:
            # Deterministic midpoint of each CDF step (no randomness at all).
            u = 0.5 * (f_at + f_left)

        # Randomised quantile: uniform inside [F(x-), F(x)].
        q_rand = f_left + u * (f_at - f_left)
        wet = q_rand > (1.0 - f_target)

        # Deterministic threshold value for reporting / tie handling.
        n_wet_target = int(round(f_target * n))
        if n_wet_target > 0:
            threshold = float(np.sort(future)[::-1][n_wet_target - 1])
        else:
            threshold = float(np.max(future)) + 1.0

        realized = float(np.mean(wet))
        result = OccurrenceResult(
            wet=wet,
            target_frequency=f_target,
            realized_frequency=realized,
            threshold_value=threshold,
            randomized_uniforms=(u if mode == OCC_STOCHASTIC else None),
            rng_state=rng_state,
            seed=seed_used,
            mode=mode,
            method=self.method,
            empirical_cdf=f_at,
            empirical_cdf_left=f_left,
        )
        result.records = self._build_records(future, q_rand, wet, f_target,
                                             f_at, f_left, u)
        return result

    @staticmethod
    def _build_records(future, q_rand, wet, f_target, f_at, f_left, u) -> List[Dict]:
        records = []
        for idx in range(future.size):
            records.append({
                "index": int(idx),
                "raw_value": float(future[idx]),
                "empirical_cdf": float(f_at[idx]),
                "empirical_cdf_left": float(f_left[idx]),
                "uniform_draw": float(u[idx]),
                "randomized_quantile": float(q_rand[idx]),
                "decision_cutoff": float(1.0 - f_target),
                "is_wet": bool(wet[idx]),
            })
        return records
