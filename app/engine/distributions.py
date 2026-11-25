"""Empirical distribution with an explicit zero point-mass and a fitted upper tail.

The distribution is constructed from *wet* values (strictly positive) and separately
tracks how often zero occurs.  Quantile / CDF evaluation uses type-7 linear
interpolation (the convention used by ``numpy.quantile``), which keeps the mapping
invertible and free of step-function plateaus across most of the support.

Values beyond the training maximum are not silently re-mapped to the maximum.  A
tail model (generalised Pareto, with an exponential fallback) describes how the
quantile function continues above the largest training value, and callers choose
how to treat extrapolation (see :mod:`app.engine.amount`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Empirical distribution
# --------------------------------------------------------------------------- #
class EmpiricalDist:
    """Empirical distribution of strictly positive values, plus a zero mass.

    Parameters
    ----------
    wet_values:
        Strictly positive sample (the wet days).  May be empty.
    wet_frequency:
        Probability ``P(X > 0)`` in the population this sample represents, which
        can differ from the in-sample fraction when occurrence is modelled
        separately.  Defaults to the in-sample fraction.
    """

    def __init__(self, wet_values: np.ndarray, wet_frequency: Optional[float] = None):
        values = np.asarray(wet_values, dtype=float)
        values = values[np.isfinite(values)]
        self.sorted = np.sort(values[values > 0.0])
        self.n_wet = self.sorted.size
        if wet_frequency is None:
            wet_frequency = 1.0 if self.n_wet else 0.0
        self.wet_frequency = float(np.clip(wet_frequency, 0.0, 1.0))
        self.tail: Optional[TailModel] = None

    # -- basic summaries ---------------------------------------------------- #
    @property
    def zero_mass(self) -> float:
        """Probability mass concentrated at exactly zero."""
        return 1.0 - self.wet_frequency

    @property
    def is_empty(self) -> bool:
        return self.n_wet == 0

    # -- CDF ---------------------------------------------------------------- #
    def cdf_wet(self, x: np.ndarray) -> np.ndarray:
        """CDF *conditional on being wet*, i.e. on ``(0, inf)``.

        Weibull plotting positions ``i/(n+1)`` with linear interpolation.  The
        largest observation therefore sits at ``n/(n+1) < 1`` rather than at 1,
        leaving an explicit ``(n/(n+1), 1]`` region in which a value is *beyond
        the training range* and the tail model takes over.
        """
        x = np.asarray(x, dtype=float)
        if self.n_wet == 0:
            return np.zeros_like(x)
        positions = (np.arange(self.n_wet) + 1.0) / (self.n_wet + 1.0)
        # Left of the smallest observation -> first plotting position extrapolated
        # toward 0; right of the largest -> kept at n/(n+1) (NOT flattened to 1),
        # so out-of-range values remain distinguishable.
        out = np.interp(x, self.sorted, positions,
                        left=positions[0] / 2.0, right=positions[-1])
        return out

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Unconditional CDF including the zero mass."""
        x = np.asarray(x, dtype=float)
        out = np.where(x <= 0.0, 0.0, self.zero_mass
                       + self.wet_frequency * self.cdf_wet(x))
        return np.where(x < 0.0, 0.0, out)

    # -- quantile ----------------------------------------------------------- #
    def quantile_wet(self, p: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`cdf_wet` for probabilities in ``[0, 1]``."""
        p = np.asarray(p, dtype=float)
        if self.n_wet == 0:
            return np.zeros_like(p)
        positions = (np.arange(self.n_wet) + 1.0) / (self.n_wet + 1.0)
        # np.interp requires increasing x; positions are strictly increasing.
        return np.interp(np.clip(p, 0.0, 1.0), positions, self.sorted)

    def quantile(self, p: np.ndarray) -> np.ndarray:
        """Unconditional quantile: probability below ``zero_mass`` maps to zero."""
        p = np.asarray(p, dtype=float)
        if self.wet_frequency <= 0.0:
            return np.zeros_like(p)
        # Conditional wet probability.
        p_wet = (p - self.zero_mass) / self.wet_frequency
        q = self.quantile_wet(p_wet)
        return np.where(p <= self.zero_mass, 0.0, q)

    # -- tail --------------------------------------------------------------- #
    def attach_tail(self, threshold_quantile: float = 0.85) -> "TailModel":
        """Fit a tail model to the upper ``1 - threshold_quantile`` of the data."""
        self.tail = fit_tail(self.sorted, threshold_quantile)
        return self.tail


# --------------------------------------------------------------------------- #
# Tail model
# --------------------------------------------------------------------------- #
@dataclass
class TailModel:
    """Upper-tail model used for extrapolation beyond the training maximum.

    A generalised Pareto distribution (GPD) is fitted by maximum likelihood to the
    exceedances over a high threshold.  If the fit fails or is non-stationary, an
    exponential distribution (shape fixed at zero) is used.  Below ``x_max`` the
    empirical quantile function is still authoritative; the tail only governs
    values above the largest observation.
    """

    threshold: float
    scale: float
    shape: float
    n_exceed: int
    kind: str  # "gpd" | "exponential" | "none"
    x_max: float

    def quantile_wet(self, p: float, cdf_at_threshold: float) -> float:
        """Quantile at conditional-wet probability ``p`` (p > empirical max pos).

        ``cdf_at_threshold`` is the empirical conditional-wet CDF value at
        :attr:`threshold`, so the GPD exceedance probability can be blended with
        the empirical body.
        """
        if self.kind == "none" or self.scale <= 0.0:
            return self.x_max
        # Residual conditional probability above the threshold.
        tail_frac = max(1e-12, 1.0 - cdf_at_threshold)
        exceed_prob = (np.clip(p, 0.0, 1.0) - cdf_at_threshold) / tail_frac
        exceed_prob = np.clip(exceed_prob, 1e-9, 1.0 - 1e-12)
        u = stats.genpareto.ppf(exceed_prob, c=self.shape, loc=0.0, scale=self.scale)
        return self.threshold + float(u)


def fit_tail(sorted_wet: np.ndarray, threshold_quantile: float = 0.85) -> TailModel:
    """Fit a GPD tail (exponential fallback) above the chosen quantile."""
    n = sorted_wet.size
    x_max = float(sorted_wet[-1]) if n else 0.0
    if n < 20:
        return TailModel(threshold=x_max, scale=0.0, shape=0.0,
                         n_exceed=0, kind="none", x_max=x_max)

    q = float(np.clip(threshold_quantile, 0.5, 0.99))
    # np.quantile on the sorted sample directly.
    threshold = float(np.quantile(sorted_wet, q))
    exceed = sorted_wet[sorted_wet > threshold] - threshold
    if exceed.size < 8:
        # Too few exceedances: exponential over the (smaller) top decile.
        threshold = float(np.quantile(sorted_wet, 0.9))
        exceed = sorted_wet[sorted_wet > threshold] - threshold

    if exceed.size < 5:
        return TailModel(threshold=x_max, scale=0.0, shape=0.0,
                         n_exceed=exceed.size, kind="none", x_max=x_max)

    # GPD maximum-likelihood fit.
    try:
        shape, loc, scale = stats.genpareto.fit(exceed, floc=0.0)
        # Reject unstable / heavy non-stationary fits; fall back to exponential.
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0.0 \
                or shape >= 0.9:
            raise ValueError("unstable GPD fit")
        kind = "gpd"
    except (ValueError, RuntimeError):
        scale = float(np.mean(exceed))
        shape, kind = 0.0, "exponential"

    return TailModel(threshold=threshold, scale=float(scale), shape=float(shape),
                     n_exceed=int(exceed.size), kind=kind, x_max=x_max)
