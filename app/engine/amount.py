"""Stage 2 -- precipitation *amount* correction conditional on a wet day.

Two transfer methods are offered:

* :data:`METHOD_QM`  -- plain quantile mapping
  ``x_corr = Q_obs( F_hindcast(x) )``.  The future marginal is pulled onto the
  observed historical distribution, which also erases any future shift of the
  distribution (the whole point of quantile mapping, but it discards the trend).

* :data:`METHOD_QDM` -- quantile *delta* mapping (Cannon, Sobie & Murdock, 2015).
  The future value's non-exceedance probability ``tau = F_future(x)`` is mapped
  through the observed quantile, then the model's own absolute delta at that
  quantile is added back:
  ``x_corr = Q_obs(tau) + [ x - Q_hindcast(tau) ]``.
  The bias-corrected marginal matches the observed climatology while the
  model-projected change signal (including at the extreme tail) is preserved.

Out-of-training-range values are handled explicitly rather than clamped without
trace.  ``tail="clip"`` saturates at the observed maximum; ``tail="extrapolate"``
continues the observed quantile function with the fitted GPD/exponential tail.
Every extrapolated point is flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .distributions import EmpiricalDist

METHOD_QM = "qm"
METHOD_QDM = "qdm"

TAIL_CLIP = "clip"
TAIL_EXTRAPOLATE = "extrapolate"


@dataclass
class AmountFit:
    obs: EmpiricalDist
    hindcast: EmpiricalDist
    n_obs_wet: int
    n_hind_wet: int


@dataclass
class AmountResult:
    corrected: np.ndarray
    method: str
    tail_mode: str
    # Wet values only, aligned to the *input* wet positions.
    wet_mask: np.ndarray
    taus: np.ndarray                 # F_future(x) on the wet subset
    extrapolated: np.ndarray         # bool on the wet subset
    records: List[Dict] = field(default_factory=list)


class AmountCorrector:
    """Bias-correct wet-day amounts with QM or trend-preserving QDM."""

    def __init__(self, method: str = METHOD_QDM,
                 tail: str = TAIL_EXTRAPOLATE,
                 tail_quantile: float = 0.85):
        if method not in (METHOD_QM, METHOD_QDM):
            raise ValueError(f"unknown amount method {method!r}")
        if tail not in (TAIL_CLIP, TAIL_EXTRAPOLATE):
            raise ValueError(f"unknown tail mode {tail!r}")
        self.method = method
        self.tail = tail
        self.tail_quantile = tail_quantile
        self.fit_: Optional[AmountFit] = None

    # ------------------------------------------------------------------ #
    def fit(self, hindcast: np.ndarray, obs: np.ndarray) -> "AmountCorrector":
        h = np.asarray(hindcast, float)
        o = np.asarray(obs, float)
        h_wet = h[h > 0.0]
        o_wet = o[o > 0.0]
        obs_dist = EmpiricalDist(o_wet, wet_frequency=1.0)
        hind_dist = EmpiricalDist(h_wet, wet_frequency=1.0)
        obs_dist.attach_tail(self.tail_quantile)
        hind_dist.attach_tail(self.tail_quantile)
        self.fit_ = AmountFit(obs=obs_dist, hindcast=hind_dist,
                              n_obs_wet=o_wet.size, n_hind_wet=h_wet.size)
        return self

    # ------------------------------------------------------------------ #
    def _observed_quantile(self, dist: EmpiricalDist, taus: np.ndarray,
                           max_position: float,
                           force_tail: Optional[np.ndarray] = None) -> np.ndarray:
        """Observed-distribution quantile, applying the chosen tail policy.

        ``force_tail`` marks values exceeding the *hindcast* training maximum;
        under ``extrapolate`` they continue through the observed tail model even
        if their own probability still lies inside the observed empirical support.
        """
        out = np.empty_like(taus)
        beyond = (taus > max_position)
        if force_tail is not None:
            beyond = beyond | np.asarray(force_tail, dtype=bool)
        in_range = ~beyond
        # In-range: empirical quantile.
        out[in_range] = dist.quantile_wet(taus[in_range])
        if np.any(beyond):
            if self.tail == TAIL_CLIP or dist.tail is None \
                    or dist.tail.kind == "none":
                out[beyond] = dist.sorted[-1]
            else:
                cdf_thr = dist.cdf_wet(np.array([dist.tail.threshold]))[0]
                out[beyond] = np.array([
                    dist.tail.quantile_wet(float(p), cdf_thr)
                    for p in taus[beyond]
                ])
        return out

    def apply(self, future: np.ndarray) -> AmountResult:
        if self.fit_ is None:
            raise RuntimeError("AmountCorrector must be fitted before apply")
        future = np.asarray(future, float)
        wet_mask = future > 0.0
        x = future[wet_mask]

        # Future empirical distribution supplies the non-exceedance probability.
        fut_dist = EmpiricalDist(x, wet_frequency=1.0)
        taus = fut_dist.cdf_wet(x)

        obs = self.fit_.obs
        hind = self.fit_.hindcast
        # Out-of-training-range: the future value exceeds the largest *hindcast
        # training* value (its magnitude, not its in-sample rank).  These points
        # require tail extrapolation rather than interpolation.
        hind_max = hind.sorted.size / (hind.sorted.size + 1.0)
        obs_max = obs.sorted.size / (obs.sorted.size + 1.0)

        q_obs = self._observed_quantile(
            obs, taus, obs_max, force_tail=(x > hind.sorted[-1]))
        extrapolated = x > hind.sorted[-1]

        if self.method == METHOD_QM:
            corrected_wet = q_obs
            delta = np.zeros_like(x)
            q_hind = np.zeros_like(x)
        else:
            # QDM: preserve the model absolute change signal at each quantile.
            q_hind = self._hindcast_quantile(hind, taus, hind_max)
            delta = x - q_hind
            corrected_wet = q_obs + delta
            corrected_wet = np.maximum(corrected_wet, 0.0)
            # In clip mode the *final* value is bounded by the observed maximum,
            # so the preserved delta cannot push the result out of range.
            if self.tail == TAIL_CLIP:
                corrected_wet = np.minimum(corrected_wet, obs.sorted[-1])

        corrected = np.zeros_like(future)
        corrected[wet_mask] = corrected_wet

        result = AmountResult(
            corrected=corrected,
            method=self.method,
            tail_mode=self.tail,
            wet_mask=wet_mask,
            taus=taus,
            extrapolated=extrapolated,
        )
        result.records = self._records(
            x, taus, q_obs, q_hind, delta, corrected_wet, extrapolated,
            hind_max)
        return result

    def _hindcast_quantile(self, dist: EmpiricalDist, taus: np.ndarray,
                           max_position: float) -> np.ndarray:
        out = np.empty_like(taus)
        in_range = taus <= max_position
        out[in_range] = dist.quantile_wet(taus[in_range])
        if np.any(~in_range):
            if dist.tail is None or dist.tail.kind == "none":
                out[~in_range] = dist.sorted[-1]
            else:
                cdf_thr = dist.cdf_wet(np.array([dist.tail.threshold]))[0]
                out[~in_range] = np.array([
                    dist.tail.quantile_wet(float(p), cdf_thr)
                    for p in taus[~in_range]
                ])
        return out

    # ------------------------------------------------------------------ #
    def _records(self, x, taus, q_obs, q_hind, delta, corrected,
                 extrapolated, max_pos) -> List[Dict]:
        records = []
        for local_idx in range(x.size):
            records.append({
                "wet_local_index": int(local_idx),
                "future_value": float(x[local_idx]),
                "non_exceedance_tau": float(taus[local_idx]),
                "obs_quantile_q_obs": float(q_obs[local_idx]),
                "hind_quantile_q_hind": float(q_hind[local_idx]),
                "preserved_delta": float(delta[local_idx]),
                "corrected_value": float(corrected[local_idx]),
                "beyond_training_max": bool(extrapolated[local_idx]),
                "training_max_position": float(max_pos),
            })
        return records
