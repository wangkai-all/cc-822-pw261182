"""End-to-end correction pipeline orchestrating occurrence + amount stages.

Responsibilities
----------------
* Fit / apply the two stages separately for every seasonal stratum and reassemble
  the result in original row order.
* Run contiguous-year cross-validation, *selecting* the transfer method on the
  validation block and reporting it only on the held-out test block.
* Produce a versioned, serialisable :class:`Calibration` that freezes the fitted
  distributions, the chosen methods, the tail policy and -- for stochastic
  occurrence -- the seed, generator state and every uniform draw.
* Bind each corrected product to the exact calibration id and the observation
  revision it was built from, so a later observation revision cannot silently
  re-bias an existing product.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .occurrence import (OccurrenceCorrector, OccurrenceResult,
                         OCC_PLAIN, OCC_TREND, OCC_DETERMINISTIC, OCC_STOCHASTIC)
from .amount import (AmountCorrector, AmountResult,
                     METHOD_QM, METHOD_QDM, TAIL_CLIP, TAIL_EXTRAPOLATE)
from .seasons import season_key, season_labels
from .splits import contiguous_year_splits, years_mask, SplitPlan
from . import metrics

# Candidate (occurrence method, amount method) combinations for auto-selection.
CANDIDATES = [
    (OCC_PLAIN, METHOD_QM),
    (OCC_PLAIN, METHOD_QDM),
    (OCC_TREND, METHOD_QM),
    (OCC_TREND, METHOD_QDM),
]


@dataclass
class CorrectionConfig:
    season_mode: str = "seasonal"          # "annual" | "seasonal"
    occ_method: str = OCC_TREND             # or "auto"
    amount_method: str = METHOD_QDM         # or "auto"
    occ_mode: str = OCC_DETERMINISTIC       # deterministic | stochastic
    tail: str = TAIL_EXTRAPOLATE
    tail_quantile: float = 0.85
    train_years: int = 6
    val_years: int = 3
    test_years: int = 3
    stride: int = 1
    seed: Optional[int] = None


@dataclass
class SeasonFit:
    label: str
    occurrence: OccurrenceCorrector
    amount: AmountCorrector
    n_train: int


@dataclass
class Calibration:
    calibration_id: str
    created_at: float
    obs_revision: str
    obs_fingerprint: str
    config: Dict
    train_year_range: List[int]
    selected_methods: Dict[str, List[str]]   # season -> [occ_method, amount_method]
    stochastic: Dict = field(default_factory=dict)  # season -> persisted RNG info
    fits: Dict[str, SeasonFit] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable fingerprint of everything that defines the mapping."""
        payload = {
            "id": self.calibration_id,
            "obs_revision": self.obs_revision,
            "obs_fingerprint": self.obs_fingerprint,
            "config": self.config,
            "methods": self.selected_methods,
            "stochastic": {k: _jsonable(v) for k, v in self.stochastic.items()},
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _fingerprint_series(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype=np.float64, order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


class CorrectionPipeline:
    def __init__(self, config: Optional[CorrectionConfig] = None):
        self.config = config or CorrectionConfig()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _season_keys(self, df: pd.DataFrame) -> np.ndarray:
        return season_key(df["month"].to_numpy(), self.config.season_mode)

    def _fit_season(self, hind, obs, occ_method, amount_method) -> SeasonFit:
        occ = OccurrenceCorrector(occ_method).fit(hind, obs)
        amt = AmountCorrector(amount_method, self.config.tail,
                              self.config.tail_quantile).fit(hind, obs)
        return SeasonFit(label="", occurrence=occ, amount=amt, n_train=hind.size)

    def _apply_season(self, future, fit: SeasonFit, occ_mode, seed,
                      fixed_uniforms):
        occ_res: OccurrenceResult = fit.occurrence.apply(
            future, mode=occ_mode, seed=seed, fixed_uniforms=fixed_uniforms)
        # Amount input: keep raw value on occurrence-wet days, zero elsewhere.
        amount_input = np.where(occ_res.wet, future, 0.0)
        # A wet decision on an exact zero gets the smallest positive future value.
        wet_zero = occ_res.wet & (amount_input <= 0.0)
        if np.any(wet_zero):
            positive = future[future > 0.0]
            fill = positive.min() if positive.size else 1e-6
            amount_input[wet_zero] = fill
        amt_res: AmountResult = fit.amount.apply(amount_input)
        corrected = amt_res.corrected
        # Guarantee occurrence-dry days are exactly zero.
        corrected = np.where(occ_res.wet, corrected, 0.0)
        return corrected, occ_res, amt_res

    # ------------------------------------------------------------------ #
    # Cross-validation with in-fold method selection
    # ------------------------------------------------------------------ #
    def run_cross_validation(self, history: pd.DataFrame) -> Dict:
        cfg = self.config
        years = history["year"].to_numpy()
        plan: SplitPlan = contiguous_year_splits(
            years, cfg.train_years, cfg.val_years, cfg.test_years, cfg.stride)
        skeys = self._season_keys(history)
        hind_all = history["hindcast"].to_numpy(float)
        obs_all = history["obs"].to_numpy(float)

        fold_reports = []
        for split in plan.folds:
            tr_m = years_mask(years, split.train_years)
            va_m = years_mask(years, split.val_years)
            te_m = years_mask(years, split.test_years)
            season_report = {}
            for label in season_labels(cfg.season_mode):
                sm = skeys == label
                season_report[label] = self._cv_season(
                    label, hind_all, obs_all, tr_m & sm, va_m & sm, te_m & sm)
            fold_reports.append({
                "fold": split.fold,
                "train_years": list(split.train_years),
                "val_years": list(split.val_years),
                "test_years": list(split.test_years),
                "seasons": season_report,
            })
        return {"plan": plan, "folds": fold_reports,
                "recommendation": self._recommend(fold_reports)}

    def _cv_season(self, label, hind_all, obs_all, tr_m, va_m, te_m) -> Dict:
        # Guard: a seasonal stratum can in principle leave a role block empty.
        if not (tr_m.any() and va_m.any() and te_m.any()):
            return {
                "label": label,
                "candidates": [], "selected": [OCC_TREND, METHOD_QDM],
                "test_metrics": None, "trend_diagnostic": None,
                "n_train": int(tr_m.sum()), "n_val": int(va_m.sum()),
                "n_test": int(te_m.sum()), "empty": True,
            }
        candidate_scores = []
        for occ_method, amount_method in CANDIDATES:
            fit = self._fit_season(hind_all[tr_m], obs_all[tr_m],
                                   occ_method, amount_method)
            # Selection signal comes ONLY from the validation block.
            val_corr, _, _ = self._apply_season(
                hind_all[va_m], fit, OCC_DETERMINISTIC, None, None)
            val_wet = (val_corr > 0).astype(float)
            brier = metrics.brier_score(val_wet, obs_all[va_m])
            amt_mae = metrics.wet_amount_mae(val_corr, obs_all[va_m])
            # Normalised composite (MAE scaled by the validation mean wet amount).
            scale = max(float(np.mean(obs_all[va_m])), 1e-9)
            score = brier + amt_mae / scale
            candidate_scores.append({
                "occ_method": occ_method, "amount_method": amount_method,
                "brier": brier, "amount_mae": amt_mae, "score": score,
            })
        chosen = min(candidate_scores, key=lambda c: c["score"])

        # Refit on train with the selected method; evaluate ONLY on test.
        final_fit = self._fit_season(
            hind_all[tr_m], obs_all[tr_m],
            chosen["occ_method"], chosen["amount_method"])
        test_corr, _, _ = self._apply_season(
            hind_all[te_m], final_fit, OCC_DETERMINISTIC, None, None)
        test_summary = metrics.full_summary(
            hind_all[te_m], test_corr, obs_all[te_m])
        return {
            "label": label,
            "candidates": candidate_scores,
            "selected": [chosen["occ_method"], chosen["amount_method"]],
            "test_metrics": test_summary,
            "n_train": int(tr_m.sum()), "n_val": int(va_m.sum()),
            "n_test": int(te_m.sum()),
        }

    def _recommend(self, fold_reports) -> Dict[str, List[str]]:
        votes: Dict[str, Dict[tuple, int]] = {}
        for fold in fold_reports:
            for label, rep in fold["seasons"].items():
                votes.setdefault(label, {})
                key = tuple(rep["selected"])
                votes[label][key] = votes[label].get(key, 0) + 1
        out = {}
        for label, tally in votes.items():
            best = max(tally.items(), key=lambda kv: kv[1])[0]
            out[label] = list(best)
        return out

    # ------------------------------------------------------------------ #
    # Final production calibration on all history years
    # ------------------------------------------------------------------ #
    def fit_final(self, history: pd.DataFrame, obs_revision: str,
                  recommendation: Optional[Dict[str, List[str]]] = None
                  ) -> Calibration:
        cfg = self.config
        skeys = self._season_keys(history)
        hind_all = history["hindcast"].to_numpy(float)
        obs_all = history["obs"].to_numpy(float)
        years = history["year"].to_numpy()
        created = time.time()
        cal_id = "cal-" + time.strftime("%Y%m%d%H%M%S") + "-" + \
            hashlib.sha256(str(created).encode()).hexdigest()[:6]

        fits: Dict[str, SeasonFit] = {}
        selected: Dict[str, List[str]] = {}
        stochastic: Dict[str, Dict] = {}
        for label in season_labels(cfg.season_mode):
            sm = skeys == label
            if recommendation and label in recommendation and \
                    cfg.occ_method == "auto" and cfg.amount_method == "auto":
                occ_method, amount_method = recommendation[label]
            else:
                occ_method = (recommendation[label][0]
                              if cfg.occ_method == "auto" and recommendation
                              else (cfg.occ_method if cfg.occ_method != "auto"
                                    else OCC_TREND))
                amount_method = (recommendation[label][1]
                                 if cfg.amount_method == "auto" and recommendation
                                 else (cfg.amount_method
                                       if cfg.amount_method != "auto" else METHOD_QDM))
            fit = self._fit_season(hind_all[sm], obs_all[sm],
                                   occ_method, amount_method)
            fit.label = label
            fits[label] = fit
            selected[label] = [occ_method, amount_method]

        cal = Calibration(
            calibration_id=cal_id,
            created_at=created,
            obs_revision=obs_revision,
            obs_fingerprint=_fingerprint_series(obs_all),
            config=asdict(cfg),
            train_year_range=[int(np.min(years)), int(np.max(years))],
            selected_methods=selected,
            stochastic=stochastic,
            fits=fits,
        )
        return cal

    # ------------------------------------------------------------------ #
    # Apply a frozen calibration to the future block
    # ------------------------------------------------------------------ #
    def apply(self, future: pd.DataFrame, cal: Calibration,
              occ_mode: Optional[str] = None,
              seed: Optional[int] = None) -> "CorrectedProduct":
        mode = occ_mode or self.config.occ_mode
        n = len(future)
        corrected = np.zeros(n)
        skeys = self._season_keys(future)
        raw = future["forecast"].to_numpy(float)
        trace = {}
        extrapolated = np.zeros(n, dtype=bool)
        for label, fit in cal.fits.items():
            idx = np.where(skeys == label)[0]
            if idx.size == 0:
                continue
            fixed_uniforms = None
            use_seed = seed
            if mode == OCC_STOCHASTIC:
                persisted = cal.stochastic.get(label)
                if persisted is not None:
                    # Reuse the exact same draws -> deterministic re-application.
                    fixed_uniforms = np.asarray(persisted["uniforms"])
                    if fixed_uniforms.size != idx.size:
                        fixed_uniforms = None  # block changed; draw fresh below
                if fixed_uniforms is None:
                    use_seed = self.config.seed if use_seed is None else use_seed
            corr, occ_res, amt_res = self._apply_season(
                raw[idx], fit, mode, use_seed, fixed_uniforms)
            corrected[idx] = corr
            extrapolated[idx[amt_res.wet_mask]] = amt_res.extrapolated

            if mode == OCC_STOCHASTIC and label not in cal.stochastic:
                cal.stochastic[label] = {
                    "seed": occ_res.seed,
                    "rng_state": occ_res.rng_state,
                    "uniforms": np.asarray(occ_res.randomized_uniforms),
                }
            trace[label] = {"occurrence": occ_res, "amount": amt_res,
                            "row_indices": idx}

        product = CorrectedProduct(
            calibration_id=cal.calibration_id,
            calibration_fingerprint=cal.fingerprint(),
            obs_revision=cal.obs_revision,
            obs_fingerprint=cal.obs_fingerprint,
            raw=raw,
            corrected=corrected,
            extrapolated=extrapolated,
            occ_mode=mode,
            applied_at=time.time(),
            future=future,
            trace=trace,
        )
        return product

    # ------------------------------------------------------------------ #
    # Trace for a single forecast day
    # ------------------------------------------------------------------ #
    @staticmethod
    def explain_row(product: "CorrectedProduct", row_index: int) -> Dict:
        label = next(lab for lab, tr in product.trace.items()
                     if row_index in set(tr["row_indices"].tolist()))
        tr = product.trace[label]
        occ: OccurrenceResult = tr["occurrence"]
        amt: AmountResult = tr["amount"]
        local = int(np.where(tr["row_indices"] == row_index)[0][0])
        out = {
            "row_index": row_index,
            "season": label,
            "raw_value": float(product.raw[row_index]),
            "corrected_value": float(product.corrected[row_index]),
            "occurrence": occ.records[local],
        }
        # Amount records only exist for occurrence-wet days; map local wet ordinal.
        if occ.wet[local]:
            wet_ordinal = int(np.sum(occ.wet[:local]))
            out["amount"] = amt.records[wet_ordinal]
        else:
            out["amount"] = None
        return out


@dataclass
class CorrectedProduct:
    calibration_id: str
    calibration_fingerprint: str
    obs_revision: str
    obs_fingerprint: str
    raw: np.ndarray
    corrected: np.ndarray
    extrapolated: np.ndarray
    occ_mode: str
    applied_at: float
    future: pd.DataFrame
    trace: Dict
