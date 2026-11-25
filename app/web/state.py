"""In-process application state and the service layer backing the web UI.

A single :class:`AppState` holds the loaded dataset, cross-validation result, the
frozen calibration(s) and the corrected products.  The state is intentionally
explicit and serialisable-friendly rather than hidden in globals, so the same
service functions can be driven from the Flask routes or a script.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..engine import (CorrectionPipeline, CorrectionConfig,
                      generate_synthetic_dataset, SyntheticConfig, metrics)
from ..engine.pipeline import Calibration, CorrectedProduct, CANDIDATES
from ..engine.synthetic import dataset_manifest
from ..engine.storage import (CalibrationStore, export_product,
                              check_revision_binding, series_fingerprint,
                              import_history, import_future)


@dataclass
class AppState:
    history: Optional[pd.DataFrame] = None
    future: Optional[pd.DataFrame] = None
    manifest: Optional[Dict] = None
    cv_result: Optional[Dict] = None
    calibrations: Dict[str, Calibration] = field(default_factory=dict)
    products: Dict[str, CorrectedProduct] = field(default_factory=dict)
    # Observation bookkeeping for revision binding.
    obs_revision: str = "rev-1"
    source: str = "none"
    lock: threading.RLock = field(default_factory=threading.RLock)

    def require_data(self):
        if self.history is None or self.future is None:
            raise RuntimeError("尚未导入数据")


# Module-level singleton used by the Flask app.
STATE = AppState()
STORE: Optional[CalibrationStore] = None


def init_store(root: str) -> CalibrationStore:
    global STORE
    STORE = CalibrationStore(root)
    return STORE


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_synthetic(config: Optional[SyntheticConfig] = None) -> Dict:
    with STATE.lock:
        ds = generate_synthetic_dataset(config or SyntheticConfig())
        STATE.history = ds["history"]
        STATE.future = ds["future"]
        STATE.manifest = dataset_manifest(ds)
        STATE.cv_result = None
        STATE.calibrations.clear()
        STATE.products.clear()
        STATE.obs_revision = "rev-synthetic-1"
        STATE.source = "synthetic"
        return STATE.manifest


def load_imported(history_path: str, future_path: str,
                  revision: str = "rev-import-1") -> Dict:
    with STATE.lock:
        hist = import_history(history_path)
        fut = import_future(future_path)
        STATE.history = hist
        STATE.future = fut
        STATE.manifest = {
            "history_years": [int(hist["year"].min()), int(hist["year"].max())],
            "future_years": [int(fut["year"].min()), int(fut["year"].max())],
            "history_wet_freq_obs": float(np.mean(hist["obs"] > 0)),
            "history_wet_freq_hindcast": float(np.mean(hist["hindcast"] > 0)),
            "known_bias": None,
            "known_trend": None,
        }
        STATE.cv_result = None
        STATE.calibrations.clear()
        STATE.products.clear()
        STATE.obs_revision = revision
        STATE.source = "imported"
        return STATE.manifest


def revise_observations(multiplier: float = 1.1,
                        addend: float = 0.0) -> str:
    """Simulate an observation revision; old products keep their old binding."""
    with STATE.lock:
        STATE.require_data()
        new_rev = STATE.obs_revision + "-r"
        hist = STATE.history.copy()
        hist["obs"] = np.maximum(0.0, hist["obs"] * multiplier + addend)
        STATE.history = hist
        STATE.obs_revision = new_rev
        STATE.cv_result = None
        return new_rev


# --------------------------------------------------------------------------- #
# Cross-validation and calibration
# --------------------------------------------------------------------------- #
def run_cv(config: CorrectionConfig) -> Dict:
    with STATE.lock:
        STATE.require_data()
        pipe = CorrectionPipeline(config)
        cv = pipe.run_cross_validation(STATE.history)
        STATE.cv_result = {"config": config, **cv}
        return serialize_cv(STATE.cv_result)


def _season_subset(df: pd.DataFrame, season_mode: str, season: str) -> pd.DataFrame:
    if season_mode == "annual" or season in (None, "ANNUAL", "all"):
        return df
    from ..engine.seasons import season_key
    keys = season_key(df["month"].to_numpy(), season_mode)
    return df[keys == season]


def calibrate_and_apply(config: CorrectionConfig, occ_mode: str,
                        tail: str, seed: Optional[int],
                        use_recommendation: bool) -> Dict:
    with STATE.lock:
        STATE.require_data()
        config.tail = tail
        pipe = CorrectionPipeline(config)
        recommendation = None
        if use_recommendation or config.occ_method == "auto" \
                or config.amount_method == "auto":
            cv = pipe.run_cross_validation(STATE.history)
            recommendation = cv["recommendation"]
        cal = pipe.fit_final(STATE.history, STATE.obs_revision, recommendation)
        STATE.calibrations[cal.calibration_id] = cal
        if STORE is not None:
            STORE.save(cal)
        product = pipe.apply(STATE.future, cal, occ_mode=occ_mode, seed=seed)
        STATE.products[cal.calibration_id] = product
        return {
            "calibration_id": cal.calibration_id,
            "fingerprint": cal.fingerprint(),
            "obs_revision": cal.obs_revision,
            "selected_methods": cal.selected_methods,
            "product_summary": summarize_product(product),
        }


def compare_methods(tail: str, occ_mode: str, season: str,
                    config: CorrectionConfig) -> Dict:
    """Run both transfer families side by side for the comparison dashboard."""
    with STATE.lock:
        STATE.require_data()
        fut = _season_subset(STATE.future, config.season_mode, season)
        hist = _season_subset(STATE.history, config.season_mode, season)
        results = {}
        for occ_m, amt_m, name in [
            ("plain", "qm", "plain_qm"),
            ("trend", "qdm", "trend_qdm"),
        ]:
            # The frames are already sliced to one season, so fit a single
            # (annual) stratum to avoid empty-season fits inside fit_final.
            cfg = CorrectionConfig(
                season_mode="annual", occ_method=occ_m,
                amount_method=amt_m, occ_mode=occ_mode, tail=tail,
                tail_quantile=config.tail_quantile,
                train_years=config.train_years, val_years=config.val_years,
                test_years=config.test_years, stride=config.stride,
                seed=config.seed)
            pipe = CorrectionPipeline(cfg)
            cal = pipe.fit_final(hist, STATE.obs_revision)
            product = pipe.apply(fut, cal, occ_mode=occ_mode)
            results[name] = _product_series(product, fut)
        truth = fut["future_truth_hidden"].to_numpy() \
            if "future_truth_hidden" in fut.columns else None
        history_wf = metrics.wet_frequency(hist["obs"].to_numpy())
        return build_comparison(results, truth, tail, season, history_wf)


def _product_series(product: CorrectedProduct, fut: pd.DataFrame) -> Dict:
    return {
        "raw": product.raw,
        "corrected": product.corrected,
        "extrapolated": product.extrapolated,
        "dates": fut["date"].dt.strftime("%Y-%m-%d").to_numpy(),
    }


def summarize_product(product: CorrectedProduct) -> Dict:
    fut = product.future
    truth = fut["future_truth_hidden"].to_numpy() \
        if "future_truth_hidden" in fut.columns else None
    summary = metrics.full_summary(product.raw, product.corrected,
                                   truth if truth is not None
                                   else np.zeros_like(product.raw))
    summary["n_extrapolated"] = int(product.extrapolated.sum())
    summary["binding"] = {
        "calibration_id": product.calibration_id,
        "calibration_fingerprint": product.calibration_fingerprint,
        "obs_revision": product.obs_revision,
        "obs_fingerprint": product.obs_fingerprint,
    }
    return summary


# --------------------------------------------------------------------------- #
# Serialisation helpers for JSON responses
# --------------------------------------------------------------------------- #
def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    return obj


def serialize_cv(cv: Dict) -> Dict:
    folds = []
    for fold in cv["folds"]:
        seasons = {}
        for label, rep in fold["seasons"].items():
            seasons[label] = {
                "selected": rep["selected"],
                "candidates": rep["candidates"],
                "test_metrics": rep["test_metrics"],
                "sizes": [rep["n_train"], rep["n_val"], rep["n_test"]],
            }
        return_fold = {
            "fold": fold["fold"],
            "train_years": fold["train_years"],
            "val_years": fold["val_years"],
            "test_years": fold["test_years"],
            "seasons": seasons,
        }
        folds.append(return_fold)
    return _to_jsonable({
        "plan": cv["plan"].describe(),
        "recommendation": cv["recommendation"],
        "folds": folds,
    })


def build_comparison(results: Dict, truth, tail: str, season: str,
                     history_wf: Optional[float] = None) -> Dict:
    probs = [0.5, 0.75, 0.9, 0.95, 0.99, 0.995]
    payload = {"season": season, "tail_mode": tail, "series": {}, "truth": None}
    for name, data in results.items():
        raw, corr = data["raw"], data["corrected"]
        series_payload = {
            "wet_frequency": {
                "raw": metrics.wet_frequency(raw),
                "corrected": metrics.wet_frequency(corr),
            },
            "quantiles": {
                "probs": probs,
                "raw": metrics.quantile_curve(raw, probs).tolist(),
                "corrected": metrics.quantile_curve(corr, probs).tolist(),
            },
            "tail_corrected": metrics.extreme_tail_summary(corr, probs[3:]),
            "n_extrapolated": int(np.sum(data["extrapolated"])),
            "dates": data["dates"].tolist(),
            "raw_series": raw.tolist(),
            "corrected_series": corr.tolist(),
        }
        # Trend retention relative to the historical observed wet frequency:
        # plain/QM collapses onto it (retention ~0); trend/QDM keeps the future
        # signal (retention ~1 toward the raw future shift).
        if history_wf is not None:
            raw_shift = metrics.wet_frequency(raw) - history_wf
            corr_shift = metrics.wet_frequency(corr) - history_wf
            series_payload["trend_retention"] = (
                float(np.clip(corr_shift / raw_shift, -2, 2))
                if abs(raw_shift) > 1e-9 else 0.0)
        payload["series"][name] = series_payload
    if truth is not None:
        payload["truth"] = {
            "wet_frequency": metrics.wet_frequency(truth),
            "quantiles": metrics.quantile_curve(truth, probs).tolist(),
            "tail": metrics.extreme_tail_summary(truth, probs[3:]),
        }
    return _to_jsonable(payload)


def explain(calibration_id: str, row_index: int) -> Dict:
    with STATE.lock:
        product = STATE.products[calibration_id]
        explanation = CorrectionPipeline.explain_row(product, row_index)
        date = product.future.iloc[row_index]["date"].strftime("%Y-%m-%d")
        explanation["date"] = date
        explanation["binding"] = check_revision_binding(
            product, STATE.history["obs"].to_numpy())
        return _to_jsonable(explanation)


def export(calibration_id: str, path: str) -> str:
    with STATE.lock:
        product = STATE.products[calibration_id]
        return export_product(product, path)


def list_calibrations() -> List[Dict]:
    with STATE.lock:
        out = []
        for cid, cal in STATE.calibrations.items():
            product = STATE.products.get(cid)
            binding = None
            if product is not None:
                binding = check_revision_binding(
                    product, STATE.history["obs"].to_numpy())
            out.append({
                "calibration_id": cid,
                "fingerprint": cal.fingerprint(),
                "obs_revision": cal.obs_revision,
                "selected_methods": cal.selected_methods,
                "created_at": cal.created_at,
                "binding": binding,
            })
        return _to_jsonable(out)
