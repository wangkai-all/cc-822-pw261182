"""Persistence: data import/export, calibration versioning and revision binding.

Design points required by the domain:

* Imported files are untrusted.  They live in their own directories and are read
  with strict column/type validation; nothing is imported for side effects.
* Every calibration is stored as an immutable, versioned artifact (engine object
  via pickle plus a JSON sidecar).  The sidecar records the observation revision
  id and a SHA-256 fingerprint of the observation array.
* A corrected product stores the calibration id *and* the observation fingerprint
  it was produced from.  If the observation series is later revised, the old
  product still resolves to its original calibration and the stale binding is
  reported rather than silently re-mapped through the new distribution.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REQUIRED_HISTORY = {"date", "obs", "hindcast"}
REQUIRED_FUTURE = {"date", "forecast"}


# --------------------------------------------------------------------------- #
# Import / validation
# --------------------------------------------------------------------------- #
def _read_csv(path: str, required: set) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} 缺少必需列: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for col in required - {"date"}:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if (df[col] < 0).any():
            raise ValueError(f"{os.path.basename(path)} 列 {col} 含负值（降水不可为负）")
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.day_of_year
    return df


def import_history(path: str) -> pd.DataFrame:
    df = _read_csv(path, REQUIRED_HISTORY)
    if not df["date"].is_unique:
        raise ValueError("历史数据日期重复")
    return df


def import_future(path: str) -> pd.DataFrame:
    df = _read_csv(path, REQUIRED_FUTURE)
    if not df["date"].is_unique:
        raise ValueError("待校正数据日期重复")
    return df


def series_fingerprint(values: np.ndarray) -> str:
    # Canonicalise dtype/order so the fingerprint is identical whether the input
    # is a raw DataFrame column or an array coerced during fitting.
    canonical = np.asarray(values, dtype=np.float64, order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def obs_revision_id(values: np.ndarray, label: Optional[str] = None) -> str:
    fp = series_fingerprint(values)
    return label or f"obs-{fp[:12]}"


# --------------------------------------------------------------------------- #
# Calibration store
# --------------------------------------------------------------------------- #
@dataclass
class StoredCalibration:
    calibration_id: str
    path_pkl: str
    meta: Dict


class CalibrationStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _paths(self, cal_id: str) -> Tuple[str, str]:
        d = os.path.join(self.root, cal_id)
        return os.path.join(d, "calibration.pkl"), os.path.join(d, "meta.json")

    def save(self, calibration) -> StoredCalibration:
        pkl_path, json_path = self._paths(calibration.calibration_id)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(calibration, fh)
        meta = {
            "calibration_id": calibration.calibration_id,
            "fingerprint": calibration.fingerprint(),
            "obs_revision": calibration.obs_revision,
            "obs_fingerprint": calibration.obs_fingerprint,
            "created_at": calibration.created_at,
            "train_year_range": calibration.train_year_range,
            "selected_methods": calibration.selected_methods,
            "config": calibration.config,
            "stochastic_seasons": list(calibration.stochastic.keys()),
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2, default=str)
        return StoredCalibration(calibration.calibration_id, pkl_path, meta)

    def load(self, cal_id: str):
        pkl_path, _ = self._paths(cal_id)
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    def list_calibrations(self) -> List[Dict]:
        out = []
        if not os.path.isdir(self.root):
            return out
        for name in sorted(os.listdir(self.root)):
            json_path = os.path.join(self.root, name, "meta.json")
            if os.path.exists(json_path):
                with open(json_path, encoding="utf-8") as fh:
                    out.append(json.load(fh))
        return out


# --------------------------------------------------------------------------- #
# Product export + revision binding check
# --------------------------------------------------------------------------- #
def export_product(product, path: str) -> str:
    """Write original vs corrected values with provenance columns."""
    fut = product.future.copy()
    out = pd.DataFrame({
        "date": fut["date"].values,
        "forecast_raw": product.raw,
        "corrected": product.corrected,
        "is_wet": product.corrected > 0.0,
        "extrapolated_beyond_training": product.extrapolated,
        "occurrence_mode": product.occ_mode,
        "calibration_id": product.calibration_id,
        "calibration_fingerprint": product.calibration_fingerprint,
        "obs_revision": product.obs_revision,
        "obs_fingerprint": product.obs_fingerprint,
    })
    out.to_csv(path, index=False)
    return path


def check_revision_binding(product, current_obs_values: np.ndarray) -> Dict:
    """Confirm a product still matches the *current* observation revision.

    Returns a status dict.  When the fingerprints differ the product is stale: it
    remains bound to its original calibration and must be re-generated on demand.
    """
    current_fp = series_fingerprint(current_obs_values)
    matches = current_fp == product.obs_fingerprint
    return {
        "status": "current" if matches else "stale",
        "product_obs_fingerprint": product.obs_fingerprint,
        "current_obs_fingerprint": current_fp,
        "bound_calibration_id": product.calibration_id,
        "message": ("产品与当前观测修订一致" if matches else
                    "观测已修订；旧产品仍绑定原标定分布，请按需用新观测重新标定"),
    }
