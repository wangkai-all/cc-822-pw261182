"""Seasonal stratification.

Calibration distributions differ markedly by season, so every fit is done within a
seasonal stratum.  Two modes are supported: annual (no stratification) and
meteorological seasons (DJF / MAM / JJA / SON).  The user selects a season on the
page; the same key is used consistently for fitting and applying.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

SEASON_MODES = ("annual", "seasonal")

# Meteorological seasons by month.
_SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")


def season_key(months: np.ndarray, mode: str) -> np.ndarray:
    """Return a per-row season label array."""
    if mode == "annual":
        return np.full(np.asarray(months).shape, "ANNUAL", dtype=object)
    if mode != "seasonal":
        raise ValueError(f"unknown season mode {mode!r}")
    months = np.asarray(months)
    return np.array([_SEASON_BY_MONTH[int(m)] for m in months], dtype=object)


def season_labels(mode: str) -> List[str]:
    return ["ANNUAL"] if mode == "annual" else list(SEASON_ORDER)


def masks_for_season(keys: np.ndarray, label: str) -> np.ndarray:
    return np.asarray(keys) == label


def describe_season(label: str) -> str:
    names = {
        "ANNUAL": "全年（不分季）",
        "DJF": "冬季 DJF（12–2月）",
        "MAM": "春季 MAM（3–5月）",
        "JJA": "夏季 JJA（6–8月）",
        "SON": "秋季 SON（9–11月）",
    }
    return names.get(label, label)
