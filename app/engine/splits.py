"""Contiguous-year isolation for training, method selection and evaluation.

Daily weather is strongly autocorrelated and carries inter-annual regime
structure, so a random or shuffled train/test split leaks information across the
boundary (the same wet spell or season appears on both sides).  Every split here
is built from **contiguous blocks of whole years**, and the three roles use three
disjoint, ordered blocks:

    |<-- train (fit distributions) -->|<-- val (select method) -->|<-- test (evaluate) -->|

Folds roll forward one year at a time.  A test block is never used to fit a
distribution or to choose a method; it only reports out-of-sample performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class YearSplit:
    fold: int
    train_years: tuple
    val_years: tuple
    test_years: tuple

    def describe(self) -> str:
        return (f"fold {self.fold}: train {_rng(self.train_years)} | "
                f"val {_rng(self.val_years)} | test {_rng(self.test_years)}")


def _rng(years: Sequence[int]) -> str:
    if not years:
        return "—"
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}–{years[-1]}"


@dataclass
class SplitPlan:
    train_years: int
    val_years: int
    test_years: int
    stride: int
    folds: List[YearSplit] = field(default_factory=list)

    def describe(self) -> List[str]:
        return [f.describe() for f in self.folds]


def contiguous_year_splits(years, train_years: int, val_years: int,
                           test_years: int, stride: int = 1) -> SplitPlan:
    """Build rolling contiguous-year folds over the sorted unique ``years``."""
    if min(train_years, val_years, test_years) < 1:
        raise ValueError("train/val/test block sizes must each be >= 1 year")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    unique = sorted(int(y) for y in np.unique(np.asarray(years)))
    window = train_years + val_years + test_years
    if len(unique) < window:
        raise ValueError(
            f"need at least {window} contiguous years, found {len(unique)} "
            f"({unique[0]}–{unique[-1]})")

    folds: List[YearSplit] = []
    fold_id = 0
    start = 0
    while start + window <= len(unique):
        tr = unique[start:start + train_years]
        va = unique[start + train_years:start + train_years + val_years]
        te = unique[start + train_years + val_years:start + window]
        folds.append(YearSplit(fold_id, tuple(tr), tuple(va), tuple(te)))
        fold_id += 1
        start += stride

    return SplitPlan(train_years, val_years, test_years, stride, folds)


def years_mask(years: np.ndarray, block_years: Sequence[int]) -> np.ndarray:
    """Boolean mask selecting rows whose year is in ``block_years``."""
    years = np.asarray(years)
    block = set(int(y) for y in block_years)
    return np.array([int(y) in block for y in years])
