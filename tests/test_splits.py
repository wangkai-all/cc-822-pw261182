"""Contiguous-year isolation: no shuffled/overlapping leakage across roles."""
import numpy as np
import pytest

from app.engine.splits import contiguous_year_splits, years_mask


def _years(n=18, start=2000):
    return np.repeat(np.arange(start, start + n), 365)


def test_folds_are_contiguous_and_disjoint():
    years = _years()
    plan = contiguous_year_splits(years, 6, 3, 3, stride=1)
    assert len(plan.folds) == 18 - 12 + 1
    for fold in plan.folds:
        tr, va, te = fold.train_years, fold.val_years, fold.test_years
        # Each block is a contiguous run.
        for block in (tr, va, te):
            assert block == tuple(range(block[0], block[-1] + 1))
        # The three roles never share a year.
        assert not (set(tr) & set(va))
        assert not (set(va) & set(te))
        assert not (set(tr) & set(te))
        # Ordering: train immediately precedes val, which precedes test.
        assert tr[-1] + 1 == va[0] and va[-1] + 1 == te[0]


def test_stride_advances_window():
    years = _years()
    plan = contiguous_year_splits(years, 5, 2, 2, stride=2)
    starts = [f.train_years[0] for f in plan.folds]
    assert starts == sorted(starts)
    assert all(b - a == 2 for a, b in zip(starts, starts[1:]))


def test_insufficient_years_raises():
    with pytest.raises(ValueError):
        contiguous_year_splits(_years(8), 6, 3, 3)


def test_mask_selects_whole_years_only():
    years = _years(5)
    m = years_mask(years, [2001, 2002])
    assert set(np.unique(years[m])) == {2001, 2002}
    assert m.sum() == 2 * 365


def test_test_block_never_used_for_selection(history):
    # A high-level guarantee: every fold's reported test block is disjoint from
    # the train and val blocks driving the fit and the method choice.
    from app.engine.pipeline import CorrectionPipeline, CorrectionConfig
    pipe = CorrectionPipeline(CorrectionConfig(train_years=6, val_years=3,
                                               test_years=3))
    cv = pipe.run_cross_validation(history)
    for fold in cv["folds"]:
        blocks = (fold["train_years"], fold["val_years"], fold["test_years"])
        joined = [y for b in blocks for y in b]
        assert len(joined) == len(set(joined))
