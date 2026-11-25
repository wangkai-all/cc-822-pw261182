"""Calibration versioning, observation revision binding and CSV round-trip."""
import os

import numpy as np
import pandas as pd

from app.engine.pipeline import CorrectionPipeline, CorrectionConfig
from app.engine.storage import (CalibrationStore, export_product,
                                check_revision_binding, import_history,
                                import_future, series_fingerprint)


def _pipe():
    return CorrectionPipeline(CorrectionConfig(
        occ_method="trend", amount_method="qdm",
        train_years=6, val_years=3, test_years=3))


def test_calibration_roundtrip_is_identical(tmp_path, history, future):
    store = CalibrationStore(str(tmp_path / "store"))
    pipe = _pipe()
    cal = pipe.fit_final(history, obs_revision="rev-A")
    store.save(cal)
    loaded = store.load(cal.calibration_id)
    assert loaded.fingerprint() == cal.fingerprint()
    p1 = pipe.apply(future, cal, occ_mode="stochastic", seed=99)
    p2 = pipe.apply(future, loaded, occ_mode="stochastic", seed=99)
    assert np.allclose(p1.corrected, p2.corrected)


def test_revision_binding_goes_stale_after_obs_change(tmp_path, history, future):
    pipe = _pipe()
    cal = pipe.fit_final(history, obs_revision="rev-A")
    product = pipe.apply(future, cal)
    # Same observation -> current.
    status = check_revision_binding(product, history["obs"].to_numpy())
    assert status["status"] == "current"
    # Revised observation -> stale, but product still names the original calibration.
    revised = history["obs"].to_numpy().copy()
    revised[revised > 0] *= 1.1
    status2 = check_revision_binding(product, revised)
    assert status2["status"] == "stale"
    assert status2["bound_calibration_id"] == cal.calibration_id


def test_old_product_keeps_original_distribution_after_revision(tmp_path,
                                                                history, future):
    pipe = _pipe()
    cal_a = pipe.fit_final(history, obs_revision="rev-A")
    product_a = pipe.apply(future, cal_a)
    # Build a new calibration from revised observations.
    hist_b = history.copy()
    hist_b["obs"] = hist_b["obs"] * 1.2 + 0.01
    cal_b = pipe.fit_final(hist_b, obs_revision="rev-B")
    product_b = pipe.apply(future, cal_b)
    # The old product's numbers are unchanged and remain tagged to rev-A.
    assert product_a.obs_revision == "rev-A"
    assert product_b.obs_revision == "rev-B"
    assert not np.allclose(product_a.corrected, product_b.corrected)


def test_csv_import_export_roundtrip(tmp_path, history, future):
    hp = tmp_path / "history.csv"
    fp = tmp_path / "future.csv"
    history[["date", "obs", "hindcast"]].to_csv(hp, index=False)
    future[["date", "forecast"]].to_csv(fp, index=False)
    h2 = import_history(str(hp))
    f2 = import_future(str(fp))
    assert len(h2) == len(history) and "year" in h2.columns
    assert len(f2) == len(future)

    pipe = _pipe()
    cal = pipe.fit_final(h2, obs_revision="rev-A")
    product = pipe.apply(f2, cal)
    out = export_product(product, str(tmp_path / "product.csv"))
    df = pd.read_csv(out)
    assert {"forecast_raw", "corrected", "calibration_id",
            "obs_revision", "extrapolated_beyond_training"} <= set(df.columns)
    assert np.allclose(df["forecast_raw"], product.raw)


def test_import_rejects_negative_and_missing_column(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2000-01-01"], "obs": [-1.0],
                  "hindcast": [0.2]}).to_csv(bad, index=False)
    try:
        import_history(str(bad))
        assert False, "expected ValueError"
    except ValueError:
        pass
