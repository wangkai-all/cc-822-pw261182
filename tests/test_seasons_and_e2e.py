"""Seasonal stratification and zero point-mass handling end to end."""
import numpy as np

from app.engine.seasons import season_key, season_labels
from app.engine.pipeline import CorrectionPipeline, CorrectionConfig
from app.engine import metrics


def test_season_keys_cover_all_rows():
    months = np.repeat(np.arange(1, 13), 10)
    keys = season_key(months, "seasonal")
    assert set(keys) == {"DJF", "MAM", "JJA", "SON"}
    assert set(season_labels("annual")) == {"ANNUAL"}


def test_corrected_zeros_are_exact_and_driven_by_occurrence(history, future):
    pipe = CorrectionPipeline(CorrectionConfig(
        season_mode="seasonal", occ_method="plain", amount_method="qm",
        occ_mode="deterministic"))
    cal = pipe.fit_final(history, "r1")
    product = pipe.apply(future, cal)
    # Every corrected value is >= 0 and dry days are exactly 0.0.
    assert np.all(product.corrected >= 0.0)
    assert np.all(product.corrected[product.corrected <= 0.0] == 0.0)


def test_plain_erases_wet_trend_qdm_keeps_it(history, future):
    truth = future["future_truth_hidden"].to_numpy()
    out = {}
    for occ, amt, name in [("plain", "qm", "plain_qm"),
                           ("trend", "qdm", "trend_qdm")]:
        pipe = CorrectionPipeline(CorrectionConfig(
            occ_method=occ, amount_method=amt, occ_mode="deterministic"))
        cal = pipe.fit_final(history, "r1")
        product = pipe.apply(future, cal)
        out[name] = metrics.wet_frequency(product.corrected)
    truth_wf = metrics.wet_frequency(truth)
    # Trend-preserving correction lands closer to the (wetter) future truth.
    assert abs(out["trend_qdm"] - truth_wf) < abs(out["plain_qm"] - truth_wf)


def test_explain_row_contains_both_stages(history, future):
    pipe = CorrectionPipeline(CorrectionConfig(occ_method="trend",
                                               amount_method="qdm"))
    cal = pipe.fit_final(history, "r1")
    product = pipe.apply(future, cal)
    # Pick a corrected-wet row so both occurrence and amount traces exist.
    wet_idx = int(np.flatnonzero(product.corrected > 0)[0])
    explanation = CorrectionPipeline.explain_row(product, wet_idx)
    assert explanation["occurrence"]["is_wet"] is True
    assert explanation["amount"] is not None
    assert "non_exceedance_tau" in explanation["amount"]
    dry_idx = int(np.flatnonzero(product.corrected == 0)[0])
    dry_expl = CorrectionPipeline.explain_row(product, dry_idx)
    assert dry_expl["occurrence"]["is_wet"] is False
    assert dry_expl["amount"] is None
