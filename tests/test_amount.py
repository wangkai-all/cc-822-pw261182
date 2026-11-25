"""Quantile mapping vs trend-preserving delta mapping behaviour."""
import numpy as np

from app.engine.amount import AmountCorrector, METHOD_QM, METHOD_QDM
from app.engine.distributions import EmpiricalDist


def test_qm_pulls_marginal_to_observed():
    rng = np.random.default_rng(1)
    obs = rng.gamma(0.9, 6.0, 4000)
    hind = 1.3 * rng.gamma(0.9, 6.0, 4000)
    future = 1.3 * rng.gamma(0.9, 6.0, 4000)
    qm = AmountCorrector(METHOD_QM, tail="clip").fit(hind, obs)
    out = qm.apply(future)
    # Median of corrected should track the observed median, not the raw one.
    assert abs(np.median(out.corrected) - np.median(obs)) < \
        abs(np.median(future) - np.median(obs))


def test_qdm_preserves_relative_shift_qm_does_not():
    rng = np.random.default_rng(2)
    n = 4000
    obs = rng.gamma(1.0, 5.0, n)
    hind = rng.gamma(1.0, 5.0, n)
    shift = 1.4
    future = shift * rng.gamma(1.0, 5.0, n)   # genuine future increase
    qm = AmountCorrector(METHOD_QM, tail="extrapolate").fit(hind, obs)
    qdm = AmountCorrector(METHOD_QDM, tail="extrapolate").fit(hind, obs)
    out_qm = qm.apply(future).corrected
    out_qdm = qdm.apply(future).corrected
    # QDM keeps more of the upward shift than QM (which erases it).
    assert np.mean(out_qdm) > np.mean(out_qm)
    # QM mean collapses back toward the observed historical mean.
    assert abs(np.mean(out_qm) - np.mean(obs)) < abs(np.mean(future) - np.mean(obs))


def test_empirical_quantile_cdf_roundtrip():
    rng = np.random.default_rng(3)
    x = rng.gamma(1.0, 4.0, 1000)
    d = EmpiricalDist(x, wet_frequency=1.0)
    probs = np.linspace(0.05, 0.95, 50)
    q = d.quantile_wet(probs)
    back = d.cdf_wet(q)
    assert np.allclose(back, probs, atol=2e-3)


def test_weibull_max_below_one_leaves_tail_region():
    d = EmpiricalDist(np.arange(1.0, 101.0), wet_frequency=1.0)
    assert d.cdf_wet(np.array([1e9]))[0] < 1.0
