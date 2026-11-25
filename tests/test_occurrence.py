"""Occurrence stage: zero mass, trend preservation, RNG persistence."""
import numpy as np

from app.engine.occurrence import (OccurrenceCorrector, OCC_PLAIN, OCC_TREND,
                                   OCC_DETERMINISTIC, OCC_STOCHASTIC)


def test_plain_matches_observed_frequency():
    rng = np.random.default_rng(0)
    obs = (rng.random(2000) < 0.30).astype(float) * rng.gamma(1, 4, 2000)
    hind = (rng.random(2000) < 0.55).astype(float) * rng.gamma(1, 4, 2000)
    future = (rng.random(1000) < 0.60).astype(float) * rng.gamma(1, 4, 1000)
    corr = OccurrenceCorrector(OCC_PLAIN).fit(hind, obs)
    res = corr.apply(future, mode=OCC_DETERMINISTIC)
    # Plain forces onto the observed historical wet frequency regardless of future.
    assert abs(res.realized_frequency - 0.30) < 0.02
    assert res.rng_state is None and res.seed is None


def test_trend_preserves_frequency_change_signal():
    rng = np.random.default_rng(1)
    obs = (rng.random(4000) < 0.30).astype(float)
    hind = (rng.random(4000) < 0.40).astype(float)
    future_drier = (rng.random(2000) < 0.20).astype(float)
    future_wetter = (rng.random(2000) < 0.70).astype(float)
    corr = OccurrenceCorrector(OCC_TREND).fit(hind, obs)
    f_drier = corr.fit_.target_frequency(0.20)
    f_wetter = corr.fit_.target_frequency(0.70)
    # Ordering preserved: a wetter future maps to a higher corrected frequency.
    assert f_drier < f_wetter
    # And it does not simply collapse both to the observed 0.30.
    assert f_drier < 0.30 < f_wetter


def test_stochastic_persists_rng_and_is_reproducible():
    rng = np.random.default_rng(2)
    obs = (rng.random(2000) < 0.35).astype(float) * rng.gamma(1, 4, 2000)
    hind = (rng.random(2000) < 0.5).astype(float) * rng.gamma(1, 4, 2000)
    future = rng.gamma(1, 4, 1500)
    corr = OccurrenceCorrector(OCC_TREND).fit(hind, obs)
    r1 = corr.apply(future, mode=OCC_STOCHASTIC, seed=12345)
    r2 = corr.apply(future, mode=OCC_STOCHASTIC, seed=12345)
    # Seed + full generator state + every draw are persisted ...
    assert r1.seed == 12345
    assert r1.rng_state is not None
    assert r1.randomized_uniforms is not None
    assert r1.randomized_uniforms.shape == future.shape
    # ... so re-running with the same persisted draws reproduces the decisions.
    r3 = corr.apply(future, mode=OCC_STOCHASTIC,
                    fixed_uniforms=r1.randomized_uniforms)
    assert np.array_equal(r1.wet, r3.wet)
    assert np.allclose(r1.randomized_uniforms, r3.randomized_uniforms)


def test_no_silent_point_deletion_frequency_near_target():
    # The wet frequency must come from the randomised quantile mechanism with a
    # stored state, never from ad-hoc dropping; realized ~ target within MC error.
    rng = np.random.default_rng(3)
    obs = (rng.random(3000) < 0.4).astype(float)
    hind = (rng.random(3000) < 0.4).astype(float)
    future = rng.gamma(1, 3, 3000)
    corr = OccurrenceCorrector(OCC_PLAIN).fit(hind, obs)
    res = corr.apply(future, mode=OCC_STOCHASTIC, seed=7)
    assert abs(res.realized_frequency - res.target_frequency) < 0.03
