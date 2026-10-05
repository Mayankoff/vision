import numpy as np
import pytest

from rppg.signal.preprocess import bandpass, detrend, fill_gaps, postprocess_pulse, resample_uniform, spatial_mean

FS = 30.0


def test_spatial_mean_uses_only_masked_pixels():
    frame = np.zeros((4, 4, 3), np.uint8)
    frame[:2] = (10, 20, 30)
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    np.testing.assert_allclose(spatial_mean(frame, mask), [10, 20, 30])
    assert np.isnan(spatial_mean(frame, np.zeros((4, 4), bool))).all()


def test_fill_gaps_interpolates_nans():
    x = np.array([[0.0], [np.nan], [2.0], [np.nan]])
    np.testing.assert_allclose(fill_gaps(x).ravel(), [0, 1, 2, 2])
    with pytest.raises(ValueError):
        fill_gaps(np.full(3, np.nan))


def test_resample_uniform_onto_regular_grid():
    t = np.array([0.0, 0.03, 0.07, 0.10])
    tn, x = resample_uniform(t, 10 * t, 50.0)
    np.testing.assert_allclose(np.diff(tn), 0.02)
    np.testing.assert_allclose(x, 10 * tn)


def test_detrend_removes_slow_drift_keeps_pulse():
    t = np.arange(900) / FS
    pulse = np.sin(2 * np.pi * 1.2 * t)
    drift = 5 * t / t[-1] + 2 * np.sin(2 * np.pi * 0.02 * t)
    y = detrend(pulse + drift)
    assert np.corrcoef(y, pulse)[0, 1] > 0.98
    assert abs(y.mean()) < 0.1


def test_bandpass_rejects_out_of_band():
    t = np.arange(900) / FS
    inband, low, high = np.sin(2 * np.pi * 1.2 * t), np.sin(2 * np.pi * 0.1 * t), np.sin(2 * np.pi * 8 * t)
    y = bandpass(inband + low + high, FS, (0.7, 3.0))
    core = slice(100, -100)  # ignore filter edge effects
    assert np.corrcoef(y[core], inband[core])[0, 1] > 0.99


def test_postprocess_is_unit_variance():
    rng = np.random.default_rng(0)
    y = postprocess_pulse(rng.standard_normal(600) + np.sin(np.arange(600) / 5), FS)
    assert abs(y.std() - 1) < 1e-9 and abs(y.mean()) < 1e-9
