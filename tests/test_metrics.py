import numpy as np
import pytest

from rppg.eval.metrics import mae, mape, pearson, rmse, snr_db, summarize

FS = 30.0


def test_error_metrics_hand_computed():
    pred, gt = [70, 80, 95], [72, 80, 90]
    assert mae(pred, gt) == pytest.approx(7 / 3)
    assert rmse(pred, gt) == pytest.approx(np.sqrt(29 / 3))
    assert mape(pred, gt) == pytest.approx((2 / 72 + 0 + 5 / 90) / 3 * 100)
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_summarize_skips_nan_and_counts():
    s = summarize([70, np.nan, 90], [72, 80, 90], snr=[1.0, 2.0, np.nan])
    assert s["n"] == 2 and s["MAE"] == pytest.approx(1.0) and s["SNR"] == pytest.approx(1.5)


def test_snr_higher_for_cleaner_pulse():
    rng = np.random.default_rng(0)
    t = np.arange(900) / FS
    pulse = np.sin(2 * np.pi * 1.2 * t)
    clean = snr_db(pulse + 0.1 * rng.standard_normal(len(t)), FS, 72)
    noisy = snr_db(pulse + 2.0 * rng.standard_normal(len(t)), FS, 72)
    assert clean > 10 > noisy
    # A pulse at the wrong rate puts its power in the "noise" region.
    assert snr_db(np.sin(2 * np.pi * 2.0 * t), FS, 72) < -10
