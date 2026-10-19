"""Filtering helpers for colour traces and pulse signals."""

from __future__ import annotations

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

from afscreen.config import SETTINGS


def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN samples (e.g. frames with no face) along axis 0."""
    x = np.array(x, dtype=np.float64, copy=True)
    flat = x.reshape(len(x), -1)
    idx = np.arange(len(x))
    for c in range(flat.shape[1]):
        bad = np.isnan(flat[:, c])
        if bad.all():
            raise ValueError("signal contains no valid samples")
        if bad.any():
            flat[bad, c] = np.interp(idx[bad], idx[~bad], flat[~bad, c])
    return flat.reshape(x.shape)


def detrend(x: np.ndarray, lam: float = SETTINGS.detrend_lambda) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen et al., 2002) along axis 0.

    Removes slow baseline drift (auto-exposure, slow lighting change) while keeping the pulse.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        return x - x.mean(axis=0)
    ones = np.ones(n)
    d2 = sparse.spdiags([ones, -2 * ones, ones], [0, 1, 2], n - 2, n)
    a = (sparse.identity(n) + lam**2 * (d2.T @ d2)).tocsc()
    trend = spsolve(a, x)
    return x - trend.reshape(x.shape)


def bandpass(x: np.ndarray, fs: float, band=SETTINGS.hr_band, order: int = SETTINGS.filter_order) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0."""
    sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
    x = np.asarray(x, dtype=np.float64)
    padlen = min(3 * (2 * len(sos) + 1), x.shape[0] - 1)
    return signal.sosfiltfilt(sos, x, axis=0, padlen=padlen)


def clean_pulse(x: np.ndarray, fs: float, band=SETTINGS.hr_band) -> np.ndarray:
    """Detrend + band-pass + zero mean / unit variance."""
    y = bandpass(detrend(x), fs, band)
    std = y.std()
    return (y - y.mean()) / std if std > 0 else y - y.mean()
