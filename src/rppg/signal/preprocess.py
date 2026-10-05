"""Block 4 - preprocessing of RGB traces and pulse signals."""

from __future__ import annotations

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

from rppg.protocol import PROTOCOL


def spatial_mean(frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean R, G, B over the masked pixels of an (H, W, 3) frame. NaN if the mask is empty."""
    if mask is None:
        return frame.reshape(-1, 3).mean(axis=0, dtype=np.float64)
    if not mask.any():
        return np.full(3, np.nan)
    return frame[mask].mean(axis=0, dtype=np.float64)


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


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample samples taken at (possibly jittery) times ``t`` onto a uniform ``fs`` grid."""
    t = np.asarray(t, dtype=np.float64)
    t_new = np.arange(t[0], t[-1] + 1e-9, 1.0 / fs)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return t_new, np.interp(t_new, t, x)
    return t_new, np.stack([np.interp(t_new, t, x[:, c]) for c in range(x.shape[1])], axis=1)


def detrend(x: np.ndarray, lam: float = PROTOCOL.detrend_lambda) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen et al., 2002) along axis 0.

    Removes slow baseline drift (auto-exposure, slow lighting change) while
    keeping the pulse. Equivalent to the toolbox's ``detrend`` but solved
    with sparse matrices, so it is fast for long videos.
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


def bandpass(x: np.ndarray, fs: float, band=PROTOCOL.hr_band, order: int = PROTOCOL.filter_order) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0."""
    sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
    x = np.asarray(x, dtype=np.float64)
    padlen = min(3 * (2 * len(sos) + 1), x.shape[0] - 1)
    return signal.sosfiltfilt(sos, x, axis=0, padlen=padlen)


def temporal_normalize(rgb: np.ndarray) -> np.ndarray:
    """Divide each channel by its temporal mean (removes skin-tone / illuminant level)."""
    return rgb / rgb.mean(axis=0, keepdims=True)


def postprocess_pulse(bvp: np.ndarray, fs: float, band=PROTOCOL.hr_band) -> np.ndarray:
    """Block 6 - the common clean-up applied to every method's pulse output:
    detrend, band-pass, zero-mean / unit-variance."""
    y = bandpass(detrend(bvp), fs, band)
    std = y.std()
    return (y - y.mean()) / std if std > 0 else y - y.mean()
