"""Block 5A - classical rPPG methods.

Every method maps an RGB trace (T, 3) sampled at ``fs`` Hz to a raw pulse
signal (T,). The common post-processing (:func:`postprocess_pulse`) is applied
afterwards by the pipeline, identically for all methods.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import signal as sig
from scipy.ndimage import uniform_filter1d

from rppg.protocol import PROTOCOL
from rppg.signal.preprocess import bandpass, detrend


def green(rgb: np.ndarray, fs: float, band=PROTOCOL.hr_band) -> np.ndarray:
    """GREEN (Verkruysse et al., 2008): the green channel alone. Simplest baseline."""
    return rgb[:, 1] / rgb[:, 1].mean()


def pos(rgb: np.ndarray, fs: float, band=PROTOCOL.hr_band, win_s: float = PROTOCOL.projection_window_s) -> np.ndarray:
    """POS - Plane-Orthogonal-to-Skin (Wang et al., 2017).

    In each short window the temporally-normalised RGB is projected onto two
    axes orthogonal to the (1, 1, 1) intensity direction, then the two
    projections are combined with an alpha-tuning step that cancels the
    remaining distortion. Windows are overlap-added.
    """
    n = rgb.shape[0]
    win = math.ceil(win_s * fs)
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    h = np.zeros(n)
    for end in range(win, n + 1):
        start = end - win
        cn = rgb[start:end] / rgb[start:end].mean(axis=0)
        s = cn @ projection.T  # (win, 2)
        s1, s2 = s[:, 0], s[:, 1]
        std2 = s2.std()
        p = s1 + (s1.std() / std2 if std2 > 0 else 0.0) * s2
        h[start:end] += p - p.mean()
    return h


def chrom(rgb: np.ndarray, fs: float, band=PROTOCOL.hr_band, win_s: float = PROTOCOL.projection_window_s) -> np.ndarray:
    """CHROM - chrominance-based rPPG (de Haan & Jeanne, 2013).

    RGB is normalised by its local (moving) mean, two chrominance signals
    X = 3R - 2G and Y = 1.5R + G - 1.5B are band-passed, and they are combined
    as X - alpha * Y with alpha = std(X) / std(Y) computed per window;
    Hann-weighted windows with 50% overlap are overlap-added.

    Unlike rPPG-Toolbox's implementation, X and Y are band-passed over the
    whole signal rather than inside each 1.6 s window: filtering such short
    segments distorts pulses below ~60 BPM (the window then holds barely one
    beat), letting the 2nd harmonic win.
    """
    n = rgb.shape[0]
    win = math.ceil(win_s * fs)
    win += win % 2
    half = win // 2
    cn = rgb / uniform_filter1d(rgb, size=win, axis=0, mode="nearest")
    xf = bandpass(3 * cn[:, 0] - 2 * cn[:, 1], fs, band)
    yf = bandpass(1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2], fs, band)
    hann = sig.windows.hann(win, sym=False)  # periodic Hann: 50%-overlapped copies sum to 1
    out = np.zeros(n)
    starts = list(range(0, n - win + 1, half))
    if starts and starts[-1] + win < n:
        starts.append(n - win)  # cover the tail
    for start in starts:
        xw, yw = xf[start : start + win], yf[start : start + win]
        std_y = yw.std()
        alpha = xw.std() / std_y if std_y > 0 else 0.0
        out[start : start + win] += (xw - alpha * yw) * hann
    return out


def ica(rgb: np.ndarray, fs: float, band=PROTOCOL.hr_band, seed: int = 0) -> np.ndarray:
    """ICA (Poh et al., 2010): FastICA on detrended, standardised RGB; the
    component whose spectrum has the most dominant peak inside the HR band is
    taken as the pulse."""
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning

    x = detrend(rgb)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    with warnings.catch_warnings():
        # Near-Gaussian noise sources often converge slowly; the spectral selection below
        # is what matters, so a non-converged unmixing is still usable.
        warnings.simplefilter("ignore", ConvergenceWarning)
        sources = FastICA(n_components=3, whiten="unit-variance", random_state=seed, max_iter=1000).fit_transform(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    best, best_score = 0, -np.inf
    for k in range(sources.shape[1]):
        power = np.abs(np.fft.rfft(sources[:, k])) ** 2
        score = power[in_band].max() / power[in_band].sum()  # peak dominance
        if score > best_score:
            best, best_score = k, score
    return sources[:, best]


METHODS = {"green": green, "ica": ica, "chrom": chrom, "pos": pos}
